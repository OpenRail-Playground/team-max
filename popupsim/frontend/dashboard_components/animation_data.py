"""Pure data transforms for the simulation animation tab.

This module intentionally avoids any Streamlit or Plotly imports so that the
geometry and timeline logic stays fully unit-testable. It turns the raw
``resource_locations`` / ``resource_states`` event logs and the scenario
``tracks`` / ``topology`` / ``workshops`` / ``train_schedule`` configuration
into:

* a synthetic yard layout with a **metric x-axis** (track length in metres) and
  a connecting throat at ``x = 0``,
* per-resource movement timelines (dwell + route-aware transit segments),
* per-frame rectangle centres + lengths + colours, ready to be drawn as flat,
  non-overlapping, train-like rectangles by Plotly.

Wagons are sized by their real length (from ``train_schedule.csv``); locomotives
use a default length. Within a track, occupants are packed into non-overlapping
slots: parking/storage tracks pack from the **far end** (furthest from the
throat) inward, retrofit tracks pack from the throat, and workshops use one slot
per retrofit bay.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any

import pandas as pd

# --- Visual constants (mirrors scenario_tab for consistency) ----------------

TRACK_TYPE_ORDER: list[str] = [
    'mainline',
    'collection',
    'retrofit',
    'workshop',
    'retrofitted',
    'parking',
    'rescource_parking',
]

TRACK_TYPE_COLORS: dict[str, str] = {
    'collection': '#e74c3c',
    'retrofit': '#f39c12',
    'workshop': '#27ae60',
    'retrofitted': '#9b59b6',
    'parking': '#34495e',
    'mainline': '#95a5a6',
    'rescource_parking': '#7f8c8d',
}

WAGON_COLOR_PENDING: str = '#e74c3c'  # red: not yet retrofitted
WAGON_COLOR_DONE: str = '#2ecc71'  # green: retrofitted
LOCO_COLOR: str = '#2c3e50'  # dark slate, distinct from wagons

# Track types that pack from the far end (furthest from the throat) inward.
STORAGE_TYPES: frozenset[str] = frozenset({'collection', 'parking', 'retrofitted', 'rescource_parking'})

# Geometry. The x-axis is in metres along the track; the throat sits at x = 0.
# (The y-axis is a lane index, so rectangle *height* is rendered nominally by
# the view layer for visibility, while rectangle *length* is true-to-metre.)
THROAT_X: float = 0.0
LANE_SPACING: float = 1.0
MIN_TRACK_LEN_M: float = 40.0  # floor so very short tracks stay visible
MAINLINE_DRAWN_M: float = 300.0  # mainline is transit-only; not drawn to scale

# Resource dimensions (metres).
DEFAULT_WAGON_LENGTH_M: float = 16.0
LOCO_LENGTH_M: float = 19.0
SLOT_GAP_M: float = 2.0  # coupling gap added to the widest resource for slots

# Fallback transit duration (simulation minutes) used when an explicit
# departure time is not available in the event log (e.g. locomotives, which
# record arrivals via ``previous_location`` but no move-start row).
DEFAULT_TRANSIT_MIN: float = 12.0


# --- Layout model -----------------------------------------------------------


@dataclass(frozen=True)
class TrackLayout:  # pylint: disable=too-many-instance-attributes
    """Geometry for a single track lane in the schematic yard."""

    track_id: str
    track_type: str
    lane_y: float
    x_start: float
    x_end: float
    length_m: float
    color: str
    is_workshop: bool = False
    bays: int | None = None


@dataclass(frozen=True)
class YardLayout:
    """Full schematic layout: track lanes plus the connecting throat."""

    tracks: dict[str, TrackLayout]
    throat_x: float
    x_max: float
    y_min: float
    y_max: float


# --- Resource timeline model ------------------------------------------------


@dataclass
class Dwell:
    """A stationary span of a resource sitting on a track."""

    track: str
    t_arrive: float
    t_depart: float  # math.inf for the final (open-ended) dwell
    slot: int = 0


@dataclass
class Move:
    """A transit span of a resource travelling between two tracks."""

    t_depart: float
    t_arrive: float
    route: tuple[str, ...]
    from_track: str
    to_track: str
    from_slot: int = 0
    to_slot: int = 0


@dataclass
class ResourceTrack:  # pylint: disable=too-many-instance-attributes
    """Full ordered timeline (dwells + moves) for a single resource."""

    resource_id: str
    resource_type: str
    length_m: float = DEFAULT_WAGON_LENGTH_M
    dwells: list[Dwell] = field(default_factory=list)
    moves: list[Move] = field(default_factory=list)
    t_retrofitted: float | None = None
    first_seen: float = 0.0
    last_seen: float = 0.0


# --- Frame model ------------------------------------------------------------


@dataclass(frozen=True)
class FrameData:  # pylint: disable=too-many-instance-attributes
    """Rectangle centres + lengths for a single animation frame."""

    t: float
    datetime_label: str
    wagon_x: list[float]
    wagon_y: list[float]
    wagon_len: list[float]
    wagon_color: list[str]
    wagon_ids: list[str]
    loco_x: list[float]
    loco_y: list[float]
    loco_len: list[float]
    loco_ids: list[str]


# === Yard layout ============================================================


def _natural_key(track_id: str) -> tuple[str, int]:
    """Return a sort key that orders ``parking2`` before ``parking10``."""
    digits = ''.join(ch for ch in track_id if ch.isdigit())
    prefix = ''.join(ch for ch in track_id if not ch.isdigit())
    return (prefix, int(digits) if digits else 0)


def _edge_length(topology: dict[str, Any], edges: list[str]) -> float:
    """Sum the lengths of the given topology edges (0.0 if unknown)."""
    edge_map = topology.get('edges', {}) if topology else {}
    total = 0.0
    for edge in edges:
        info = edge_map.get(edge)
        if isinstance(info, dict) and 'length' in info:
            total += float(info['length'])
    return total


def build_layout(  # pylint: disable=too-many-locals
    tracks_config: list[dict[str, Any]],
    topology: dict[str, Any],
    workshops_config: list[dict[str, Any]] | None = None,
) -> YardLayout:
    """Build a schematic yard layout from scenario configuration.

    Tracks are grouped top-to-bottom by :data:`TRACK_TYPE_ORDER`; within a
    group they are ordered naturally by id. The x-axis is metric: each track
    spans ``[0, length_m]`` metres from the throat. The mainline is transit-only
    and is drawn at a fixed (non-scaled) length. Workshop tracks are tagged and
    annotated with their bay count.
    """
    workshops_config = workshops_config or []
    bays_by_id: dict[str, int] = {}
    for shop in workshops_config:
        shop_id = str(shop.get('id', ''))
        if shop_id:
            bays_by_id[shop_id] = int(shop.get('retrofit_stations', 0))

    enriched: list[dict[str, Any]] = []
    for track in tracks_config:
        track_id = str(track['id'])
        track_type = str(track.get('type', 'parking'))
        length_m = _edge_length(topology, track.get('edges', [track_id]))
        enriched.append({'id': track_id, 'type': track_type, 'length_m': length_m})

    def order_index(track_type: str) -> int:
        return TRACK_TYPE_ORDER.index(track_type) if track_type in TRACK_TYPE_ORDER else len(TRACK_TYPE_ORDER)

    enriched.sort(key=lambda t: (order_index(t['type']), _natural_key(t['id'])))

    n_lanes = len(enriched)
    tracks: dict[str, TrackLayout] = {}
    for index, track in enumerate(enriched):
        lane_y = (n_lanes - 1 - index) * LANE_SPACING
        track_type = track['type']
        track_id = track['id']
        length_m = track['length_m']
        drawn = MAINLINE_DRAWN_M if track_type == 'mainline' else max(MIN_TRACK_LEN_M, length_m)
        tracks[track_id] = TrackLayout(
            track_id=track_id,
            track_type=track_type,
            lane_y=lane_y,
            x_start=THROAT_X,
            x_end=THROAT_X + drawn,
            length_m=length_m,
            color=TRACK_TYPE_COLORS.get(track_type, '#7f7f7f'),
            is_workshop=track_type == 'workshop',
            bays=bays_by_id.get(track_id),
        )

    y_values = [t.lane_y for t in tracks.values()] or [0.0]
    x_values = [t.x_end for t in tracks.values()] or [MAINLINE_DRAWN_M]
    return YardLayout(
        tracks=tracks,
        throat_x=THROAT_X,
        x_max=max(x_values),
        y_min=min(y_values),
        y_max=max(y_values),
    )


# === Movement-segment extraction ============================================


def _parse_route(value: Any) -> tuple[str, ...]:
    """Parse a pipe-delimited ``route_path`` value into a tuple of track ids."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    return tuple(part for part in text.split('|') if part)


def _is_blank(value: Any) -> bool:
    """Return True for missing / empty cell values."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ''


def _retrofit_times(states: pd.DataFrame | None) -> dict[str, float]:
    """Map each wagon id to the timestamp it first reaches ``retrofitted``."""
    if states is None or states.empty or 'state' not in states.columns:
        return {}
    done = states[states['state'] == 'retrofitted']
    if done.empty:
        return {}
    grouped = done.groupby('resource_id')['timestamp'].min()
    return {str(rid): float(ts) for rid, ts in grouped.items()}


def wagon_lengths(train_schedule: pd.DataFrame | None) -> dict[str, float]:
    """Map each wagon id to its real length (metres) from the train schedule."""
    if train_schedule is None or train_schedule.empty:
        return {}
    if 'wagon_id' not in train_schedule.columns or 'length' not in train_schedule.columns:
        return {}
    lengths: dict[str, float] = {}
    for _, row in train_schedule.iterrows():
        try:
            lengths[str(row['wagon_id'])] = float(row['length'])
        except (ValueError, TypeError):
            continue
    return lengths


def _build_timeline(resource_id: str, resource_type: str, rows: pd.DataFrame) -> ResourceTrack | None:
    """Build a single resource timeline from its ordered location rows."""
    rows = rows.sort_values('timestamp')
    track = ResourceTrack(resource_id=resource_id, resource_type=resource_type)

    cur_track: str | None = None
    cur_arrive = 0.0
    pending_route: tuple[str, ...] = ()
    pending_depart: float | None = None

    for _, row in rows.iterrows():
        ts = float(row['timestamp'])
        location = None if _is_blank(row.get('location')) else str(row['location'])
        route = _parse_route(row.get('route_path'))

        if cur_track is None:
            if location is None:
                continue
            cur_track, cur_arrive = location, ts
            track.first_seen = ts
            continue

        if route:  # move-start marker (wagons)
            pending_route, pending_depart = route, ts

        if location is not None and location != cur_track:
            depart = pending_depart if pending_depart is not None else max(cur_arrive, ts - DEFAULT_TRANSIT_MIN)
            depart = min(max(depart, cur_arrive), ts)
            track.dwells.append(Dwell(track=cur_track, t_arrive=cur_arrive, t_depart=depart))
            route_used = pending_route if pending_route else (cur_track, location)
            track.moves.append(
                Move(t_depart=depart, t_arrive=ts, route=route_used, from_track=cur_track, to_track=location)
            )
            cur_track, cur_arrive = location, ts
            pending_route, pending_depart = (), None

    if cur_track is not None:
        track.dwells.append(Dwell(track=cur_track, t_arrive=cur_arrive, t_depart=math.inf))
        track.last_seen = float(rows['timestamp'].max())
        return track
    return None


def extract_timelines(
    resource_locations: pd.DataFrame | None,
    resource_states: pd.DataFrame | None,
    lengths: dict[str, float] | None = None,
) -> dict[str, ResourceTrack]:
    """Extract per-resource dwell/move timelines from the event logs.

    Wagons use the populated ``route_path`` rows to determine the departure
    time and the path taken. Locomotives (which only record arrivals via
    ``previous_location``) fall back to :data:`DEFAULT_TRANSIT_MIN`. ``lengths``
    maps wagon ids to real lengths (metres); missing wagons fall back to
    :data:`DEFAULT_WAGON_LENGTH_M` and locomotives to :data:`LOCO_LENGTH_M`.
    """
    if resource_locations is None or resource_locations.empty:
        return {}

    lengths = lengths or {}
    retrofit = _retrofit_times(resource_states)
    timelines: dict[str, ResourceTrack] = {}
    for resource_id, rows in resource_locations.groupby('resource_id'):
        rid = str(resource_id)
        rtype = str(rows['resource_type'].iloc[0]) if 'resource_type' in rows.columns else 'wagon'
        timeline = _build_timeline(rid, rtype, rows)
        if timeline is None:
            continue
        timeline.t_retrofitted = retrofit.get(rid)
        timeline.length_m = LOCO_LENGTH_M if rtype == 'locomotive' else lengths.get(rid, DEFAULT_WAGON_LENGTH_M)
        timelines[rid] = timeline
    assign_slots(timelines)
    return timelines


# === Slot assignment + position resolver ====================================


def assign_slots(timelines: dict[str, ResourceTrack]) -> dict[str, int]:
    """Assign a non-overlapping slot index to every dwell, per track.

    Uses interval-graph greedy colouring: a slot is reused once its previous
    occupant has departed. Returns the number of slots required per track.
    """
    by_track: dict[str, list[tuple[float, float, Dwell]]] = defaultdict(list)
    for rt in timelines.values():
        for dwell in rt.dwells:
            by_track[dwell.track].append((dwell.t_arrive, dwell.t_depart, dwell))

    slot_counts: dict[str, int] = {}
    for track_id, items in by_track.items():
        items.sort(key=lambda item: item[0])
        slot_free_at: list[float] = []
        for t_arrive, t_depart, dwell in items:
            assigned: int | None = None
            for i, free_t in enumerate(slot_free_at):
                if free_t <= t_arrive:
                    assigned = i
                    slot_free_at[i] = t_depart
                    break
            if assigned is None:
                assigned = len(slot_free_at)
                slot_free_at.append(t_depart)
            dwell.slot = assigned
        slot_counts[track_id] = max(1, len(slot_free_at))

    for rt in timelines.values():
        for i, move in enumerate(rt.moves):
            move.from_slot = rt.dwells[i].slot
            move.to_slot = rt.dwells[i + 1].slot
    return slot_counts


def compute_slot_width(timelines: dict[str, ResourceTrack]) -> float:
    """Return the uniform slot width (metres): widest resource + coupling gap."""
    widest = max((rt.length_m for rt in timelines.values()), default=DEFAULT_WAGON_LENGTH_M)
    return max(widest, LOCO_LENGTH_M) + SLOT_GAP_M


def _slot_pos(layout: YardLayout, track_id: str, slot: int, slot_width: float) -> tuple[float, float]:
    """Return the (x, y) centre of a slot on a track, packed by track type.

    Storage tracks pack from the far end (furthest from the throat) inward;
    retrofit tracks pack from the throat; workshops use one slot per bay; the
    mainline (transit only) returns its centre.
    """
    tl = layout.tracks[track_id]
    if tl.track_type == 'workshop':
        n_bays = max(1, tl.bays or 1)
        x = tl.x_end * (slot + 0.5) / n_bays
    elif tl.track_type in STORAGE_TYPES:
        x = tl.x_end - (slot + 0.5) * slot_width  # far-end packing
    elif tl.track_type == 'retrofit':
        x = (slot + 0.5) * slot_width  # near-throat packing
    else:  # mainline / unknown: centre
        x = tl.x_end / 2.0

    lo = min(slot_width * 0.5, tl.x_end / 2.0)
    hi = max(tl.x_end - slot_width * 0.5, tl.x_end / 2.0)
    return (max(lo, min(x, hi)), tl.lane_y)


def _move_waypoints(layout: YardLayout, move: Move, slot_width: float) -> list[tuple[float, float]]:
    """Build the route-aware waypoint path for a transit segment.

    Path: source slot -> throat at each intermediate lane height -> dest slot.
    Routing through the throat at each route lane reproduces the
    "out to the lead, along the throat, into the destination" motion.
    """
    points: list[tuple[float, float]] = [_slot_pos(layout, move.from_track, move.from_slot, slot_width)]
    for track_id in move.route:
        tl = layout.tracks.get(track_id)
        if tl is not None:
            points.append((layout.throat_x, tl.lane_y))
    points.append(_slot_pos(layout, move.to_track, move.to_slot, slot_width))
    return points


def _interp_path(points: list[tuple[float, float]], frac: float) -> tuple[float, float]:
    """Linearly interpolate a position at ``frac`` (0..1) along a polyline."""
    if frac <= 0.0 or len(points) == 1:
        return points[0]
    if frac >= 1.0:
        return points[-1]
    seg_lengths = [math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    total = sum(seg_lengths)
    if total == 0.0:
        return points[0]
    target = frac * total
    walked = 0.0
    for i, seg_len in enumerate(seg_lengths):
        if walked + seg_len >= target:
            local = 0.0 if seg_len == 0 else (target - walked) / seg_len
            (x0, y0), (x1, y1) = points[i], points[i + 1]
            return (x0 + local * (x1 - x0), y0 + local * (y1 - y0))
        walked += seg_len
    return points[-1]


def position_at(rt: ResourceTrack, layout: YardLayout, slot_width: float, t: float) -> tuple[float, float] | None:
    """Resolve a resource's (x, y) centre at simulation time ``t``.

    Returns ``None`` before the resource first appears. While stationary it
    sits in its slot; while moving it is interpolated along its route.
    """
    if t < rt.first_seen:
        return None
    for dwell in rt.dwells:
        if dwell.track in layout.tracks and dwell.t_arrive <= t <= dwell.t_depart:
            return _slot_pos(layout, dwell.track, dwell.slot, slot_width)
    for move in rt.moves:
        if move.t_depart <= t <= move.t_arrive:
            denom = move.t_arrive - move.t_depart
            frac = 0.0 if denom <= 0 else (t - move.t_depart) / denom
            return _interp_path(_move_waypoints(layout, move, slot_width), frac)
    last = rt.dwells[-1]
    if last.track in layout.tracks:
        return _slot_pos(layout, last.track, last.slot, slot_width)
    return None


# === Frame builder ==========================================================


def sim_time_bounds(timelines: dict[str, ResourceTrack]) -> tuple[float, float]:
    """Return the (start, end) simulation time spanning all resources."""
    if not timelines:
        return (0.0, 0.0)
    start = min(rt.first_seen for rt in timelines.values())
    end = max(rt.last_seen for rt in timelines.values())
    if end <= start:
        end = start + 1.0
    return (start, end)


def _wagon_color(rt: ResourceTrack, t: float) -> str:
    """Return the wagon colour at time ``t`` (green once retrofitted)."""
    if rt.t_retrofitted is not None and t >= rt.t_retrofitted:
        return WAGON_COLOR_DONE
    return WAGON_COLOR_PENDING


def _empty_acc() -> dict[str, list[Any]]:
    """Return empty per-category arrays for a frame under construction."""
    return {
        'wagon_x': [],
        'wagon_y': [],
        'wagon_len': [],
        'wagon_color': [],
        'wagon_ids': [],
        'loco_x': [],
        'loco_y': [],
        'loco_len': [],
        'loco_ids': [],
    }


def build_frames(  # pylint: disable=too-many-locals
    layout: YardLayout,
    timelines: dict[str, ResourceTrack],
    num_frames: int,
    bounds: tuple[float, float] | None = None,
) -> list[FrameData]:
    """Build the per-frame rectangle arrays over a uniform simulation-time grid.

    ``num_frames`` (the user-selected resolution) sets the grid density. Each
    frame carries wagon rectangle centres + lengths + colours and locomotive
    rectangle centres + lengths.
    """
    if not timelines or num_frames < 1:
        return []
    t_start, t_end = bounds if bounds is not None else sim_time_bounds(timelines)
    span = t_end - t_start
    slot_width = compute_slot_width(timelines)

    frames: list[FrameData] = []
    for index in range(num_frames):
        frac = 0.0 if num_frames == 1 else index / (num_frames - 1)
        t = t_start + frac * span
        acc = _empty_acc()
        for rt in timelines.values():
            pos = position_at(rt, layout, slot_width, t)
            if pos is None:
                continue
            if rt.resource_type == 'locomotive':
                acc['loco_x'].append(pos[0])
                acc['loco_y'].append(pos[1])
                acc['loco_len'].append(rt.length_m)
                acc['loco_ids'].append(rt.resource_id)
            else:
                acc['wagon_x'].append(pos[0])
                acc['wagon_y'].append(pos[1])
                acc['wagon_len'].append(rt.length_m)
                acc['wagon_color'].append(_wagon_color(rt, t))
                acc['wagon_ids'].append(rt.resource_id)
        frames.append(
            FrameData(
                t=t,
                datetime_label=_format_time(t_start, t),
                wagon_x=acc['wagon_x'],
                wagon_y=acc['wagon_y'],
                wagon_len=acc['wagon_len'],
                wagon_color=acc['wagon_color'],
                wagon_ids=acc['wagon_ids'],
                loco_x=acc['loco_x'],
                loco_y=acc['loco_y'],
                loco_len=acc['loco_len'],
                loco_ids=acc['loco_ids'],
            )
        )
    return frames


def _format_time(t_start: float, t: float) -> str:
    """Format a simulation timestamp (minutes) as an elapsed D/H:M label."""
    elapsed = t - t_start
    days = int(elapsed // (24 * 60))
    hours = int((elapsed % (24 * 60)) // 60)
    minutes = int(elapsed % 60)
    if days > 0:
        return f'Day {days + 1} {hours:02d}:{minutes:02d}'
    return f'{hours:02d}:{minutes:02d}'
