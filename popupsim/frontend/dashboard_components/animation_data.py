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
use a default length. Each parked resource is given a **fixed** position when it
arrives (it never shifts while stationary): occupants pack flush, end-to-end
(no gap — coupled wagons have no space between them). Parking/storage tracks
pack from the far end (furthest from the throat) inward, retrofit tracks pack
from the throat, and workshops use one slot per retrofit bay. A departing
resource leaves its slot behind (parked wagons do not slide), and the freed
space is reused by later arrivals. Adjacent wagons are told apart by alternating
fill shade plus a border rather than by a gap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any

import pandas as pd

from dashboard_components import routes_graph as rg

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

WAGON_COLOR_PENDING: str = '#0072B2'  # blue (Okabe-Ito): not yet retrofitted
WAGON_COLOR_DONE: str = '#009E73'  # green (Okabe-Ito): retrofitted
LOCO_COLOR: str = '#2c3e50'  # dark slate, distinct from wagons

# Alternating fill shades so flush (gap-less) neighbours stay individually
# countable. Index 0 is the base colour; index 1 is a lighter sibling.
WAGON_SHADES_PENDING: tuple[str, str] = ('#0072B2', '#56B4E9')  # blue / sky-blue
WAGON_SHADES_DONE: tuple[str, str] = ('#009E73', '#66C2A5')  # green / light green

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
# Wagons pack flush (no gap): in reality there is no space between coupled
# wagons. Individual wagons are told apart by alternating fill shade + border
# rather than by a gap. Kept as a tunable constant (0 = flush).
UNIFORM_GAP_M: float = 0.0

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
    cluster: str = 'local'
    throat_x: float = 0.0  # x of the connecting (ladder) end of the track
    zone: str = 'single'  # 'single' | 'left' | 'middle' | 'right'


@dataclass(frozen=True)
class YardLayout:  # pylint: disable=too-many-instance-attributes
    """Full schematic layout: track lanes plus connecting throat(s)."""

    tracks: dict[str, TrackLayout]
    throat_x: float
    x_max: float
    y_min: float
    y_max: float
    mode: str = 'single'  # 'single' (one vertical throat) | 'zones' (left/Mainline/right)
    left_throat_x: float = 0.0
    right_throat_x: float = 0.0
    corridor_y: float = 0.0


# --- Resource timeline model ------------------------------------------------


@dataclass
class Dwell:
    """A stationary span of a resource sitting on a track (fixed centre)."""

    track: str
    t_arrive: float
    t_depart: float  # math.inf for the final (open-ended) dwell
    center_x: float = 0.0


@dataclass
class Move:
    """A transit span of a resource travelling between two tracks."""

    t_depart: float
    t_arrive: float
    route: tuple[str, ...]
    from_track: str
    to_track: str
    anchor_from_x: float = 0.0
    anchor_to_x: float = 0.0


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
class FrameStats:
    """Aggregate live statistics for a single animation frame."""

    to_retrofit: int  # wagons present and not yet retrofitted
    present_retrofitted: int  # retrofitted wagons currently in the system
    cumulative_retrofitted: int  # wagons retrofitted up to this time
    cumulative_rejected: int  # wagons rejected up to this time
    utilization: dict[str, float]  # track id -> occupied length / real length


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
    wagon_shade: list[int]
    loco_x: list[float]
    loco_y: list[float]
    loco_len: list[float]
    loco_ids: list[str]
    stats: FrameStats


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


def _resolve_track_meta(
    track_id: str,
    tracks_by_id: dict[str, dict[str, Any]],
    topology: dict[str, Any],
    bays_by_id: dict[str, int],
    default_ws_len: float,
) -> tuple[str, float, int | None]:
    """Resolve (track_type, length_m, bays) for a track id, robust to id drift.

    Handles scenarios where the runtime/route id (e.g. workshop ``WS_01``) does
    not match the ``tracks.json`` id (``WS1``): such ids fall back to the
    workshop config for type/bays and a default workshop length.
    """
    if track_id in tracks_by_id:
        track = tracks_by_id[track_id]
        track_type = str(track.get('type', 'parking'))
        length_m = _edge_length(topology, track.get('edges', [track_id]))
        bays = bays_by_id.get(track_id) if track_type == 'workshop' else None
        return track_type, length_m, bays
    if track_id in bays_by_id:  # a workshop id with no matching track entry
        return 'workshop', default_ws_len, bays_by_id[track_id]
    if track_id == rg.MAINLINE:
        return 'mainline', _edge_length(topology, [track_id]), None
    return 'parking', _edge_length(topology, [track_id]) or MIN_TRACK_LEN_M, None


def build_layout(  # pylint: disable=too-many-locals
    tracks_config: list[dict[str, Any]],
    topology: dict[str, Any],
    workshops_config: list[dict[str, Any]] | None = None,
    route_graph: rg.RouteGraph | None = None,
    active_ids: list[str] | None = None,
) -> YardLayout:
    """Build a schematic yard layout from scenario configuration.

    Lanes are grouped top-to-bottom by connectivity cluster (remote / arrival
    side on top, the ``Mainline`` spine in the middle, the local retrofit yard
    below) and then by :data:`TRACK_TYPE_ORDER` and natural id. When no
    ``route_graph`` is given every track is treated as ``local`` so the ordering
    reduces to the original type-based layout. The x-axis is metric: each track
    spans ``[0, length_m]`` metres from the throat; the mainline is transit-only
    and drawn at a fixed (non-scaled) length.

    ``active_ids`` (when given) selects which tracks to lay out — typically the
    union of ids referenced by routes and the event logs — so runtime ids that
    differ from ``tracks.json`` (e.g. workshop ``WS_01`` vs ``WS1``) still render.
    """
    workshops_config = workshops_config or []
    tracks_by_id = {str(t['id']): t for t in tracks_config}
    bays_by_id = {str(s['id']): int(s.get('retrofit_stations', 0)) for s in workshops_config if s.get('id')}
    ws_lengths = [
        _edge_length(topology, t.get('edges', [t['id']])) for t in tracks_config if t.get('type') == 'workshop'
    ]
    default_ws_len = max((length for length in ws_lengths if length > 0), default=260.0)

    ids = active_ids if active_ids else [str(t['id']) for t in tracks_config]
    cluster_of = route_graph.cluster if route_graph else {}

    enriched: list[dict[str, Any]] = []
    for track_id in dict.fromkeys(ids):  # de-duplicate, preserve order
        track_type, length_m, bays = _resolve_track_meta(track_id, tracks_by_id, topology, bays_by_id, default_ws_len)
        enriched.append(
            {
                'id': track_id,
                'type': track_type,
                'length_m': length_m,
                'bays': bays,
                'cluster': cluster_of.get(track_id, 'local'),
            }
        )

    if route_graph is not None and any(t['cluster'] == 'remote' for t in enriched):
        return _layout_zones(enriched)
    return _layout_single(enriched)


def _order_index(track_type: str) -> int:
    """Return the top-to-bottom rank of a track type."""
    return TRACK_TYPE_ORDER.index(track_type) if track_type in TRACK_TYPE_ORDER else len(TRACK_TYPE_ORDER)


def _drawn_len(track_type: str, length_m: float) -> float:
    """Return the drawn x-extent (metres) of a track (mainline is fixed)."""
    return MAINLINE_DRAWN_M if track_type == 'mainline' else max(MIN_TRACK_LEN_M, length_m)


def _make_track(
    track: dict[str, Any], lane_y: float, x_range: tuple[float, float], throat_x: float, zone: str
) -> TrackLayout:
    """Build a TrackLayout from an enriched track dict and geometry."""
    track_type = track['type']
    return TrackLayout(
        track_id=track['id'],
        track_type=track_type,
        lane_y=lane_y,
        x_start=x_range[0],
        x_end=x_range[1],
        length_m=track['length_m'],
        color=TRACK_TYPE_COLORS.get(track_type, '#7f7f7f'),
        is_workshop=track_type == 'workshop',
        bays=track['bays'],
        cluster=track['cluster'],
        throat_x=throat_x,
        zone=zone,
    )


def _layout_single(enriched: list[dict[str, Any]]) -> YardLayout:
    """Lay out all tracks as a single vertically-stacked column (legacy)."""
    enriched.sort(key=lambda t: (rg.CLUSTER_RANK.get(t['cluster'], 2), _order_index(t['type']), _natural_key(t['id'])))
    n_lanes = len(enriched)
    tracks: dict[str, TrackLayout] = {}
    for index, track in enumerate(enriched):
        drawn = _drawn_len(track['type'], track['length_m'])
        tracks[track['id']] = _make_track(
            track, (n_lanes - 1 - index) * LANE_SPACING, (THROAT_X, THROAT_X + drawn), THROAT_X, 'single'
        )
    y_values = [t.lane_y for t in tracks.values()] or [0.0]
    x_values = [t.x_end for t in tracks.values()] or [MAINLINE_DRAWN_M]
    return YardLayout(
        tracks=tracks,
        throat_x=THROAT_X,
        x_max=max(x_values),
        y_min=min(y_values),
        y_max=max(y_values),
        mode='single',
        left_throat_x=THROAT_X,
    )


def _layout_zones(enriched: list[dict[str, Any]]) -> YardLayout:  # pylint: disable=too-many-locals
    """Lay out tracks in three columns: local yard | Mainline | remote storage.

    The local yard sits on the left (ladder on its right edge), the Mainline is
    a fixed-width corridor in the middle (<=25% of the width), and the remote
    storage sits on the right (ladder on its left edge). Cross-zone moves pass
    through the Mainline corridor.
    """
    left = sorted(
        (t for t in enriched if t['type'] != 'mainline' and t['cluster'] in ('local', 'hub')),
        key=lambda t: (_order_index(t['type']), _natural_key(t['id'])),
    )
    right = sorted(
        (t for t in enriched if t['type'] != 'mainline' and t['cluster'] == 'remote'),
        key=lambda t: (_order_index(t['type']), _natural_key(t['id'])),
    )
    middle = [t for t in enriched if t['type'] == 'mainline']

    left_w = max((_drawn_len(t['type'], t['length_m']) for t in left), default=MIN_TRACK_LEN_M)
    right_w = max((_drawn_len(t['type'], t['length_m']) for t in right), default=MIN_TRACK_LEN_M)
    middle_w = (left_w + right_w) / 3.0  # exactly 25% of total width
    left_throat = left_w
    right_throat = left_w + middle_w
    total_w = left_w + middle_w + right_w
    n_lanes = max(len(left), len(right), 1)
    corridor_y = (n_lanes - 1) / 2.0 * LANE_SPACING

    tracks: dict[str, TrackLayout] = {}

    def lane_for(group: list[dict[str, Any]], index: int) -> float:
        offset = (n_lanes - len(group)) / 2.0  # vertically centre the shorter column
        return (n_lanes - 1 - index - offset) * LANE_SPACING

    for index, track in enumerate(left):
        drawn = _drawn_len(track['type'], track['length_m'])
        tracks[track['id']] = _make_track(
            track, lane_for(left, index), (left_throat - drawn, left_throat), left_throat, 'left'
        )
    for index, track in enumerate(right):
        drawn = _drawn_len(track['type'], track['length_m'])
        tracks[track['id']] = _make_track(
            track, lane_for(right, index), (right_throat, right_throat + drawn), right_throat, 'right'
        )
    for track in middle:
        tracks[track['id']] = _make_track(
            track, corridor_y, (left_throat, right_throat), (left_throat + right_throat) / 2.0, 'middle'
        )

    y_values = [t.lane_y for t in tracks.values()] or [0.0]
    return YardLayout(
        tracks=tracks,
        throat_x=left_throat,
        x_max=total_w,
        y_min=min(y_values),
        y_max=max(y_values),
        mode='zones',
        left_throat_x=left_throat,
        right_throat_x=right_throat,
        corridor_y=corridor_y,
    )


# === Movement-segment extraction ============================================


def active_track_ids(resource_locations: pd.DataFrame | None, route_graph: rg.RouteGraph | None) -> list[str]:
    """Return the union of track ids referenced by routes and the event log.

    This is the set of lanes worth drawing — it includes runtime ids (e.g.
    workshop ``WS_01``) that may not appear verbatim in ``tracks.json``.
    """
    ids: set[str] = set()
    if route_graph is not None:
        ids |= set(route_graph.track_ids)
    if resource_locations is not None and not resource_locations.empty and 'location' in resource_locations.columns:
        ids |= {str(v) for v in resource_locations['location'].dropna().unique() if str(v).strip()}
    return sorted(ids)


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


def rejected_times(rejected_wagons: pd.DataFrame | None) -> list[float]:
    """Return sorted rejection timestamps (minutes) from the rejected-wagons log."""
    if rejected_wagons is None or rejected_wagons.empty or 'timestamp' not in rejected_wagons.columns:
        return []
    times: list[float] = []
    for value in rejected_wagons['timestamp']:
        try:
            times.append(float(value))
        except (ValueError, TypeError):
            continue
    return sorted(times)


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
    return timelines


# === Stable position allocation =============================================


def _merge_free(free: list[list[float]]) -> None:
    """Merge adjacent free intervals (in place); each item is ``[start, len]``."""
    free.sort()
    merged: list[list[float]] = []
    for start, length in free:
        if merged and abs(merged[-1][0] + merged[-1][1] - start) < 1e-6:
            merged[-1][1] += length
        else:
            merged.append([start, length])
    free[:] = merged


def _assign_linear(  # pylint: disable=too-many-locals
    tl: TrackLayout, items: list[tuple[float, float, float, Dwell]], gap: float
) -> None:
    """Assign fixed centres for a linear track via an event-driven allocator.

    Each dwell reserves ``length + gap`` of axis space when it arrives (first
    fit from the packing end) and frees it on departure, so parked resources
    keep a constant position and adjacent ones sit flush (``gap`` defaults to 0).
    Storage tracks pack from the outer (far) end; other tracks pack from the
    throat. A trailing region absorbs any over-capacity overflow sequentially so
    wagons never stack on the same coordinate. Works for any zone via the
    track's ``throat_x`` / endpoints.
    """
    track_len = tl.x_end - tl.x_start
    outer_x = tl.x_start if abs(tl.throat_x - tl.x_end) < 1e-9 else tl.x_end
    if tl.track_type in STORAGE_TYPES:
        origin, direction = outer_x, (1.0 if tl.throat_x > outer_x else -1.0)
    else:  # retrofit (and similar) pack from the throat
        origin, direction = tl.throat_x, (1.0 if outer_x > tl.throat_x else -1.0)

    events: list[tuple[float, int, int]] = []
    for i, (t_arrive, t_depart, _length, _dwell) in enumerate(items):
        events.append((t_arrive, 0, i))  # arrival (before departures at the same time)
        events.append((t_depart, 1, i))  # departure
    events.sort(key=lambda e: (e[0], e[1]))

    # The second segment is an (effectively unbounded) overflow region so that
    # over-capacity arrivals pack sequentially beyond the track instead of
    # collapsing onto a single coordinate.
    free: list[list[float]] = [[0.0, track_len], [track_len, track_len * 10.0 + 1000.0]]
    alloc: dict[int, list[float]] = {}
    for _time, kind, i in events:
        _ta, _td, length, dwell = items[i]
        need = length + gap
        if kind == 0:
            start = _first_fit(free, need, track_len)
            alloc[i] = [start, need]
            dwell.center_x = origin + direction * (start + length / 2.0)
        elif i in alloc:
            free.append(alloc.pop(i))
            _merge_free(free)


def _first_fit(free: list[list[float]], need: float, capacity: float) -> float:
    """Return the start offset of the first free interval that fits ``need``."""
    for seg in free:
        if seg[1] >= need - 1e-9:
            start = seg[0]
            seg[0] += need
            seg[1] -= need
            return start
    return capacity  # overflow: stack beyond the usable length (rare)


def _peak_concurrency(events: list[tuple[float, int, int]]) -> int:
    """Return the maximum number of simultaneously-present items over the events."""
    current = peak = 0
    for _time, kind, _i in events:
        current += 1 if kind == 0 else -1
        peak = max(peak, current)
    return peak


def _assign_workshop(tl: TrackLayout, items: list[tuple[float, float, float, Dwell]]) -> None:
    """Assign fixed slot centres for a workshop, always inside the box.

    There is one slot per retrofit bay; if more wagons are ever present at once
    than there are bays, the box is divided into as many equal slots as the peak
    occupancy so every wagon stays within ``[x_start, x_end]`` and none overlap.
    Arrivals are processed before departures at tied timestamps so a momentary
    (zero-duration) visit never leaks its slot.
    """
    events: list[tuple[float, int, int]] = []
    for i, (t_arrive, t_depart, _length, _dwell) in enumerate(items):
        events.append((t_arrive, 0, i))
        events.append((t_depart, 1, i))
    events.sort(key=lambda e: (e[0], e[1]))

    n_slots = max(1, tl.bays or 1, _peak_concurrency(events))
    width = tl.x_end - tl.x_start
    free_slots = list(range(n_slots))
    assigned: dict[int, int] = {}
    for _time, kind, i in events:
        _ta, _td, _length, dwell = items[i]
        if kind == 0:
            slot = free_slots.pop(0) if free_slots else n_slots - 1
            assigned[i] = slot
            dwell.center_x = tl.x_start + width * (slot + 0.5) / n_slots
        elif i in assigned:
            free_slots.append(assigned.pop(i))
            free_slots.sort()


def _anchor_moves(timelines: dict[str, ResourceTrack]) -> None:
    """Set each move's from/to anchor to its bracketing dwell centres."""
    for rt in timelines.values():
        for i, move in enumerate(rt.moves):
            move.anchor_from_x = rt.dwells[i].center_x
            move.anchor_to_x = rt.dwells[i + 1].center_x


def assign_positions(layout: YardLayout, timelines: dict[str, ResourceTrack], gap: float = UNIFORM_GAP_M) -> None:
    """Assign a fixed centre to every dwell and anchor every move (in place).

    Positions are computed once per dwell so a stationary resource never shifts.
    """
    per_track: dict[str, list[tuple[float, float, float, Dwell]]] = defaultdict(list)
    for rt in timelines.values():
        for dwell in rt.dwells:
            per_track[dwell.track].append((dwell.t_arrive, dwell.t_depart, rt.length_m, dwell))

    for track_id, items in per_track.items():
        tl = layout.tracks.get(track_id)
        if tl is None:
            for *_unused, dwell in items:
                dwell.center_x = 0.0
        elif tl.track_type == 'workshop':
            _assign_workshop(tl, items)
        elif tl.track_type in STORAGE_TYPES or tl.track_type == 'retrofit':
            _assign_linear(tl, items, gap)
        else:  # mainline / unknown: centre
            for *_unused, dwell in items:
                dwell.center_x = tl.x_end / 2.0

    _anchor_moves(timelines)


# === Position resolver ======================================================


def _move_waypoints(layout: YardLayout, move: Move) -> list[tuple[float, float]]:
    """Build the route-aware waypoint path for a transit segment.

    In single-column mode the path routes out to the shared vertical throat at
    each route lane. In three-zone mode it routes to the source track's ladder,
    and — for cross-zone moves — along the Mainline corridor to the destination
    zone's ladder, so the resource visibly passes through the Mainline.
    """
    a = layout.tracks.get(move.from_track)
    b = layout.tracks.get(move.to_track)
    lane_from = a.lane_y if a is not None else 0.0
    lane_to = b.lane_y if b is not None else 0.0
    start = (move.anchor_from_x, lane_from)
    end = (move.anchor_to_x, lane_to)

    if layout.mode != 'zones' or a is None or b is None:
        points: list[tuple[float, float]] = [start]
        for track_id in move.route:
            tl = layout.tracks.get(track_id)
            if tl is not None:
                points.append((layout.throat_x, tl.lane_y))
        points.append(end)
        return points

    points = [start, (a.throat_x, lane_from)]
    if a.zone == b.zone:
        points.append((a.throat_x, lane_to))
    else:  # cross-zone: traverse the Mainline corridor between the two ladders
        points.extend(
            [
                (a.throat_x, layout.corridor_y),
                (b.throat_x, layout.corridor_y),
                (b.throat_x, lane_to),
            ]
        )
    points.append(end)
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


def position_at(rt: ResourceTrack, layout: YardLayout, t: float) -> tuple[float, float] | None:
    """Resolve a resource's (x, y) centre at simulation time ``t``.

    Returns ``None`` before the resource first appears. While stationary it sits
    at its fixed dwell centre; while moving it is interpolated along its route.
    Requires :func:`assign_positions` to have been run on ``timelines``.
    """
    if t < rt.first_seen:
        return None
    for dwell in rt.dwells:
        if dwell.track in layout.tracks and dwell.t_arrive <= t <= dwell.t_depart:
            return (dwell.center_x, layout.tracks[dwell.track].lane_y)
    for move in rt.moves:
        if move.t_depart <= t <= move.t_arrive:
            denom = move.t_arrive - move.t_depart
            frac = 0.0 if denom <= 0 else (t - move.t_depart) / denom
            return _interp_path(_move_waypoints(layout, move), frac)
    last = rt.dwells[-1]
    if last.track in layout.tracks:
        return (last.center_x, layout.tracks[last.track].lane_y)
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


def _shade_indices(xs: list[float], ys: list[float]) -> list[int]:
    """Assign an alternating 0/1 shade per wagon, by position along each lane.

    Wagons on the same lane (same ``y``) are ranked left-to-right; the shade is
    the rank parity, so flush neighbours render in alternating brightness and
    can be told apart without a gap.
    """
    shades = [0] * len(xs)
    order = sorted(range(len(xs)), key=lambda i: (ys[i], xs[i]))
    prev_y: float | None = None
    rank = 0
    for i in order:
        if ys[i] != prev_y:
            prev_y, rank = ys[i], 0
        shades[i] = rank % 2
        rank += 1
    return shades


def _dwell_track_at(rt: ResourceTrack, t: float) -> str | None:
    """Return the track a resource is parked on at ``t`` (None while in transit)."""
    if t < rt.first_seen:
        return None
    for dwell in rt.dwells:
        if dwell.t_arrive <= t <= dwell.t_depart:
            return dwell.track
    return None


def _frame_stats(  # pylint: disable=too-many-locals
    layout: YardLayout, timelines: dict[str, ResourceTrack], t: float, rejected_at: list[float]
) -> FrameStats:
    """Compute the aggregate live statistics for simulation time ``t``."""
    to_retrofit = present_done = cumulative_done = 0
    occupied: dict[str, float] = defaultdict(float)
    for rt in timelines.values():
        if rt.resource_type == 'locomotive':
            continue
        done = rt.t_retrofitted is not None and t >= rt.t_retrofitted
        cumulative_done += 1 if done else 0
        present = t >= rt.first_seen
        if present:
            present_done += 1 if done else 0
            to_retrofit += 0 if done else 1
            track = _dwell_track_at(rt, t)
            if track is not None:
                occupied[track] += rt.length_m

    utilization: dict[str, float] = {}
    for track_id, tl in layout.tracks.items():
        if tl.track_type == 'mainline' or tl.length_m <= 0:
            continue
        utilization[track_id] = occupied.get(track_id, 0.0) / tl.length_m

    cumulative_rejected = sum(1 for rt_t in rejected_at if rt_t <= t)
    return FrameStats(
        to_retrofit=to_retrofit,
        present_retrofitted=present_done,
        cumulative_retrofitted=cumulative_done,
        cumulative_rejected=cumulative_rejected,
        utilization=utilization,
    )


def _collect_frame_arrays(layout: YardLayout, timelines: dict[str, ResourceTrack], t: float) -> dict[str, list[Any]]:
    """Accumulate per-resource rectangle arrays for a single frame."""
    acc = _empty_acc()
    for rt in timelines.values():
        pos = position_at(rt, layout, t)
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
    return acc


def _build_one_frame(
    layout: YardLayout,
    timelines: dict[str, ResourceTrack],
    t: float,
    t_start: float,
    rejected_at: list[float],
) -> FrameData:
    """Assemble a single :class:`FrameData` (rectangles + shades + stats) at ``t``."""
    acc = _collect_frame_arrays(layout, timelines, t)
    return FrameData(
        t=t,
        datetime_label=_format_time(t_start, t),
        wagon_x=acc['wagon_x'],
        wagon_y=acc['wagon_y'],
        wagon_len=acc['wagon_len'],
        wagon_color=acc['wagon_color'],
        wagon_ids=acc['wagon_ids'],
        wagon_shade=_shade_indices(acc['wagon_x'], acc['wagon_y']),
        loco_x=acc['loco_x'],
        loco_y=acc['loco_y'],
        loco_len=acc['loco_len'],
        loco_ids=acc['loco_ids'],
        stats=_frame_stats(layout, timelines, t, rejected_at),
    )


def build_frames(
    layout: YardLayout,
    timelines: dict[str, ResourceTrack],
    num_frames: int,
    bounds: tuple[float, float] | None = None,
    rejected_at: list[float] | None = None,
) -> list[FrameData]:
    """Build the per-frame rectangle arrays over a uniform simulation-time grid.

    ``num_frames`` (the user-selected resolution) sets the grid density. Each
    frame carries wagon / locomotive rectangle centres + lengths + colours,
    per-wagon shade indices, and aggregate live :class:`FrameStats`.
    ``rejected_at`` lists rejection timestamps (minutes) for the cumulative
    rejected-wagon counter.
    """
    if not timelines or num_frames < 1:
        return []
    assign_positions(layout, timelines)
    rejected_at = rejected_at or []
    t_start, t_end = bounds if bounds is not None else sim_time_bounds(timelines)
    span = t_end - t_start

    frames: list[FrameData] = []
    for index in range(num_frames):
        frac = 0.0 if num_frames == 1 else index / (num_frames - 1)
        t = t_start + frac * span
        frames.append(_build_one_frame(layout, timelines, t, t_start, rejected_at))
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
