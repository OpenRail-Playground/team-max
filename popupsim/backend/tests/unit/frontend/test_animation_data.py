"""Unit tests for the animation data transforms (animation_data.py).

These tests use small synthetic event logs mirroring the real
``resource_locations`` / ``resource_states`` shape (see wagon W0001 trace).
"""

import itertools
import math

from dashboard_components import animation_data as ad
from dashboard_components import routes_graph as rg
import pandas as pd
import pytest

# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def tracks_config() -> list[dict]:
    """Return a minimal tracks config covering several track types."""
    return [
        {'id': 'collection1', 'edges': ['collection1'], 'type': 'collection'},
        {'id': 'retrofit1', 'edges': ['retrofit1'], 'type': 'retrofit'},
        {'id': 'WS_01', 'edges': ['WS_01'], 'type': 'workshop'},
        {'id': 'retrofitted1', 'edges': ['retrofitted1'], 'type': 'retrofitted'},
        {'id': 'Mainline', 'edges': ['Mainline'], 'type': 'mainline'},
        {'id': 'track_19', 'edges': ['track_19'], 'type': 'rescource_parking'},
    ]


@pytest.fixture
def topology() -> dict:
    """Return a matching topology with edge lengths."""
    return {
        'nodes': [1, 2],
        'edges': {
            'collection1': {'nodes': [1, 2], 'length': 500.0},
            'retrofit1': {'nodes': [1, 2], 'length': 426.0},
            'WS_01': {'nodes': [1, 2], 'length': 260.0},
            'retrofitted1': {'nodes': [1, 2], 'length': 704.0},
            'Mainline': {'nodes': [1, 2], 'length': 8000.0},
            'track_19': {'nodes': [1, 2], 'length': 169.0},
        },
    }


@pytest.fixture
def workshops_config() -> list[dict]:
    """Return a workshop config providing bay counts."""
    return [{'id': 'WS_01', 'name': 'First workshop', 'retrofit_stations': 2, 'track': 'WS1'}]


@pytest.fixture
def layout(tracks_config, topology, workshops_config) -> ad.YardLayout:
    """Return a built yard layout for the synthetic config."""
    return ad.build_layout(tracks_config, topology, workshops_config)


@pytest.fixture
def train_schedule() -> pd.DataFrame:
    """Return a train schedule with per-wagon real lengths."""
    return pd.DataFrame(
        [
            {'train_id': 'T1', 'wagon_id': 'W0001', 'length': 15.9},
            {'train_id': 'T1', 'wagon_id': 'W0002', 'length': 23.5},
        ]
    )


@pytest.fixture
def w0001_locations() -> pd.DataFrame:
    """Return a location log mirroring wagon W0001's real trace."""
    return pd.DataFrame(
        [
            {
                'timestamp': 360.0,
                'resource_id': 'W0001',
                'resource_type': 'wagon',
                'location': 'collection1',
                'route_path': None,
            },
            {
                'timestamp': 433.0,
                'resource_id': 'W0001',
                'resource_type': 'wagon',
                'location': 'collection1',
                'route_path': 'collection1|Mainline|retrofit1',
            },
            {
                'timestamp': 493.0,
                'resource_id': 'W0001',
                'resource_type': 'wagon',
                'location': 'retrofit1',
                'route_path': None,
            },
            {
                'timestamp': 503.0,
                'resource_id': 'W0001',
                'resource_type': 'wagon',
                'location': 'retrofit1',
                'route_path': 'retrofit1|WS_01',
            },
            {
                'timestamp': 508.0,
                'resource_id': 'W0001',
                'resource_type': 'wagon',
                'location': 'WS_01',
                'route_path': None,
            },
        ]
    )


@pytest.fixture
def w0001_states() -> pd.DataFrame:
    """Return a state log marking the retrofit-completion timestamp."""
    return pd.DataFrame(
        [
            {'timestamp': 360.0, 'resource_id': 'W0001', 'resource_type': 'wagon', 'state': 'arrived'},
            {'timestamp': 433.0, 'resource_id': 'W0001', 'resource_type': 'wagon', 'state': 'moving'},
            {'timestamp': 493.0, 'resource_id': 'W0001', 'resource_type': 'wagon', 'state': 'queued'},
            {'timestamp': 508.0, 'resource_id': 'W0001', 'resource_type': 'wagon', 'state': 'in_workshop'},
            {'timestamp': 574.0, 'resource_id': 'W0001', 'resource_type': 'wagon', 'state': 'retrofitted'},
        ]
    )


@pytest.fixture
def loco_locations() -> pd.DataFrame:
    """Return a locomotive location log (uses previous_location, no route_path)."""
    return pd.DataFrame(
        [
            {
                'timestamp': 360.0,
                'resource_id': 'LOCO_01',
                'resource_type': 'locomotive',
                'location': 'track_19',
                'previous_location': None,
                'route_path': None,
            },
            {
                'timestamp': 420.0,
                'resource_id': 'LOCO_01',
                'resource_type': 'locomotive',
                'location': 'collection1',
                'previous_location': 'track_19',
                'route_path': None,
            },
            {
                'timestamp': 500.0,
                'resource_id': 'LOCO_01',
                'resource_type': 'locomotive',
                'location': 'retrofit1',
                'previous_location': 'collection1',
                'route_path': None,
            },
        ]
    )


def _two_wagons(track: str, len_a: float, len_b: float, depart_b: float = 100.0) -> dict[str, ad.ResourceTrack]:
    """Build two concurrent wagons dwelling on a track from t=0."""
    rt_a = ad.ResourceTrack('A', 'wagon', length_m=len_a, dwells=[ad.Dwell(track, 0.0, 200.0)])
    rt_b = ad.ResourceTrack('B', 'wagon', length_m=len_b, dwells=[ad.Dwell(track, 0.0, depart_b)])
    return {'A': rt_a, 'B': rt_b}


def _one_track_layout(track_type: str, length: float, bays: int | None = None) -> ad.YardLayout:
    """Build a single-track yard layout (throat at x=0) for allocator tests."""
    tl = ad.TrackLayout(
        track_id='T',
        track_type=track_type,
        lane_y=0.0,
        x_start=0.0,
        x_end=length,
        length_m=length,
        color='#000000',
        is_workshop=track_type == 'workshop',
        bays=bays,
        throat_x=0.0,
        zone='single',
    )
    return ad.YardLayout(
        tracks={'T': tl}, throat_x=0.0, x_max=length, y_min=0.0, y_max=0.0, mode='single', left_throat_x=0.0
    )


def _n_wagons(track: str, n: int, length: float) -> dict[str, ad.ResourceTrack]:
    """Build ``n`` concurrent wagons dwelling on ``track`` from t=0."""
    return {
        f'W{i}': ad.ResourceTrack(f'W{i}', 'wagon', length_m=length, dwells=[ad.Dwell(track, 0.0, 200.0)])
        for i in range(n)
    }


def _intervals_on(timelines: dict[str, ad.ResourceTrack], track: str, t: float) -> list[tuple[float, float]]:
    """Return sorted (start, end) x-extents of wagons present on ``track`` at ``t``."""
    spans: list[tuple[float, float]] = []
    for rt in timelines.values():
        for dwell in rt.dwells:
            if dwell.track == track and dwell.t_arrive <= t <= dwell.t_depart:
                spans.append((dwell.center_x - rt.length_m / 2.0, dwell.center_x + rt.length_m / 2.0))
    return sorted(spans)


def _has_overlap(intervals: list[tuple[float, float]]) -> bool:
    """Return True if any two sorted intervals overlap by more than a rounding epsilon."""
    return any(a[1] - b[0] > 1e-6 for a, b in itertools.pairwise(intervals))


# --- Segment extraction -----------------------------------------------------


class TestExtractTimelines:
    """Tests for movement-segment extraction."""

    def test_wagon_segments_follow_route_path(self, w0001_locations, w0001_states):
        """Wagon dwells/moves should follow the recorded location sequence."""
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        rt = timelines['W0001']
        assert rt.resource_type == 'wagon'
        assert [d.track for d in rt.dwells] == ['collection1', 'retrofit1', 'WS_01']
        assert len(rt.moves) == 2

    def test_wagon_departure_uses_moving_event(self, w0001_locations, w0001_states):
        """The move should start at the route_path/moving row and end at arrival."""
        rt = ad.extract_timelines(w0001_locations, w0001_states)['W0001']
        first_move = rt.moves[0]
        assert first_move.t_depart == 433.0
        assert first_move.t_arrive == 493.0
        assert first_move.route == ('collection1', 'Mainline', 'retrofit1')

    def test_retrofit_timestamp_detected(self, w0001_locations, w0001_states):
        """The retrofit-completion timestamp should be captured for colouring."""
        rt = ad.extract_timelines(w0001_locations, w0001_states)['W0001']
        assert rt.t_retrofitted == 574.0

    def test_locomotive_uses_default_transit_and_length(self, loco_locations):
        """Locomotives use the default transit fallback and default length."""
        rt = ad.extract_timelines(loco_locations, None)['LOCO_01']
        assert rt.resource_type == 'locomotive'
        assert rt.length_m == ad.LOCO_LENGTH_M
        assert rt.moves[0].t_depart == pytest.approx(420.0 - ad.DEFAULT_TRANSIT_MIN)

    def test_wagon_length_from_schedule(self, w0001_locations, w0001_states, train_schedule):
        """Wagon length should be taken from the train schedule when provided."""
        lengths = ad.wagon_lengths(train_schedule)
        rt = ad.extract_timelines(w0001_locations, w0001_states, lengths)['W0001']
        assert rt.length_m == pytest.approx(15.9)

    def test_wagon_length_falls_back_to_default(self, w0001_locations, w0001_states):
        """Without a schedule entry the wagon length defaults."""
        rt = ad.extract_timelines(w0001_locations, w0001_states)['W0001']
        assert rt.length_m == ad.DEFAULT_WAGON_LENGTH_M

    def test_empty_input_returns_empty(self):
        """Empty or missing location logs should yield no timelines."""
        assert ad.extract_timelines(None, None) == {}
        assert ad.extract_timelines(pd.DataFrame(), None) == {}


class TestWagonLengths:
    """Tests for the wagon-length lookup helper."""

    def test_maps_ids_to_lengths(self, train_schedule):
        """Each wagon id should map to its float length."""
        lengths = ad.wagon_lengths(train_schedule)
        assert lengths == {'W0001': pytest.approx(15.9), 'W0002': pytest.approx(23.5)}

    def test_missing_columns_returns_empty(self):
        """A schedule without the expected columns yields an empty map."""
        assert ad.wagon_lengths(pd.DataFrame([{'foo': 1}])) == {}
        assert ad.wagon_lengths(None) == {}


# --- Yard layout ------------------------------------------------------------


class TestBuildLayout:
    """Tests for the synthetic yard layout."""

    def test_track_type_ordering_top_to_bottom(self, layout):
        """Mainline sits at the top lane and rescource_parking at the bottom."""
        assert layout.tracks['Mainline'].lane_y == max(t.lane_y for t in layout.tracks.values())
        assert layout.tracks['track_19'].lane_y == min(t.lane_y for t in layout.tracks.values())

    def test_workshop_adjacent_to_retrofit(self, layout):
        """The workshop lane should sit directly below its retrofit lane."""
        assert layout.tracks['retrofit1'].lane_y - layout.tracks['WS_01'].lane_y == pytest.approx(ad.LANE_SPACING)

    def test_workshop_tagged_with_bays(self, layout):
        """Workshop tracks should be tagged and carry their bay count."""
        ws = layout.tracks['WS_01']
        assert ws.is_workshop is True
        assert ws.bays == 2

    def test_x_axis_is_metric(self, layout):
        """Non-mainline tracks span their real length in metres from the throat."""
        assert layout.tracks['collection1'].x_end == pytest.approx(500.0)
        assert layout.tracks['retrofitted1'].x_end == pytest.approx(704.0)

    def test_mainline_not_to_scale(self, layout):
        """The 8000 m mainline is drawn at the fixed transit length."""
        assert layout.tracks['Mainline'].x_end == pytest.approx(ad.MAINLINE_DRAWN_M)


# --- Stable position allocation ---------------------------------------------


class TestAssignPositions:
    """Tests for the stable, uniform-gap position allocator."""

    def test_uniform_gap_between_adjacent_wagons(self, layout):
        """Adjacent parked wagons sit exactly UNIFORM_GAP_M apart, regardless of length."""
        timelines = _two_wagons('collection1', 15.9, 23.5)
        ad.assign_positions(layout, timelines)
        a, b = timelines['A'].dwells[0], timelines['B'].dwells[0]
        # Storage packs from the far end: A (first) is furthest, B is closer to the throat.
        a_near = a.center_x - timelines['A'].length_m / 2  # edge toward the throat
        b_far = b.center_x + timelines['B'].length_m / 2  # edge toward the far end
        assert a_near - b_far == pytest.approx(ad.UNIFORM_GAP_M)

    def test_storage_packs_from_far_end(self, layout):
        """The first arrival on a storage track sits flush against the far end."""
        timelines = _two_wagons('collection1', 16.0, 16.0)
        ad.assign_positions(layout, timelines)
        x_end = layout.tracks['collection1'].x_end
        assert timelines['A'].dwells[0].center_x == pytest.approx(x_end - 8.0)
        assert timelines['B'].dwells[0].center_x < timelines['A'].dwells[0].center_x

    def test_retrofit_packs_from_throat(self, layout):
        """The first arrival on a retrofit track sits nearest the throat."""
        timelines = _two_wagons('retrofit1', 16.0, 16.0)
        ad.assign_positions(layout, timelines)
        assert timelines['A'].dwells[0].center_x == pytest.approx(8.0)
        assert timelines['B'].dwells[0].center_x > timelines['A'].dwells[0].center_x

    def test_workshop_uses_bay_slots(self, layout):
        """Workshop occupants take distinct bay-slot centres."""
        timelines = _two_wagons('WS_01', 16.0, 16.0)
        ad.assign_positions(layout, timelines)
        x_end = layout.tracks['WS_01'].x_end
        centres = sorted([timelines['A'].dwells[0].center_x, timelines['B'].dwells[0].center_x])
        assert centres == pytest.approx([x_end * 0.25, x_end * 0.75])

    def test_freed_space_is_reused(self, layout):
        """After a wagon departs, a later arrival reuses its far-end slot."""
        rt_a = ad.ResourceTrack('A', 'wagon', length_m=16.0, dwells=[ad.Dwell('collection1', 0.0, 50.0)])
        rt_c = ad.ResourceTrack('C', 'wagon', length_m=16.0, dwells=[ad.Dwell('collection1', 60.0, 200.0)])
        timelines = {'A': rt_a, 'C': rt_c}
        ad.assign_positions(layout, timelines)
        # A left before C arrived, so C reuses the far-end position.
        assert rt_c.dwells[0].center_x == pytest.approx(rt_a.dwells[0].center_x)

    def test_position_is_stable_during_dwell(self, layout):
        """A stationary resource keeps the same position across its whole dwell."""
        timelines = _two_wagons('collection1', 16.0, 24.0, depart_b=50.0)
        ad.assign_positions(layout, timelines)
        rt_a = timelines['A']
        # A dwells [0, 200]; B leaves at 50. A's position must not change.
        assert ad.position_at(rt_a, layout, 10.0) == ad.position_at(rt_a, layout, 120.0)


class TestPackingGlitches:
    """Tests locking the no-stacking / within-box packing fixes."""

    def test_overflow_positions_are_distinct(self):
        """Over-capacity wagons pack sequentially (distinct centres), never stacked."""
        layout = _one_track_layout('collection', 10.0)
        timelines = _n_wagons('T', 15, 1.0)  # 15 m of wagons on a 10 m track
        ad.assign_positions(layout, timelines)
        centres = [rt.dwells[0].center_x for rt in timelines.values()]
        assert len({round(c, 6) for c in centres}) == len(centres)

    def test_storage_wagons_never_stack_when_they_fit(self):
        """Wagons whose true lengths fit the track must not overlap (no gap inflation)."""
        layout = _one_track_layout('collection', 10.0)
        timelines = _n_wagons('T', 7, 1.0)  # 7 m of wagons on a 10 m track: fits flush
        ad.assign_positions(layout, timelines)
        assert not _has_overlap(_intervals_on(timelines, 'T', 100.0))

    def test_retrofit_wagons_never_stack_when_they_fit(self):
        """Retrofit-track packing also avoids overlap when the wagons fit."""
        layout = _one_track_layout('retrofit', 10.0)
        timelines = _n_wagons('T', 7, 1.0)
        ad.assign_positions(layout, timelines)
        assert not _has_overlap(_intervals_on(timelines, 'T', 100.0))

    def test_workshop_overflow_stays_within_box(self):
        """More wagons than bays must still render inside the workshop box."""
        layout = _one_track_layout('workshop', 260.0, bays=2)
        timelines = _n_wagons('T', 3, 16.0)  # 3 wagons, only 2 bays
        ad.assign_positions(layout, timelines)
        tl = layout.tracks['T']
        for rt in timelines.values():
            center = rt.dwells[0].center_x
            assert center - rt.length_m / 2.0 >= tl.x_start - 1e-6
            assert center + rt.length_m / 2.0 <= tl.x_end + 1e-6


class TestFrameStats:
    """Tests for the per-frame live statistics."""

    @staticmethod
    def _timelines() -> dict[str, ad.ResourceTrack]:
        """Two wagons on one track; W1 retrofitted at t=50, W2 never."""
        w1 = ad.ResourceTrack('W1', 'wagon', length_m=20.0, dwells=[ad.Dwell('T', 0.0, math.inf)])
        w1.first_seen, w1.t_retrofitted = 0.0, 50.0
        w2 = ad.ResourceTrack('W2', 'wagon', length_m=30.0, dwells=[ad.Dwell('T', 10.0, math.inf)])
        w2.first_seen = 10.0
        return {'W1': w1, 'W2': w2}

    def test_counts_before_and_after_retrofit(self):
        """to_retrofit / retrofitted counts reflect presence and completion."""
        layout = _one_track_layout('collection', 100.0)
        timelines = self._timelines()
        early = ad._frame_stats(layout, timelines, 5.0, [])
        assert (early.to_retrofit, early.present_retrofitted, early.cumulative_retrofitted) == (1, 0, 0)
        late = ad._frame_stats(layout, timelines, 60.0, [])
        assert (late.to_retrofit, late.present_retrofitted, late.cumulative_retrofitted) == (1, 1, 1)

    def test_track_utilization_is_occupied_over_real_length(self):
        """Utilization sums real wagon lengths over the real track length."""
        layout = _one_track_layout('collection', 100.0)
        stats = ad._frame_stats(layout, self._timelines(), 60.0, [])
        assert stats.utilization['T'] == pytest.approx(0.5)  # (20 + 30) / 100

    def test_cumulative_rejected_is_monotonic(self):
        """Rejected counter accumulates as timestamps are passed."""
        layout = _one_track_layout('collection', 100.0)
        timelines = self._timelines()
        rejected = [40.0, 70.0]
        assert ad._frame_stats(layout, timelines, 5.0, rejected).cumulative_rejected == 0
        assert ad._frame_stats(layout, timelines, 60.0, rejected).cumulative_rejected == 1
        assert ad._frame_stats(layout, timelines, 100.0, rejected).cumulative_rejected == 2

    def test_mainline_excluded_from_utilization(self, layout):
        """The Mainline is excluded from per-track utilization."""
        stats = ad._frame_stats(layout, self._timelines(), 60.0, [])
        assert 'Mainline' not in stats.utilization

    def test_build_frames_attaches_stats_and_shades(self):
        """build_frames wires stats and alternating shades into every frame."""
        layout = _one_track_layout('collection', 100.0)
        frames = ad.build_frames(layout, self._timelines(), num_frames=5, rejected_at=[40.0])
        assert all(isinstance(f.stats, ad.FrameStats) for f in frames)
        last = frames[-1]
        assert len(last.wagon_shade) == len(last.wagon_x)
        # Two flush neighbours on the same lane get alternating shades.
        if len(last.wagon_shade) == 2:
            assert set(last.wagon_shade) == {0, 1}


# --- Position resolver ------------------------------------------------------


class TestPositionAt:
    """Tests for the position resolver."""

    def test_endpoints_match_dwell_centres(self, layout, w0001_locations, w0001_states):
        """At departure/arrival the resource sits at its source/dest dwell centre."""
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        ad.assign_positions(layout, timelines)
        rt = timelines['W0001']
        at_depart = ad.position_at(rt, layout, 433.0)
        assert at_depart == pytest.approx((rt.dwells[0].center_x, layout.tracks['collection1'].lane_y))
        at_arrive = ad.position_at(rt, layout, 493.0)
        assert at_arrive == pytest.approx((rt.dwells[1].center_x, layout.tracks['retrofit1'].lane_y))

    def test_midpoint_routes_through_throat(self, layout, w0001_locations, w0001_states):
        """Mid-transit the marker routes through the throat (x near zero)."""
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        ad.assign_positions(layout, timelines)
        rt = timelines['W0001']
        mid = ad.position_at(rt, layout, 463.0)
        assert mid is not None
        assert mid[0] <= rt.dwells[0].center_x + 1e-9

    def test_none_before_first_seen(self, layout, w0001_locations, w0001_states):
        """A resource has no position before it first appears."""
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        ad.assign_positions(layout, timelines)
        assert ad.position_at(timelines['W0001'], layout, 100.0) is None


# --- Frame builder ----------------------------------------------------------


class TestBuildFrames:
    """Tests for the per-frame builder."""

    def test_frame_count_matches_resolution(self, layout, w0001_locations, w0001_states):
        """The number of frames equals the requested resolution."""
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        assert len(ad.build_frames(layout, timelines, num_frames=50)) == 50

    def test_first_frame_carries_position_and_length(self, layout, w0001_locations, w0001_states, train_schedule):
        """The first frame places the wagon at its dwell centre and carries its length."""
        lengths = ad.wagon_lengths(train_schedule)
        timelines = ad.extract_timelines(w0001_locations, w0001_states, lengths)
        frames = ad.build_frames(layout, timelines, num_frames=10)
        first = frames[0]
        assert first.t == pytest.approx(360.0)
        assert first.wagon_ids == ['W0001']
        assert first.wagon_len[0] == pytest.approx(15.9)
        assert first.wagon_x[0] == pytest.approx(timelines['W0001'].dwells[0].center_x)

    def test_color_flips_at_retrofit_time(self, layout, w0001_locations, w0001_states):
        """Wagon colour switches from red to green at the retrofit timestamp."""
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        frames = ad.build_frames(layout, timelines, num_frames=3, bounds=(560.0, 580.0))
        assert frames[0].wagon_color[0] == ad.WAGON_COLOR_PENDING
        assert frames[-1].wagon_color[0] == ad.WAGON_COLOR_DONE

    def test_locomotives_separated_from_wagons(self, layout, w0001_locations, loco_locations):
        """Locomotives appear only in the locomotive arrays, never the wagon arrays."""
        combined = pd.concat([w0001_locations, loco_locations], ignore_index=True)
        timelines = ad.extract_timelines(combined, None)
        frames = ad.build_frames(layout, timelines, num_frames=5)
        assert any('LOCO_01' in f.loco_ids for f in frames)
        assert all('LOCO_01' not in f.wagon_ids for f in frames)
        assert all(length == ad.LOCO_LENGTH_M for f in frames for length in f.loco_len)

    def test_empty_timelines_returns_empty(self, layout):
        """No timelines yields no frames."""
        assert ad.build_frames(layout, {}, num_frames=10) == []

    def test_sim_time_bounds(self, w0001_locations, loco_locations):
        """Simulation bounds span the earliest and latest resource timestamps."""
        combined = pd.concat([w0001_locations, loco_locations], ignore_index=True)
        timelines = ad.extract_timelines(combined, None)
        assert ad.sim_time_bounds(timelines) == (360.0, 508.0)

    def test_single_frame_does_not_divide_by_zero(self, layout, w0001_locations, w0001_states):
        """A single-frame request produces one finite-time frame."""
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        frames = ad.build_frames(layout, timelines, num_frames=1)
        assert len(frames) == 1
        assert math.isfinite(frames[0].t)


# --- Routes-aware layout ----------------------------------------------------


class TestRoutesAwareLayout:
    """Tests for routes-aware, id-robust layout building."""

    @staticmethod
    def _h4r_config() -> tuple[list[dict], dict, list[dict], dict]:
        """Return hack4rail-style config with a workshop id mismatch (WS1 vs WS_01)."""
        tracks = [
            {'id': 'WS1', 'edges': ['WS1'], 'type': 'workshop'},
            {'id': 'collection1', 'edges': ['collection1'], 'type': 'collection'},
            {'id': 'track_19', 'edges': ['track_19'], 'type': 'rescource_parking'},
            {'id': 'Mainline', 'edges': ['Mainline'], 'type': 'mainline'},
            {'id': 'retrofit', 'edges': ['retrofit'], 'type': 'retrofit'},
        ]
        topology = {
            'edges': {
                'WS1': {'length': 260.0},
                'collection1': {'length': 740.0},
                'track_19': {'length': 169.0},
                'Mainline': {'length': 8000.0},
                'retrofit': {'length': 740.0},
            }
        }
        workshops = [{'id': 'WS_01', 'retrofit_stations': 2, 'track': 'track_WS1'}]
        routes = {
            'routes': [
                {'id': 'a', 'duration': 5.0, 'path': ['track_19', 'WS_01']},
                {'id': 'b', 'duration': 5.0, 'path': ['track_19', 'retrofit']},
                {'id': 'c', 'duration': 60.0, 'path': ['track_19', 'Mainline', 'collection1']},
            ]
        }
        return tracks, topology, workshops, routes

    def test_workshop_id_mismatch_resolved(self):
        """A workshop id (WS_01) absent from tracks.json still gets a lane with bays."""
        tracks, topology, workshops, routes = self._h4r_config()
        graph = rg.parse_routes(routes)
        layout = ad.build_layout(tracks, topology, workshops, graph, sorted(graph.track_ids))
        assert 'WS_01' in layout.tracks
        ws = layout.tracks['WS_01']
        assert ws.is_workshop is True
        assert ws.bays == 2
        assert ws.length_m == pytest.approx(260.0)  # default from WS1's length

    def test_three_zone_placement(self):
        """With a remote cluster the layout uses three zones (left / Mainline / right)."""
        tracks, topology, workshops, routes = self._h4r_config()
        graph = rg.parse_routes(routes)
        layout = ad.build_layout(tracks, topology, workshops, graph, sorted(graph.track_ids))
        assert layout.mode == 'zones'
        # Local yard on the left (right of throat), remote on the right, Mainline in the middle.
        assert layout.tracks['retrofit'].zone == 'left'
        assert layout.tracks['collection1'].zone == 'right'
        assert layout.tracks['Mainline'].zone == 'middle'
        assert layout.tracks['retrofit'].x_end <= layout.left_throat_x + 1e-9
        assert layout.tracks['collection1'].x_start >= layout.right_throat_x - 1e-9
        assert layout.left_throat_x < layout.right_throat_x < layout.x_max

    def test_mainline_corridor_at_most_quarter_width(self):
        """The Mainline corridor occupies no more than 25% of the total width."""
        tracks, topology, workshops, routes = self._h4r_config()
        graph = rg.parse_routes(routes)
        layout = ad.build_layout(tracks, topology, workshops, graph, sorted(graph.track_ids))
        middle_w = layout.right_throat_x - layout.left_throat_x
        assert middle_w <= 0.25 * layout.x_max + 1e-9

    def test_cross_zone_move_passes_through_mainline_corridor(self):
        """A local<->remote move routes through the Mainline corridor height."""
        tracks, topology, workshops, routes = self._h4r_config()
        graph = rg.parse_routes(routes)
        layout = ad.build_layout(tracks, topology, workshops, graph, sorted(graph.track_ids))
        rt = ad.ResourceTrack('W', 'wagon', length_m=16.0)
        rt.dwells = [ad.Dwell('collection1', 0.0, 100.0), ad.Dwell('retrofit', 200.0, math.inf)]
        rt.moves = [
            ad.Move(
                t_depart=100.0,
                t_arrive=200.0,
                route=('collection1', 'Mainline', 'retrofit'),
                from_track='collection1',
                to_track='retrofit',
            )
        ]
        ad.assign_positions(layout, {'W': rt})
        waypoints = ad._move_waypoints(layout, rt.moves[0])
        corridor_pts = [p for p in waypoints if abs(p[1] - layout.corridor_y) < 1e-9]
        # The path runs along the corridor from the right ladder to the left ladder.
        assert any(abs(p[0] - layout.left_throat_x) < 1e-9 for p in corridor_pts)
        assert any(abs(p[0] - layout.right_throat_x) < 1e-9 for p in corridor_pts)

    def test_layout_without_routes_is_backward_compatible(self):
        """Without a route graph the layout falls back to a single stacked column."""
        tracks, topology, workshops, _routes = self._h4r_config()
        layout = ad.build_layout(tracks, topology, workshops)
        assert layout.mode == 'single'
        assert set(layout.tracks) == {'WS1', 'collection1', 'track_19', 'Mainline', 'retrofit'}
        assert layout.tracks['Mainline'].lane_y == max(t.lane_y for t in layout.tracks.values())


class TestActiveTrackIds:
    """Tests for the active-id collector."""

    def test_union_of_routes_and_locations(self):
        """Active ids combine route track ids and event-log locations."""
        graph = rg.parse_routes({'routes': [{'id': 'a', 'duration': 5.0, 'path': ['track_19', 'WS_01']}]})
        locs = pd.DataFrame([{'location': 'collection1'}, {'location': 'WS_01'}, {'location': None}])
        ids = ad.active_track_ids(locs, graph)
        assert set(ids) == {'track_19', 'WS_01', 'collection1'}

    def test_handles_missing_inputs(self):
        """Missing locations / graph degrade gracefully."""
        assert ad.active_track_ids(None, None) == []
