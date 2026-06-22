"""Unit tests for the animation data transforms (animation_data.py).

These tests use small synthetic event logs mirroring the real
``resource_locations`` / ``resource_states`` shape (see wagon W0001 trace).
"""

import math

from dashboard_components import animation_data as ad
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


# --- Segment extraction -----------------------------------------------------


class TestExtractTimelines:
    """Tests for movement-segment extraction."""

    def test_wagon_segments_follow_route_path(self, w0001_locations, w0001_states):
        """Wagon dwells/moves should follow the recorded location sequence."""
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        assert 'W0001' in timelines
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
        assert first_move.from_track == 'collection1'
        assert first_move.to_track == 'retrofit1'

    def test_retrofit_timestamp_detected(self, w0001_locations, w0001_states):
        """The retrofit-completion timestamp should be captured for colouring."""
        rt = ad.extract_timelines(w0001_locations, w0001_states)['W0001']
        assert rt.t_retrofitted == 574.0

    def test_no_retrofit_state_leaves_none(self, w0001_locations):
        """Without a retrofitted state the retrofit timestamp stays None."""
        rt = ad.extract_timelines(w0001_locations, None)['W0001']
        assert rt.t_retrofitted is None

    def test_locomotive_uses_default_transit_and_length(self, loco_locations):
        """Locomotives use the default transit fallback and default length."""
        rt = ad.extract_timelines(loco_locations, None)['LOCO_01']
        assert rt.resource_type == 'locomotive'
        assert rt.length_m == ad.LOCO_LENGTH_M
        assert [d.track for d in rt.dwells] == ['track_19', 'collection1', 'retrofit1']
        first_move = rt.moves[0]
        assert first_move.t_arrive == 420.0
        assert first_move.t_depart == pytest.approx(420.0 - ad.DEFAULT_TRANSIT_MIN)
        assert first_move.route == ('track_19', 'collection1')

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

    def test_track_type_ordering_top_to_bottom(self, tracks_config, topology, workshops_config):
        """Mainline sits at the top lane and rescource_parking at the bottom."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        assert layout.tracks['Mainline'].lane_y == max(t.lane_y for t in layout.tracks.values())
        assert layout.tracks['track_19'].lane_y == min(t.lane_y for t in layout.tracks.values())

    def test_workshop_adjacent_to_retrofit(self, tracks_config, topology, workshops_config):
        """The workshop lane should sit directly below its retrofit lane."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        assert layout.tracks['retrofit1'].lane_y - layout.tracks['WS_01'].lane_y == pytest.approx(ad.LANE_SPACING)

    def test_workshop_tagged_with_bays(self, tracks_config, topology, workshops_config):
        """Workshop tracks should be tagged and carry their bay count."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        ws = layout.tracks['WS_01']
        assert ws.is_workshop is True
        assert ws.bays == 2

    def test_x_axis_is_metric(self, tracks_config, topology, workshops_config):
        """Non-mainline tracks span their real length in metres from the throat."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        collection = layout.tracks['collection1']
        assert collection.x_start == ad.THROAT_X
        assert collection.x_end == pytest.approx(500.0)
        assert layout.tracks['retrofitted1'].x_end == pytest.approx(704.0)

    def test_mainline_not_to_scale(self, tracks_config, topology, workshops_config):
        """The 8000 m mainline is drawn at the fixed transit length."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        assert layout.tracks['Mainline'].x_end == pytest.approx(ad.MAINLINE_DRAWN_M)

    def test_colors_match_type(self, tracks_config, topology, workshops_config):
        """Track colours should come from the shared type colour map."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        assert layout.tracks['collection1'].color == ad.TRACK_TYPE_COLORS['collection']


# --- Slot assignment + position resolver ------------------------------------


class TestSlotAssignment:
    """Tests for slot assignment and position resolution."""

    def test_overlapping_dwells_get_distinct_slots(self):
        """Two resources on a track at the same time get different slots."""
        rt_a = ad.ResourceTrack('A', 'wagon')
        rt_a.dwells = [ad.Dwell('collection1', 0.0, 100.0)]
        rt_b = ad.ResourceTrack('B', 'wagon')
        rt_b.dwells = [ad.Dwell('collection1', 50.0, 150.0)]
        counts = ad.assign_slots({'A': rt_a, 'B': rt_b})
        assert {rt_a.dwells[0].slot, rt_b.dwells[0].slot} == {0, 1}
        assert counts['collection1'] == 2

    def test_non_overlapping_dwells_reuse_slot(self):
        """Sequential occupants of a track reuse the same slot."""
        rt_a = ad.ResourceTrack('A', 'wagon')
        rt_a.dwells = [ad.Dwell('collection1', 0.0, 100.0)]
        rt_b = ad.ResourceTrack('B', 'wagon')
        rt_b.dwells = [ad.Dwell('collection1', 100.0, 200.0)]
        counts = ad.assign_slots({'A': rt_a, 'B': rt_b})
        assert rt_a.dwells[0].slot == rt_b.dwells[0].slot == 0
        assert counts['collection1'] == 1

    def test_storage_packs_from_far_end(self, tracks_config, topology, workshops_config):
        """On storage tracks, slot 0 sits furthest from the throat (near x_end)."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        slot_width = 26.0
        x0, _ = ad._slot_pos(layout, 'collection1', 0, slot_width)
        x1, _ = ad._slot_pos(layout, 'collection1', 1, slot_width)
        x_end = layout.tracks['collection1'].x_end
        assert x0 == pytest.approx(x_end - 0.5 * slot_width)  # furthest from throat
        assert x1 < x0  # next slot is closer to the throat

    def test_retrofit_packs_from_throat(self, tracks_config, topology, workshops_config):
        """On retrofit tracks, slot 0 sits nearest the throat."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        slot_width = 26.0
        x0, _ = ad._slot_pos(layout, 'retrofit1', 0, slot_width)
        x1, _ = ad._slot_pos(layout, 'retrofit1', 1, slot_width)
        assert x0 == pytest.approx(0.5 * slot_width)
        assert x1 > x0

    def test_workshop_uses_bay_slots(self, tracks_config, topology, workshops_config):
        """Workshop slots are spaced one per bay across the track."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        x_end = layout.tracks['WS_01'].x_end
        x0, _ = ad._slot_pos(layout, 'WS_01', 0, 26.0)
        x1, _ = ad._slot_pos(layout, 'WS_01', 1, 26.0)
        assert x0 == pytest.approx(x_end * 0.25)  # 2 bays -> centres at 1/4 and 3/4
        assert x1 == pytest.approx(x_end * 0.75)

    def test_position_endpoints_match_slots(
        self, tracks_config, topology, workshops_config, w0001_locations, w0001_states
    ):
        """At departure/arrival the resource sits in its source/dest slot."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        slot_width = ad.compute_slot_width(timelines)
        rt = timelines['W0001']

        at_depart = ad.position_at(rt, layout, slot_width, 433.0)
        expected_source = ad._slot_pos(layout, 'collection1', rt.moves[0].from_slot, slot_width)
        assert at_depart == pytest.approx(expected_source)

        at_arrive = ad.position_at(rt, layout, slot_width, 493.0)
        expected_dest = ad._slot_pos(layout, 'retrofit1', rt.moves[0].to_slot, slot_width)
        assert at_arrive == pytest.approx(expected_dest)

    def test_position_midpoint_routes_through_throat(
        self, tracks_config, topology, workshops_config, w0001_locations, w0001_states
    ):
        """Mid-transit the marker routes through the throat (x near zero)."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        slot_width = ad.compute_slot_width(timelines)
        rt = timelines['W0001']
        mid = ad.position_at(rt, layout, slot_width, 463.0)
        assert mid is not None
        source_x = ad._slot_pos(layout, 'collection1', rt.moves[0].from_slot, slot_width)[0]
        assert mid[0] <= source_x + 1e-9

    def test_position_none_before_first_seen(
        self, tracks_config, topology, workshops_config, w0001_locations, w0001_states
    ):
        """A resource has no position before it first appears."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        slot_width = ad.compute_slot_width(timelines)
        assert ad.position_at(timelines['W0001'], layout, slot_width, 100.0) is None


# --- Frame builder ----------------------------------------------------------


class TestBuildFrames:
    """Tests for the per-frame builder."""

    def test_frame_count_matches_resolution(
        self, tracks_config, topology, workshops_config, w0001_locations, w0001_states
    ):
        """The number of frames equals the requested resolution."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        frames = ad.build_frames(layout, timelines, num_frames=50)
        assert len(frames) == 50

    def test_first_frame_carries_position_and_length(
        self, tracks_config, topology, workshops_config, w0001_locations, w0001_states, train_schedule
    ):
        """The first frame places the wagon at its slot and carries its length."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        lengths = ad.wagon_lengths(train_schedule)
        timelines = ad.extract_timelines(w0001_locations, w0001_states, lengths)
        slot_width = ad.compute_slot_width(timelines)
        frames = ad.build_frames(layout, timelines, num_frames=10)
        first = frames[0]
        assert first.t == pytest.approx(360.0)
        assert first.wagon_ids == ['W0001']
        assert first.wagon_len[0] == pytest.approx(15.9)
        expected = ad._slot_pos(layout, 'collection1', timelines['W0001'].dwells[0].slot, slot_width)
        assert (first.wagon_x[0], first.wagon_y[0]) == pytest.approx(expected)

    def test_color_flips_at_retrofit_time(
        self, tracks_config, topology, workshops_config, w0001_locations, w0001_states
    ):
        """Wagon colour switches from red to green at the retrofit timestamp."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        frames = ad.build_frames(layout, timelines, num_frames=3, bounds=(560.0, 580.0))
        assert frames[0].wagon_color[0] == ad.WAGON_COLOR_PENDING
        assert frames[-1].wagon_color[0] == ad.WAGON_COLOR_DONE

    def test_locomotives_separated_from_wagons(
        self, tracks_config, topology, workshops_config, w0001_locations, loco_locations
    ):
        """Locomotives appear only in the locomotive arrays, never the wagon arrays."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        combined = pd.concat([w0001_locations, loco_locations], ignore_index=True)
        timelines = ad.extract_timelines(combined, None)
        frames = ad.build_frames(layout, timelines, num_frames=5)
        assert any('LOCO_01' in f.loco_ids for f in frames)
        assert all('LOCO_01' not in f.wagon_ids for f in frames)
        assert all(length == ad.LOCO_LENGTH_M for f in frames for length in f.loco_len)

    def test_empty_timelines_returns_empty(self, tracks_config, topology, workshops_config):
        """No timelines yields no frames."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        assert ad.build_frames(layout, {}, num_frames=10) == []

    def test_sim_time_bounds(self, w0001_locations, loco_locations):
        """Simulation bounds span the earliest and latest resource timestamps."""
        combined = pd.concat([w0001_locations, loco_locations], ignore_index=True)
        timelines = ad.extract_timelines(combined, None)
        start, end = ad.sim_time_bounds(timelines)
        assert start == 360.0
        assert end == 508.0

    def test_single_frame_does_not_divide_by_zero(
        self, tracks_config, topology, workshops_config, w0001_locations, w0001_states
    ):
        """A single-frame request produces one finite-time frame."""
        layout = ad.build_layout(tracks_config, topology, workshops_config)
        timelines = ad.extract_timelines(w0001_locations, w0001_states)
        frames = ad.build_frames(layout, timelines, num_frames=1)
        assert len(frames) == 1
        assert math.isfinite(frames[0].t)
