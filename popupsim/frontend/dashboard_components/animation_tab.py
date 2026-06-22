"""Animation tab - animated playback of a simulation run.

Renders a schematic yard (metric track lanes + a connecting throat) and plays
back wagon / locomotive movements over time using Plotly's native frame
animation (client-side play / pause / scrub). Resources are drawn as flat,
non-overlapping, train-like rectangles sized by their real length.

All heavy data preparation lives in
:mod:`dashboard_components.animation_data` (pure, unit-tested); this module
only handles Streamlit controls and Plotly figure assembly.
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_components import animation_data as ad

# Rectangle height in lane units (nominal: the y-axis is a lane index, not
# metric, so true 3 m would render as an invisible hairline against a yard that
# is hundreds of metres tall). Length, by contrast, is true-to-metre.
_RECT_HEIGHT = 0.5
_LANE_HALF_HEIGHT = 0.36
_TRACK_LINE_WIDTH = 5
_LABEL_OFFSET_M = 14.0
_MIN_FRAME_MS = 20


@st.cache_data(show_spinner=False)
def _compute_animation(
    resource_locations: pd.DataFrame | None,
    resource_states: pd.DataFrame | None,
    layout_config: dict[str, Any],
    num_frames: int,
) -> tuple[ad.YardLayout, list[ad.FrameData]]:
    """Build (and cache) the yard layout and per-frame rectangle arrays.

    Cached on the raw inputs + resolution so scrubbing/replaying is instant.
    ``layout_config`` bundles the ``tracks`` / ``topology`` / ``workshops``
    config plus the ``wagon_lengths`` map.
    """
    layout = ad.build_layout(
        layout_config.get('tracks', []),
        layout_config.get('topology', {}),
        layout_config.get('workshops', []),
    )
    timelines = ad.extract_timelines(resource_locations, resource_states, layout_config.get('wagon_lengths', {}))
    frames = ad.build_frames(layout, timelines, num_frames)
    return layout, frames


def _rect_polygons(
    xs: list[float], ys: list[float], lengths: list[float]
) -> tuple[list[float | None], list[float | None]]:
    """Build None-separated polygon corner arrays for a set of rectangles."""
    poly_x: list[float | None] = []
    poly_y: list[float | None] = []
    half_h = _RECT_HEIGHT / 2.0
    for xc, yc, length in zip(xs, ys, lengths, strict=False):
        half_l = length / 2.0
        poly_x.extend([xc - half_l, xc + half_l, xc + half_l, xc - half_l, xc - half_l, None])
        poly_y.extend([yc - half_h, yc - half_h, yc + half_h, yc + half_h, yc - half_h, None])
    return poly_x, poly_y


def _rect_trace(xs: list[float], ys: list[float], lengths: list[float], color: str, line_color: str) -> go.Scatter:
    """Build a single filled-rectangle Scatter trace (one colour group)."""
    poly_x, poly_y = _rect_polygons(xs, ys, lengths)
    return go.Scatter(
        x=poly_x,
        y=poly_y,
        mode='lines',
        fill='toself',
        fillcolor=color,
        line={'color': line_color, 'width': 0.5},
        hoverinfo='skip',
        showlegend=False,
    )


def _split_wagons(frame: ad.FrameData) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Split a frame's wagons into pending (red) and done (green) groups."""
    pending: dict[str, list[float]] = {'x': [], 'y': [], 'len': []}
    done: dict[str, list[float]] = {'x': [], 'y': [], 'len': []}
    for x, y, length, color in zip(frame.wagon_x, frame.wagon_y, frame.wagon_len, frame.wagon_color, strict=False):
        target = done if color == ad.WAGON_COLOR_DONE else pending
        target['x'].append(x)
        target['y'].append(y)
        target['len'].append(length)
    return pending, done


def _dynamic_traces(frame: ad.FrameData) -> list[go.Scatter]:
    """Build the four dynamic traces (red wagons, green wagons, locos, hover)."""
    pending, done = _split_wagons(frame)
    hover_x = frame.wagon_x + frame.loco_x
    hover_y = frame.wagon_y + frame.loco_y
    hover_ids = frame.wagon_ids + frame.loco_ids
    return [
        _rect_trace(pending['x'], pending['y'], pending['len'], ad.WAGON_COLOR_PENDING, '#7b241c'),
        _rect_trace(done['x'], done['y'], done['len'], ad.WAGON_COLOR_DONE, '#1e8449'),
        _rect_trace(frame.loco_x, frame.loco_y, frame.loco_len, ad.LOCO_COLOR, '#f1c40f'),
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode='markers',
            marker={'size': 12, 'color': 'rgba(0,0,0,0)'},
            text=hover_ids,
            hovertemplate='%{text}<extra></extra>',
            showlegend=False,
        ),
    ]


def _legend_traces() -> list[go.Scatter]:
    """Legend-only marker traces (no data points)."""
    entries = [
        ('Wagon · to retrofit', ad.WAGON_COLOR_PENDING, 'square'),
        ('Wagon · retrofitted', ad.WAGON_COLOR_DONE, 'square'),
        ('Locomotive', ad.LOCO_COLOR, 'square'),
        ('Workshop', ad.TRACK_TYPE_COLORS['workshop'], 'square'),
    ]
    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            name=name,
            marker={'color': color, 'symbol': symbol, 'size': 12, 'line': {'color': '#222', 'width': 1}},
            showlegend=True,
            hoverinfo='skip',
        )
        for name, color, symbol in entries
    ]


def _add_static_geometry(fig: go.Figure, layout: ad.YardLayout) -> None:
    """Draw track lines, workshop boxes with bay dividers, throat and labels."""
    y_lo, y_hi = layout.y_min - 0.6, layout.y_max + 0.6
    fig.add_shape(
        type='line',
        x0=layout.throat_x,
        y0=y_lo,
        x1=layout.throat_x,
        y1=y_hi,
        line={'color': '#bdc3c7', 'width': 3, 'dash': 'dot'},
        layer='below',
    )

    for tl in layout.tracks.values():
        if tl.is_workshop:
            _draw_workshop(fig, tl)
        else:
            fig.add_shape(
                type='line',
                x0=tl.x_start,
                y0=tl.lane_y,
                x1=tl.x_end,
                y1=tl.lane_y,
                line={'color': tl.color, 'width': _TRACK_LINE_WIDTH},
                opacity=0.5,
                layer='below',
            )
        fig.add_annotation(
            x=layout.throat_x - _LABEL_OFFSET_M,
            y=tl.lane_y,
            text=tl.track_id,
            showarrow=False,
            xanchor='right',
            font={'size': 9, 'color': '#555'},
        )


def _draw_workshop(fig: go.Figure, tl: ad.TrackLayout) -> None:
    """Draw a workshop as a labeled building box with one divider per bay."""
    fig.add_shape(
        type='rect',
        x0=tl.x_start,
        y0=tl.lane_y - _LANE_HALF_HEIGHT,
        x1=tl.x_end,
        y1=tl.lane_y + _LANE_HALF_HEIGHT,
        line={'color': tl.color, 'width': 2},
        fillcolor=tl.color,
        opacity=0.18,
        layer='below',
    )
    bays = tl.bays or 1
    for i in range(1, bays):
        x = tl.x_start + (tl.x_end - tl.x_start) * i / bays
        fig.add_shape(
            type='line',
            x0=x,
            y0=tl.lane_y - _LANE_HALF_HEIGHT,
            x1=x,
            y1=tl.lane_y + _LANE_HALF_HEIGHT,
            line={'color': tl.color, 'width': 1, 'dash': 'dot'},
            layer='below',
        )
    bay_label = f' · {tl.bays} bays' if tl.bays else ''
    fig.add_annotation(
        x=(tl.x_start + tl.x_end) / 2,
        y=tl.lane_y + _LANE_HALF_HEIGHT,
        text=f'🏭 {tl.track_id}{bay_label}',
        showarrow=False,
        yanchor='bottom',
        font={'size': 9, 'color': '#1e5631'},
    )


def _animation_controls(frames: list[ad.FrameData], frame_ms: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the play/pause buttons (updatemenus) and the time slider."""
    play_args = {'frame': {'duration': frame_ms, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}
    pause_args = {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}
    slider_steps = [
        {
            'args': [
                [str(i)],
                {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}},
            ],
            'label': f.datetime_label,
            'method': 'animate',
        }
        for i, f in enumerate(frames)
    ]
    updatemenus = [
        {
            'type': 'buttons',
            'direction': 'left',
            'x': 0,
            'y': -0.04,
            'xanchor': 'left',
            'yanchor': 'top',
            'pad': {'r': 8, 't': 8},
            'buttons': [
                {'label': '▶ Play', 'method': 'animate', 'args': [None, play_args]},
                {'label': '⏸ Pause', 'method': 'animate', 'args': [[None], pause_args]},
            ],
        }
    ]
    sliders = [
        {
            'active': 0,
            'x': 0.12,
            'len': 0.88,
            'xanchor': 'left',
            'y': -0.04,
            'yanchor': 'top',
            'currentvalue': {'prefix': 'Time: ', 'font': {'size': 13}},
            'pad': {'t': 8},
            'steps': slider_steps,
        }
    ]
    return updatemenus, sliders


def _build_figure(layout: ad.YardLayout, frames: list[ad.FrameData], frame_ms: int) -> go.Figure:
    """Assemble the full animated Plotly figure."""
    first = frames[0]
    fig = go.Figure(data=[*_dynamic_traces(first), *_legend_traces()])
    _add_static_geometry(fig, layout)

    fig.frames = [go.Frame(name=str(i), data=_dynamic_traces(f), traces=[0, 1, 2, 3]) for i, f in enumerate(frames)]

    updatemenus, sliders = _animation_controls(frames, frame_ms)
    fig.update_layout(
        height=max(520, len(layout.tracks) * 28),
        margin={'l': 10, 'r': 10, 't': 30, 'b': 10},
        plot_bgcolor='white',
        xaxis={'visible': False, 'range': [-_LABEL_OFFSET_M * 5, layout.x_max + _LABEL_OFFSET_M]},
        yaxis={'visible': False, 'range': [layout.y_min - 0.8, layout.y_max + 1.0]},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
        updatemenus=updatemenus,
        sliders=sliders,
    )
    return fig


def _render_controls() -> tuple[int, int, float]:
    """Render the playback control widgets; return (num_frames, length_s, speed)."""
    col1, col2, col3 = st.columns(3)
    with col1:
        num_frames = st.slider(
            'Resolution (frames)',
            min_value=50,
            max_value=600,
            value=200,
            step=50,
            help='More frames = smoother motion but a larger figure.',
        )
    with col2:
        anim_length_s = st.slider(
            'Animation length (s)',
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            help='Wall-clock duration of one full playback.',
        )
    with col3:
        speed = st.select_slider('Playback speed', options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)
    return num_frames, anim_length_s, speed


def render_animation_tab(data: dict[str, Any]) -> None:
    """Render the animated simulation playback tab."""
    st.header('🎬 Simulation Animation')
    st.caption(
        'Playback of wagon and locomotive movements through the yard. Resources are flat, '
        'length-accurate rectangles: wagons red until retrofitted then green, locomotives dark. '
        'Parked wagons pack to the far end of storage tracks; workshops show one slot per bay.'
    )

    resource_locations = data.get('resource_locations')
    resource_states = data.get('resource_states')
    if resource_locations is None or resource_locations.empty:
        st.warning('⚠️ No `resource_locations.csv` found for this scenario — cannot build the animation.')
        return

    scenario_config = data.get('scenario_config', {})
    tracks_config = scenario_config.get('tracks', {}).get('tracks', [])
    topology = scenario_config.get('topology', {})
    workshops_config = scenario_config.get('workshops', {}).get('workshops', [])
    lengths = ad.wagon_lengths(scenario_config.get('train_schedule'))
    if not tracks_config:
        st.warning('⚠️ No track configuration found — cannot build the yard layout.')
        return

    num_frames, anim_length_s, speed = _render_controls()

    with st.spinner('Preparing animation...'):
        layout, frames = _compute_animation(
            resource_locations,
            resource_states,
            {
                'tracks': tracks_config,
                'topology': topology,
                'workshops': workshops_config,
                'wagon_lengths': lengths,
            },
            num_frames,
        )

    if not frames:
        st.info('No movement data available to animate for this scenario.')
        return

    frame_ms = max(_MIN_FRAME_MS, int(anim_length_s * 1000 / num_frames / speed))
    fig = _build_figure(layout, frames, frame_ms)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png', 'filename': 'yard_animation'}},
    )
    st.caption(
        f'{len(frames)} frames · {len(layout.tracks)} tracks · x-axis in metres along track · '
        f'press ▶ Play, drag the slider to scrub, hover a rectangle for its id.'
    )
