"""Visualization helpers for optimizer results.

All functions accept a ``pd.DataFrame`` as returned by ``run_grid_search`` and
produce interactive Plotly figures.

The search space is 13-dimensional, so plots always aggregate over the
dimensions that are *not* on the x/y axes — by default taking the **best**
(max) value in each 2-D bin so that the plot shows the achievable frontier.
"""

from __future__ import annotations

from typing import Literal

import numpy as _np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Use the browser renderer so the figure opens as a standalone HTML page.
# This works regardless of whether nbformat is installed.
pio.renderers.default = "browser"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric: Literal["wagons_parked", "locomotive_active_time_min", "score"] = "score",
    agg: Literal["max", "min", "mean"] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Heatmap of one metric over two parameter axes.

    Aggregates all other dimensions by taking the *best* value in each cell
    (max for wagons/score, min for loco time) unless overridden via ``agg``.

    Args:
        df: Results DataFrame from ``run_grid_search``.
        x_col: Column name for the x-axis (any SearchPoint field).
        y_col: Column name for the y-axis (any SearchPoint field).
        metric: Which output metric to colour by.
        agg: Aggregation function over all other dimensions. Defaults to
             ``"max"`` for wagons/score and ``"min"`` for loco time.
        title: Figure title (auto-generated if None).

    Returns:
        Plotly Figure.
    """
    if agg is None:
        agg = "min" if metric == "locomotive_active_time_min" else "max"

    pivot = df.pivot_table(values=metric, index=y_col, columns=x_col, aggfunc=agg)

    colorscale = "RdYlGn" if metric != "locomotive_active_time_min" else "RdYlGn_r"
    auto_title = title or f"{metric}  ({agg})  —  {x_col}  ×  {y_col}"

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[str(v) for v in pivot.columns],
            y=[str(v) for v in pivot.index],
            colorscale=colorscale,
            colorbar=dict(title=metric),
            hoverongaps=False,
            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{metric}: %{{z:.1f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=auto_title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
    )
    return fig


def plot_dual_heatmaps(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> go.Figure:
    """Side-by-side heatmaps: wagons_parked (left) and loco active time (right).

    Args:
        df: Results DataFrame from ``run_grid_search``.
        x_col: Column name for the x-axis.
        y_col: Column name for the y-axis.

    Returns:
        Plotly Figure with two subplots.
    """
    pivot_wagons = df.pivot_table(values="wagons_parked", index=y_col, columns=x_col, aggfunc="max")
    pivot_loco = df.pivot_table(values="locomotive_active_time_min", index=y_col, columns=x_col, aggfunc="min")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["wagons_parked  (max)", "loco active time  (min, minutes)"],
        horizontal_spacing=0.15,
    )

    fig.add_trace(
        go.Heatmap(
            z=pivot_wagons.values,
            x=[str(v) for v in pivot_wagons.columns],
            y=[str(v) for v in pivot_wagons.index],
            colorscale="RdYlGn",
            colorbar=dict(title="wagons", x=0.44),
            hoverongaps=False,
            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>wagons: %{{z}}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=pivot_loco.values,
            x=[str(v) for v in pivot_loco.columns],
            y=[str(v) for v in pivot_loco.index],
            colorscale="RdYlGn_r",
            colorbar=dict(title="loco min", x=1.0),
            hoverongaps=False,
            hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>loco min: %{{z:.0f}}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y_col, col=1)
    fig.update_layout(
        title=f"Dual objectives  —  {x_col}  ×  {y_col}",
        template="plotly_white",
        width=1100,
        height=500,
    )
    return fig


def plot_pareto(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of wagons_parked vs loco active time — highlights Pareto front.

    Each point is one evaluated SearchPoint configuration. The Pareto-optimal
    points (maximise wagons, minimise loco time simultaneously) are highlighted.

    Args:
        df: Results DataFrame from ``run_grid_search``.

    Returns:
        Plotly Figure.
    """
    dominated = _pareto_mask(df)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.loc[~dominated, "locomotive_active_time_min"],
            y=df.loc[~dominated, "wagons_parked"],
            mode="markers",
            name="Pareto-optimal",
            marker=dict(color="crimson", size=9, symbol="star"),
            hovertemplate="wagons: %{y}<br>loco: %{x:.0f} min<extra>Pareto</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[dominated, "locomotive_active_time_min"],
            y=df.loc[dominated, "wagons_parked"],
            mode="markers",
            name="Dominated",
            marker=dict(color="steelblue", size=5, opacity=0.4),
            hovertemplate="wagons: %{y}<br>loco: %{x:.0f} min<extra></extra>",
        )
    )
    fig.update_layout(
        title="Pareto front: wagons parked vs locomotive active time",
        xaxis_title="Locomotive active time (min)  ← lower is better",
        yaxis_title="Wagons parked  ↑ higher is better",
        template="plotly_white",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )
    return fig


def plot_surface_3d(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric: Literal["wagons_parked", "locomotive_active_time_min", "score"] = "score",
    agg: Literal["max", "min", "mean"] | None = None,
    interp_resolution: int = 60,
    title: str | None = None,
) -> go.Figure:
    """3D surface of one metric over two parameter axes with smooth gradient.

    The discrete grid values are interpolated onto a fine mesh using inverse
    distance weighting so the surface appears as a smooth continuous gradient
    rather than a step function.

    Args:
        df: Results DataFrame from ``run_grid_search``.
        x_col: Column for the x-axis (any SearchPoint field).
        y_col: Column for the y-axis (any SearchPoint field).
        metric: Output metric for the z-axis.
        agg: Aggregation over all other dimensions. Defaults to ``"max"`` for
             wagons/score and ``"min"`` for loco time.
        interp_resolution: Number of grid points along each axis for the
                           interpolated surface (higher = smoother).
        title: Figure title (auto-generated if None).

    Returns:
        Plotly Figure with a ``go.Surface`` trace.
    """
    if agg is None:
        agg = "min" if metric == "locomotive_active_time_min" else "max"

    pivot = df.pivot_table(values=metric, index=y_col, columns=x_col, aggfunc=agg)

    # Scatter points for IDW: use the centre of each cell
    xs = pivot.columns.to_numpy(dtype=float)
    ys = pivot.index.to_numpy(dtype=float)
    xx, yy = _np.meshgrid(xs, ys)
    zz = pivot.values.astype(float)

    # Build fine interpolation grid
    xi = _np.linspace(xs.min(), xs.max(), interp_resolution)
    yi = _np.linspace(ys.min(), ys.max(), interp_resolution)
    xi_grid, yi_grid = _np.meshgrid(xi, yi)

    zi_grid = _idw_interpolate(
        xx.ravel(), yy.ravel(), zz.ravel(), xi_grid, yi_grid
    )

    colorscale = "RdYlGn" if metric != "locomotive_active_time_min" else "RdYlGn_r"
    auto_title = title or f"{metric} surface  ({agg})  —  {x_col}  ×  {y_col}"

    fig = go.Figure(
        go.Surface(
            x=xi,
            y=yi,
            z=zi_grid,
            colorscale=colorscale,
            colorbar=dict(title=metric),
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=True, highlightcolor="white", width=1),
            ),
            hovertemplate=f"{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<br>{metric}: %{{z:.1f}}<extra></extra>",
            opacity=0.9,
        )
    )

    # Overlay the actual measured points as scatter
    scatter_z = zz.ravel()
    scatter_x = xx.ravel()
    scatter_y = yy.ravel()
    fig.add_trace(
        go.Scatter3d(
            x=scatter_x,
            y=scatter_y,
            z=scatter_z,
            mode="markers",
            marker=dict(size=5, color="black", symbol="circle"),
            name="Evaluated points",
            hovertemplate=f"{x_col}: %{{x:.3f}}<br>{y_col}: %{{y:.3f}}<br>{metric}: %{{z:.1f}}<extra>Measured</extra>",
        )
    )

    fig.update_layout(
        title=auto_title,
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=metric,
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        template="plotly_white",
        width=900,
        height=650,
    )
    return fig


def top_table(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top-n rows sorted by score, with key columns only.

    Args:
        df: Results DataFrame from ``run_grid_search``.
        n: Number of rows to return.

    Returns:
        Slimmed DataFrame suitable for display.
    """
    cols = [
        "score",
        "wagons_parked",
        "locomotive_active_time_min",
        "ctr_base",
        "ctr_hold_thresh",
        "ctr_max_hold",
        "ctr_r0_thresh",
        "ctr_r1_thresh",
        "rtw_base",
        "rtw_deprio_thresh",
        "wtr_base",
        "rtp_base",
        "rtp_hold_thresh",
        "rtp_max_hold",
        "rtp_r0_thresh",
        "rtp_r1_thresh",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing].head(n)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _idw_interpolate(
    px: _np.ndarray,
    py: _np.ndarray,
    pz: _np.ndarray,
    xi: _np.ndarray,
    yi: _np.ndarray,
    power: float = 2.0,
) -> _np.ndarray:
    """Inverse distance weighting interpolation onto a meshgrid.

    Args:
        px, py: Flat arrays of known x/y coordinates.
        pz: Flat array of known z values.
        xi, yi: 2-D meshgrids of target coordinates.
        power: IDW exponent — higher = sharper local influence.

    Returns:
        2-D array of interpolated z values matching the shape of xi/yi.
    """
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()
    zi = _np.empty(len(xi_flat))

    for k in range(len(xi_flat)):
        dists = _np.sqrt((px - xi_flat[k]) ** 2 + (py - yi_flat[k]) ** 2)
        exact = dists == 0.0
        if exact.any():
            zi[k] = pz[exact][0]
        else:
            w = 1.0 / dists**power
            zi[k] = (w * pz).sum() / w.sum()

    return zi.reshape(xi.shape)


def _pareto_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask; True = point is dominated (not on Pareto front).

    A point p is dominated if there exists another point q such that:
      q.wagons_parked >= p.wagons_parked  AND
      q.loco_time     <= p.loco_time      (at least one strict).
    """
    wagons = df["wagons_parked"].to_numpy()
    loco = df["locomotive_active_time_min"].to_numpy()
    dominated = pd.Series(False, index=df.index)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if wagons[j] >= wagons[i] and loco[j] <= loco[i]:
                if wagons[j] > wagons[i] or loco[j] < loco[i]:
                    dominated.iloc[i] = True
                    break
    return dominated
