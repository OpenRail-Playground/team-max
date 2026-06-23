"""Grid-search algorithm for the PopUpSim optimizer.

Strategy
--------
Phase 1 — Coarse grid
    Enumerate all valid SearchPoints at ``Resolution.COARSE`` (~13 k points).
    Run each through the harness. Collect results into a DataFrame.

Phase 2 — Refinement (planned)
    Take the top-N points from Phase 1 and generate a local fine grid around
    each by linearly interpolating between the coarse step values. Re-run those
    points and keep the overall best.

Objective
---------
Two criteria, ranked lexicographically:
  1. Maximise ``wagons_parked``  (primary — throughput)
  2. Minimise ``locomotive_active_time_min``  (secondary — resource cost)

Both are combined into a single scalar score so that comparisons and sorting
are straightforward::

    score = wagons_parked * W_WAGONS - locomotive_active_time_min * W_LOCO

Higher score is better.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from popupsim.optimizer.harness import SimulationResult
from popupsim.optimizer.harness import compute_cost
from popupsim.optimizer.harness import run_simulation
from popupsim.optimizer.modeller import Resolution
from popupsim.optimizer.modeller import SearchPoint
from popupsim.optimizer.modeller import grid_size
from popupsim.optimizer.modeller import iter_grid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Objective weights — passed directly to harness.compute_cost().
#
#   score = W_WAGONS * wagon_completion_pct/100
#         − W_LOCO   * loco_utilization_pct/100
#
# Both metrics are normalised to [0, 1].  Higher score is better.
# Default: wagons dominate (90 %), loco use is a 10 % penalty.
# ---------------------------------------------------------------------------
W_WAGONS: float = 0.9
W_LOCO: float = 0.1

# ---------------------------------------------------------------------------
# Search depth — controls how many refinement phases are run after the coarse
# pass.  0 = only coarse grid.  Each phase re-runs around the top results of
# the previous phase with a tighter local grid.
# ---------------------------------------------------------------------------
SEARCH_DEPTH: int = 1


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    """A single evaluated point: parameters + simulation outcome + score."""

    point: SearchPoint
    wagons_parked: int
    locomotive_active_time_min: float
    score: float
    error: str | None  # non-None if the simulation raised an exception


def _score(sim: SimulationResult) -> float:
    return compute_cost(sim, w_wagons=W_WAGONS, w_loco=W_LOCO)


# ---------------------------------------------------------------------------
# Core search loop
# ---------------------------------------------------------------------------


def run_grid_search(
    scenario_dir: Path,
    resolution: Resolution = Resolution.COARSE,
    top_n: int = 20,
    progress_every: int = 100,
) -> tuple[pd.DataFrame, list[EvalResult]]:
    """Run a full grid search at the requested resolution.

    Args:
        scenario_dir: Base scenario directory (passed unchanged to the harness).
        resolution: ``COARSE`` (~13 k) or ``FINE`` (~17 M) grid.
        top_n: How many top results to return in the second return value.
        progress_every: Log a progress message every this many evaluations.

    Returns:
        A tuple ``(results_df, top_results)`` where:
        - ``results_df`` is a pandas DataFrame with one row per evaluated point
          (all SearchPoint fields + wagons_parked, loco_time, score, error).
        - ``top_results`` is a list of the best ``top_n`` EvalResults sorted
          by descending score.
    """
    total, valid = grid_size(resolution)
    logger.info("Starting %s grid search: %d valid points (of %d total)", resolution.value, valid, total)
    print(f"Grid search ({resolution.value}): {valid:,} points to evaluate …")

    rows: list[dict[str, Any]] = []
    for i, point in enumerate(iter_grid(resolution), start=1):
        if i % progress_every == 0:
            pct = i / valid * 100
            print(f"  {i:>7,} / {valid:,}  ({pct:.1f}%)  …", end="\r", flush=True)

        overrides = point.to_scenario_overrides()
        error: str | None = None
        sim_result: SimulationResult | None = None
        try:
            sim_result = run_simulation(scenario_dir, scenario_overrides=overrides)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.warning("Point %d failed: %s", i, error)

        score = _score(sim_result) if sim_result is not None else float("-inf")
        wagons = sim_result.wagons_parked if sim_result is not None else 0
        loco = sim_result.locomotive_active_time_min if sim_result is not None else float("nan")

        row: dict[str, Any] = asdict(point)
        row["wagons_parked"] = wagons
        row["locomotive_active_time_min"] = loco
        row["score"] = score
        row["error"] = error
        rows.append(row)

    print(f"  Done — {len(rows):,} points evaluated.         ")

    df = pd.DataFrame(rows)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    top_eval = [
        EvalResult(
            point=SearchPoint(**{k: row[k] for k in SearchPoint.__dataclass_fields__}),
            wagons_parked=int(row["wagons_parked"]),
            locomotive_active_time_min=float(row["locomotive_active_time_min"]),
            score=float(row["score"]),
            error=row["error"],
        )
        for _, row in df.head(top_n).iterrows()
    ]

    logger.info("Best score: %.1f  (wagons=%d, loco=%.0f min)", top_eval[0].score, top_eval[0].wagons_parked, top_eval[0].locomotive_active_time_min)
    return df, top_eval


# ---------------------------------------------------------------------------
# Refinement helper (Phase 2)
# ---------------------------------------------------------------------------


def refine_around(
    scenario_dir: Path,
    top_results: list[EvalResult],
    n_steps: int = 3,
) -> tuple[pd.DataFrame, list[EvalResult]]:
    """Generate and evaluate a local fine grid around each top result.

    For each numeric axis in the SearchPoint, values are interpolated between
    the coarse step below and above the current value.  Integer axes
    (base_priority) are kept as-is plus ±1.

    Args:
        scenario_dir: Base scenario directory.
        top_results: Best points from the coarse search.
        n_steps: Number of interpolation steps between coarse values (default 3).

    Returns:
        Same format as ``run_grid_search``: ``(df, top_n_results)``.
    """
    import itertools
    from dataclasses import fields as dc_fields

    FLOAT_DELTA = 0.05
    INT_DELTA = 1

    candidate_points: set[SearchPoint] = set()
    for er in top_results:
        p = er.point
        axis_variants: dict[str, list[Any]] = {}
        for field in dc_fields(p):
            val = getattr(p, field.name)
            if isinstance(val, float):
                variants = sorted({
                    round(max(0.0, val - FLOAT_DELTA * 2), 3),
                    round(max(0.0, val - FLOAT_DELTA), 3),
                    val,
                    round(min(1.0, val + FLOAT_DELTA), 3),
                    round(min(1.0, val + FLOAT_DELTA * 2), 3),
                })
                axis_variants[field.name] = variants
            elif isinstance(val, int):
                variants = sorted({max(0, val - INT_DELTA), val, val + INT_DELTA})
                axis_variants[field.name] = variants
            else:
                axis_variants[field.name] = [val]

        for combo in itertools.product(*axis_variants.values()):
            candidate = SearchPoint(**dict(zip(axis_variants.keys(), combo)))
            # Re-apply monotonicity constraint
            if candidate.ctr_r1_thresh > candidate.ctr_r0_thresh and candidate.rtp_r1_thresh > candidate.rtp_r0_thresh:
                candidate_points.add(candidate)

    print(f"Refinement: {len(candidate_points):,} candidate points around top-{len(top_results)} …")

    rows: list[dict[str, Any]] = []
    for i, point in enumerate(candidate_points, start=1):
        overrides = point.to_scenario_overrides()
        error: str | None = None
        sim_result = None
        try:
            sim_result = run_simulation(scenario_dir, scenario_overrides=overrides)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        score = _score(sim_result) if sim_result is not None else float("-inf")
        wagons = sim_result.wagons_parked if sim_result is not None else 0
        loco = sim_result.locomotive_active_time_min if sim_result is not None else float("nan")

        row: dict[str, Any] = asdict(point)
        row["wagons_parked"] = wagons
        row["locomotive_active_time_min"] = loco
        row["score"] = score
        row["error"] = error
        rows.append(row)

        if i % 100 == 0:
            print(f"  {i:>6,} / {len(candidate_points):,} …", end="\r", flush=True)

    print(f"  Done — {len(rows):,} refinement points evaluated.   ")

    df = pd.DataFrame(rows)
    df.sort_values("score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    top_n = len(top_results)
    top_eval = [
        EvalResult(
            point=SearchPoint(**{k: row[k] for k in SearchPoint.__dataclass_fields__}),
            wagons_parked=int(row["wagons_parked"]),
            locomotive_active_time_min=float(row["locomotive_active_time_min"]),
            score=float(row["score"]),
            error=row["error"],
        )
        for _, row in df.head(top_n).iterrows()
    ]
    return df, top_eval


# ---------------------------------------------------------------------------
# Convenience entry point — honours SEARCH_DEPTH
# ---------------------------------------------------------------------------


def run_search(
    scenario_dir: Path,
    top_n: int = 20,
    depth: int | None = None,
) -> tuple[list[pd.DataFrame], list[EvalResult]]:
    """Run the full search pipeline (coarse grid + optional refinement phases).

    Args:
        scenario_dir: Base scenario directory.
        top_n: Number of top results to carry forward between phases.
        depth: Number of refinement phases after the coarse pass.
               Defaults to the module-level ``SEARCH_DEPTH`` constant.
               Set to 0 for coarse-only.

    Returns:
        ``(phase_dfs, best_results)`` where:
        - ``phase_dfs`` is a list of DataFrames, one per phase (phase 0 = coarse).
        - ``best_results`` is the final top-N list.
    """
    n_refine = SEARCH_DEPTH if depth is None else depth

    df_coarse, top = run_grid_search(scenario_dir, Resolution.COARSE, top_n=top_n)
    phase_dfs: list[pd.DataFrame] = [df_coarse]

    for phase in range(1, n_refine + 1):
        print(f"\n--- Refinement phase {phase} / {n_refine} ---")
        df_refined, top = refine_around(scenario_dir, top)
        phase_dfs.append(df_refined)

    return phase_dfs, top

