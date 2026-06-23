"""Search-space modeller for the PopUpSim optimizer.

The modeller spans the parameter space over ``task_priorities`` in a
scenario.json and translates each point into a ``scenario_overrides`` dict
that the harness can consume.

Design decisions
----------------
* **Fixed rule topology** — the *which* conditions are evaluated per task is
  fixed (matching the structure of the existing priority_dispatch examples).
  Only the numeric parameters (thresholds, base priorities, hold times) vary.
* **Coarse / fine resolution** — a ``Resolution`` enum controls step-sizes,
  enabling a two-phase grid search without changing calling code.
* **Constraint filtering** — invalid combinations (e.g. non-monotone
  thresholds) are dropped before the grid is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any
from typing import Iterator


# ---------------------------------------------------------------------------
# Resolution enum
# ---------------------------------------------------------------------------


class Resolution(Enum):
    """Controls how many discrete levels each axis gets."""

    COARSE = "coarse"  # 2-3 levels → small grid, fast exploration
    FINE = "fine"  # 4-5 levels → large grid, detailed search


# ---------------------------------------------------------------------------
# SearchPoint — one point in the parameter space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchPoint:
    """A single configuration in the task-priority search space.

    Naming convention: ``{task_abbrev}_{field}``
    Tasks: ctr = collection_to_retrofit, rtw = retrofit_to_workshop,
           wtr = workshop_to_retrofitted, rtp = retrofitted_to_parking
    """

    # --- collection_to_retrofit ---
    ctr_base: int
    """base_priority (no rule matched)."""
    ctr_hold_thresh: float
    """hold_until target_fill_below threshold (1.0 = disabled)."""
    ctr_max_hold: float
    """max_hold_time in minutes."""
    ctr_r0_thresh: float
    """Rule 0: source_fill_above → priority 1."""
    ctr_r1_thresh: float
    """Rule 1: source_fill_above → priority 0 (escalation). Must be > r0."""

    # --- retrofit_to_workshop ---
    rtw_base: int
    """base_priority."""
    rtw_deprio_thresh: float
    """Rule 1: source_fill_below threshold → priority 5 (de-prioritise)."""

    # --- workshop_to_retrofitted ---
    wtr_base: int
    """base_priority (no rules for this task)."""

    # --- retrofitted_to_parking ---
    rtp_base: int
    """base_priority."""
    rtp_hold_thresh: float
    """hold_until source_fill_above threshold."""
    rtp_max_hold: float
    """max_hold_time in minutes."""
    rtp_r0_thresh: float
    """Rule 0: source_fill_above → priority 1."""
    rtp_r1_thresh: float
    """Rule 1: source_fill_above → priority 0 (escalation). Must be > r0."""

    def to_scenario_overrides(self) -> dict[str, Any]:
        """Convert this point to a ``scenario_overrides`` dict for the harness."""
        ctr: dict[str, Any] = {
            "base_priority": self.ctr_base,
            "max_hold_time": self.ctr_max_hold,
            "rules": [
                {"condition": "source_fill_above", "threshold": self.ctr_r0_thresh, "priority": 1},
                {"condition": "source_fill_above", "threshold": self.ctr_r1_thresh, "priority": 0},
            ],
        }
        if self.ctr_hold_thresh < 1.0:
            ctr["hold_until"] = {"condition": "target_fill_below", "threshold": self.ctr_hold_thresh}
        else:
            ctr.pop("hold_until", None)

        rtw: dict[str, Any] = {
            "base_priority": self.rtw_base,
            "rules": [
                {"condition": "target_idle", "threshold": 0.0, "priority": 0},
                {"condition": "source_fill_below", "threshold": self.rtw_deprio_thresh, "priority": 5},
            ],
        }

        wtr: dict[str, Any] = {
            "base_priority": self.wtr_base,
            "rules": [],
        }

        rtp: dict[str, Any] = {
            "base_priority": self.rtp_base,
            "hold_until": {"condition": "source_fill_above", "threshold": self.rtp_hold_thresh},
            "max_hold_time": self.rtp_max_hold,
            "rules": [
                {"condition": "source_fill_above", "threshold": self.rtp_r0_thresh, "priority": 1},
                {"condition": "source_fill_above", "threshold": self.rtp_r1_thresh, "priority": 0},
            ],
        }

        return {
            "task_priorities": {
                "collection_to_retrofit": ctr,
                "retrofit_to_workshop": rtw,
                "workshop_to_retrofitted": wtr,
                "retrofitted_to_parking": rtp,
            }
        }


# ---------------------------------------------------------------------------
# Axis definitions per resolution
# ---------------------------------------------------------------------------

_AXES_COARSE: dict[str, list[Any]] = {
    # collection_to_retrofit
    "ctr_base": [1, 2, 3],
    "ctr_hold_thresh": [0.7, 0.85, 1.0],   # 1.0 = no hold
    "ctr_max_hold": [20.0, 45.0],
    "ctr_r0_thresh": [0.4, 0.7],
    "ctr_r1_thresh": [0.7, 0.9],
    # retrofit_to_workshop
    "rtw_base": [2, 3],
    "rtw_deprio_thresh": [0.05, 0.15],
    # workshop_to_retrofitted (fixed topology, only base_priority varies)
    "wtr_base": [1, 2],
    # retrofitted_to_parking
    "rtp_base": [2, 3],
    "rtp_hold_thresh": [0.15, 0.35],
    "rtp_max_hold": [30.0, 60.0],
    "rtp_r0_thresh": [0.4, 0.65],
    "rtp_r1_thresh": [0.7, 0.9],
}

_AXES_FINE: dict[str, list[Any]] = {
    # collection_to_retrofit
    "ctr_base": [1, 2, 3, 4],
    "ctr_hold_thresh": [0.6, 0.75, 0.85, 0.95, 1.0],
    "ctr_max_hold": [15.0, 30.0, 45.0, 60.0],
    "ctr_r0_thresh": [0.3, 0.5, 0.65, 0.75],
    "ctr_r1_thresh": [0.65, 0.75, 0.85, 0.95],
    # retrofit_to_workshop
    "rtw_base": [2, 3, 4],
    "rtw_deprio_thresh": [0.05, 0.1, 0.2],
    # workshop_to_retrofitted
    "wtr_base": [1, 2, 3],
    # retrofitted_to_parking
    "rtp_base": [2, 3, 4],
    "rtp_hold_thresh": [0.1, 0.2, 0.3, 0.45],
    "rtp_max_hold": [20.0, 40.0, 60.0, 90.0],
    "rtp_r0_thresh": [0.4, 0.55, 0.7, 0.8],
    "rtp_r1_thresh": [0.7, 0.8, 0.9, 0.95],
}


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------


def _is_valid(point: SearchPoint) -> bool:
    """Return False for structurally invalid or redundant parameter combinations."""
    # Monotonicity: escalation rule must have a higher threshold than the base rule
    if point.ctr_r1_thresh <= point.ctr_r0_thresh:
        return False
    if point.rtp_r1_thresh <= point.rtp_r0_thresh:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def iter_grid(resolution: Resolution = Resolution.COARSE) -> Iterator[SearchPoint]:
    """Yield all valid SearchPoints for the given resolution.

    Args:
        resolution: ``COARSE`` for a fast first-pass exploration,
                    ``FINE`` for a detailed follow-up search.

    Yields:
        SearchPoint instances that satisfy all constraints.
    """
    axes = _AXES_COARSE if resolution == Resolution.COARSE else _AXES_FINE
    keys = list(axes.keys())
    for values in product(*axes.values()):
        point = SearchPoint(**dict(zip(keys, values)))
        if _is_valid(point):
            yield point


def grid_size(resolution: Resolution = Resolution.COARSE) -> tuple[int, int]:
    """Return (total_combinations, valid_combinations) for the given resolution.

    Useful for estimating runtime before kicking off a search.
    """
    axes = _AXES_COARSE if resolution == Resolution.COARSE else _AXES_FINE
    total = 1
    for v in axes.values():
        total *= len(v)
    valid = sum(1 for _ in iter_grid(resolution))
    return total, valid
