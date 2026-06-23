from __future__ import annotations

import random
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from itertools import product
from typing import Any
from typing import ClassVar
from typing import Iterator
from typing import cast

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


# ---------------------------------------------------------------------------
# Enums  (str first so .value serialises transparently to JSON)
# ---------------------------------------------------------------------------


class TrackType(str, Enum):
    PARKING = "parking"
    RESOURCE_PARKING = "resource_parking"
    COLLECTION = "collection"
    RETROFIT = "retrofit"
    RETROFITTED = "retrofitted"
    WORKSHOP = "workshop"
    MAINLINE = "mainline"


class SelectionStrategy(str, Enum):
    LEAST_OCCUPIED = "least_occupied"
    # MOST_AVAILABLE = "most_available"  # NOTE: alias of LEAST_OCCUPIED — left out
    FIRST_AVAILABLE = "first_available"
    BEST_FIT = "best_fit"
    ROUND_ROBIN = "round_robin"
    SHORTEST_QUEUE = "shortest_queue"
    RANDOM = "random"


class Condition(str, Enum):
    SOURCE_FILL_ABOVE = "source_fill_above"
    SOURCE_FILL_BELOW = "source_fill_below"
    TARGET_FILL_ABOVE = "target_fill_above"
    TARGET_FILL_BELOW = "target_fill_below"
    TARGET_IDLE = "target_idle"


# ---------------------------------------------------------------------------
# Domain models with validation
# ---------------------------------------------------------------------------


class Track(BaseModel):
    type: TrackType


class PriorityRule(BaseModel):
    condition: Condition  # typed enum, not bare str
    threshold: float = Field(ge=0.0, le=1.0)
    priority: int = Field(ge=0)

    @model_validator(mode="after")
    def target_idle_requires_zero_threshold(self) -> PriorityRule:
        """TARGET_IDLE carries no fill-level information — fix threshold to 0.0.

        Enforcing a canonical value eliminates semantically identical variants
        that differ only in an otherwise irrelevant threshold field.
        """
        if self.condition == Condition.TARGET_IDLE and self.threshold != 0.0:
            raise ValueError("TARGET_IDLE condition must use threshold=0.0")
        return self


class HoldCondition(BaseModel):
    condition: Condition
    threshold: float = Field(le=1.0, ge=0.0)


class TaskPriorityRule(BaseModel):
    base_priority: int = Field(ge=0)
    rules: list[PriorityRule] = Field(max_length=5, default_factory=list)
    hold_until: HoldCondition | None = None
    max_hold_time: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_task_rule(self) -> TaskPriorityRule:
        """Enforce structural validity and canonical form.

        Canonical-form rules
        --------------------
        1. ``max_hold_time`` is only meaningful together with ``hold_until``.
        2. For each condition, rules must appear in **strictly ascending
           threshold order** with no duplicates.  This is the canonical form:
           it eliminates permutation duplicates so that two logically identical
           rule sets always produce the same object and the same hash.

        Why canonical form matters for grid search
        ------------------------------------------
        Without it, (rule_A_0.3, rule_A_0.7) and (rule_A_0.7, rule_A_0.3)
        would both be accepted as different grid points even though they
        produce identical simulation behaviour.  Enforcing ascending order
        means each logical configuration has exactly one representation.
        """
        if self.max_hold_time is not None and self.hold_until is None:
            raise ValueError("max_hold_time requires hold_until to be set")

        by_cond: dict[str, list[float]] = {}
        for rule in self.rules:
            by_cond.setdefault(rule.condition.value, []).append(rule.threshold)

        for cond, thresholds in by_cond.items():
            sorted_unique = sorted(set(thresholds))
            if len(sorted_unique) < len(thresholds):
                raise ValueError(
                    f"Duplicate threshold for condition '{cond}': {thresholds}"
                )
            if thresholds != sorted_unique:
                raise ValueError(
                    f"Rules for '{cond}' must be in strictly ascending threshold order "
                    f"(got {thresholds}).  Sort rules by threshold before constructing."
                )

        return self


class TaskPriorities(BaseModel):
    collection_to_retrofit: TaskPriorityRule
    retrofit_to_workshop: TaskPriorityRule
    workshop_to_retrofitted: TaskPriorityRule
    retrofitted_to_parking: TaskPriorityRule


class Configuration(BaseModel):
    tracks: list[Track]  # NOTE: fixed per scenario — not varied in search
    task_priorities: TaskPriorities
    collection_track_strategy: SelectionStrategy
    retrofit_selection_strategy: SelectionStrategy
    retrofitted_selection_strategy: SelectionStrategy
    workshop_selection_strategy: SelectionStrategy
    parking_selection_strategy: SelectionStrategy

    def to_scenario_overrides(self) -> dict[str, Any]:
        """Serialise to the ``scenario_overrides`` dict expected by the simulation harness."""

        def _rule(r: PriorityRule) -> dict[str, Any]:
            return {"condition": r.condition.value, "threshold": r.threshold, "priority": r.priority}

        def _task(t: TaskPriorityRule) -> dict[str, Any]:
            d: dict[str, Any] = {
                "base_priority": t.base_priority,
                "rules": [_rule(r) for r in t.rules],
            }
            if t.hold_until is not None:
                d["hold_until"] = {
                    "condition": t.hold_until.condition.value,
                    "threshold": t.hold_until.threshold,
                }
            if t.max_hold_time is not None:
                d["max_hold_time"] = t.max_hold_time
            return d

        tp = self.task_priorities
        return {
            "task_priorities": {
                "collection_to_retrofit": _task(tp.collection_to_retrofit),
                "retrofit_to_workshop": _task(tp.retrofit_to_workshop),
                "workshop_to_retrofitted": _task(tp.workshop_to_retrofitted),
                "retrofitted_to_parking": _task(tp.retrofitted_to_parking),
            },
            "collection_track_strategy": self.collection_track_strategy.value,
            "retrofit_selection_strategy": self.retrofit_selection_strategy.value,
            "retrofitted_selection_strategy": self.retrofitted_selection_strategy.value,
            "workshop_selection_strategy": self.workshop_selection_strategy.value,
            "parking_selection_strategy": self.parking_selection_strategy.value,
        }


# ---------------------------------------------------------------------------
# Search space machinery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RuleSlot:
    """Discrete axis for one rule slot within a task's rule list.

    A slot generates a set of candidate :class:`PriorityRule` objects (one per
    threshold × priority combination) plus ``None`` when the slot is optional
    (= rule may be absent in some configurations).
    """

    condition: Condition
    thresholds: tuple[float, ...]
    priorities: tuple[int, ...]
    optional: bool = True  # if True, None (slot inactive) is also valid

    def options(self) -> list[PriorityRule | None]:
        """All valid :class:`PriorityRule` values for this slot, plus ``None`` when optional."""
        opts: list[PriorityRule | None] = [None] if self.optional else []
        for thresh in self.thresholds:
            for prio in self.priorities:
                try:
                    opts.append(
                        PriorityRule(condition=self.condition, threshold=thresh, priority=prio)
                    )
                except Exception:
                    pass  # validator rejected (e.g. TARGET_IDLE with non-zero threshold)
        return opts


@dataclass
class _TaskSpace:
    """Search axes for one :class:`TaskPriorityRule`."""

    base_priorities: tuple[int, ...]
    rule_slots: tuple[_RuleSlot, ...] = field(default_factory=tuple)
    hold_condition: Condition | None = None  # None = no hold option at all
    hold_thresholds: tuple[float, ...] = ()            # only used when hold_condition is set
    hold_optional: bool = True                         # allow hold_until = None
    max_hold_times: tuple[float | None, ...] = (None,) # None entry = no max_hold_time cap

    def _hold_options(self) -> list[HoldCondition | None]:
        opts: list[HoldCondition | None] = (
            [None] if (self.hold_condition is None or self.hold_optional) else []
        )
        if self.hold_condition is not None:
            for thresh in self.hold_thresholds:
                opts.append(HoldCondition(condition=self.hold_condition, threshold=thresh))
        return opts

    def iter_task_rules(self) -> Iterator[TaskPriorityRule]:
        """Yield every valid :class:`TaskPriorityRule` for this task's axes."""
        slot_options = [slot.options() for slot in self.rule_slots]
        hold_opts = self._hold_options()
        n_slots = len(self.rule_slots)

        for combo in product(self.base_priorities, *slot_options, hold_opts, self.max_hold_times):
            base_prio = cast(int, combo[0])
            slot_vals = combo[1 : 1 + n_slots]
            hold = cast("HoldCondition | None", combo[1 + n_slots])
            max_hold_t = cast("float | None", combo[2 + n_slots])

            if max_hold_t is not None and hold is None:
                continue  # structurally invalid: max_hold_time without hold_until

            rules: list[PriorityRule] = [r for r in slot_vals if isinstance(r, PriorityRule)]
            try:
                yield TaskPriorityRule(
                    base_priority=base_prio,
                    rules=rules,
                    hold_until=hold,
                    max_hold_time=max_hold_t,
                )
            except Exception:
                continue  # validator rejected (e.g. non-ascending thresholds within a condition)


@dataclass
class SearchSpace:
    """Generates a grid of valid :class:`Configuration` objects.

    The grid is the cartesian product of all valid task-rule combinations and
    strategy options.  Invalid combinations are filtered by the Pydantic
    validators on :class:`TaskPriorityRule`.

    Use :meth:`coarse` or :meth:`fine` for built-in presets, or supply custom
    ``task_spaces`` for full control over which axes are varied.

    Example::

        tracks = [Track(type=TrackType.COLLECTION), ...]
        for cfg in SearchSpace.coarse(tracks):
            result = run_simulation(scenario_dir, cfg.to_scenario_overrides())

    The space is a lazy iterator — no :class:`Configuration` objects are
    created until you iterate.
    """

    _TASK_NAMES: ClassVar[tuple[str, ...]] = (
        "collection_to_retrofit",
        "retrofit_to_workshop",
        "workshop_to_retrofitted",
        "retrofitted_to_parking",
    )

    tracks: list[Track]
    task_spaces: dict[str, _TaskSpace]
    strategy_options: dict[str, list[SelectionStrategy]] = field(
        default_factory=lambda: {
            "collection_track_strategy": [SelectionStrategy.FIRST_AVAILABLE],
            "retrofit_selection_strategy": [SelectionStrategy.LEAST_OCCUPIED],
            "retrofitted_selection_strategy": [SelectionStrategy.LEAST_OCCUPIED],
            "workshop_selection_strategy": [SelectionStrategy.LEAST_OCCUPIED],
            "parking_selection_strategy": [SelectionStrategy.LEAST_OCCUPIED],
        }
    )

    def iter_grid(self) -> Iterator[Configuration]:
        """Lazily yield every valid :class:`Configuration` in this search space."""
        # Materialise per-task rule lists once (small, O(k) per task)
        task_rule_lists = [
            list(self.task_spaces[name].iter_task_rules()) for name in self._TASK_NAMES
        ]
        s_opts = [
            self.strategy_options.get("collection_track_strategy", [SelectionStrategy.FIRST_AVAILABLE]),
            self.strategy_options.get("retrofit_selection_strategy", [SelectionStrategy.LEAST_OCCUPIED]),
            self.strategy_options.get("retrofitted_selection_strategy", [SelectionStrategy.LEAST_OCCUPIED]),
            self.strategy_options.get("workshop_selection_strategy", [SelectionStrategy.LEAST_OCCUPIED]),
            self.strategy_options.get("parking_selection_strategy", [SelectionStrategy.LEAST_OCCUPIED]),
        ]

        for ctr, rtw, wtr, rtp in product(*task_rule_lists):
            for s_coll, s_retro, s_retrofitted, s_ws, s_park in product(*s_opts):
                yield Configuration(
                    tracks=self.tracks,
                    task_priorities=TaskPriorities(
                        collection_to_retrofit=ctr,
                        retrofit_to_workshop=rtw,
                        workshop_to_retrofitted=wtr,
                        retrofitted_to_parking=rtp,
                    ),
                    collection_track_strategy=s_coll,
                    retrofit_selection_strategy=s_retro,
                    retrofitted_selection_strategy=s_retrofitted,
                    workshop_selection_strategy=s_ws,
                    parking_selection_strategy=s_park,
                )

    def __iter__(self) -> Iterator[Configuration]:
        return self.iter_grid()

    def sample(self, n: int, rng: random.Random | None = None) -> list[Configuration]:
        """Return ``n`` configurations drawn uniformly at random (without replacement if possible).

        Materialises the per-task rule lists once (small) and picks random
        indices — never materialises the full Cartesian product grid.

        Args:
            n: Number of configurations to sample.
            rng: Optional :class:`random.Random` instance for reproducibility.
                 Defaults to the global random state.

        Returns:
            List of ``n`` distinct :class:`Configuration` objects chosen
            uniformly at random from the full search space.
        """
        _rng = rng or random.Random()
        task_rule_lists = [
            list(self.task_spaces[name].iter_task_rules()) for name in self._TASK_NAMES
        ]
        s_opts = [
            self.strategy_options.get("collection_track_strategy", [SelectionStrategy.FIRST_AVAILABLE]),
            self.strategy_options.get("retrofit_selection_strategy", [SelectionStrategy.LEAST_OCCUPIED]),
            self.strategy_options.get("retrofitted_selection_strategy", [SelectionStrategy.LEAST_OCCUPIED]),
            self.strategy_options.get("workshop_selection_strategy", [SelectionStrategy.LEAST_OCCUPIED]),
            self.strategy_options.get("parking_selection_strategy", [SelectionStrategy.LEAST_OCCUPIED]),
        ]
        sizes = [len(lst) for lst in task_rule_lists] + [len(o) for o in s_opts]
        total = 1
        for s in sizes:
            total *= s

        # Clamp n to the total space size
        n = min(n, total)

        # Sample unique flat indices, then decode each
        indices = _rng.sample(range(total), n)
        configs: list[Configuration] = []
        for idx in indices:
            coords: list[int] = []
            remainder = idx
            for s in reversed(sizes):
                coords.append(remainder % s)
                remainder //= s
            coords.reverse()

            ctr = task_rule_lists[0][coords[0]]
            rtw = task_rule_lists[1][coords[1]]
            wtr = task_rule_lists[2][coords[2]]
            rtp = task_rule_lists[3][coords[3]]
            s_coll = s_opts[0][coords[4]]
            s_retro = s_opts[1][coords[5]]
            s_retrofitted = s_opts[2][coords[6]]
            s_ws = s_opts[3][coords[7]]
            s_park = s_opts[4][coords[8]]

            configs.append(
                Configuration(
                    tracks=self.tracks,
                    task_priorities=TaskPriorities(
                        collection_to_retrofit=ctr,
                        retrofit_to_workshop=rtw,
                        workshop_to_retrofitted=wtr,
                        retrofitted_to_parking=rtp,
                    ),
                    collection_track_strategy=s_coll,
                    retrofit_selection_strategy=s_retro,
                    retrofitted_selection_strategy=s_retrofitted,
                    workshop_selection_strategy=s_ws,
                    parking_selection_strategy=s_park,
                )
            )
        return configs

    def task_neighbors(self, cfg: Configuration, task_name: str) -> list[Configuration]:
        """All configurations that differ from *cfg* only in the given task's rule.

        Used by coordinate descent: fix all other tasks and enumerate every
        valid variant for ``task_name``.

        Args:
            cfg: The current configuration.
            task_name: One of the four task keys (e.g. ``"collection_to_retrofit"``).

        Returns:
            List of :class:`Configuration` objects, one per valid
            :class:`TaskPriorityRule` for ``task_name`` (excluding the
            current rule).
        """
        tp = cfg.task_priorities
        current_task: TaskPriorityRule = getattr(tp, task_name)
        neighbors: list[Configuration] = []
        for candidate in self.task_spaces[task_name].iter_task_rules():
            if candidate == current_task:
                continue
            updated_tp = tp.model_copy(update={task_name: candidate})
            neighbors.append(cfg.model_copy(update={"task_priorities": updated_tp}))
        return neighbors

    def grid_size(self) -> dict[str, int]:
        """Return valid rule counts per task and the total estimated grid size.

        Iterates each task's rule space independently (cheap) and multiplies.
        Does **not** iterate the full Configuration grid.
        """
        counts: dict[str, int] = {}
        total = 1
        for name in self._TASK_NAMES:
            n = sum(1 for _ in self.task_spaces[name].iter_task_rules())
            counts[name] = n
            total *= n
        for opts in self.strategy_options.values():
            total *= len(opts)
        counts["total"] = total
        return counts

    @classmethod
    def coarse(cls, tracks: list[Track]) -> SearchSpace:
        """~10–50k valid configurations for initial broad exploration.

        Rule thresholds use 2 levels per slot, base priorities use 2–3 levels.
        Hold conditions are optional.  Strategies are fixed at scenario defaults.
        """
        return cls(
            tracks=tracks,
            task_spaces={
                "collection_to_retrofit": _TaskSpace(
                    base_priorities=(1, 2, 3),
                    rule_slots=(
                        _RuleSlot(Condition.SOURCE_FILL_ABOVE, (0.4, 0.7), (1,)),
                        _RuleSlot(Condition.SOURCE_FILL_ABOVE, (0.7, 0.9), (0,)),
                    ),
                    hold_condition=Condition.TARGET_FILL_BELOW,
                    hold_thresholds=(0.7, 0.85),
                    hold_optional=True,
                    max_hold_times=(None, 30.0),
                ),
                "retrofit_to_workshop": _TaskSpace(
                    base_priorities=(2, 3),
                    rule_slots=(
                        _RuleSlot(Condition.TARGET_IDLE, (0.0,), (0,)),
                        _RuleSlot(Condition.SOURCE_FILL_BELOW, (0.1, 0.2), (5,)),
                    ),
                ),
                "workshop_to_retrofitted": _TaskSpace(
                    base_priorities=(1, 2),
                ),
                "retrofitted_to_parking": _TaskSpace(
                    base_priorities=(2, 3),
                    rule_slots=(
                        _RuleSlot(Condition.SOURCE_FILL_ABOVE, (0.4, 0.65), (1,)),
                        _RuleSlot(Condition.SOURCE_FILL_ABOVE, (0.7, 0.85), (0,)),
                    ),
                    hold_condition=Condition.SOURCE_FILL_ABOVE,
                    hold_thresholds=(0.15, 0.35),
                    hold_optional=True,
                    max_hold_times=(None, 45.0),
                ),
            },
        )

    @classmethod
    def fine(cls, tracks: list[Track]) -> SearchSpace:
        """~100k–1M valid configurations for focused refinement around promising regions.

        Rule thresholds use 4 levels per slot, base priorities use 4 levels.
        """
        return cls(
            tracks=tracks,
            task_spaces={
                "collection_to_retrofit": _TaskSpace(
                    base_priorities=(1, 2, 3, 4),
                    rule_slots=(
                        _RuleSlot(Condition.SOURCE_FILL_ABOVE, (0.3, 0.5, 0.65, 0.75), (1,)),
                        _RuleSlot(Condition.SOURCE_FILL_ABOVE, (0.65, 0.75, 0.85, 0.95), (0,)),
                    ),
                    hold_condition=Condition.TARGET_FILL_BELOW,
                    hold_thresholds=(0.6, 0.75, 0.85, 0.95),
                    hold_optional=True,
                    max_hold_times=(None, 15.0, 30.0, 60.0),
                ),
                "retrofit_to_workshop": _TaskSpace(
                    base_priorities=(2, 3, 4),
                    rule_slots=(
                        _RuleSlot(Condition.TARGET_IDLE, (0.0,), (0,)),
                        _RuleSlot(Condition.SOURCE_FILL_BELOW, (0.05, 0.1, 0.2), (5,)),
                    ),
                ),
                "workshop_to_retrofitted": _TaskSpace(
                    base_priorities=(1, 2, 3),
                ),
                "retrofitted_to_parking": _TaskSpace(
                    base_priorities=(2, 3, 4),
                    rule_slots=(
                        _RuleSlot(Condition.SOURCE_FILL_ABOVE, (0.4, 0.55, 0.7, 0.8), (1,)),
                        _RuleSlot(Condition.SOURCE_FILL_ABOVE, (0.7, 0.8, 0.9, 0.95), (0,)),
                    ),
                    hold_condition=Condition.SOURCE_FILL_ABOVE,
                    hold_thresholds=(0.1, 0.2, 0.3, 0.45),
                    hold_optional=True,
                    max_hold_times=(None, 20.0, 40.0, 60.0, 90.0),
                ),
            },
        )

