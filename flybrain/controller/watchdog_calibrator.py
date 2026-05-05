"""Round-9: derive per-task-type watchdog budgets from manual_graph traces.

Round-8 shipped the ``int | dict[str, int]`` widening on
:class:`flybrain.controller.finalizer_watchdog.FinalizerWatchdogController`
and a hand-tuned ``DEFAULT_FORCE_AFTER_BY_TASK = {coding: 28, math: 12,
research: 16, tool_use: 12}``. The round-8 caveat (§6 of
``docs/round8_pertasktype.md``) flagged the obvious next step:

> *"Per-task-type budgets are tuned, not learned. New benchmarks with
>  different optimal-route lengths would still need a manual entry."*

Round-9 replaces the hand-tuned table with a **calibration step**:
read ``manual_graph`` traces from one or more bench output dirs,
group ``totals.llm_calls`` by ``task_type`` (only over
``verification.passed=True`` traces — i.e. successful canonical routes),
and pick a high-percentile budget per group. The watchdog factory
``FinalizerWatchdogController.from_bench_dirs`` wires this output back
into the per-task-type ``force_after`` dict so the v3 baseline is
**zero-shot to new benchmarks** as long as ``manual_graph`` traces for
those benchmarks are available.

Stall budgets are derived as a fixed fraction of the force budget
(``ceil(force_after * 0.25)`` clamped to ≥2) — empirical inspection
of round-7 traces shows ``manual_graph`` rarely stalls more than ~25%
of its productive depth before progressing.

The calibrator is a **pure function** of the bench output directory
contents — no LLM calls, no training, deterministic given the input
traces. This is why round-9 stays at 0 ₽ even though it is a strict
generalisation of the round-8 lever.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Round-9 calibration defaults. ``percentile=0.90`` matches the
# heuristic the round-8 hand-tuned table approximated (humaneval P90
# ≈ 28-30 in round-7 manual_graph; round-8's coding=28 is the same
# value rounded down).
DEFAULT_PERCENTILE: float = 0.90
# A task type with <3 successful manual_graph traces falls back to
# the hand-tuned defaults — 2 samples can collapse to a misleadingly
# low percentile (e.g. tool_use [3, 3] → P90 = 3) that would starve
# the trained controller.
DEFAULT_MIN_SAMPLES: int = 3
DEFAULT_STALL_RATIO: float = 0.25
DEFAULT_FALLBACK_FORCE: int = 12
DEFAULT_FALLBACK_STALL: int = 3
# Even with adequate samples the calibrator never returns a budget
# below this floor — the trained controller has its own warm-up and
# a budget of 3-4 would clip every route before progress is possible.
DEFAULT_FORCE_FLOOR: int = 8
DEFAULT_STALL_FLOOR: int = 2
DEFAULT_BASELINE_NAME: str = "manual_graph"


@dataclass(frozen=True, slots=True)
class WatchdogCalibration:
    """Per-task-type budgets derived from successful manual_graph traces.

    Attributes:
        force_after: ``{task_type: P{percentile} of llm_calls}``,
            rounded up. Falls back to ``fallback_force`` for
            task types with fewer than ``min_samples`` successful
            traces (so unknown task types still get a safe budget).
        stall_after: ``ceil(force_after[t] * stall_ratio)`` clamped
            to ≥2.
        n_samples_per_task: how many successful traces fed each
            task type's budget — useful for the round-9 write-up
            and for downstream "did this calibration have enough
            evidence" checks.
        percentile: the percentile actually used (echoed back so
            the calibration is self-describing).
    """

    force_after: dict[str, int] = field(default_factory=dict)
    stall_after: dict[str, int] = field(default_factory=dict)
    n_samples_per_task: dict[str, int] = field(default_factory=dict)
    percentile: float = DEFAULT_PERCENTILE

    @classmethod
    def from_bench_dirs(
        cls,
        dirs: Iterable[str | Path],
        *,
        baseline_name: str = DEFAULT_BASELINE_NAME,
        percentile: float = DEFAULT_PERCENTILE,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        fallback_force: int = DEFAULT_FALLBACK_FORCE,
        fallback_stall: int = DEFAULT_FALLBACK_STALL,
        stall_ratio: float = DEFAULT_STALL_RATIO,
        force_floor: int = DEFAULT_FORCE_FLOOR,
        stall_floor: int = DEFAULT_STALL_FLOOR,
    ) -> WatchdogCalibration:
        """Calibrate from bench output dirs (one or more).

        Each ``dir`` is expected to follow the layout produced by
        ``flybrain-py bench`` — i.e.
        ``<dir>/<baseline_name>/<benchmark>/*.trace.json``. All other
        baselines/benchmarks are ignored. Files that fail to parse or
        miss the expected keys are skipped (calibration is best-
        effort and never crashes the controller factory).
        """
        if not 0.0 < percentile < 1.0:
            raise ValueError(f"percentile must be in (0, 1), got {percentile}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be ≥1, got {min_samples}")

        calls_by_type: dict[str, list[int]] = {}
        for d in dirs:
            base = Path(d) / baseline_name
            if not base.is_dir():
                continue
            for bench_dir in base.iterdir():
                if not bench_dir.is_dir():
                    continue
                for trace_path in bench_dir.glob("*.trace.json"):
                    record = _load_trace(trace_path)
                    if record is None:
                        continue
                    task_type, n_calls = record
                    calls_by_type.setdefault(task_type, []).append(n_calls)

        force_after: dict[str, int] = {}
        stall_after: dict[str, int] = {}
        n_samples: dict[str, int] = {}
        for task_type, samples in calls_by_type.items():
            n_samples[task_type] = len(samples)
            if len(samples) < min_samples:
                force_after[task_type] = fallback_force
                stall_after[task_type] = fallback_stall
                continue
            raw = _percentile_int(samples, percentile)
            f = max(force_floor, raw)
            force_after[task_type] = f
            stall_after[task_type] = max(stall_floor, math.ceil(f * stall_ratio))

        return cls(
            force_after=force_after,
            stall_after=stall_after,
            n_samples_per_task=n_samples,
            percentile=percentile,
        )


def _load_trace(path: Path) -> tuple[str, int] | None:
    """Parse a single trace.json and return ``(task_type, llm_calls)``.

    Returns ``None`` if the trace failed verification (so a degenerate
    route doesn't pollute the calibration), if required keys are
    missing, or if the file is unreadable. The calibrator must never
    raise on bad data — calibration is a best-effort enhancement of
    the watchdog, not a hard dependency.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    verification = payload.get("verification") or {}
    if not verification.get("passed"):
        return None
    task_type = payload.get("task_type")
    totals = payload.get("totals") or {}
    n_calls = totals.get("llm_calls")
    if not isinstance(task_type, str) or not isinstance(n_calls, int):
        return None
    if n_calls <= 0:
        return None
    return task_type, n_calls


def _percentile_int(samples: list[int], q: float) -> int:
    """Linear-interpolation percentile, rounded **up** to int.

    Rounding up is the conservative choice for a force_after budget:
    if the calibrator is unsure between 27 and 28, we'd rather give
    the controller one extra step than cut Debugger short by one.
    """
    if not samples:
        raise ValueError("samples is empty")
    sorted_samples = sorted(samples)
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = q * (len(sorted_samples) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_samples[lo]
    frac = rank - lo
    interp = sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo])
    return math.ceil(interp)


__all__ = [
    "DEFAULT_BASELINE_NAME",
    "DEFAULT_FALLBACK_FORCE",
    "DEFAULT_FALLBACK_STALL",
    "DEFAULT_FORCE_FLOOR",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_PERCENTILE",
    "DEFAULT_STALL_FLOOR",
    "DEFAULT_STALL_RATIO",
    "WatchdogCalibration",
]
