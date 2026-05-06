"""Round-7/8 wrapper: force ``Finalizer`` then ``terminate`` after stalls.

Round-5 Finalizer-route fix made every supervised optimal route end
with ``Finalizer`` — but round-5 traces showed the trained
``flybrain_sim_pretrain v6`` controller still **never** emits
``Finalizer`` at inference. Instead it loops on ``Planner`` /
``SchemaValidator`` until ``max_steps`` is hit, which costs LLM
tokens and never produces the ``final_answer`` component the
runtime verifier requires for math/research/tool_use tasks.

This wrapper is a **post-processing watchdog** around any wrapped
controller. On each call to :meth:`select_action` it:

1. Tracks ``produced_components`` snapshots across calls (per
   ``state.task_id``).
2. If ``state.step_id >= force_after`` **or** the produced set
   hasn't grown for ``stall_after`` consecutive steps, it overrides
   the wrapped controller's action with ``activate_agent(Finalizer)``
   when ``"final_answer"`` isn't yet produced, otherwise with
   ``terminate``.

Round-8 update: ``force_after`` and ``stall_after`` may be either a
single ``int`` (round-7 behaviour: same threshold for every task
type) or a ``dict[str, int]`` keyed by ``task_type`` (round-8: each
task type gets its own budget). Round-7 N=10 showed the round-7
default ``force_after=12`` matched manual_graph on
synthetic_routing (0.900) but regressed humaneval (0.500 vs 0.900)
because coding tasks legitimately need ~20 steps of plan→code→test
→debug iterations. Round-8 ships
``DEFAULT_FORCE_AFTER_BY_TASK = {coding: 28, math: 12, research: 16,
tool_use: 12}`` so the watchdog only short-circuits when a task
type is past its empirical "honest progress" window.

The wrapper has zero learned parameters and adds no LLM calls — the
``builder.from_runtime_sync`` cost is the same as the underlying
controller. It is therefore valid as a pure-CPU baseline that can be
benchmarked against ``flybrain_sim_pretrain`` (round-5 v6 head) and
``manual_graph`` on the same N=10 OpenRouter free-tier shard used in
round-6/round-7 to prove the fix is **architectural** rather than
training-data-bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flybrain.controller.base import Controller
from flybrain.runtime.state import RuntimeState

# Round-8 defaults — calibrated against manual_graph LLM-call depth
# observed in round-7 N=10 (humaneval 20.6, synthetic_routing 11.1).
# coding gets the largest budget because the optimal route includes
# Coder → TestRunner → Debugger × retries before Finalizer; math /
# research / tool_use have shorter optimal routes and so a smaller
# budget is sufficient.
DEFAULT_FORCE_AFTER_BY_TASK: dict[str, int] = {
    "coding": 28,
    "math": 12,
    "research": 16,
    "tool_use": 12,
}

DEFAULT_STALL_AFTER_BY_TASK: dict[str, int] = {
    "coding": 6,
    "math": 3,
    "research": 4,
    "tool_use": 3,
}


def _resolve_threshold(
    threshold: int | dict[str, int],
    task_type: str | None,
    fallback: int,
) -> int:
    """Pick the right threshold for the current task type.

    Falls back to ``fallback`` if the task type isn't in the dict
    (e.g. an unknown / new benchmark) so the watchdog stays safe for
    out-of-distribution callers."""
    if isinstance(threshold, int):
        return threshold
    if task_type is None:
        return fallback
    return threshold.get(task_type, fallback)


@dataclass(slots=True)
class _TaskTrace:
    """Per-task history used by the watchdog (one entry per task_id)."""

    last_size: int = 0
    stall_count: int = 0
    finalizer_emitted: bool = False


@dataclass
class FinalizerWatchdogController:
    """Wrap a :class:`Controller`, override stalls with Finalizer + terminate.

    The wrapped controller's logic is preserved when the run is making
    progress; the watchdog only intervenes once the trained controller
    has clearly stalled. Implements the ``Controller`` Protocol
    structurally — no base class needed.

    Round-8: ``force_after`` and ``stall_after`` may be a per-task-type
    dict (see ``DEFAULT_FORCE_AFTER_BY_TASK``). Passing a plain int
    reproduces the round-7 behaviour exactly.
    """

    inner: Controller
    finalizer_name: str = "Finalizer"
    final_answer_tag: str = "final_answer"
    force_after: int | dict[str, int] = 12
    stall_after: int | dict[str, int] = 3
    name: str = "flybrain_sim_pretrain_watchdog"
    _traces: dict[str, _TaskTrace] = field(default_factory=dict)

    @classmethod
    def from_bench_dirs(
        cls,
        inner: Controller,
        bench_dirs: list[str] | tuple[str, ...],
        *,
        baseline_name: str = "manual_graph",
        percentile: float = 0.90,
        min_samples: int = 2,
        fallback_force: int = 12,
        fallback_stall: int = 3,
        name: str = "flybrain_sim_pretrain_watchdog",
        finalizer_name: str = "Finalizer",
        final_answer_tag: str = "final_answer",
    ) -> FinalizerWatchdogController:
        """Round-9: build a watchdog whose per-task-type budgets are
        **calibrated** from manual_graph traces in ``bench_dirs``.

        See :mod:`flybrain.controller.watchdog_calibrator` for the
        calibration math. The classmethod is the canonical entry point
        for round-9: the trained controller stays the same, but the
        watchdog's ``force_after`` / ``stall_after`` dicts are derived
        empirically from the bench output (zero hand-tuning).
        """
        from flybrain.controller.watchdog_calibrator import WatchdogCalibration

        cal = WatchdogCalibration.from_bench_dirs(
            bench_dirs,
            baseline_name=baseline_name,
            percentile=percentile,
            min_samples=min_samples,
            fallback_force=fallback_force,
            fallback_stall=fallback_stall,
        )
        force = dict(cal.force_after) if cal.force_after else fallback_force
        stall = dict(cal.stall_after) if cal.stall_after else fallback_stall
        return cls(
            inner=inner,
            finalizer_name=finalizer_name,
            final_answer_tag=final_answer_tag,
            force_after=force,
            stall_after=stall,
            name=name,
        )

    def select_action(self, state: RuntimeState) -> dict[str, Any]:
        trace = self._traces.setdefault(state.task_id, _TaskTrace())
        produced = set(state.produced_components or set())
        if len(produced) > trace.last_size:
            trace.stall_count = 0
            trace.last_size = len(produced)
        else:
            trace.stall_count += 1

        # Once we have force-fired Finalizer and the runtime confirms
        # final_answer is now produced, terminate immediately. This is
        # checked first so the stall/budget gates can't accidentally
        # restart inner-controller passthrough after the forced finish.
        if trace.finalizer_emitted and self.final_answer_tag in produced:
            return {"kind": "terminate"}

        force_after = _resolve_threshold(self.force_after, state.task_type, fallback=12)
        stall_after = _resolve_threshold(self.stall_after, state.task_type, fallback=3)
        force = state.step_id >= force_after or trace.stall_count >= stall_after

        if force:
            if (
                self.final_answer_tag not in produced
                and self.finalizer_name in (state.available_agents or [])
                and not trace.finalizer_emitted
            ):
                trace.finalizer_emitted = True
                trace.stall_count = 0
                return {
                    "kind": "activate_agent",
                    "agent": self.finalizer_name,
                }
            return {"kind": "terminate"}

        return self.inner.select_action(state)


__all__ = [
    "DEFAULT_FORCE_AFTER_BY_TASK",
    "DEFAULT_STALL_AFTER_BY_TASK",
    "FinalizerWatchdogController",
]
