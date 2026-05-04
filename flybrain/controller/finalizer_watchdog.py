"""Round-7 wrapper: force ``Finalizer`` then ``terminate`` after stalls.

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
    """

    inner: Controller
    finalizer_name: str = "Finalizer"
    final_answer_tag: str = "final_answer"
    force_after: int = 12
    stall_after: int = 3
    name: str = "flybrain_sim_pretrain_watchdog"
    _traces: dict[str, _TaskTrace] = field(default_factory=dict)

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

        force = (
            state.step_id >= self.force_after
            or trace.stall_count >= self.stall_after
        )

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


__all__ = ["FinalizerWatchdogController"]
