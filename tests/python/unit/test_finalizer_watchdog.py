"""Round-7: tests for ``FinalizerWatchdogController``."""

from __future__ import annotations

from typing import Any

from flybrain.controller.finalizer_watchdog import FinalizerWatchdogController
from flybrain.runtime.state import RuntimeState


class _StuckController:
    """Always picks the same agent — a stand-in for the round-5 v6
    failure mode where the trained controller loops on a non-progress
    agent (Planner / SchemaValidator) until ``max_steps``."""

    name = "stuck"

    def __init__(self, agent: str = "Planner") -> None:
        self._agent = agent

    def select_action(self, _state: RuntimeState) -> dict[str, Any]:
        return {"kind": "activate_agent", "agent": self._agent}


def _state(
    *,
    step_id: int,
    produced: set[str] | None = None,
    task_id: str = "t1",
    available_agents: list[str] | None = None,
) -> RuntimeState:
    return RuntimeState(
        task_id=task_id,
        task_type="math",
        prompt="2+2=?",
        step_id=step_id,
        available_agents=available_agents or ["Planner", "MathSolver", "Finalizer"],
        pending_inbox={},
        last_active_agent=None,
        produced_components=produced or set(),
    )


def test_passes_through_actions_while_progress_is_made() -> None:
    """If ``produced_components`` keeps growing, watchdog must NOT
    intervene — the wrapped controller's choices are forwarded as-is."""
    inner = _StuckController("Planner")
    wd = FinalizerWatchdogController(inner=inner, force_after=10, stall_after=3)

    # Each call produces a new component → no stall.
    a0 = wd.select_action(_state(step_id=0, produced={"plan"}))
    a1 = wd.select_action(_state(step_id=1, produced={"plan", "code"}))
    a2 = wd.select_action(_state(step_id=2, produced={"plan", "code", "tests_run"}))

    assert a0 == {"kind": "activate_agent", "agent": "Planner"}
    assert a1 == {"kind": "activate_agent", "agent": "Planner"}
    assert a2 == {"kind": "activate_agent", "agent": "Planner"}


def test_forces_finalizer_after_stall_then_terminate() -> None:
    """When the produced set hasn't grown for ``stall_after`` calls,
    the watchdog forces ``Finalizer``; once ``final_answer`` is in
    ``produced``, it forces ``terminate``."""
    inner = _StuckController("Planner")
    wd = FinalizerWatchdogController(inner=inner, force_after=99, stall_after=3)

    # 3 consecutive stall calls (same produced set).
    wd.select_action(_state(step_id=0, produced={"plan"}))
    wd.select_action(_state(step_id=1, produced={"plan"}))
    wd.select_action(_state(step_id=2, produced={"plan"}))

    # 4th call → stall_count == 3, watchdog forces Finalizer.
    forced = wd.select_action(_state(step_id=3, produced={"plan"}))
    assert forced == {"kind": "activate_agent", "agent": "Finalizer"}

    # Next state: runtime has executed Finalizer, so produced now
    # contains "final_answer". Watchdog forces terminate.
    after_final = wd.select_action(_state(step_id=4, produced={"plan", "final_answer"}))
    assert after_final == {"kind": "terminate"}


def test_forces_terminate_after_step_budget() -> None:
    """Once ``state.step_id >= force_after`` the watchdog short-circuits
    even if the wrapped controller would otherwise keep going."""
    inner = _StuckController("Planner")
    wd = FinalizerWatchdogController(inner=inner, force_after=5, stall_after=99)

    # Below the budget: watchdog defers to inner controller.
    a0 = wd.select_action(_state(step_id=0, produced={"plan", "code"}))
    assert a0 == {"kind": "activate_agent", "agent": "Planner"}

    # At the budget: watchdog forces Finalizer.
    a5 = wd.select_action(_state(step_id=5, produced={"plan", "code"}))
    assert a5 == {"kind": "activate_agent", "agent": "Finalizer"}

    # After Finalizer fires: watchdog forces terminate.
    a6 = wd.select_action(_state(step_id=6, produced={"plan", "code", "final_answer"}))
    assert a6 == {"kind": "terminate"}


def test_per_task_state_is_isolated() -> None:
    """Two different ``task_id``s must not share stall counters."""
    inner = _StuckController("Planner")
    wd = FinalizerWatchdogController(inner=inner, force_after=99, stall_after=2)

    # task A stalls.
    wd.select_action(_state(step_id=0, produced={"plan"}, task_id="A"))
    wd.select_action(_state(step_id=1, produced={"plan"}, task_id="A"))
    forced_a = wd.select_action(_state(step_id=2, produced={"plan"}, task_id="A"))
    assert forced_a["agent"] == "Finalizer"

    # task B is fresh — first call should NOT be forced.
    fresh = wd.select_action(_state(step_id=0, produced={"plan"}, task_id="B"))
    assert fresh == {"kind": "activate_agent", "agent": "Planner"}


def test_per_task_type_force_after_dict() -> None:
    """Round-8: ``force_after`` may be a dict keyed by ``task_type``."""
    inner = _StuckController("Planner")
    wd = FinalizerWatchdogController(
        inner=inner,
        force_after={"coding": 28, "math": 12},
        stall_after=99,
    )

    # math task at step 12 → forced.
    math_state = RuntimeState(
        task_id="m1",
        task_type="math",
        prompt="2+2=?",
        step_id=12,
        available_agents=["Planner", "Finalizer"],
        pending_inbox={},
        last_active_agent=None,
        produced_components={"plan"},
    )
    forced_math = wd.select_action(math_state)
    assert forced_math == {"kind": "activate_agent", "agent": "Finalizer"}

    # coding task at step 12 → NOT forced (budget is 28).
    coding_state = RuntimeState(
        task_id="c1",
        task_type="coding",
        prompt="def fib():",
        step_id=12,
        available_agents=["Planner", "Finalizer"],
        pending_inbox={},
        last_active_agent=None,
        produced_components={"plan"},
    )
    not_forced = wd.select_action(coding_state)
    assert not_forced == {"kind": "activate_agent", "agent": "Planner"}


def test_unknown_task_type_falls_back_to_default() -> None:
    """If a task_type isn't in the dict, the wrapper uses fallback (12)."""
    inner = _StuckController("Planner")
    wd = FinalizerWatchdogController(
        inner=inner,
        force_after={"coding": 28},  # no entry for math
        stall_after=99,
    )

    # math task at step 12 → falls back to default 12 → forced.
    math_state = RuntimeState(
        task_id="m1",
        task_type="math",
        prompt="2+2=?",
        step_id=12,
        available_agents=["Planner", "Finalizer"],
        pending_inbox={},
        last_active_agent=None,
        produced_components={"plan"},
    )
    forced = wd.select_action(math_state)
    assert forced == {"kind": "activate_agent", "agent": "Finalizer"}
