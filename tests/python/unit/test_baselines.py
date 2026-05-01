"""Phase-9 baseline registry + graph-builder smoke tests."""

from __future__ import annotations

import pytest

from flybrain.agents.specs import MINIMAL_15
from flybrain.baselines import (
    BUILTIN_SUITES,
    RoundRobinController,
    builtin_baselines,
    degree_preserving_random_graph,
    empty_graph,
    fully_connected_graph,
    list_baselines,
    random_sparse_graph,
)
from flybrain.runtime.state import RuntimeState

AGENT_NAMES = [a.name for a in MINIMAL_15]


# -- graph builders ------------------------------------------------------------


def test_empty_graph_has_no_edges() -> None:
    g = empty_graph(AGENT_NAMES)
    assert g["nodes"] == AGENT_NAMES
    assert g["edges"] == {}


def test_fully_connected_graph_excludes_self_loops() -> None:
    g = fully_connected_graph(AGENT_NAMES)
    n = len(AGENT_NAMES)
    # Each source has exactly n-1 outgoing edges.
    for src, edges in g["edges"].items():
        assert src not in edges
        assert len(edges) == n - 1


def test_random_sparse_graph_is_deterministic_for_seed() -> None:
    a = random_sparse_graph(AGENT_NAMES, edge_prob=0.3, seed=42)
    b = random_sparse_graph(AGENT_NAMES, edge_prob=0.3, seed=42)
    c = random_sparse_graph(AGENT_NAMES, edge_prob=0.3, seed=43)
    assert a == b
    assert a != c


def test_degree_preserving_respects_target_degree() -> None:
    g = degree_preserving_random_graph(AGENT_NAMES, target_out_degree=3, seed=0)
    for src, edges in g["edges"].items():
        assert src not in edges
        assert len(edges) <= 3


def test_degree_preserving_uses_fly_adjacency() -> None:
    fly_adj = {
        AGENT_NAMES[0]: AGENT_NAMES[1:4],
        AGENT_NAMES[1]: [AGENT_NAMES[2]],
    }
    g = degree_preserving_random_graph(
        AGENT_NAMES, fly_adjacency=fly_adj, target_out_degree=99, seed=0
    )
    assert len(g["edges"][AGENT_NAMES[0]]) == 3
    assert len(g["edges"][AGENT_NAMES[1]]) == 1


# -- registry ------------------------------------------------------------------


def test_builtin_baselines_yields_nine_specs() -> None:
    specs = builtin_baselines()
    # 9 README §15 baselines + flybrain_graph_ssl_pretrain + 5 embedding-ablation
    # rows + 4 verifier-ablation rows.
    assert len(specs) == 9 + 1 + 5 + 4
    # Canonical README §15 order is preserved at the top of the list.
    full_min_names = [s.name for s in specs[:9]]
    assert full_min_names == BUILTIN_SUITES["full_min"]


def test_graph_ssl_baseline_registered() -> None:
    """README §18 Exp 4 row '+graph SSL pretrain'."""
    by_name = {s.name: s for s in builtin_baselines()}
    spec = by_name["flybrain_graph_ssl_pretrain"]
    assert "graph-ssl" in spec.tags
    assert "training_ablation" in BUILTIN_SUITES
    assert "flybrain_graph_ssl_pretrain" in BUILTIN_SUITES["training_ablation"]


def test_emb_ablation_specs_have_distinct_masks() -> None:
    """README §18 Exp 2: each level should be a registered baseline."""
    by_name = {s.name: s for s in builtin_baselines()}
    levels = [
        ("emb_ablation_none", 5),
        ("emb_ablation_task", 4),
        ("emb_ablation_task_agent", 3),
        ("emb_ablation_task_agent_trace", 2),
        ("emb_ablation_full", 0),
    ]
    for name, _ in levels:
        assert name in by_name, name


def test_verif_ablation_specs_carry_mas_overrides() -> None:
    """README §18 Exp 3: each level should carry the right MASConfig override."""
    by_name = {s.name: s for s in builtin_baselines()}
    expected = {
        "verif_ablation_off": "off",
        "verif_ablation_final": "final",
        "verif_ablation_step": "step",
        "verif_ablation_full": "full",
    }
    for name, mode in expected.items():
        spec = by_name[name]
        assert spec.mas_config_overrides == {"verification_mode": mode}


def test_baseline_make_mas_config_applies_overrides() -> None:
    from flybrain.baselines import BaselineSpec
    from flybrain.runtime.runner import MASConfig

    base = MASConfig(max_steps=8)
    spec = BaselineSpec(
        name="x",
        description="-",
        factory=lambda _: (None, None),  # type: ignore[arg-type]
        mas_config_overrides={"verification_mode": "off"},
    )
    cfg = spec.make_mas_config(base)
    assert cfg.verification_mode == "off"
    assert cfg.max_steps == 8

    # No override → identity (same instance).
    spec2 = BaselineSpec(name="y", description="-", factory=lambda _: (None, None))  # type: ignore[arg-type]
    assert spec2.make_mas_config(base) is base


@pytest.mark.parametrize("suite_name", sorted(BUILTIN_SUITES))
def test_list_baselines_matches_suite(suite_name: str) -> None:
    specs = list_baselines(suite_name)
    assert [s.name for s in specs] == BUILTIN_SUITES[suite_name]


def test_list_baselines_extra_appended() -> None:
    from flybrain.baselines import BaselineSpec

    extra = BaselineSpec(name="ablation_x", description="-", factory=lambda _: (None, None))  # type: ignore[arg-type]
    out = list_baselines("smoke", extra=[extra])
    assert out[-1].name == "ablation_x"


def test_list_baselines_rejects_unknown_suite() -> None:
    with pytest.raises(KeyError):
        list_baselines("not_a_real_suite")


@pytest.mark.parametrize("name", BUILTIN_SUITES["static"])
def test_static_baselines_construct(name: str) -> None:
    """Static-graph baselines must instantiate without torch / Yandex."""
    spec = next(s for s in builtin_baselines() if s.name == name)
    ctrl, graph = spec.factory(AGENT_NAMES)
    assert ctrl is not None
    assert graph is not None
    assert "nodes" in graph
    assert "edges" in graph


# -- round-robin controller ----------------------------------------------------


def _state(step_id: int = 0, last_active: str | None = None) -> RuntimeState:
    return RuntimeState(
        task_id="t1",
        task_type="coding",
        prompt="x",
        step_id=step_id,
        available_agents=AGENT_NAMES,
        pending_inbox={},
        last_active_agent=last_active,
    )


def test_round_robin_cycles_then_verifies_then_terminates() -> None:
    ctrl = RoundRobinController()
    actions = []
    for i in range(len(AGENT_NAMES) + 2):
        actions.append(ctrl.select_action(_state(step_id=i)))
    # First N actions activate each agent in order.
    for i, name in enumerate(AGENT_NAMES):
        assert actions[i] == {"kind": "activate_agent", "agent": name}
    # Then a verifier call, then terminate.
    assert actions[-2] == {"kind": "call_verifier"}
    assert actions[-1] == {"kind": "terminate"}


def test_round_robin_resets_on_new_task() -> None:
    ctrl = RoundRobinController()
    ctrl.select_action(_state(step_id=0))
    ctrl.select_action(_state(step_id=1))
    # New task: the cursor must reset.
    new = RuntimeState(
        task_id="t2",
        task_type="coding",
        prompt="y",
        step_id=0,
        available_agents=AGENT_NAMES,
        pending_inbox={},
        last_active_agent=None,
    )
    assert ctrl.select_action(new) == {
        "kind": "activate_agent",
        "agent": AGENT_NAMES[0],
    }
