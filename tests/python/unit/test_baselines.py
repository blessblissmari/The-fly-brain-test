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
    flybrain_prior_graph,
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


# -- flybrain prior graph ------------------------------------------------------


def test_flybrain_prior_graph_is_non_empty_and_excludes_self_loops() -> None:
    g = flybrain_prior_graph(AGENT_NAMES)
    assert g["nodes"] == AGENT_NAMES
    total_edges = sum(len(dsts) for dsts in g["edges"].values())
    assert total_edges > 0
    for src, dsts in g["edges"].items():
        assert src not in dsts, f"self-loop on {src}"


def test_flybrain_prior_graph_is_deterministic_for_seed() -> None:
    a = flybrain_prior_graph(AGENT_NAMES, seed=0)
    b = flybrain_prior_graph(AGENT_NAMES, seed=0)
    assert a == b


def test_flybrain_prior_graph_respects_max_out_degree() -> None:
    cap = 3
    g = flybrain_prior_graph(AGENT_NAMES, max_out_degree=cap)
    for dsts in g["edges"].values():
        assert len(dsts) <= cap


def test_flybrain_prior_graph_row_normalises_weights() -> None:
    g = flybrain_prior_graph(AGENT_NAMES)
    for dsts in g["edges"].values():
        s = sum(dsts.values())
        assert s == pytest.approx(1.0, abs=1e-6)


def test_flybrain_prior_graph_handles_empty_agents() -> None:
    g = flybrain_prior_graph([])
    assert g == {"nodes": [], "edges": {}}


def test_flybrain_prior_graph_skips_self_loop_for_single_agent() -> None:
    """Single-agent runs have no possible non-self destination — the
    builder must not invent one (HANDOFF.md §4.a Q1 fly-prior contract
    forbids self-loops)."""
    g = flybrain_prior_graph(["solo"])
    assert g["nodes"] == ["solo"]
    assert g["edges"].get("solo", {}) == {}


# -- registry ------------------------------------------------------------------


def test_builtin_baselines_yields_nine_specs() -> None:
    specs = builtin_baselines()
    # 9 README §15 baselines + flybrain_sim_pretrain_watchdog (round-7)
    # + flybrain_sim_pretrain_watchdog_v2 (round-8)
    # + flybrain_sim_pretrain_watchdog_v3 (round-9)
    # + flybrain_graph_ssl_pretrain + 5 embedding-ablation rows
    # + 4 verifier-ablation rows
    # + 3 round-10 connectome-prior-ablation rows
    # + 3 round-11 null-priors-with-watchdog rows.
    assert len(specs) == 9 + 1 + 1 + 1 + 1 + 5 + 4 + 3 + 3
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


def test_round11_priors_with_watchdog_suite_has_all_four_priors() -> None:
    """Round-11 cross-product: real_fly + 3 null priors, all wrapped in
    watchdog v2. The suite must include exactly one ``manual_graph``
    control + 4 prior variants."""
    names = BUILTIN_SUITES["round11_priors_with_watchdog"]
    assert names[0] == "manual_graph"
    prior_baselines = set(names[1:])
    assert prior_baselines == {
        "flybrain_sim_pretrain_watchdog_v2",
        "er_prior_watchdog_v2",
        "shuffled_fly_watchdog_v2",
        "reverse_fly_watchdog_v2",
    }


@pytest.mark.parametrize(
    "name",
    [
        "er_prior_watchdog_v2",
        "shuffled_fly_watchdog_v2",
        "reverse_fly_watchdog_v2",
    ],
)
def test_round11_null_prior_watchdog_baselines_registered(name: str) -> None:
    """Each round-11 null-prior + watchdog v2 baseline must be in the
    registry with the round-11 tag."""
    spec = next((s for s in builtin_baselines() if s.name == name), None)
    assert spec is not None, f"missing round-11 baseline {name}"
    assert "round-11" in spec.tags
    assert "watchdog" in spec.tags
    assert "null-prior" in spec.tags


def test_round13_paid_yandex_suite_has_four_baselines() -> None:
    """Round-13 final paid YandexGPT bench picks the 4 baselines that
    carry the project's main story: manual_graph control, raw GNN
    (cost-Pareto), watchdog v3 (production), and one null+watchdog
    (Yandex-side replication of round-11)."""
    names = BUILTIN_SUITES["round13_paid_yandex"]
    assert names == [
        "manual_graph",
        "flybrain_sim_pretrain",
        "flybrain_sim_pretrain_watchdog_v3",
        "er_prior_watchdog_v2",
    ]
    # Each baseline must already be registered.
    by_name = {s.name for s in builtin_baselines()}
    for n in names:
        assert n in by_name, f"round13 references unknown baseline {n!r}"


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


@pytest.mark.parametrize(
    "name",
    [
        "flybrain_prior_untrained",
        "flybrain_sim_pretrain",
        "flybrain_imitation",
        "flybrain_rl",
        "flybrain_graph_ssl_pretrain",
    ],
)
def test_flybrain_baselines_ship_non_empty_initial_graph(name: str) -> None:
    """HANDOFF.md §4.a Q1: trained / fly-prior baselines must ship a
    non-empty initial AgentGraph so the GNN/router has something to
    read from on step 0."""
    pytest.importorskip("torch")  # GNN / router controllers need torch
    spec = next(s for s in builtin_baselines() if s.name == name)
    _ctrl, graph = spec.factory(AGENT_NAMES)
    assert graph is not None
    assert graph["nodes"] == AGENT_NAMES
    total_edges = sum(len(dsts) for dsts in graph["edges"].values())
    assert total_edges > 0, f"{name} factory produced empty initial graph"


def test_resolve_checkpoint_path_prefers_env_var(tmp_path, monkeypatch) -> None:
    from flybrain.baselines.registry import _resolve_checkpoint_path

    fake = tmp_path / "custom.pt"
    fake.write_bytes(b"")
    monkeypatch.setenv("FLYBRAIN_BASELINE_SIM_PRETRAIN", str(fake))
    assert _resolve_checkpoint_path("SIM_PRETRAIN") == fake


def test_resolve_checkpoint_path_falls_back_to_default(tmp_path, monkeypatch) -> None:
    """When the env var is unset and no default path exists, returns None."""
    from flybrain.baselines.registry import _resolve_checkpoint_path

    monkeypatch.delenv("FLYBRAIN_BASELINE_IMITATION", raising=False)
    # Run from an empty cwd so the default `data/checkpoints/...` paths
    # resolve to non-existent files.
    monkeypatch.chdir(tmp_path)
    assert _resolve_checkpoint_path("IMITATION") is None


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
