"""Phase-9 baseline registry (PLAN.md §603-605, README §15).

Each entry is a small description of a baseline plus a *factory*
that produces ``(controller, initial_graph)`` for a given agent
roster. The registry lets ``scripts/run_baselines.py`` iterate the
suite without hand-coding 9 separate setups.

Static graph baselines (#1-#4) pair an ``ManualController`` with a
fixed initial graph; learned baselines (#5-#9) instantiate one of
the Phase-5 controllers and optionally load a checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flybrain.baselines.graphs import (
    degree_preserving_random_graph,
    empty_graph,
    flybrain_prior_graph,
    fully_connected_graph,
    random_sparse_graph,
)
from flybrain.controller import ManualController, RandomController
from flybrain.controller.base import Controller
from flybrain.runtime.runner import MASConfig

BaselineFactory = Callable[
    [list[str]],
    tuple[Controller, dict[str, Any] | None],
]


@dataclass(slots=True)
class BaselineSpec:
    """Lightweight description of one baseline.

    ``factory`` accepts ``agent_names`` and returns the
    ``(controller, initial_graph)`` pair the runtime ingests.
    ``initial_graph`` is ``None`` for baselines that rely on the
    runtime's default empty graph.
    """

    name: str
    """Stable identifier; used as the column header in the comparison table."""
    description: str
    factory: BaselineFactory
    tags: list[str] = field(default_factory=list)
    """Free-form tags (e.g. ``"static-graph"``, ``"untrained"``,
    ``"trained"``) so suites can filter."""
    mas_config_overrides: dict[str, Any] = field(default_factory=dict)
    """Optional ``MASConfig`` field overrides (e.g.
    ``{"verification_mode": "off"}`` for the Experiment-3 verifier
    ablation). The bench runner merges these into the suite-level
    ``MASConfig`` when constructing ``MAS.from_specs``."""

    def make_mas_config(self, base: MASConfig) -> MASConfig:
        """Return a new ``MASConfig`` with this baseline's overrides applied."""
        if not self.mas_config_overrides:
            return base
        # Use dataclasses.replace via direct field copy because MASConfig
        # uses ``slots=True`` and we want to avoid pulling extra deps here.
        kwargs = {f: getattr(base, f) for f in MASConfig.__slots__}
        kwargs.update(self.mas_config_overrides)
        return MASConfig(**kwargs)


# -- factories -----------------------------------------------------------------


def _manual_with_graph(builder: Callable[[list[str]], dict[str, Any]]) -> BaselineFactory:
    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        return ManualController(), builder(agent_names)

    return factory


def _manual_baseline(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
    """#1 Manual MAS graph — runtime-default empty initial graph,
    plus the hand-tuned ManualController plan."""
    return ManualController(), empty_graph(agent_names)


def _fully_connected_baseline(
    agent_names: list[str],
) -> tuple[Controller, dict[str, Any] | None]:
    """#2 Fully connected MAS — broadcast graph + ManualController."""
    return ManualController(), fully_connected_graph(agent_names)


def _random_sparse_baseline(seed: int = 0) -> BaselineFactory:
    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        return RandomController(seed=seed), random_sparse_graph(agent_names, seed=seed)

    return factory


def _degree_preserving_baseline(seed: int = 0) -> BaselineFactory:
    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        return ManualController(), degree_preserving_random_graph(agent_names, seed=seed)

    return factory


def _learned_router_no_prior(
    builder_factory: Callable[[], Any] | None = None,
) -> BaselineFactory:
    """#5 LearnedRouter without fly-prior init."""

    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        from flybrain.controller import LearnedRouterController
        from flybrain.embeddings import (
            AgentEmbedder,
            AgentGraphEmbedder,
            ControllerStateBuilder,
            FlyGraphEmbedder,
            MockEmbeddingClient,
            TaskEmbedder,
            TraceEmbedder,
        )

        client = MockEmbeddingClient(output_dim=32)
        agents_emb = AgentEmbedder(client)
        from flybrain.agents.specs import MINIMAL_15

        agents_emb.precompute_sync(MINIMAL_15)
        builder = ControllerStateBuilder(
            task=TaskEmbedder(client),
            agents=agents_emb,
            trace=TraceEmbedder(client),
            fly=FlyGraphEmbedder(dim=8),
            agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
        )
        ctrl = LearnedRouterController(
            builder=builder,
            task_dim=32,
            agent_dim=32,
            graph_dim=32,
            trace_dim=32 + 13,
            fly_dim=8,
            produced_dim=6,
            hidden_dim=32,
        )
        # Important: skip init_from_fly_graph to keep this baseline
        # honest — that's the whole point of #5.
        return ctrl, empty_graph(agent_names)

    return factory


def _flybrain_prior_untrained() -> BaselineFactory:
    """#6 FlyBrain prior without training.

    Ships a fly-derived initial AgentGraph (see
    :func:`flybrain.baselines.graphs.flybrain_prior_graph`) so the
    untrained GNN/router has something non-trivial to read from on
    step 0 — without that, the controller can't traverse the agent
    space and its success rate stays at 0% even with a perfect
    architecture (see HANDOFF.md §4.a).
    """

    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        from flybrain.controller import FlyBrainGNNController
        from flybrain.embeddings import (
            AgentEmbedder,
            AgentGraphEmbedder,
            ControllerStateBuilder,
            FlyGraphEmbedder,
            MockEmbeddingClient,
            TaskEmbedder,
            TraceEmbedder,
        )

        client = MockEmbeddingClient(output_dim=32)
        agents_emb = AgentEmbedder(client)
        from flybrain.agents.specs import MINIMAL_15

        agents_emb.precompute_sync(MINIMAL_15)
        builder = ControllerStateBuilder(
            task=TaskEmbedder(client),
            agents=agents_emb,
            trace=TraceEmbedder(client),
            fly=FlyGraphEmbedder(dim=8),
            agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
        )
        ctrl = FlyBrainGNNController(
            builder=builder,
            task_dim=32,
            agent_dim=32,
            graph_dim=32,
            trace_dim=32 + 13,
            fly_dim=8,
            produced_dim=6,
            hidden_dim=32,
        )
        return ctrl, flybrain_prior_graph(agent_names)

    return factory


def _flybrain_with_graph_ssl(checkpoint_path: Path | None = None) -> BaselineFactory:
    """README §18 Experiment 4 row "+ graph self-supervised pretraining".

    Wires a graph-SSL-pretrained encoder (link prediction +
    masked-node reconstruction, see ``flybrain.training.graph_ssl``)
    into the FlyBrain prior controller. ``checkpoint_path`` defaults
    to ``data/checkpoints/graph_ssl_K64.npz`` produced by
    ``scripts/run_graph_ssl_pretrain.py``; if absent the factory
    falls back to the deterministic Gaussian-projection encoder so
    the baseline is still runnable in CI without artefacts.
    """

    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        from flybrain.controller import FlyBrainGNNController
        from flybrain.embeddings import (
            AgentEmbedder,
            AgentGraphEmbedder,
            ControllerStateBuilder,
            FlyGraphEmbedder,
            MockEmbeddingClient,
            TaskEmbedder,
            TraceEmbedder,
        )

        client = MockEmbeddingClient(output_dim=32)
        agents_emb = AgentEmbedder(client)
        from flybrain.agents.specs import MINIMAL_15

        agents_emb.precompute_sync(MINIMAL_15)
        agent_graph = AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32)

        # Try to load the SSL checkpoint; silently fall through if missing
        # so the baseline factory is still safe to register at import time.
        path = checkpoint_path or Path("data/checkpoints/graph_ssl_K64.npz")
        if path.exists():
            from flybrain.training.graph_ssl import (
                apply_to_embedder,
                load_checkpoint,
            )

            try:
                weights = load_checkpoint(path)
                apply_to_embedder(agent_graph, weights)
            except (ValueError, OSError):
                # Shape mismatch or unreadable file — keep the default
                # numpy projection so the run still produces metrics.
                pass

        builder = ControllerStateBuilder(
            task=TaskEmbedder(client),
            agents=agents_emb,
            trace=TraceEmbedder(client),
            fly=FlyGraphEmbedder(dim=8),
            agent_graph=agent_graph,
        )
        ctrl = FlyBrainGNNController(
            builder=builder,
            task_dim=32,
            agent_dim=32,
            graph_dim=32,
            trace_dim=32 + 13,
            fly_dim=8,
            produced_dim=6,
            hidden_dim=32,
        )
        return ctrl, flybrain_prior_graph(agent_names)

    return factory


def _learned_router_with_mask(mask: frozenset[str]) -> BaselineFactory:
    """README §18 Experiment 2 — embedding ablation factory.

    Returns a ``LearnedRouterController`` whose ``ControllerStateBuilder``
    zeros the listed feature groups (``"task"``, ``"agent"``, ``"trace"``,
    ``"graph"``, ``"fly"``) before the controller sees them. ``init_from_fly``
    is intentionally skipped — the ablation is about input features only.
    """

    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        from flybrain.controller import LearnedRouterController
        from flybrain.embeddings import (
            AgentEmbedder,
            AgentGraphEmbedder,
            ControllerStateBuilder,
            FlyGraphEmbedder,
            MockEmbeddingClient,
            TaskEmbedder,
            TraceEmbedder,
        )

        client = MockEmbeddingClient(output_dim=32)
        agents_emb = AgentEmbedder(client)
        from flybrain.agents.specs import MINIMAL_15

        agents_emb.precompute_sync(MINIMAL_15)
        builder = ControllerStateBuilder(
            task=TaskEmbedder(client),
            agents=agents_emb,
            trace=TraceEmbedder(client),
            fly=FlyGraphEmbedder(dim=8),
            agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
            feature_mask=mask,
        )
        ctrl = LearnedRouterController(
            builder=builder,
            task_dim=32,
            agent_dim=32,
            graph_dim=32,
            trace_dim=32 + 13,
            fly_dim=8,
            produced_dim=6,
            hidden_dim=32,
        )
        return ctrl, empty_graph(agent_names)

    return factory


# Default checkpoint discovery paths used when the
# ``FLYBRAIN_BASELINE_{LABEL}`` env var is unset. Keep these in sync
# with the canonical save locations of
# ``scripts/run_simulation_pretrain.py`` /
# ``scripts/run_imitation.py`` / ``scripts/run_rl.py``.
_DEFAULT_CHECKPOINT_PATHS: dict[str, tuple[Path, ...]] = {
    "SIM_PRETRAIN": (
        # Round-5 headline (Finalizer-route fix, +20pp humaneval). Prefer
        # this over the unsuffixed v1 file when present.
        Path("data/checkpoints/sim_pretrain_gnn_v6.pt"),
        Path("data/checkpoints/sim_pretrain_gnn.pt"),
        Path("data/checkpoints/sim_pretrain.pt"),
    ),
    "IMITATION": (
        Path("data/checkpoints/imitation_gnn.pt"),
        Path("data/checkpoints/imitation.pt"),
    ),
    "RL": (
        Path("data/checkpoints/rl_gnn.pt"),
        Path("data/checkpoints/rl.pt"),
    ),
}


def _resolve_checkpoint_path(label: str) -> Path | None:
    """Return the first existing checkpoint for ``label``.

    Lookup order:

    1. ``FLYBRAIN_BASELINE_{LABEL}`` env var, if set and the file
       exists.
    2. Each path in ``_DEFAULT_CHECKPOINT_PATHS[label]``, in order.

    Returns ``None`` if none of the candidates resolve to an
    existing file, in which case the factory falls back to the
    untrained variant (HANDOFF.md §4.a Q1.3).
    """
    import os

    env_var = f"FLYBRAIN_BASELINE_{label.upper()}"
    candidates: list[Path] = []
    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(_DEFAULT_CHECKPOINT_PATHS.get(label.upper(), ()))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _flybrain_with_checkpoint_and_watchdog(
    controller_name: str,
    label: str,
    *,
    force_after: int | dict[str, int] = 12,
    stall_after: int | dict[str, int] = 3,
    baseline_name: str | None = None,
) -> BaselineFactory:
    """Round-7/8 variant: wrap the trained checkpoint baseline in
    :class:`FinalizerWatchdogController` so stalls are forced into
    Finalizer + terminate. See ``flybrain/controller/finalizer_watchdog.py``
    for the watchdog rationale (round-5 trained controllers never
    actually emit Finalizer at inference). Round-8 supports per-
    task-type ``force_after`` / ``stall_after`` dicts.
    """

    inner_factory = _flybrain_with_checkpoint(controller_name, label)

    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        from flybrain.controller.finalizer_watchdog import (
            FinalizerWatchdogController,
        )

        inner, init_graph = inner_factory(agent_names)
        wrapped = FinalizerWatchdogController(
            inner=inner,
            force_after=force_after,
            stall_after=stall_after,
            name=baseline_name or "flybrain_sim_pretrain_watchdog",
        )
        return wrapped, init_graph

    return factory


def _flybrain_with_checkpoint(
    controller_name: str,
    label: str,
) -> BaselineFactory:
    """Generic factory for #7-#9: same architecture as #6 but with a
    pre-loaded checkpoint produced by Phase-6 / Phase-7 / Phase-8.

    The checkpoint is resolved by :func:`_resolve_checkpoint_path` —
    first the ``FLYBRAIN_BASELINE_{LABEL}`` env var, then a fallback
    list of standard ``data/checkpoints/`` locations. If neither is
    present the factory degrades to the untrained variant so
    ``run_baselines.py`` is still safe to invoke without artefacts.

    The initial AgentGraph is the fly-prior
    (:func:`flybrain.baselines.graphs.flybrain_prior_graph`), matching
    the architecture the trainers see in
    ``flybrain.training.simulation_pretrain`` / ``imitation_train`` /
    ``rl_train`` (HANDOFF.md §4.a Q1).
    """

    def factory(agent_names: list[str]) -> tuple[Controller, dict[str, Any] | None]:
        from flybrain.embeddings import (
            AgentEmbedder,
            AgentGraphEmbedder,
            ControllerStateBuilder,
            FlyGraphEmbedder,
            MockEmbeddingClient,
            TaskEmbedder,
            TraceEmbedder,
        )

        client = MockEmbeddingClient(output_dim=32)
        agents_emb = AgentEmbedder(client)
        from flybrain.agents.specs import MINIMAL_15

        agents_emb.precompute_sync(MINIMAL_15)
        builder = ControllerStateBuilder(
            task=TaskEmbedder(client),
            agents=agents_emb,
            trace=TraceEmbedder(client),
            fly=FlyGraphEmbedder(dim=8),
            agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
        )
        if controller_name == "gnn":
            from flybrain.controller import FlyBrainGNNController

            ctrl_cls: Any = FlyBrainGNNController
        elif controller_name == "rnn":
            from flybrain.controller import FlyBrainRNNController

            ctrl_cls = FlyBrainRNNController
        elif controller_name == "router":
            from flybrain.controller import LearnedRouterController

            ctrl_cls = LearnedRouterController
        else:
            raise ValueError(f"unknown controller {controller_name!r}")

        ctrl = ctrl_cls(
            builder=builder,
            task_dim=32,
            agent_dim=32,
            graph_dim=32,
            trace_dim=32 + 13,
            fly_dim=8,
            produced_dim=6,
            hidden_dim=32,
        )

        ckpt_path = _resolve_checkpoint_path(label)
        if ckpt_path is not None:
            try:
                import torch

                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                ctrl.load_state_dict(sd, strict=False)
            except Exception as e:  # pragma: no cover - best effort
                import warnings

                warnings.warn(
                    f"failed to load {label} checkpoint at {ckpt_path}: {e}",
                    stacklevel=2,
                )
        return ctrl, flybrain_prior_graph(agent_names)

    return factory


# -- registry ------------------------------------------------------------------


def builtin_baselines() -> list[BaselineSpec]:
    """The canonical 9-baseline list from README §15."""
    return [
        BaselineSpec(
            name="manual_graph",
            description="#1 Manual MAS graph + ManualController plan.",
            factory=_manual_baseline,
            tags=["static-graph", "no-llm-controller"],
        ),
        BaselineSpec(
            name="fully_connected",
            description="#2 Fully connected broadcast graph + ManualController.",
            factory=_fully_connected_baseline,
            tags=["static-graph"],
        ),
        BaselineSpec(
            name="random_sparse",
            description="#3 Erdos-Renyi sparse graph + RandomController.",
            factory=_random_sparse_baseline(),
            tags=["static-graph", "random"],
        ),
        BaselineSpec(
            name="degree_preserving",
            description=(
                "#4 Random graph with each node's out-degree fixed; ManualController on top."
            ),
            factory=_degree_preserving_baseline(),
            tags=["static-graph", "random"],
        ),
        BaselineSpec(
            name="learned_router_no_prior",
            description=("#5 Phase-5 LearnedRouter without `init_from_fly_graph`."),
            factory=_learned_router_no_prior(),
            tags=["learned", "untrained", "no-fly-prior"],
        ),
        BaselineSpec(
            name="flybrain_prior_untrained",
            description="#6 Phase-5 GNN with fly-prior init but no training.",
            factory=_flybrain_prior_untrained(),
            tags=["learned", "untrained", "fly-prior"],
        ),
        BaselineSpec(
            name="flybrain_sim_pretrain",
            description="#7 FlyBrain GNN + Phase-6 simulation pretraining.",
            factory=_flybrain_with_checkpoint("gnn", "SIM_PRETRAIN"),
            tags=["learned", "trained", "fly-prior"],
        ),
        BaselineSpec(
            name="flybrain_imitation",
            description="#8 FlyBrain GNN + Phase-7 imitation learning.",
            factory=_flybrain_with_checkpoint("gnn", "IMITATION"),
            tags=["learned", "trained", "fly-prior"],
        ),
        BaselineSpec(
            name="flybrain_rl",
            description="#9 FlyBrain GNN + Phase-8 RL/bandit finetuning.",
            factory=_flybrain_with_checkpoint("gnn", "RL"),
            tags=["learned", "trained", "fly-prior"],
        ),
        BaselineSpec(
            name="flybrain_sim_pretrain_watchdog",
            description=(
                "Round-7: #7 wrapped in FinalizerWatchdogController — "
                "forces Finalizer then terminate after 12 steps or 3 "
                "consecutive non-progress steps. Closes the round-5 gap "
                "where the trained controller never actually emits "
                "Finalizer at inference."
            ),
            factory=_flybrain_with_checkpoint_and_watchdog(
                "gnn", "SIM_PRETRAIN", force_after=12, stall_after=3
            ),
            tags=["learned", "trained", "fly-prior", "watchdog", "round-7"],
        ),
        BaselineSpec(
            name="flybrain_sim_pretrain_watchdog_v2",
            description=(
                "Round-8: per-task-type FinalizerWatchdogController — "
                "force_after={coding:28, math:12, research:16, "
                "tool_use:12} so coding tasks get the full plan→code"
                "→test→debug depth manual_graph uses (~21 calls/task) "
                "while math/research/tool_use short-circuit early. "
                "Fixes the round-7 humaneval regression while keeping "
                "the synthetic_routing parity with manual_graph."
            ),
            factory=_flybrain_with_checkpoint_and_watchdog(
                "gnn",
                "SIM_PRETRAIN",
                force_after={
                    "coding": 28,
                    "math": 12,
                    "research": 16,
                    "tool_use": 12,
                },
                stall_after={
                    "coding": 6,
                    "math": 3,
                    "research": 4,
                    "tool_use": 3,
                },
                baseline_name="flybrain_sim_pretrain_watchdog_v2",
            ),
            tags=["learned", "trained", "fly-prior", "watchdog", "round-8"],
        ),
        BaselineSpec(
            name="flybrain_graph_ssl_pretrain",
            description=(
                "Exp4 row '+graph SSL' — FlyBrain GNN with the agent-graph "
                "encoder pretrained via link prediction + masked-node "
                "reconstruction (README §12.5)."
            ),
            factory=_flybrain_with_graph_ssl(),
            tags=["learned", "trained", "fly-prior", "graph-ssl"],
        ),
        # README §18 Experiment 2 — embedding ablation (5 levels).
        # L1 (no embeddings) zeros every feature; L5 keeps all features.
        BaselineSpec(
            name="emb_ablation_none",
            description="Exp2 L1 — LearnedRouter with all embeddings zeroed.",
            factory=_learned_router_with_mask(
                frozenset({"task", "agent", "trace", "graph", "fly"})
            ),
            tags=["ablation", "embedding"],
        ),
        BaselineSpec(
            name="emb_ablation_task",
            description="Exp2 L2 — LearnedRouter with task embedding only.",
            factory=_learned_router_with_mask(frozenset({"agent", "trace", "graph", "fly"})),
            tags=["ablation", "embedding"],
        ),
        BaselineSpec(
            name="emb_ablation_task_agent",
            description="Exp2 L3 — LearnedRouter with task + agent embeddings.",
            factory=_learned_router_with_mask(frozenset({"trace", "graph", "fly"})),
            tags=["ablation", "embedding"],
        ),
        BaselineSpec(
            name="emb_ablation_task_agent_trace",
            description="Exp2 L4 — LearnedRouter with task + agent + trace embeddings.",
            factory=_learned_router_with_mask(frozenset({"graph", "fly"})),
            tags=["ablation", "embedding"],
        ),
        BaselineSpec(
            name="emb_ablation_full",
            description="Exp2 L5 — LearnedRouter with all five embeddings (task+agent+trace+graph+fly).",
            factory=_learned_router_with_mask(frozenset()),
            tags=["ablation", "embedding"],
        ),
        # README §18 Experiment 3 — verifier ablation (4 levels).
        # The controller is held fixed at #6 (FlyBrain prior, untrained)
        # so the only thing varying across these rows is the verifier
        # configuration (set via MAS-config overrides on the spec).
        BaselineSpec(
            name="verif_ablation_off",
            description="Exp3 L1 — verifier disabled (verification_mode=off).",
            factory=_flybrain_prior_untrained(),
            tags=["ablation", "verifier"],
            mas_config_overrides={"verification_mode": "off"},
        ),
        BaselineSpec(
            name="verif_ablation_final",
            description="Exp3 L2 — final-task verifier only.",
            factory=_flybrain_prior_untrained(),
            tags=["ablation", "verifier"],
            mas_config_overrides={"verification_mode": "final"},
        ),
        BaselineSpec(
            name="verif_ablation_step",
            description="Exp3 L3 — per-step verifier only (final is rule-based).",
            factory=_flybrain_prior_untrained(),
            tags=["ablation", "verifier"],
            mas_config_overrides={"verification_mode": "step"},
        ),
        BaselineSpec(
            name="verif_ablation_full",
            description="Exp3 L4 — full verification layer (per-step + final).",
            factory=_flybrain_prior_untrained(),
            tags=["ablation", "verifier"],
            mas_config_overrides={"verification_mode": "full"},
        ),
    ]


BUILTIN_SUITES: dict[str, list[str]] = {
    # PLAN.md §605: `--suite full_min` should run all 9.
    "full_min": [
        "manual_graph",
        "fully_connected",
        "random_sparse",
        "degree_preserving",
        "learned_router_no_prior",
        "flybrain_prior_untrained",
        "flybrain_sim_pretrain",
        "flybrain_imitation",
        "flybrain_rl",
    ],
    "static": [
        "manual_graph",
        "fully_connected",
        "random_sparse",
        "degree_preserving",
    ],
    "learned": [
        "learned_router_no_prior",
        "flybrain_prior_untrained",
        "flybrain_sim_pretrain",
        "flybrain_imitation",
        "flybrain_rl",
        "flybrain_graph_ssl_pretrain",
    ],
    "smoke": ["manual_graph", "random_sparse"],
    # README §18 ablation suites.
    "embedding_ablation": [
        "emb_ablation_none",
        "emb_ablation_task",
        "emb_ablation_task_agent",
        "emb_ablation_task_agent_trace",
        "emb_ablation_full",
    ],
    "verifier_ablation": [
        "verif_ablation_off",
        "verif_ablation_final",
        "verif_ablation_step",
        "verif_ablation_full",
    ],
    # README §18 Experiment 4 — training ablation (5 levels).
    # The four trained-controller rows above + the SSL-pretrained row.
    "training_ablation": [
        "flybrain_prior_untrained",
        "flybrain_graph_ssl_pretrain",
        "flybrain_sim_pretrain",
        "flybrain_imitation",
        "flybrain_rl",
    ],
    # Round-7 — sim_pretrain v6 vs Finalizer-watchdog wrapper.
    # Used by `bench_round7_watchdog` to A/B the post-processing
    # fix for the "trained controller never emits Finalizer" gap
    # documented in round-5 traces.
    "round7_watchdog": [
        "manual_graph",
        "flybrain_sim_pretrain",
        "flybrain_sim_pretrain_watchdog",
    ],
    # Round-8 — adds the per-task-type tuned watchdog (v2) to the
    # round-7 trio so we can A/B/C the regression-fix in a single
    # bench run.
    "round8_watchdog_v2": [
        "manual_graph",
        "flybrain_sim_pretrain",
        "flybrain_sim_pretrain_watchdog",
        "flybrain_sim_pretrain_watchdog_v2",
    ],
}


def list_baselines(
    suite: str = "full_min",
    *,
    extra: list[BaselineSpec] | None = None,
) -> list[BaselineSpec]:
    """Materialise a suite into a list of `BaselineSpec`s.

    Pass ``extra`` to inject custom baselines (e.g. an ablation under
    test) without modifying the registry."""
    if suite not in BUILTIN_SUITES:
        raise KeyError(f"unknown suite {suite!r}; choose one of {sorted(BUILTIN_SUITES)}")
    by_name = {b.name: b for b in builtin_baselines()}
    out = [by_name[n] for n in BUILTIN_SUITES[suite] if n in by_name]
    if extra:
        out.extend(extra)
    return out


__all__ = [
    "BUILTIN_SUITES",
    "BaselineFactory",
    "BaselineSpec",
    "builtin_baselines",
    "list_baselines",
]
