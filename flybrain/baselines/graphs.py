"""Static AgentGraph builders used by Phase-9 baselines (README §15).

Each builder returns a JSON-shaped graph dict ``{"nodes": [...],
"edges": {src: {dst: weight}}}`` matching ``flybrain_core::AgentGraph``
so the runtime ingests it via ``MAS.run(initial_graph=...)``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flybrain.graph.dataclasses import FlyGraph

# Canonical paths the FlyWire 783 K-compressed prior is written to by
# ``scripts/build_flywire_csv.py`` + ``flybrain build``. The first one
# that exists is used by :func:`flybrain_prior_graph` when the caller
# does not pass an explicit ``fly_graph``. CI / dev environments that
# don't have the 813 MB Zenodo artefact downloaded fall through to the
# synthetic generator.
DEFAULT_FLYBRAIN_GRAPH_PATHS: tuple[Path, ...] = (
    Path("data/flybrain/fly_graph_64.fbg"),
    Path("data/flybrain/fly_graph_128.fbg"),
    Path("data/flybrain/fly_graph_256.fbg"),
    Path("data/flybrain/fly_graph_32.fbg"),
)


_FLYBRAIN_PRIOR_CACHE: FlyGraph | None = None
_FLYBRAIN_PRIOR_RESOLVED: bool = False


def _load_default_fly_graph() -> FlyGraph | None:
    """Return the first on-disk ``fly_graph_*.fbg`` we can find.

    Cached at module level so repeated factory calls don't re-parse
    the .fbg file. Returns ``None`` if no canonical prior is on disk
    (in which case the caller falls back to ``build_synthetic``).
    """
    global _FLYBRAIN_PRIOR_CACHE, _FLYBRAIN_PRIOR_RESOLVED
    if _FLYBRAIN_PRIOR_RESOLVED:
        return _FLYBRAIN_PRIOR_CACHE
    for candidate in DEFAULT_FLYBRAIN_GRAPH_PATHS:
        if not candidate.exists():
            continue
        try:
            from flybrain.graph import load as _load

            graph = _load(candidate)
        except (OSError, ValueError):  # pragma: no cover - best effort
            continue
        _FLYBRAIN_PRIOR_CACHE = graph
        _FLYBRAIN_PRIOR_RESOLVED = True
        return graph
    _FLYBRAIN_PRIOR_RESOLVED = True
    return None


def empty_graph(agent_names: list[str]) -> dict[str, Any]:
    """The default graph the runtime uses when no initial graph is
    supplied: a node set with no edges. Equivalent to baseline #1
    (Manual MAS graph) when paired with ``ManualController``."""
    return {"nodes": list(agent_names), "edges": {}}


def fully_connected_graph(
    agent_names: list[str],
    *,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Baseline #2 — every agent broadcasts to every other agent.
    Equivalent to a complete directed graph on ``len(agent_names)``
    nodes. Self-loops are omitted."""
    edges: dict[str, dict[str, float]] = {}
    for src in agent_names:
        edges[src] = {dst: weight for dst in agent_names if dst != src}
    return {"nodes": list(agent_names), "edges": edges}


def random_sparse_graph(
    agent_names: list[str],
    *,
    edge_prob: float = 0.2,
    seed: int = 0,
) -> dict[str, Any]:
    """Baseline #3 — Erdős–Rényi style random sparse graph.

    Each directed edge is sampled independently with probability
    ``edge_prob``; weight is fixed at 1.0. The seed makes the result
    reproducible across runs.
    """
    rng = random.Random(seed)
    edges: dict[str, dict[str, float]] = {}
    for src in agent_names:
        bucket: dict[str, float] = {}
        for dst in agent_names:
            if src == dst:
                continue
            if rng.random() < edge_prob:
                bucket[dst] = 1.0
        if bucket:
            edges[src] = bucket
    return {"nodes": list(agent_names), "edges": edges}


def flybrain_prior_graph(
    agent_names: list[str],
    *,
    seed: int = 0,
    fly_graph: FlyGraph | None = None,
    max_out_degree: int = 4,
    min_weight: float = 1e-3,
) -> dict[str, Any]:
    """Initial AgentGraph derived from the FlyBrain connectome prior.

    Phase-9 baselines #6-#9 (``flybrain_*``) use a learned controller
    that *reads* the agent graph as a feature; if the runtime starts
    from ``empty_graph`` the controller has no signal to act on. This
    builder materialises the same fly-prior the controllers' GNN
    layer is initialised from, so:

    * every agent shares its fly-derived neighbourhood from the start;
    * the GNN/router immediately sees a non-trivial structure;
    * the resulting initial graph is **deterministic** for a given
      ``(agent_names, seed, fly_graph)`` tuple.

    The mapping uses ``agent_i ↔ fly_node_(i mod num_nodes)``. Each
    fly edge ``(s, t, w)`` contributes ``w`` to the agent edge
    ``(agent_names[s mod K], agent_names[t mod K])`` (self-loops
    skipped). Per-agent edges are then truncated to the
    ``max_out_degree`` strongest neighbours and L1-normalised so the
    weights sum to 1.0 — which matches the row-normalised priors used
    by ``LearnedRouterController.init_from_fly_graph``.

    ``fly_graph`` defaults to the on-disk K-compressed FlyWire 783
    prior (``data/flybrain/fly_graph_64.fbg``) produced by
    ``scripts/build_flywire_csv.py`` + ``flybrain build``. If no
    on-disk artefact is available, the builder falls back to a
    deterministic ``build_synthetic`` graph at
    ``num_nodes = max(len(agent_names), 32)`` so unit tests / CI
    don't require the 813 MB Zenodo download.
    """
    if not agent_names:
        return {"nodes": [], "edges": {}}

    if fly_graph is None:
        fly_graph = _load_default_fly_graph()
    if fly_graph is None:
        from flybrain.graph import build_synthetic

        fly_graph = build_synthetic(
            num_nodes=max(len(agent_names), 32),
            seed=seed,
        )

    k = int(fly_graph.num_nodes)
    if k == 0:
        return {"nodes": list(agent_names), "edges": {}}

    # Rank fly nodes by total non-self-loop weighted degree (in + out)
    # and assign each fly node a slot ``rank % n_agents``. This way
    # every agent represents a "super-cluster" of fly clusters with
    # similar prominence — small Louvain singletons (which dominate
    # the long tail of FlyWire 783 K=64) are folded into the same
    # agents as the well-connected hubs, so every agent ends up with
    # at least one meaningful outgoing edge.
    deg = [0.0] * k
    for (s, t), w in zip(fly_graph.edge_index, fly_graph.edge_weight, strict=True):
        if s == t:
            continue
        weight = float(w)
        if weight <= 0.0:
            continue
        deg[int(s)] += weight
        deg[int(t)] += weight
    # Sort by degree desc, then node id asc so the mapping is
    # deterministic when several nodes share a degree.
    ranked = sorted(range(k), key=lambda i: (-deg[i], i))
    n_agents = len(agent_names)
    fly_to_agent: dict[int, str] = {
        fly_idx: agent_names[rank % n_agents] for rank, fly_idx in enumerate(ranked)
    }

    # Aggregate (src, dst) → summed weight in the agent name space.
    aggregated: dict[tuple[str, str], float] = {}
    for (s, t), w in zip(fly_graph.edge_index, fly_graph.edge_weight, strict=True):
        a = fly_to_agent.get(int(s))
        b = fly_to_agent.get(int(t))
        if a is None or b is None or a == b:
            continue
        weight = float(w)
        if weight <= 0.0:
            continue
        key = (a, b)
        aggregated[key] = aggregated.get(key, 0.0) + weight

    # Truncate each source to its top ``max_out_degree`` neighbours
    # by raw weight, then row-normalise so the weights are scale-free.
    rng = random.Random(seed)
    edges: dict[str, dict[str, float]] = {}
    # Identify the most-connected agent (by total fly-derived in-weight)
    # to use as a fallback target for any agent the rank-mod scheme
    # left without outgoing edges. Without this fallback the GNN /
    # router can't propagate signal out of those agents, which
    # silently degrades the trained baselines for ~1-3 nodes per run.
    in_weight: dict[str, float] = {}
    for (_s, dst), w in aggregated.items():
        in_weight[dst] = in_weight.get(dst, 0.0) + w
    if in_weight:
        hub = max(in_weight.items(), key=lambda kv: (kv[1], kv[0]))[0]
    else:
        hub = agent_names[0]

    for src in agent_names:
        bucket = [(dst, w) for (s, dst), w in aggregated.items() if s == src and w >= min_weight]
        if not bucket:
            # No fly-derived outgoing edge survived the threshold —
            # fall back to a single-edge link to the global hub so
            # the agent isn't orphaned. With a single-agent run no
            # non-self edge exists; skip rather than emit a self-loop.
            fallback = hub if hub != src else next((n for n in agent_names if n != src), None)
            if fallback is None:
                continue
            edges[src] = {fallback: 1.0}
            continue
        # Stable tie-breaking: weight desc, then name asc, plus a
        # tiny rng-driven jitter so seed=0 vs seed=1 produce
        # different orderings when many edges share a weight.
        jitter = {dst: rng.random() * 1e-6 for dst, _ in bucket}
        bucket.sort(key=lambda kv: (-kv[1] - jitter[kv[0]], kv[0]))
        chosen = bucket[: max(1, max_out_degree)]
        total = sum(w for _, w in chosen)
        if total <= 0.0:
            continue
        edges[src] = {dst: float(w) / total for dst, w in chosen}

    return {"nodes": list(agent_names), "edges": edges}


def degree_preserving_random_graph(
    agent_names: list[str],
    *,
    fly_adjacency: dict[str, list[str]] | None = None,
    target_out_degree: float = 2.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Baseline #4 — random graph that *matches the out-degree* of an
    underlying graph (defaults to a target average). Edges are
    rewired uniformly at random while keeping each source's out-degree
    intact.

    Pass ``fly_adjacency`` (a ``{src: [dst, ...]}`` map derived from
    the FlyBrain graph) to preserve the actual fly-prior degree
    sequence; otherwise every node gets ``target_out_degree``
    out-edges to random non-self neighbours.
    """
    rng = random.Random(seed)
    edges: dict[str, dict[str, float]] = {}
    for src in agent_names:
        if fly_adjacency is not None and src in fly_adjacency:
            k = len(fly_adjacency[src])
        else:
            k = max(1, round(target_out_degree))
        candidates = [n for n in agent_names if n != src]
        rng.shuffle(candidates)
        chosen = candidates[: min(k, len(candidates))]
        if chosen:
            edges[src] = {dst: 1.0 for dst in chosen}
    return {"nodes": list(agent_names), "edges": edges}
