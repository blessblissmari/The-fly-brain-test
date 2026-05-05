"""Null-prior generators for the round-10 connectome ablation.

Round 10 (``docs/round10_prior_ablation.md``) asks the
fundamental falsifiability question of README §17:

    *Is the trained controller's success driven by the biological
    topology of the FlyWire 783 connectome, or would any prior
    with the same coarse statistics work equally well?*

To answer it we generate three null-priors at K=64 that share
progressively more structure with the real FlyWire Louvain prior:

* ``erdos_renyi_prior`` (matches only ``num_nodes`` and edge count)
* ``shuffled_prior`` (configuration-model double-edge swap; matches
  in/out-degree of every node and edge count, breaks correlations)
* ``reverse_prior`` (transpose of the real prior; matches the
  undirected adjacency exactly, breaks directionality)

All generators return a :class:`flybrain.graph.FlyGraph` instance
compatible with the existing controller stack: the same .fbg
schema, same ``num_nodes``, same ``edge_index`` / ``edge_weight``
/ ``is_excitatory`` layout. ``provenance`` is augmented with a
``'null_model'`` field so downstream code (and humans reading the
file) can tell what they're looking at.

References:

* Maslov & Sneppen, "Specificity and stability in topology of
  protein networks", *Science* 296 (2002) — double-edge swap as the
  canonical degree-preserving null in network neuroscience.
* Milo et al., "Network motifs", *Science* 298 (2002) — same null
  for motif detection.
* Towlson et al., "The rich club of the C. elegans neuronal
  connectome", *J. Neurosci.* 33 (2013) — application of
  configuration-model nulls to a real connectome.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from flybrain.graph.dataclasses import FlyGraph, NodeMetadata


def _clone_nodes(graph: FlyGraph) -> list[NodeMetadata]:
    """Return a deep copy of ``graph.nodes`` (mutating callers safe)."""
    return [
        NodeMetadata(
            id=n.id,
            node_type=n.node_type,
            region=n.region,
            features=list(n.features),
        )
        for n in graph.nodes
    ]


def _empty_nodes(num_nodes: int) -> list[NodeMetadata]:
    """Default node metadata when we don't have a reference graph."""
    return [NodeMetadata(id=i, node_type="", region="", features=[]) for i in range(num_nodes)]


def erdos_renyi_prior(
    num_nodes: int,
    num_edges: int,
    *,
    seed: int = 0,
    weight: float = 1.0,
    is_excitatory: bool = True,
    nodes: list[NodeMetadata] | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> FlyGraph:
    """Erdős–Rényi G(n, m) directed graph with no self-loops.

    Samples ``num_edges`` distinct ``(src, dst)`` pairs uniformly from
    the ``num_nodes * (num_nodes - 1)`` possible non-self-loop pairs.
    Each edge gets the constant ``weight`` and ``is_excitatory`` flag.

    Use this as the **null model with no structure** — it controls
    only for ``num_nodes`` and edge count. If a baseline performs
    well with this prior, the FlyBrain claim "biological topology
    matters" is severely weakened.
    """
    if num_nodes <= 1:
        raise ValueError(f"num_nodes must be > 1, got {num_nodes}")
    max_edges = num_nodes * (num_nodes - 1)
    if num_edges < 0 or num_edges > max_edges:
        raise ValueError(f"num_edges must be in [0, {max_edges}], got {num_edges}")

    rng = random.Random(seed)

    # Sample without replacement from all (src, dst) pairs with src != dst.
    # For sparse graphs (m << n*(n-1)) rejection sampling is fastest.
    chosen: set[tuple[int, int]] = set()
    if num_edges <= max_edges // 2:
        while len(chosen) < num_edges:
            s = rng.randrange(num_nodes)
            t = rng.randrange(num_nodes)
            if s == t:
                continue
            chosen.add((s, t))
    else:
        # Dense regime: enumerate all pairs and shuffle.
        all_pairs = [(s, t) for s in range(num_nodes) for t in range(num_nodes) if s != t]
        rng.shuffle(all_pairs)
        chosen = set(all_pairs[:num_edges])

    edge_index = sorted(chosen)
    edge_weight = [float(weight)] * len(edge_index)
    is_exc = [bool(is_excitatory)] * len(edge_index)

    provenance: dict[str, Any] = {
        "source": "null_prior",
        "null_model": "erdos_renyi",
        "seed": seed,
        "num_nodes": num_nodes,
        "num_edges": len(edge_index),
        "weight": weight,
    }
    if extra_provenance:
        provenance.update(extra_provenance)

    return FlyGraph(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        is_excitatory=is_exc,
        nodes=nodes if nodes is not None else _empty_nodes(num_nodes),
        provenance=provenance,
    )


def shuffled_prior(
    graph: FlyGraph,
    *,
    seed: int = 0,
    num_swaps: int | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> FlyGraph:
    """Maslov–Sneppen double-edge-swap null model.

    Repeatedly picks two edges ``(a -> b)`` and ``(c -> d)`` uniformly
    at random and rewires them to ``(a -> d)`` and ``(c -> b)``. The
    swap is accepted only if (i) all four endpoints are distinct,
    (ii) neither rewired edge already exists, and (iii) neither is a
    self-loop. This preserves the **in-degree and out-degree of
    every node** as well as the total edge count and total weight,
    while randomising who-connects-to-whom — the standard null in
    network neuroscience for distinguishing "topology matters" from
    "node degree matters" (Maslov & Sneppen 2002, Milo et al. 2002).

    For directed multi-edge sources (FlyWire's per-pair aggregated
    weights are already single-edge after :func:`flybrain.graph.compress`)
    we treat the input as a simple directed graph: weights are kept
    paired with their original ``is_excitatory`` flag through the
    rewiring so excitatory / inhibitory ratios per source-node are
    preserved.

    ``num_swaps`` defaults to ``10 * num_edges`` which is the
    convention used by NetworkX's ``directed_edge_swap`` and is
    sufficient to reach mixing on graphs of up to a few thousand
    edges (Greene & Cunningham, "Producing accurate interpretable
    null networks", 2010). At K=64, m≈200, this is ~2000 swap
    attempts and finishes in well under a second.
    """
    if graph.num_nodes <= 1:
        raise ValueError(f"graph must have at least 2 nodes for swap, got {graph.num_nodes}")

    rng = random.Random(seed)

    # Materialise edges with their associated payload so we can rewire
    # without losing weights / excitatory tags.
    edges: list[tuple[int, int, float, bool]] = [
        (int(s), int(t), float(w), bool(e))
        for (s, t), w, e in zip(
            graph.edge_index, graph.edge_weight, graph.is_excitatory, strict=True
        )
    ]
    edge_set: set[tuple[int, int]] = {(s, t) for s, t, _, _ in edges}

    target_swaps = num_swaps if num_swaps is not None else 10 * len(edges)
    accepted = 0
    attempted = 0

    # Cap attempts so a pathologically dense graph can't hang. The
    # FlyWire K=64 prior has ~83% of edges as part of a mutual
    # ``(a -> b, b -> a)`` pair, which forces rejection of most
    # candidate rewires (see scripts/build_null_priors.py for the
    # empirical success rate ≈ 0.8%). 1000× over-provisioning lets
    # mixing complete in well under a second on K=64 connectomes
    # while still failing fast on degenerate inputs.
    max_attempts = max(target_swaps * 1000, 10_000)

    while accepted < target_swaps and attempted < max_attempts:
        attempted += 1
        if len(edges) < 2:
            break
        i = rng.randrange(len(edges))
        j = rng.randrange(len(edges))
        if i == j:
            continue
        s1, t1, w1, e1 = edges[i]
        s2, t2, w2, e2 = edges[j]

        # Endpoints must be all distinct: no self-loops, no parallel edges.
        if len({s1, t1, s2, t2}) != 4:
            continue

        new_a = (s1, t2)
        new_b = (s2, t1)
        if new_a in edge_set or new_b in edge_set:
            continue
        if new_a[0] == new_a[1] or new_b[0] == new_b[1]:
            continue

        edge_set.discard((s1, t1))
        edge_set.discard((s2, t2))
        edge_set.add(new_a)
        edge_set.add(new_b)

        # Pair each rewired edge with the original payload of the
        # edge whose source it inherits, so per-source excitatory /
        # weight statistics are preserved.
        edges[i] = (s1, t2, w1, e1)
        edges[j] = (s2, t1, w2, e2)
        accepted += 1

    edge_index = sorted((s, t) for s, t, _, _ in edges)
    payload = {(s, t): (w, e) for s, t, w, e in edges}
    edge_weight = [payload[(s, t)][0] for s, t in edge_index]
    is_excitatory = [payload[(s, t)][1] for s, t in edge_index]

    provenance: dict[str, Any] = {
        "source": "null_prior",
        "null_model": "shuffled_configuration",
        "seed": seed,
        "swaps_target": target_swaps,
        "swaps_accepted": accepted,
        "swaps_attempted": attempted,
        "num_nodes": graph.num_nodes,
        "num_edges": len(edge_index),
        "based_on": graph.provenance.get("source"),
    }
    if extra_provenance:
        provenance.update(extra_provenance)

    return FlyGraph(
        num_nodes=graph.num_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        is_excitatory=is_excitatory,
        nodes=_clone_nodes(graph),
        provenance=provenance,
    )


def reverse_prior(
    graph: FlyGraph,
    *,
    extra_provenance: dict[str, Any] | None = None,
) -> FlyGraph:
    """Adjacency transpose: every edge ``a -> b`` becomes ``b -> a``.

    Preserves the *undirected* adjacency, every weight, every
    ``is_excitatory`` flag, and the total in / out-degree distribution
    (just swapped between in- and out-).  The directional structure of
    the connectome — which neuron fires onto which — is destroyed.

    Tests whether the controller's GNN actually uses connectome
    directionality. If ``reverse_prior`` ≈ ``flybrain``, direction
    is decorative for the agent-routing task; if it's strictly
    worse, the GCN's directed message passing is exploiting real
    information.
    """
    edges_by_pair: dict[tuple[int, int], tuple[float, bool]] = {}
    # Sum weights for any (b, a) collisions that arise from parallel
    # edges in the source graph (none expected for the K=64 prior, but
    # we keep the merge logic for safety).
    for (s, t), w, e in zip(graph.edge_index, graph.edge_weight, graph.is_excitatory, strict=True):
        key = (int(t), int(s))
        if key in edges_by_pair:
            prev_w, prev_e = edges_by_pair[key]
            edges_by_pair[key] = (prev_w + float(w), prev_e or bool(e))
        else:
            edges_by_pair[key] = (float(w), bool(e))

    edge_index = sorted(edges_by_pair.keys())
    edge_weight = [edges_by_pair[k][0] for k in edge_index]
    is_excitatory = [edges_by_pair[k][1] for k in edge_index]

    provenance: dict[str, Any] = {
        "source": "null_prior",
        "null_model": "reverse_transpose",
        "num_nodes": graph.num_nodes,
        "num_edges": len(edge_index),
        "based_on": graph.provenance.get("source"),
    }
    if extra_provenance:
        provenance.update(extra_provenance)

    return FlyGraph(
        num_nodes=graph.num_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        is_excitatory=is_excitatory,
        nodes=_clone_nodes(graph),
        provenance=provenance,
    )


def degree_summary(graph: FlyGraph) -> dict[str, Any]:
    """Return per-node and aggregate degree statistics for ``graph``.

    Used by :func:`scripts/build_null_priors.py` to verify that
    ``shuffled_prior`` actually preserves the in/out-degree
    distribution it's supposed to and to populate the
    ``round10_prior_ablation`` provenance JSON sidecar.
    """
    in_deg: dict[int, int] = defaultdict(int)
    out_deg: dict[int, int] = defaultdict(int)
    for s, t in graph.edge_index:
        out_deg[int(s)] += 1
        in_deg[int(t)] += 1
    in_list = [in_deg.get(i, 0) for i in range(graph.num_nodes)]
    out_list = [out_deg.get(i, 0) for i in range(graph.num_nodes)]
    return {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "in_degree_min": min(in_list, default=0),
        "in_degree_max": max(in_list, default=0),
        "in_degree_mean": sum(in_list) / max(1, len(in_list)),
        "out_degree_min": min(out_list, default=0),
        "out_degree_max": max(out_list, default=0),
        "out_degree_mean": sum(out_list) / max(1, len(out_list)),
        "in_degree": in_list,
        "out_degree": out_list,
    }


__all__ = [
    "degree_summary",
    "erdos_renyi_prior",
    "reverse_prior",
    "shuffled_prior",
]
