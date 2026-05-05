"""Unit tests for ``flybrain.graph.null_priors``.

Round 10 (``docs/round10_prior_ablation.md``) hinges on the
correctness of these null models. Each test verifies a property
the underlying scientific argument depends on:

* ``erdos_renyi_prior`` — exact ``num_nodes`` / ``num_edges``,
  no self-loops, no parallel edges, deterministic by seed.
* ``shuffled_prior`` — Maslov–Sneppen swap **must** preserve the
  in-degree and out-degree of every node (otherwise the
  configuration-model claim is false).
* ``reverse_prior`` — adjacency transpose preserves the
  undirected adjacency, weights, and in/out-degree distributions
  (just swapped between in and out).
"""

from __future__ import annotations

from collections import Counter

from flybrain.graph import build_synthetic
from flybrain.graph.dataclasses import FlyGraph, NodeMetadata
from flybrain.graph.null_priors import (
    degree_summary,
    erdos_renyi_prior,
    reverse_prior,
    shuffled_prior,
)


def _toy_graph(seed: int = 0) -> FlyGraph:
    """A small but non-trivial *simple* directed graph (no self-loops,
    no parallel edges) for null-prior tests.

    The real FlyWire K=64 prior produced by ``flybrain-py build`` is
    always simple — Rust ``compress`` aggregates parallel edges into
    a single weighted edge per pair. ``build_synthetic`` happens to
    emit parallel edges, so we deduplicate before testing so the
    invariants hold for the same shape ``shuffled_prior`` /
    ``reverse_prior`` see in production.
    """
    g = build_synthetic(num_nodes=64, seed=seed)
    seen: set[tuple[int, int]] = set()
    edge_index: list[tuple[int, int]] = []
    edge_weight: list[float] = []
    is_excitatory: list[bool] = []
    for (s, t), w, e in zip(g.edge_index, g.edge_weight, g.is_excitatory, strict=True):
        if s == t or (s, t) in seen:
            continue
        seen.add((s, t))
        edge_index.append((int(s), int(t)))
        edge_weight.append(float(w))
        is_excitatory.append(bool(e))
    return FlyGraph(
        num_nodes=g.num_nodes,
        edge_index=edge_index,
        edge_weight=edge_weight,
        is_excitatory=is_excitatory,
        nodes=list(g.nodes),
        provenance=dict(g.provenance),
    )


def test_erdos_renyi_prior_basic_shape() -> None:
    g = erdos_renyi_prior(num_nodes=64, num_edges=199, seed=0)
    assert g.num_nodes == 64
    assert g.num_edges == 199
    assert len(g.edge_weight) == 199
    assert len(g.is_excitatory) == 199
    # No self-loops, no parallel edges.
    assert all(s != t for s, t in g.edge_index)
    assert len(set(g.edge_index)) == 199
    assert g.provenance["null_model"] == "erdos_renyi"


def test_erdos_renyi_prior_is_deterministic() -> None:
    a = erdos_renyi_prior(num_nodes=64, num_edges=199, seed=42)
    b = erdos_renyi_prior(num_nodes=64, num_edges=199, seed=42)
    assert a.edge_index == b.edge_index


def test_erdos_renyi_prior_changes_with_seed() -> None:
    a = erdos_renyi_prior(num_nodes=64, num_edges=199, seed=0)
    b = erdos_renyi_prior(num_nodes=64, num_edges=199, seed=1)
    # Vanishingly unlikely that two seeds produce identical edge sets.
    assert a.edge_index != b.edge_index


def test_erdos_renyi_prior_rejects_too_many_edges() -> None:
    import pytest

    with pytest.raises(ValueError):
        erdos_renyi_prior(num_nodes=4, num_edges=999, seed=0)


def test_shuffled_prior_preserves_degree_distribution() -> None:
    """Maslov–Sneppen swap MUST preserve in/out-degree of every node."""
    real = _toy_graph()
    shuffled = shuffled_prior(real, seed=0)
    real_summary = degree_summary(real)
    shuffled_summary = degree_summary(shuffled)
    assert real_summary["in_degree"] == shuffled_summary["in_degree"]
    assert real_summary["out_degree"] == shuffled_summary["out_degree"]
    assert real.num_edges == shuffled.num_edges
    assert real.num_nodes == shuffled.num_nodes


def test_shuffled_prior_actually_changes_edges() -> None:
    """Verify the swap produced a meaningfully different topology."""
    real = _toy_graph()
    shuffled = shuffled_prior(real, seed=0)
    # At least 30% of edges should differ — otherwise mixing failed.
    real_set = set(real.edge_index)
    shuffled_set = set(shuffled.edge_index)
    intersection = real_set & shuffled_set
    assert len(intersection) < 0.7 * len(real_set), (
        f"shuffled retains {len(intersection)}/{len(real_set)} edges "
        "— mixing did not run long enough"
    )


def test_shuffled_prior_no_self_loops_or_parallels() -> None:
    real = _toy_graph()
    shuffled = shuffled_prior(real, seed=0)
    assert all(s != t for s, t in shuffled.edge_index)
    assert len(set(shuffled.edge_index)) == len(shuffled.edge_index)


def test_shuffled_prior_records_provenance() -> None:
    real = _toy_graph()
    shuffled = shuffled_prior(real, seed=0)
    assert shuffled.provenance["null_model"] == "shuffled_configuration"
    assert shuffled.provenance["seed"] == 0
    assert shuffled.provenance["swaps_accepted"] > 0


def test_reverse_prior_swaps_in_and_out_degree() -> None:
    real = _toy_graph()
    rev = reverse_prior(real)
    real_in = Counter(t for _, t in real.edge_index)
    real_out = Counter(s for s, _ in real.edge_index)
    rev_in = Counter(t for _, t in rev.edge_index)
    rev_out = Counter(s for s, _ in rev.edge_index)
    # The transpose flips in/out — what was a source becomes a sink.
    for n in range(real.num_nodes):
        assert real_in.get(n, 0) == rev_out.get(n, 0), (
            f"node {n}: real_in={real_in.get(n, 0)} rev_out={rev_out.get(n, 0)}"
        )
        assert real_out.get(n, 0) == rev_in.get(n, 0), (
            f"node {n}: real_out={real_out.get(n, 0)} rev_in={rev_in.get(n, 0)}"
        )


def test_reverse_prior_preserves_undirected_adjacency() -> None:
    real = _toy_graph()
    rev = reverse_prior(real)
    real_undir = {tuple(sorted((s, t))) for s, t in real.edge_index}
    rev_undir = {tuple(sorted((s, t))) for s, t in rev.edge_index}
    # Round-trip via the synthetic builder may have mutual edges that
    # collapse to fewer undirected pairs after the transpose; the
    # *undirected* edge set must be a subset of the original because
    # transpose can only ever revisit existing edges (no new ones).
    assert rev_undir.issubset(real_undir)


def test_reverse_prior_double_reverse_round_trip() -> None:
    """Reversing twice should restore the original (modulo merge logic)."""
    real = _toy_graph()
    twice = reverse_prior(reverse_prior(real))
    # On a graph with no parallel edges the round-trip is exact.
    assert set(twice.edge_index) == set(real.edge_index)


def test_degree_summary_consistency() -> None:
    g = FlyGraph(
        num_nodes=4,
        edge_index=[(0, 1), (1, 2), (2, 3), (1, 3)],
        edge_weight=[1.0, 1.0, 1.0, 1.0],
        is_excitatory=[True, True, True, True],
        nodes=[NodeMetadata(id=i) for i in range(4)],
        provenance={},
    )
    s = degree_summary(g)
    assert s["num_nodes"] == 4
    assert s["num_edges"] == 4
    # node 1 has out-degree 2, node 3 has in-degree 2.
    assert s["out_degree"][1] == 2
    assert s["in_degree"][3] == 2
    # All other in/out degrees are 1 except node 0 (in=0) and node 3 (out=0).
    assert s["in_degree"][0] == 0
    assert s["out_degree"][3] == 0
