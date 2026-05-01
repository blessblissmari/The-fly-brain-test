"""Unit tests for the README §12.5 graph self-supervised pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="graph_ssl needs torch")

from flybrain.embeddings import AgentGraphEmbedder  # noqa: E402
from flybrain.training import (  # noqa: E402
    GraphSSLConfig,
    GraphSSLResult,
    apply_to_embedder,
    graph_ssl_pretrain,
    load_checkpoint,
    save_checkpoint,
)


def _two_block_graph(n: int = 32, p_in: float = 0.6, p_out: float = 0.05, seed: int = 0):
    """Stochastic block model: two equal halves with intra > inter density.

    Returns ``(adjacency, features)``."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=np.float32)
    half = n // 2
    for i in range(n):
        for j in range(i + 1, n):
            same_block = (i < half) == (j < half)
            if rng.random() < (p_in if same_block else p_out):
                adj[i, j] = 1.0
    adj = adj + adj.T
    features = rng.standard_normal((n, 16)).astype(np.float32)
    return adj, features


def test_graph_ssl_result_shape() -> None:
    adj, feats = _two_block_graph(n=16)
    cfg = GraphSSLConfig(epochs=5, hidden_dim=8, out_dim=8)
    weights, res = graph_ssl_pretrain(adj, feats, cfg)

    assert isinstance(res, GraphSSLResult)
    assert res.num_nodes == 16
    assert res.num_edges > 0
    assert len(res.losses) == 5
    assert len(res.link_aucs) == 5
    assert len(res.mask_mses) == 5
    assert weights["w0"].shape == (16, 8)  # in_dim → hidden_dim
    assert weights["w1"].shape == (8, 8)  # hidden_dim → out_dim


def test_graph_ssl_link_auc_improves_on_structured_graph() -> None:
    """README §18 Exp 4 row 2: SSL must learn a non-trivial link
    representation. We allow generous tolerance because the loss is
    stochastic and we only run a handful of epochs."""
    adj, feats = _two_block_graph(n=32)
    cfg = GraphSSLConfig(epochs=200, lr=1e-2, hidden_dim=16, out_dim=16)
    _weights, res = graph_ssl_pretrain(adj, feats, cfg)

    # Random AUC is 0.5; a working SSL trainer should beat 0.75 on this graph.
    assert res.final_link_auc >= 0.75, (
        f"final AUC {res.final_link_auc} below the 0.75 baseline "
        f"(starting AUC was {res.link_aucs[0]:.3f})"
    )


def test_graph_ssl_handles_empty_graph() -> None:
    """Edge case: a graph with no edges should not crash; the trainer
    should return zero-weight matrices and an empty result."""
    n = 8
    adj = np.zeros((n, n), dtype=np.float32)
    feats = np.zeros((n, 4), dtype=np.float32)
    weights, res = graph_ssl_pretrain(adj, feats, GraphSSLConfig(epochs=2))
    assert res.num_edges == 0
    # Weights still have the right shape (in_dim=4, hidden_dim=32, out_dim=32 defaults)
    assert weights["w0"].shape == (4, 32)
    assert weights["w1"].shape == (32, 32)


def test_graph_ssl_handles_tiny_graph() -> None:
    """Graphs with fewer than 4 nodes should short-circuit gracefully."""
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    feats = np.eye(3, dtype=np.float32)
    weights, res = graph_ssl_pretrain(adj, feats, GraphSSLConfig(epochs=2))
    assert res.num_nodes == 3
    assert weights["w0"].shape == (3, 32)


def test_graph_ssl_weights_save_load_roundtrip(tmp_path: Path) -> None:
    adj, feats = _two_block_graph(n=16)
    weights, _ = graph_ssl_pretrain(adj, feats, GraphSSLConfig(epochs=2, hidden_dim=8, out_dim=8))
    p = tmp_path / "ssl.npz"
    save_checkpoint(weights, p)
    loaded = load_checkpoint(p)
    np.testing.assert_array_equal(loaded["w0"], weights["w0"])
    np.testing.assert_array_equal(loaded["w1"], weights["w1"])


def test_graph_ssl_apply_to_embedder_swaps_weights() -> None:
    """Trained weights must replace the deterministic Gaussian projection
    in ``AgentGraphEmbedder``; the embedder still produces the right
    output shape afterwards."""
    emb = AgentGraphEmbedder(in_dim=8, hidden_dim=4, out_dim=4)
    original_w0 = emb._w0.copy()
    weights = {
        "w0": np.ones_like(emb._w0) * 0.5,
        "w1": np.ones_like(emb._w1) * 0.25,
    }
    apply_to_embedder(emb, weights)
    assert not np.array_equal(emb._w0, original_w0)
    np.testing.assert_array_equal(emb._w0, np.ones_like(original_w0) * 0.5)

    # The embedder still functions on a sample graph.
    g = {"nodes": ["A", "B"], "edges": {"A": {"B": 1.0}}}
    feats = np.ones((2, 8), dtype=np.float32)
    graph_vec, node_vecs = emb.embed(g, ["A", "B"], feats)
    assert graph_vec.shape == (4,)
    assert node_vecs.shape == (2, 4)


def test_graph_ssl_apply_rejects_shape_mismatch() -> None:
    emb = AgentGraphEmbedder(in_dim=8, hidden_dim=4, out_dim=4)
    bad = {"w0": np.zeros((2, 2), dtype=np.float32)}
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_to_embedder(emb, bad)


def test_graph_ssl_pretrain_is_deterministic_for_seed() -> None:
    adj, feats = _two_block_graph(n=16)
    cfg = GraphSSLConfig(epochs=3, hidden_dim=8, out_dim=8, seed=7)
    w_a, res_a = graph_ssl_pretrain(adj, feats, cfg)
    w_b, res_b = graph_ssl_pretrain(adj, feats, cfg)
    np.testing.assert_allclose(w_a["w0"], w_b["w0"], rtol=1e-5)
    np.testing.assert_allclose(w_a["w1"], w_b["w1"], rtol=1e-5)
    assert res_a.losses == res_b.losses
