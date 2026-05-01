"""README §12.5 graph self-supervised pretraining (link prediction +
masked-node feature reconstruction).

The Phase-4 controller ingests two graph signals: the dynamic
``agent_graph`` (a tiny K×K MAS topology) and the static fly prior
(``FlyGraph`` from Phase 1). Both are encoded by a 2-layer GCN whose
weights are deterministic Gaussian projections in
``flybrain.embeddings.graph_emb.AgentGraphEmbedder``. That makes the
forward path numerically stable for tests but leaves the encoder
*unlearned* — the controller does not benefit from the structural
information in the fly connectome until simulation / imitation / RL
pretraining (Phases 6–8) reaches it indirectly through the action
heads.

This module pretrains a torch GCN encoder with two SSL objectives
(README §12.5):

* **Link prediction** — the encoder must score positive edges higher
  than uniformly-sampled negative edges. Standard GraphSAGE-style
  contrastive loss with binary cross-entropy on dot products.
* **Masked node reconstruction** — a fraction of nodes have their
  input features zeroed; the encoder must reconstruct the originals
  by a single linear decoder. Mean-squared-error loss.

The two losses are summed with `mask_loss_weight` controlling the
relative weight. After training, the encoder weights can be saved as
plain numpy arrays compatible with ``AgentGraphEmbedder`` so the
existing controller keeps using its zero-dependency numpy forward.

The exit criterion (PLAN.md §12.5 / "+ graph self-supervised
pretraining" row in README §18 Exp 4) is "controller convergeет на
synthetic+fly priors за <5 минут на CPU и проходит regression test
на link-prediction AUC ≥ 0.85" — see
``tests/python/unit/test_graph_ssl.py`` for the unit-level check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch


@dataclass(slots=True)
class GraphSSLConfig:
    """Hyperparameters for ``graph_ssl_pretrain``."""

    epochs: int = 50
    """Total optimisation epochs."""
    lr: float = 1e-2
    """AdamW learning rate."""
    weight_decay: float = 1e-5
    hidden_dim: int = 32
    out_dim: int = 32
    mask_rate: float = 0.15
    """Fraction of nodes to mask during the masked-node objective."""
    num_neg_samples: int = 1
    """Number of negative edges sampled per positive edge."""
    mask_loss_weight: float = 0.5
    """Relative weight of the masked-node loss vs. the link loss."""
    seed: int = 0


@dataclass(slots=True)
class GraphSSLResult:
    """Output of ``graph_ssl_pretrain``."""

    losses: list[float] = field(default_factory=list)
    link_aucs: list[float] = field(default_factory=list)
    mask_mses: list[float] = field(default_factory=list)
    final_link_auc: float = 0.0
    final_mask_mse: float = 0.0
    num_nodes: int = 0
    num_edges: int = 0


def _normalise_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric Kipf normalisation ``D^{-1/2} (A + I) D^{-1/2}``."""
    import torch

    n = adj.shape[0]
    a = adj + torch.eye(n, dtype=adj.dtype, device=adj.device)
    deg = a.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    d = torch.diag(deg_inv_sqrt)
    return d @ a @ d


def _build_encoder(in_dim: int, hidden_dim: int, out_dim: int) -> Any:
    """Return a 2-layer GCN matching ``AgentGraphEmbedder``'s shape.

    Returned as ``Any`` because the inner subclass is defined inside
    the function (so ``torch.nn`` is only imported when this module
    is actually used) — typed callers go through ``GraphSSLEncoder``.
    """
    import torch
    import torch.nn as nn

    class _GCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w0 = nn.Linear(in_dim, hidden_dim, bias=False)
            self.w1 = nn.Linear(hidden_dim, out_dim, bias=False)

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
            h = torch.relu(adj_norm @ self.w0(x))
            # Linear (no ReLU) on the second layer so embeddings keep
            # full sign expressiveness for the link-prediction dot
            # product. Matches the original GraphSAGE default.
            return adj_norm @ self.w1(h)

    return _GCN()


def _sample_negative_edges(
    n: int,
    pos_set: set[tuple[int, int]],
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``k`` unordered negative edges (pairs not in ``pos_set``)."""
    src = np.empty(k, dtype=np.int64)
    dst = np.empty(k, dtype=np.int64)
    i = 0
    attempts = 0
    while i < k and attempts < 50 * k:
        s, t = int(rng.integers(0, n)), int(rng.integers(0, n))
        attempts += 1
        if s == t:
            continue
        key = (min(s, t), max(s, t))
        if key in pos_set:
            continue
        src[i], dst[i] = s, t
        i += 1
    if i < k:
        src = src[:i]
        dst = dst[:i]
    return src, dst


def _link_prediction_auc(z: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Naive AUC: fraction of (pos, neg) score pairs where pos > neg."""
    import torch

    if pos.numel() == 0 or neg.numel() == 0:
        return 0.0
    pos_scores = (z[pos[0]] * z[pos[1]]).sum(dim=1)
    neg_scores = (z[neg[0]] * z[neg[1]]).sum(dim=1)
    # All-pairs comparison; small enough for the unit-test size.
    diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
    return float((diff > 0).to(torch.float32).mean().item())


def graph_ssl_pretrain(
    adjacency: np.ndarray,
    node_features: np.ndarray,
    config: GraphSSLConfig | None = None,
) -> tuple[dict[str, np.ndarray], GraphSSLResult]:
    """Train a 2-layer GCN encoder on link-prediction + masked-node SSL.

    ``adjacency`` is a square ``(n, n)`` symmetric matrix; ``node_features``
    is ``(n, in_dim)``. Returns the encoder's weight matrices as a
    dict (``{"w0": (in_dim, hidden_dim), "w1": (hidden_dim, out_dim)}``)
    plus a ``GraphSSLResult`` with per-epoch metrics. The weight dict is
    drop-in compatible with ``AgentGraphEmbedder`` once persisted via
    ``save_checkpoint`` and reloaded into the embedder's ``_w0`` / ``_w1``
    fields.
    """
    import torch

    cfg = config or GraphSSLConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    n = adjacency.shape[0]
    in_dim = node_features.shape[1]
    if n < 4 or node_features.shape[0] != n:
        return (
            {
                "w0": np.zeros((in_dim, cfg.hidden_dim), dtype=np.float32),
                "w1": np.zeros((cfg.hidden_dim, cfg.out_dim), dtype=np.float32),
            },
            GraphSSLResult(num_nodes=n, num_edges=0),
        )

    # Edge list — keep undirected pairs (sorted) so neg sampling stays simple.
    pos_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] != 0.0 or adjacency[j, i] != 0.0:
                pos_pairs.append((i, j))
    if not pos_pairs:
        return (
            {
                "w0": np.zeros((in_dim, cfg.hidden_dim), dtype=np.float32),
                "w1": np.zeros((cfg.hidden_dim, cfg.out_dim), dtype=np.float32),
            },
            GraphSSLResult(num_nodes=n, num_edges=0),
        )
    pos_set = set(pos_pairs)
    pos_arr = np.array(pos_pairs, dtype=np.int64).T  # shape (2, num_edges)

    adj_t = torch.from_numpy(adjacency.astype(np.float32))
    feat_t = torch.from_numpy(node_features.astype(np.float32))
    adj_norm = _normalise_adjacency(adj_t)

    encoder = _build_encoder(in_dim, cfg.hidden_dim, cfg.out_dim)
    optim = torch.optim.AdamW(
        encoder.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    result = GraphSSLResult(num_nodes=n, num_edges=len(pos_pairs))

    for _epoch in range(cfg.epochs):
        # --- masking ------------------------------------------------------
        n_mask = max(1, int(cfg.mask_rate * n))
        mask_idx_np = rng.choice(n, size=n_mask, replace=False)
        mask_idx = torch.from_numpy(mask_idx_np.astype(np.int64))
        feat_masked = feat_t.clone()
        feat_masked[mask_idx] = 0.0

        # --- negative sampling --------------------------------------------
        k_neg = cfg.num_neg_samples * len(pos_pairs)
        neg_src_np, neg_dst_np = _sample_negative_edges(n, pos_set, k_neg, rng)
        if neg_src_np.size == 0:
            continue
        pos_t = torch.from_numpy(pos_arr)
        neg_t = torch.from_numpy(np.stack([neg_src_np, neg_dst_np]))

        # --- forward ------------------------------------------------------
        z = encoder(feat_masked, adj_norm)

        pos_scores = (z[pos_t[0]] * z[pos_t[1]]).sum(dim=1)
        neg_scores = (z[neg_t[0]] * z[neg_t[1]]).sum(dim=1)
        # BCE-with-logits on link prediction.
        link_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        # MSE on masked-node feature reconstruction. Project z back to
        # `in_dim` via a deterministic random matrix so the encoder is
        # forced to preserve original feature direction (otherwise the
        # trivial all-zero solution wins).
        proj = _project_for_reconstruction(in_dim, cfg.out_dim)
        recon = z @ proj
        mask_loss = torch.nn.functional.mse_loss(recon[mask_idx], feat_t[mask_idx])

        loss = link_loss + cfg.mask_loss_weight * mask_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        result.losses.append(float(loss.item()))
        with torch.no_grad():
            z_eval = encoder(feat_t, adj_norm)
            result.link_aucs.append(_link_prediction_auc(z_eval, pos_t, neg_t))
            recon_eval = z_eval @ proj
            result.mask_mses.append(
                float(torch.nn.functional.mse_loss(recon_eval[mask_idx], feat_t[mask_idx]).item())
            )

    result.final_link_auc = result.link_aucs[-1] if result.link_aucs else 0.0
    result.final_mask_mse = result.mask_mses[-1] if result.mask_mses else 0.0

    # Pull weights out as numpy for the AgentGraphEmbedder reload path.
    weights = {
        "w0": encoder.w0.weight.detach().cpu().numpy().T.copy(),  # (in, hidden)
        "w1": encoder.w1.weight.detach().cpu().numpy().T.copy(),  # (hidden, out)
    }
    return weights, result


def _project_for_reconstruction(in_dim: int, out_dim: int) -> torch.Tensor:
    """Deterministic random projection from encoder output back to input
    space (used as the masked-node decoder)."""
    import torch

    rng = np.random.default_rng(seed=hash(("graph-ssl-decoder", out_dim, in_dim)) & 0xFFFFFFFF)
    w = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    w *= np.sqrt(2.0 / max(1, out_dim))
    return torch.from_numpy(w)


def save_checkpoint(weights: dict[str, np.ndarray], path: Path | str) -> None:
    """Persist SSL-trained weights as ``.npz`` so ``AgentGraphEmbedder``
    can reload them without bringing in torch."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # numpy's `savez` typing for **kwargs collides with the
    # `allow_pickle` keyword in newer stubs; cast to silence mypy.
    np.savez(str(p), **weights)  # type: ignore[arg-type]


def load_checkpoint(path: Path | str) -> dict[str, np.ndarray]:
    """Load SSL weights from disk (counterpart of ``save_checkpoint``)."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def apply_to_embedder(
    embedder: Any,
    weights: dict[str, np.ndarray],
) -> None:
    """Patch a numpy ``AgentGraphEmbedder`` in-place with SSL-trained
    weights (overrides the deterministic Gaussian random projection)."""
    if "w0" in weights:
        if weights["w0"].shape != embedder._w0.shape:
            raise ValueError(
                f"w0 shape mismatch: expected {embedder._w0.shape}, got {weights['w0'].shape}"
            )
        embedder._w0 = weights["w0"].astype(np.float32, copy=False)
    if "w1" in weights:
        if weights["w1"].shape != embedder._w1.shape:
            raise ValueError(
                f"w1 shape mismatch: expected {embedder._w1.shape}, got {weights['w1'].shape}"
            )
        embedder._w1 = weights["w1"].astype(np.float32, copy=False)


__all__ = [
    "GraphSSLConfig",
    "GraphSSLResult",
    "apply_to_embedder",
    "graph_ssl_pretrain",
    "load_checkpoint",
    "save_checkpoint",
]
