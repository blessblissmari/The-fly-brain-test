#!/usr/bin/env python3
"""README §12.5 — graph self-supervised pretraining entrypoint.

Trains a 2-layer GCN encoder on link-prediction + masked-node objectives
over a fly graph (or a synthetic two-block prior) and writes the
resulting weights to ``--output`` as a ``.npz`` file. Downstream the
weights are reloaded into ``AgentGraphEmbedder`` so the controller's
graph encoder benefits from the structural prior even before
simulation / imitation / RL pretraining (Phases 6–8) reaches it via
the action heads.

Usage::

    python scripts/run_graph_ssl_pretrain.py \\
        --graph data/graphs/fly_K128.fbg \\
        --epochs 200 --hidden-dim 32 --out-dim 32 \\
        --output runs/graph_ssl_K128.npz

Reads no Yandex / network. CPU-only (5 minutes for K=128 connectome).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain.training import GraphSSLConfig, graph_ssl_pretrain, save_checkpoint


def _build_synthetic_two_block(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic block model fallback for runs without a real fly graph."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=np.float32)
    half = n // 2
    for i in range(n):
        for j in range(i + 1, n):
            same_block = (i < half) == (j < half)
            p = 0.6 if same_block else 0.05
            if rng.random() < p:
                adj[i, j] = 1.0
    adj = adj + adj.T
    features = rng.standard_normal((n, 16)).astype(np.float32)
    return adj, features


def _load_fly_graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from flybrain.graph.pipeline import load as load_fly_graph

    fly = load_fly_graph(path)
    n = int(fly.num_nodes)
    adj = np.zeros((n, n), dtype=np.float32)
    for (s, t), w in zip(fly.edge_index, fly.edge_weight, strict=True):
        if 0 <= s < n and 0 <= t < n:
            adj[s, t] += float(w)
    adj = 0.5 * (adj + adj.T)

    # Node features: spectral coordinates if available, else random.
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, 16)).astype(np.float32)
    return adj, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Path to a `.fbg` fly graph. If omitted, falls back to a synthetic two-block graph.",
    )
    parser.add_argument(
        "--synthetic-n",
        type=int,
        default=64,
        help="Number of nodes in the synthetic fallback graph.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--out-dim", type=int, default=32)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, default=None)
    args = parser.parse_args()

    if args.graph and args.graph.exists():
        print(f"[graph_ssl] loading fly graph from {args.graph}")
        adj, feats = _load_fly_graph(args.graph)
    else:
        if args.graph:
            print(f"[graph_ssl] {args.graph} not found, falling back to synthetic SBM")
        else:
            print("[graph_ssl] no --graph supplied, using synthetic SBM")
        adj, feats = _build_synthetic_two_block(args.synthetic_n, args.seed)

    cfg = GraphSSLConfig(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        mask_rate=args.mask_rate,
        seed=args.seed,
    )
    print(f"[graph_ssl] cfg={cfg}")
    print(f"[graph_ssl] graph: {adj.shape[0]} nodes, {int((adj > 0).sum() // 2)} undirected edges")
    t0 = time.perf_counter()
    weights, res = graph_ssl_pretrain(adj, feats, cfg)
    elapsed = time.perf_counter() - t0
    print(
        f"[graph_ssl] done loss[first/last]={res.losses[0]:.3f}/{res.losses[-1]:.3f} "
        f"AUC[first/last]={res.link_aucs[0]:.3f}/{res.link_aucs[-1]:.3f} "
        f"wall={elapsed:.1f}s"
    )

    save_checkpoint(weights, args.output)
    print(f"[graph_ssl] saved checkpoint to {args.output}")

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_output.open("w") as fh:
            json.dump(
                {
                    "num_nodes": res.num_nodes,
                    "num_edges": res.num_edges,
                    "epochs": cfg.epochs,
                    "first_loss": res.losses[0] if res.losses else None,
                    "final_loss": res.losses[-1] if res.losses else None,
                    "first_link_auc": res.link_aucs[0] if res.link_aucs else None,
                    "final_link_auc": res.final_link_auc,
                    "first_mask_mse": res.mask_mses[0] if res.mask_mses else None,
                    "final_mask_mse": res.final_mask_mse,
                    "wall_seconds": elapsed,
                },
                fh,
                indent=2,
            )
        print(f"[graph_ssl] saved metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
