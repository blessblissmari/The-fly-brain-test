#!/usr/bin/env python3
"""Round-12 — train a LoRA adapter on top of the frozen
``sim_pretrain_gnn_v6`` checkpoint, using ``manual_graph`` traces
collected across rounds 7-13 as the supervised signal.

Why round-12 needs a separate script (rather than reusing
``scripts/run_imitation.py``):

1. The base controller is **frozen**; only the LoRA adapter is
   trained. ``run_imitation.py`` builds an unfrozen
   :class:`FlyBrainGNNController` and then optimises every
   parameter, which would overwrite the round-7 weights and break
   the cost-Pareto win.
2. The output checkpoint format is the small adapter-only blob
   (``flybrain.training.lora_adapter.save_lora_adapter``), not the
   full ``state_dict``.
3. The default trace dirs are pinned to the project's stored
   manual_graph runs from rounds 7-13 — so retraining is
   reproducible without an explicit ``--traces`` arg.

Run with::

    python scripts/train_round12_lora.py \
        --base-checkpoint data/checkpoints/sim_pretrain_gnn_v6.pt \
        --output data/checkpoints/lora_adapter_round12.pt \
        --rank 4 --epochs 12 --lr 5e-3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import torch

from flybrain.agents.specs import MINIMAL_15
from flybrain.embeddings import (
    AgentEmbedder,
    AgentGraphEmbedder,
    ControllerStateBuilder,
    FlyGraphEmbedder,
    MockEmbeddingClient,
    TaskEmbedder,
    TraceEmbedder,
)
from flybrain.training.expert_dataset import collect_examples, iter_traces
from flybrain.training.imitation import ImitationConfig, imitation_train
from flybrain.training.lora_adapter import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    FlyBrainGNNLoRAController,
    save_lora_adapter,
)

# Trace dirs used by default — every manual_graph trace stored
# under ``data/experiments/bench_round*/manual_graph/`` qualifies as
# expert demonstration, regardless of the round it came from.
_DEFAULT_TRACE_DIRS: tuple[Path, ...] = (
    Path("data/experiments/bench_round7_watchdog/manual_graph"),
    Path("data/experiments/bench_round8_pertasktype/manual_graph"),
    Path("data/experiments/bench_round9_autotuned/manual_graph"),
    Path("data/experiments/bench_round10_prior_ablation/manual_graph"),
    Path("data/experiments/bench_round11_priors_watchdog/manual_graph"),
    Path("data/experiments/bench_round13_paid_yandex/manual_graph"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("data/checkpoints/sim_pretrain_gnn_v6.pt"),
    )
    parser.add_argument(
        "--trace-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Manual_graph trace directory; can be passed multiple times. "
            "Default: rounds 7+8+9+10+11+13 manual_graph dirs."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/checkpoints/lora_adapter_round12.pt"),
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--alpha", type=float, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-passed", action="store_true", default=True)
    args = parser.parse_args()

    trace_dirs = args.trace_dir or list(_DEFAULT_TRACE_DIRS)

    client = MockEmbeddingClient(output_dim=32)
    agent_emb = AgentEmbedder(client)
    asyncio.run(agent_emb.precompute(MINIMAL_15))
    builder = ControllerStateBuilder(
        task=TaskEmbedder(client),
        agents=agent_emb,
        trace=TraceEmbedder(client),
        fly=FlyGraphEmbedder(dim=8),
        agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
    )

    ctrl = FlyBrainGNNLoRAController(
        builder=builder,
        task_dim=32,
        agent_dim=32,
        graph_dim=32,
        trace_dim=32 + 13,
        fly_dim=8,
        produced_dim=6,
        hidden_dim=32,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
    )

    if args.base_checkpoint.exists():
        blob = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = ctrl.load_state_dict(sd, strict=False)
        print(
            f"[round12] loaded base checkpoint {args.base_checkpoint} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    else:
        print(
            f"[round12] WARNING: base checkpoint {args.base_checkpoint} not found, training from random init."
        )

    ctrl.freeze_base()
    trainable = ctrl.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in ctrl.parameters())
    print(
        f"[round12] LoRA rank={args.rank} alpha={args.alpha} dropout={args.dropout}; "
        f"trainable={n_trainable}/{n_total} ({n_trainable / n_total:.4%})"
    )

    examples = []
    agent_names = [a.name for a in MINIMAL_15]
    for d in trace_dirs:
        if not d.exists():
            print(f"[round12] skip missing trace dir {d}")
            continue
        before = len(examples)
        examples.extend(
            collect_examples(
                iter_traces(d),
                agent_names=agent_names,
                only_passed=args.only_passed,
            )
        )
        print(f"[round12] {d}: +{len(examples) - before} examples")

    if not examples:
        raise SystemExit("[round12] no training examples found across requested trace dirs")

    cfg = ImitationConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        only_passed=args.only_passed,
    )
    print(f"[round12] cfg={cfg} num_examples={len(examples)}")

    t0 = time.perf_counter()
    res = imitation_train(
        ctrl,
        traces_dir=Path("/dev/null"),  # unused — examples are passed directly
        agent_names=agent_names,
        config=cfg,
        examples=examples,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[round12] examples={res.num_examples} "
        f"train={res.num_train} eval={res.num_eval} "
        f"loss[first/last]={res.losses[0]:.3f}/{res.losses[-1]:.3f} "
        f"final_acc={res.final_accuracy:.3f} wall={elapsed:.1f}s"
    )

    save_lora_adapter(
        ctrl,
        args.output,
        metadata={
            "round": 12,
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "num_examples": res.num_examples,
            "num_train": res.num_train,
            "num_eval": res.num_eval,
            "final_accuracy": res.final_accuracy,
            "epoch_accuracy": list(res.epoch_accuracy),
            "wall_seconds": elapsed,
            "trace_dirs": [str(d) for d in trace_dirs if d.exists()],
            "base_checkpoint": str(args.base_checkpoint),
        },
    )
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "round": 12,
                "rank": args.rank,
                "alpha": args.alpha,
                "epochs": args.epochs,
                "lr": args.lr,
                "num_examples": res.num_examples,
                "num_train": res.num_train,
                "num_eval": res.num_eval,
                "final_accuracy": res.final_accuracy,
                "epoch_accuracy": list(res.epoch_accuracy),
                "wall_seconds": elapsed,
                "trace_dirs": [str(d) for d in trace_dirs if d.exists()],
                "base_checkpoint": str(args.base_checkpoint),
            },
            indent=2,
        )
    )
    print(f"[round12] saved adapter to {args.output} (+ sidecar {sidecar.name})")


if __name__ == "__main__":
    main()
