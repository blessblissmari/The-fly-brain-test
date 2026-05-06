#!/usr/bin/env python3
"""Materialise round-10 null-prior ``.fbg`` files from the real FlyWire prior.

Round 10 (``docs/round10_prior_ablation.md``) compares the trained
``flybrain_sim_pretrain`` controller against itself with three
substituted priors:

* **erdos_renyi**  — matches only ``num_nodes`` + edge count.
* **shuffled**     — Maslov-Sneppen double-edge swap (matches every
                     node's in/out-degree, breaks correlations).
* **reverse**      — adjacency transpose (preserves undirected
                     adjacency, breaks directionality).

This script reads the canonical ``data/flybrain/fly_graph_64.fbg``
prior produced by ``flybrain-py build`` and writes the three nulls
to ``data/flybrain/null_priors/{er,shuffled,reverse}_K64_seed{N}.fbg``
along with a ``provenance.json`` summarising the resulting
graphs.

Run::

    python scripts/build_null_priors.py
    python scripts/build_null_priors.py --seeds 0 1 2 --reference data/flybrain/fly_graph_64.fbg

Idempotent: skips files that already exist unless ``--force`` is
passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flybrain.graph import load, save
from flybrain.graph.null_priors import (
    degree_summary,
    erdos_renyi_prior,
    reverse_prior,
    shuffled_prior,
)


def _summary_for_disk(graph) -> dict[str, object]:
    """Compact, JSON-friendly summary of a FlyGraph."""
    deg = degree_summary(graph)
    deg.pop("in_degree", None)
    deg.pop("out_degree", None)
    deg["provenance"] = dict(graph.provenance)
    return deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/flybrain/fly_graph_64.fbg"),
        help="Real-FlyWire .fbg used as the source for shuffled / reverse priors "
        "and for the (num_nodes, num_edges) target of the ER prior.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/flybrain/null_priors"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for ER / shuffled priors (reverse is deterministic).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .fbg files.",
    )
    args = parser.parse_args()

    if not args.reference.exists():
        raise SystemExit(
            f"reference graph not found: {args.reference}. "
            "Run `flybrain-py build --source zenodo_csv ...` first."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reference = load(args.reference)
    print(
        f"[ref] {args.reference} num_nodes={reference.num_nodes} "
        f"num_edges={reference.num_edges} "
        f"provenance.source={reference.provenance.get('source')!r}"
    )

    summaries: list[dict[str, object]] = []
    summaries.append(
        {"name": "real_flywire_K64", "path": str(args.reference), **_summary_for_disk(reference)}
    )

    # 1. ER null priors — same K, same edge count, no biological topology.
    for seed in args.seeds:
        path = args.out_dir / f"er_K{reference.num_nodes}_seed{seed}.fbg"
        if path.exists() and not args.force:
            print(f"[er] {path} exists, skipping (use --force to overwrite)")
        else:
            er = erdos_renyi_prior(
                num_nodes=reference.num_nodes,
                num_edges=reference.num_edges,
                seed=seed,
                extra_provenance={"reference": str(args.reference)},
            )
            save(er, path)
            print(f"[er] seed={seed} num_edges={er.num_edges} -> {path}")
        summaries.append(
            {
                "name": f"er_K{reference.num_nodes}_seed{seed}",
                "path": str(path),
                **_summary_for_disk(load(path)),
            }
        )

    # 2. Shuffled (config-model) null priors — preserve degree distribution.
    for seed in args.seeds:
        path = args.out_dir / f"shuffled_K{reference.num_nodes}_seed{seed}.fbg"
        if path.exists() and not args.force:
            print(f"[shuffled] {path} exists, skipping (use --force to overwrite)")
        else:
            shuffled = shuffled_prior(
                reference,
                seed=seed,
                extra_provenance={"reference": str(args.reference)},
            )
            save(shuffled, path)
            ref_deg = degree_summary(reference)
            new_deg = degree_summary(shuffled)
            assert ref_deg["in_degree"] == new_deg["in_degree"], (
                "configuration-model swap must preserve in-degree per node"
            )
            assert ref_deg["out_degree"] == new_deg["out_degree"], (
                "configuration-model swap must preserve out-degree per node"
            )
            print(
                f"[shuffled] seed={seed} swaps_accepted="
                f"{shuffled.provenance.get('swaps_accepted')} "
                f"num_edges={shuffled.num_edges} -> {path}"
            )
        summaries.append(
            {
                "name": f"shuffled_K{reference.num_nodes}_seed{seed}",
                "path": str(path),
                **_summary_for_disk(load(path)),
            }
        )

    # 3. Reverse (transpose) null prior — deterministic single artefact.
    rev_path = args.out_dir / f"reverse_K{reference.num_nodes}.fbg"
    if rev_path.exists() and not args.force:
        print(f"[reverse] {rev_path} exists, skipping (use --force to overwrite)")
    else:
        rev = reverse_prior(
            reference,
            extra_provenance={"reference": str(args.reference)},
        )
        save(rev, rev_path)
        print(f"[reverse] num_edges={rev.num_edges} -> {rev_path}")
    summaries.append(
        {
            "name": f"reverse_K{reference.num_nodes}",
            "path": str(rev_path),
            **_summary_for_disk(load(rev_path)),
        }
    )

    # Provenance sidecar.
    provenance_path = args.out_dir / "provenance.json"
    provenance_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"[done] wrote {provenance_path} with {len(summaries)} entries")


if __name__ == "__main__":
    main()
