#!/usr/bin/env python3
"""Statistical analysis of round-10 connectome-prior ablation results.

Reads the per-trace JSON files emitted by ``flybrain-py bench
--suite round10_prior_ablation`` and produces:

* A success-rate table by (baseline, benchmark) with 95 %
  bootstrap CI per cell.
* Pre-registered hypothesis tests:
  - Wilcoxon signed-rank test between ``flybrain_sim_pretrain``
    and each null-prior baseline, paired by task (Bonferroni-
    corrected for the 3 comparisons).
  - One-sided directional test for H1 (biology >> null).
* A markdown summary that can be pasted into
  ``docs/round10_prior_ablation.md`` §4.

Run::

    python scripts/analyse_round10.py
    python scripts/analyse_round10.py \\
        --bench-dir data/experiments/bench_round10_prior_ablation \\
        --output    data/experiments/bench_round10_prior_ablation/round10_summary.md

Pure CPU, deterministic given a fixed bench dir + ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROUND10_BASELINES = [
    "manual_graph",
    "flybrain_sim_pretrain",
    "er_prior_sim_pretrain",
    "shuffled_fly_sim_pretrain",
    "reverse_fly_sim_pretrain",
]
NULL_PRIORS = [
    "er_prior_sim_pretrain",
    "shuffled_fly_sim_pretrain",
    "reverse_fly_sim_pretrain",
]


def load_traces(bench_dir: Path) -> dict[str, dict[str, dict[str, bool]]]:
    """``{baseline: {benchmark: {task_id: success_bool}}}``."""
    out: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    for baseline_dir in sorted(bench_dir.iterdir()):
        if not baseline_dir.is_dir():
            continue
        baseline = baseline_dir.name
        if baseline not in ROUND10_BASELINES:
            continue
        for bench_subdir in sorted(baseline_dir.iterdir()):
            if not bench_subdir.is_dir():
                continue
            benchmark = bench_subdir.name
            for trace_file in sorted(bench_subdir.glob("*.trace.json")):
                with trace_file.open() as f:
                    trace = json.load(f)
                # The trace stem is unique per (benchmark, task). Verifier
                # outcome lives at ``trace['verification']['passed']`` in the
                # schema produced by
                # ``flybrain.benchmarks.run.run_baseline_on_benchmark``.
                task_id = trace_file.stem.removesuffix(".trace")
                verification = trace.get("verification") or {}
                passed = bool(verification.get("passed", False))
                out[baseline][benchmark][task_id] = passed
    return out


def bootstrap_ci(
    values: list[bool], *, num_resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95 % bootstrap CI of a binary success vector."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(num_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(values) / n
    lo = means[int(0.025 * num_resamples)]
    hi = means[int(0.975 * num_resamples)]
    return mean, lo, hi


def wilcoxon_signed_rank_paired(a: list[bool], b: list[bool]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired binary vectors.

    Returns (statistic_W+, two-sided p-value) using the normal
    approximation with continuity correction. Paired observations
    where both samples agree are dropped (Wilcoxon convention).
    """
    if len(a) != len(b):
        raise ValueError("paired vectors must be the same length")
    diffs = [int(x) - int(y) for x, y in zip(a, b, strict=True) if x != y]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0
    # All non-zero, all magnitude 1 → ranks 1..n with ties on the same rank.
    # Mean rank = (n + 1) / 2; W+ counts the positive-difference pairs.
    pos = sum(1 for d in diffs if d > 0)
    w_plus = pos * (n + 1) / 2
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    if var == 0:
        return w_plus, 1.0
    z = (w_plus - mean - 0.5 * (1 if w_plus > mean else -1)) / math.sqrt(var)
    # Two-sided normal-approximation p-value.
    p = 2 * (1 - _phi(abs(z)))
    return w_plus, max(0.0, min(1.0, p))


def _phi(z: float) -> float:
    """Standard-normal CDF using erf."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def render_table(
    traces: dict[str, dict[str, dict[str, bool]]],
) -> tuple[str, dict[str, dict[str, tuple[float, float, float]]]]:
    """Render the per-(baseline, benchmark) success-rate table."""
    benchmarks = sorted({b for d in traces.values() for b in d})
    cell: dict[str, dict[str, tuple[float, float, float]]] = {}
    for baseline in ROUND10_BASELINES:
        cell[baseline] = {}
        for benchmark in benchmarks:
            successes = [int(s) for s in traces.get(baseline, {}).get(benchmark, {}).values()]
            mean, lo, hi = bootstrap_ci(
                [bool(s) for s in successes], seed=hash((baseline, benchmark)) & 0xFFFF
            )
            cell[baseline][benchmark] = (mean, lo, hi)
        # overall row across all benchmarks
        all_successes = [
            int(s)
            for benchmark in benchmarks
            for s in traces.get(baseline, {}).get(benchmark, {}).values()
        ]
        mean, lo, hi = bootstrap_ci(
            [bool(s) for s in all_successes],
            seed=hash((baseline, "_overall")) & 0xFFFF,
        )
        cell[baseline]["_overall"] = (mean, lo, hi)

    columns = [*benchmarks, "_overall"]
    header = "| Baseline | " + " | ".join(columns) + " |"
    sep = "|---" + "|---:" * len(columns) + "|"
    lines = [header, sep]
    for baseline in ROUND10_BASELINES:
        row_cells = []
        for benchmark in columns:
            mean, lo, hi = cell[baseline][benchmark]
            row_cells.append(f"{mean:.3f} ({lo:.2f}-{hi:.2f})")
        lines.append(f"| `{baseline}` | " + " | ".join(row_cells) + " |")
    return "\n".join(lines), cell


def render_pre_reg_tests(
    traces: dict[str, dict[str, dict[str, bool]]],
) -> str:
    """Run the pre-registered Wilcoxon paired tests."""
    fly = traces.get("flybrain_sim_pretrain", {})
    fly_keys = sorted((b, t) for b in fly for t in fly[b])
    fly_vec = [fly[b][t] for (b, t) in fly_keys]
    if not fly_vec:
        return "_(no flybrain traces — skipping hypothesis tests)_"
    lines = [
        "| Comparison | n_pairs | mean(flybrain) | mean(null) | "
        "diff | W+ | p (two-sided, uncorrected) | p (Bonf-3) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for null_name in NULL_PRIORS:
        null = traces.get(null_name, {})
        a, b = [], []
        for key in fly_keys:
            bm, t = key
            if bm in null and t in null[bm]:
                a.append(fly[bm][t])
                b.append(null[bm][t])
        if not a:
            lines.append(f"| flybrain vs {null_name} | 0 | - | - | - | - | - | - |")
            continue
        w, p = wilcoxon_signed_rank_paired(a, b)
        p_bonf = min(1.0, p * 3)
        mean_fly = sum(int(x) for x in a) / len(a)
        mean_null = sum(int(x) for x in b) / len(b)
        lines.append(
            f"| flybrain vs `{null_name}` | {len(a)} | {mean_fly:.3f} | "
            f"{mean_null:.3f} | {mean_fly - mean_null:+.3f} | {w:.1f} | "
            f"{p:.3f} | {p_bonf:.3f} |"
        )
    return "\n".join(lines)


def render_hypothesis_verdict(
    cell: dict[str, dict[str, tuple[float, float, float]]],
) -> str:
    """Pre-registered verdict block for round-10 doc §4."""
    fly = cell.get("flybrain_sim_pretrain", {}).get("_overall", (0.0, 0.0, 0.0))
    er = cell.get("er_prior_sim_pretrain", {}).get("_overall", (0.0, 0.0, 0.0))
    sh = cell.get("shuffled_fly_sim_pretrain", {}).get("_overall", (0.0, 0.0, 0.0))
    rev = cell.get("reverse_fly_sim_pretrain", {}).get("_overall", (0.0, 0.0, 0.0))

    fly_m, sh_m, rev_m, er_m = fly[0], sh[0], rev[0], er[0]
    eps = 0.05  # treat |Δ| < 5 pp as "≈"

    def cmp(a: float, b: float) -> str:
        if a > b + eps:
            return ">"
        if a < b - eps:
            return "<"
        return "≈"

    pattern = (
        f"flybrain ({fly_m:.3f}) {cmp(fly_m, sh_m)} "
        f"shuffled ({sh_m:.3f}) {cmp(sh_m, rev_m)} "
        f"reverse ({rev_m:.3f}) {cmp(rev_m, er_m)} "
        f"er ({er_m:.3f})"
    )

    if cmp(fly_m, sh_m) == ">" and cmp(sh_m, er_m) == ">":
        verdict = "**H1 supported (biology matters strictly):**"
    elif cmp(fly_m, sh_m) == "≈" and cmp(sh_m, er_m) == ">":
        verdict = "**H0 supported (degree distribution sufficient):**"
    elif cmp(fly_m, er_m) == "≈" and cmp(sh_m, er_m) == "≈" and cmp(rev_m, er_m) == "≈":
        verdict = "**H-null supported (controller insensitive to prior):**"
    else:
        verdict = "**Mixed / inconclusive:**"

    direction = "**direction matters**" if cmp(fly_m, rev_m) == ">" else "direction immaterial"

    return (
        f"{verdict}\n\n"
        f"Observed pattern: {pattern}.\n\n"
        f"Directionality finding: {direction} "
        f"(flybrain {fly_m:.3f} vs reverse {rev_m:.3f}; Δ={fly_m - rev_m:+.3f})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("data/experiments/bench_round10_prior_ablation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the markdown summary to this file in addition to stdout.",
    )
    args = parser.parse_args()

    if not args.bench_dir.exists():
        raise SystemExit(f"bench dir not found: {args.bench_dir}")

    traces = load_traces(args.bench_dir)
    if not traces:
        raise SystemExit(f"no traces found under {args.bench_dir}; did the bench finish?")

    table, cell = render_table(traces)
    tests = render_pre_reg_tests(traces)
    verdict = render_hypothesis_verdict(cell)

    summary = (
        "# Round 10 — connectome-prior ablation analysis\n\n"
        f"Source: `{args.bench_dir}` "
        f"(N_total={sum(len(b) for d in traces.values() for b in d.values())} "
        f"task-runs across {len(traces)} baselines).\n\n"
        "## Success-rate table (mean, 95 % bootstrap CI)\n\n"
        f"{table}\n\n"
        "## Pre-registered Wilcoxon paired tests\n\n"
        f"{tests}\n\n"
        "## Verdict\n\n"
        f"{verdict}\n"
    )
    print(summary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary)
        print(f"\n[wrote] {args.output}", flush=True)


if __name__ == "__main__":
    main()
