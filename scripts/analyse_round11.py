#!/usr/bin/env python3
"""Statistical analysis of round-11 null-prior × watchdog v2 results.

Round-11 wires the round-8/9 ``FinalizerWatchdogController`` v2
(per-task-type ``force_after``) over each round-10 null-prior
(``er_prior``, ``shuffled_fly``, ``reverse_fly``) plus the canonical
real-fly prior, all evaluated in the same bench process.

Round-10 found the *raw* GNN is insensitive to its prior at
inference. Round-11 answers the natural follow-up: does the
watchdog v2 wrapper rescue **only** the biological prior, or does
it rescue **any** prior equally?

Pre-registered hypotheses
-------------------------

* **H1 (biology matters even with watchdog):**
  ``real_fly+wd2 > er_prior+wd2 ≈ shuffled+wd2 ≈ reverse+wd2``
  — biology adds value beyond what the scaffold provides; the
  prior is necessary for the controller to learn task-specific
  routing patterns the watchdog cannot supply.

* **H0 (scaffold is the value, prior is dispensable):**
  ``real_fly+wd2 ≈ er_prior+wd2 ≈ shuffled+wd2 ≈ reverse+wd2``
  — the watchdog is the universal rescuer; biology was a detour
  on the way to the right scaffold.

* **H-direction (direction still matters):**
  ``real_fly+wd2 > reverse_fly+wd2``
  — biological connectome direction encodes information beyond
  what an undirected adjacency can supply, even when the
  watchdog provides termination scaffolding.

Reads the per-trace JSON files emitted by ``flybrain-py bench
--suite round11_priors_with_watchdog`` and produces:

* A success-rate table by (baseline, benchmark) with 95 %
  bootstrap CI per cell.
* Pre-registered Wilcoxon paired tests between
  ``flybrain_sim_pretrain_watchdog_v2`` and each null-prior+wd2
  baseline (Bonferroni-corrected for the 3 comparisons).
* A markdown summary that can be pasted into
  ``docs/round11_prior_with_watchdog.md`` §4.

Run::

    python scripts/analyse_round11.py
    python scripts/analyse_round11.py \\
        --bench-dir data/experiments/bench_round11_priors_watchdog \\
        --output    data/experiments/bench_round11_priors_watchdog/round11_summary.md

Pure CPU, deterministic given a fixed bench dir.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROUND11_BASELINES = [
    "manual_graph",
    "flybrain_sim_pretrain_watchdog_v2",
    "er_prior_watchdog_v2",
    "shuffled_fly_watchdog_v2",
    "reverse_fly_watchdog_v2",
]
NULL_PRIORS_WD = [
    "er_prior_watchdog_v2",
    "shuffled_fly_watchdog_v2",
    "reverse_fly_watchdog_v2",
]
REAL_FLY_WD = "flybrain_sim_pretrain_watchdog_v2"


def load_traces(bench_dir: Path) -> dict[str, dict[str, dict[str, bool]]]:
    """``{baseline: {benchmark: {task_id: success_bool}}}``."""
    out: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    for baseline_dir in sorted(bench_dir.iterdir()):
        if not baseline_dir.is_dir():
            continue
        baseline = baseline_dir.name
        if baseline not in ROUND11_BASELINES:
            continue
        for bench_subdir in sorted(baseline_dir.iterdir()):
            if not bench_subdir.is_dir():
                continue
            benchmark = bench_subdir.name
            for trace_file in sorted(bench_subdir.glob("*.trace.json")):
                with trace_file.open() as f:
                    trace = json.load(f)
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
    pos = sum(1 for d in diffs if d > 0)
    w_plus = pos * (n + 1) / 2
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    if var == 0:
        return w_plus, 1.0
    z = (w_plus - mean - 0.5 * (1 if w_plus > mean else -1)) / math.sqrt(var)
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
    for baseline in ROUND11_BASELINES:
        cell[baseline] = {}
        for benchmark in benchmarks:
            successes = [int(s) for s in traces.get(baseline, {}).get(benchmark, {}).values()]
            mean, lo, hi = bootstrap_ci(
                [bool(s) for s in successes], seed=hash((baseline, benchmark)) & 0xFFFF
            )
            cell[baseline][benchmark] = (mean, lo, hi)
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
    for baseline in ROUND11_BASELINES:
        row_cells = []
        for benchmark in columns:
            mean, lo, hi = cell[baseline][benchmark]
            row_cells.append(f"{mean:.3f} ({lo:.2f}-{hi:.2f})")
        lines.append(f"| `{baseline}` | " + " | ".join(row_cells) + " |")
    return "\n".join(lines), cell


def render_pre_reg_tests(
    traces: dict[str, dict[str, dict[str, bool]]],
) -> str:
    """Wilcoxon paired tests: real-fly+wd2 vs each null-prior+wd2."""
    real = traces.get(REAL_FLY_WD, {})
    real_keys = sorted((b, t) for b in real for t in real[b])
    real_vec = [real[b][t] for (b, t) in real_keys]
    if not real_vec:
        return f"_(no {REAL_FLY_WD} traces — skipping hypothesis tests)_"
    lines = [
        "| Comparison | n_pairs | mean(real_fly+wd2) | mean(null+wd2) | "
        "diff | W+ | p (two-sided, uncorrected) | p (Bonf-3) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for null_name in NULL_PRIORS_WD:
        null = traces.get(null_name, {})
        a, b = [], []
        for key in real_keys:
            bm, t = key
            if bm in null and t in null[bm]:
                a.append(real[bm][t])
                b.append(null[bm][t])
        if not a:
            lines.append(f"| {REAL_FLY_WD} vs {null_name} | 0 | - | - | - | - | - | - |")
            continue
        w, p = wilcoxon_signed_rank_paired(a, b)
        p_bonf = min(1.0, p * 3)
        mean_real = sum(int(x) for x in a) / len(a)
        mean_null = sum(int(x) for x in b) / len(b)
        lines.append(
            f"| `{REAL_FLY_WD}` vs `{null_name}` | {len(a)} | {mean_real:.3f} | "
            f"{mean_null:.3f} | {mean_real - mean_null:+.3f} | {w:.1f} | "
            f"{p:.3f} | {p_bonf:.3f} |"
        )
    return "\n".join(lines)


def render_hypothesis_verdict(
    cell: dict[str, dict[str, tuple[float, float, float]]],
) -> str:
    """Pre-registered verdict block for round-11 doc §4."""
    real = cell.get(REAL_FLY_WD, {}).get("_overall", (0.0, 0.0, 0.0))
    er = cell.get("er_prior_watchdog_v2", {}).get("_overall", (0.0, 0.0, 0.0))
    sh = cell.get("shuffled_fly_watchdog_v2", {}).get("_overall", (0.0, 0.0, 0.0))
    rev = cell.get("reverse_fly_watchdog_v2", {}).get("_overall", (0.0, 0.0, 0.0))
    manual = cell.get("manual_graph", {}).get("_overall", (0.0, 0.0, 0.0))

    real_m, sh_m, rev_m, er_m, manual_m = real[0], sh[0], rev[0], er[0], manual[0]
    eps = 0.05  # treat |Δ| < 5 pp as "≈" given N=10 noise floor

    def cmp(a: float, b: float) -> str:
        if a > b + eps:
            return ">"
        if a < b - eps:
            return "<"
        return "≈"

    pattern = (
        f"manual ({manual_m:.3f}) | "
        f"real_fly+wd2 ({real_m:.3f}) {cmp(real_m, er_m)} "
        f"er+wd2 ({er_m:.3f}); "
        f"real_fly+wd2 ({real_m:.3f}) {cmp(real_m, sh_m)} "
        f"shuffled+wd2 ({sh_m:.3f}); "
        f"real_fly+wd2 ({real_m:.3f}) {cmp(real_m, rev_m)} "
        f"reverse+wd2 ({rev_m:.3f})"
    )

    null_means_close = (
        cmp(real_m, er_m) == "≈" and cmp(real_m, sh_m) == "≈" and cmp(real_m, rev_m) == "≈"
    )
    real_strictly_better = (
        cmp(real_m, er_m) == ">" and cmp(real_m, sh_m) == ">" and cmp(real_m, rev_m) == ">"
    )

    if real_strictly_better:
        verdict = (
            "**H1 supported (biology matters even with watchdog scaffold):** "
            "real-fly+wd2 strictly outperforms every null-prior+wd2 by "
            ">5 pp. The watchdog rescues termination but cannot supply "
            "task-specific routing patterns that biological topology "
            "encodes. Round-10's null result was a representational "
            "artefact of the raw GNN; with proper scaffolding biology "
            "**does** add value."
        )
    elif null_means_close:
        verdict = (
            "**H0 supported (scaffold is the value, prior is "
            "dispensable):** all four FlyBrain-architecture rows fall "
            "within a 5-pp band, so even with the watchdog v2 wrapper "
            "the controller does not exploit biological topology over "
            "ER / shuffled / reverse alternatives. The 105-LoC watchdog "
            "scaffold is the dominant value-driver; biology was a "
            "detour. Round-12 should explore learned adapters as the "
            "next quality lever."
        )
    else:
        verdict = (
            "**Mixed / partial support:** real-fly+wd2 separates from "
            "*some* but not *all* null priors. Either biology helps "
            "selectively (e.g. on humaneval but not synthetic_routing) "
            "or noise at N=10 obscures a small but real effect. "
            "Re-running at N=30 (round-13 paid) is the natural follow-up."
        )

    direction = (
        "**direction matters**"
        if cmp(real_m, rev_m) == ">"
        else "direction immaterial under watchdog scaffold"
    )

    return (
        f"{verdict}\n\n"
        f"Observed pattern: {pattern}.\n\n"
        f"Directionality finding: {direction} "
        f"(real_fly+wd2 {real_m:.3f} vs reverse+wd2 {rev_m:.3f}; "
        f"Δ={real_m - rev_m:+.3f})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("data/experiments/bench_round11_priors_watchdog"),
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
        "# Round 11 — null-prior x watchdog v2 cross-bench analysis\n\n"
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
