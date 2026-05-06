#!/usr/bin/env python3
"""Fill the round-12 results / costs tables in
``docs/round12_lora_adapter.md`` from a completed bench output.

Usage::

    python scripts/fill_round12_tables.py \
        --bench-dir data/experiments/bench_round12_lora_adapter \
        --doc       docs/round12_lora_adapter.md

Replaces the two HTML-comment-delimited blocks
``<!-- BEGIN_RESULTS_TABLE -->`` and ``<!-- BEGIN_COSTS_TABLE -->``
with markdown tables computed from the per-task ``trace.json`` files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASELINE_ORDER = [
    "manual_graph",
    "flybrain_sim_pretrain",
    "flybrain_sim_pretrain_watchdog_v3",
    "flybrain_sim_pretrain_lora",
    "flybrain_sim_pretrain_lora_watchdog_v3",
]
BASELINE_HEADERS = [
    "manual_graph",
    "raw GNN",
    "watchdog v3",
    "lora",
    "lora + wd v3",
]
BENCHMARKS = ["synthetic_routing", "humaneval", "gsm8k", "bbh_mini"]


def _aggregate(bench_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for baseline in BASELINE_ORDER:
        out[baseline] = {}
        for bench in BENCHMARKS:
            d = bench_dir / baseline / bench
            if not d.is_dir():
                out[baseline][bench] = {
                    "success": float("nan"),
                    "calls": float("nan"),
                }
                continue
            traces = list(d.glob("*.trace.json"))
            if not traces:
                out[baseline][bench] = {
                    "success": float("nan"),
                    "calls": float("nan"),
                }
                continue
            n_pass = 0
            calls: list[int] = []
            for t in traces:
                try:
                    j = json.loads(t.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if j.get("verification", {}).get("passed"):
                    n_pass += 1
                totals = j.get("totals") or {}
                if isinstance(totals.get("llm_calls"), int):
                    calls.append(totals["llm_calls"])
            n = len(traces)
            out[baseline][bench] = {
                "success": n_pass / n if n else float("nan"),
                "calls": (sum(calls) / len(calls)) if calls else float("nan"),
            }
    return out


def _fmt(v: float, kind: str) -> str:
    if v != v:
        return "—"
    if kind == "success":
        return f"{v:.3f}"
    if kind == "calls":
        return f"{v:.2f}"
    return str(v)


def _row(bench: str, agg: dict[str, dict[str, dict[str, float]]], kind: str) -> str:
    cells = [bench]
    for b in BASELINE_ORDER:
        cells.append(_fmt(agg[b][bench][kind], kind))
    return "| " + " | ".join(cells) + " |"


def _overall_row(agg: dict[str, dict[str, dict[str, float]]], kind: str) -> str:
    cells = ["**overall**"]
    for b in BASELINE_ORDER:
        vals = [agg[b][bench][kind] for bench in BENCHMARKS]
        vals = [v for v in vals if v == v]  # drop NaN
        mean = sum(vals) / len(vals) if vals else float("nan")
        cells.append("**" + _fmt(mean, kind) + "**")
    return "| " + " | ".join(cells) + " |"


def _build_table(agg: dict[str, dict[str, dict[str, float]]], kind: str) -> str:
    head = "| benchmark | " + " | ".join(BASELINE_HEADERS) + " |"
    sep = "|" + "|".join(["---"] + ["---:"] * len(BASELINE_HEADERS)) + "|"
    rows = [_row(b, agg, kind) for b in BENCHMARKS]
    rows.append(_overall_row(agg, kind))
    return "\n".join([head, sep, *rows])


def _replace_block(text: str, begin: str, end: str, body: str) -> str:
    pat = re.compile(
        r"<!-- " + re.escape(begin) + r" -->.*?<!-- " + re.escape(end) + r" -->",
        re.DOTALL,
    )
    repl = f"<!-- {begin} -->\n\n{body}\n\n<!-- {end} -->"
    new, n = pat.subn(repl, text)
    if n != 1:
        raise RuntimeError(f"{begin}/{end} block not found exactly once in doc")
    return new


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", required=True, type=Path)
    p.add_argument("--doc", required=True, type=Path)
    args = p.parse_args()

    agg = _aggregate(args.bench_dir)
    text = args.doc.read_text()

    text = _replace_block(
        text,
        "BEGIN_RESULTS_TABLE",
        "END_RESULTS_TABLE",
        _build_table(agg, "success"),
    )
    text = _replace_block(
        text,
        "BEGIN_COSTS_TABLE",
        "END_COSTS_TABLE",
        _build_table(agg, "calls"),
    )
    args.doc.write_text(text)
    print(f"updated {args.doc}")


if __name__ == "__main__":
    main()
