#!/usr/bin/env python3
"""Fill the round-9 success / calls / wall-clock tables from bench output.

Usage:
    python scripts/fill_round9_tables.py \
        --bench-dir data/experiments/bench_round9_autotuned \
        --doc       docs/round9_autotuned.md

Replaces the three placeholder blocks delimited by HTML comments
``<!-- ROUND9_*_TABLE_START -->`` / ``<!-- ROUND9_*_TABLE_END -->``
with markdown tables computed from the per-task trace.json files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BASELINE_ORDER = [
    "manual_graph",
    "flybrain_sim_pretrain",
    "flybrain_sim_pretrain_watchdog",
    "flybrain_sim_pretrain_watchdog_v2",
    "flybrain_sim_pretrain_watchdog_v3",
]
BASELINE_HEADERS = [
    "manual_graph",
    "sim_pretrain",
    "watchdog v1 (round-7)",
    "watchdog v2 (round-8)",
    "watchdog v3 (round-9)",
]
SHORT_HEADERS = [
    "manual_graph",
    "sim_pretrain",
    "watchdog v1",
    "watchdog v2",
    "watchdog v3",
]
BENCHMARKS = ["synthetic_routing", "humaneval"]


def _aggregate(bench_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-baseline x per-benchmark aggregates."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for baseline in BASELINE_ORDER:
        out[baseline] = {}
        for bench in BENCHMARKS:
            d = bench_dir / baseline / bench
            if not d.is_dir():
                out[baseline][bench] = {
                    "success": float("nan"),
                    "calls": float("nan"),
                    "wall": float("nan"),
                }
                continue
            traces = list(d.glob("*.trace.json"))
            if not traces:
                out[baseline][bench] = {
                    "success": float("nan"),
                    "calls": float("nan"),
                    "wall": float("nan"),
                }
                continue
            n_pass = 0
            calls = []
            wall = []
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
                if isinstance(totals.get("latency_ms"), int):
                    wall.append(totals["latency_ms"])
            n = len(traces)
            out[baseline][bench] = {
                "success": n_pass / n if n else float("nan"),
                "calls": (sum(calls) / len(calls)) if calls else float("nan"),
                "wall": (sum(wall) / len(wall) / 1000.0) if wall else float("nan"),
            }
    return out


def _fmt(v: float, kind: str) -> str:
    if v != v:  # NaN
        return "—"
    if kind == "success":
        return f"{v:.3f}"
    if kind == "calls":
        return f"{v:.2f}"
    if kind == "wall":
        return f"{v:.1f}s"
    return str(v)


def _row(bench: str, agg: dict[str, dict[str, dict[str, float]]], kind: str) -> str:
    cells = [bench]
    for b in BASELINE_ORDER:
        cells.append(_fmt(agg[b][bench][kind], kind))
    return "| " + " | ".join(cells) + " |"


def _build_table(headers: list[str], agg: dict[str, dict[str, dict[str, float]]], kind: str) -> str:
    head = "| benchmark | " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] + ["---:"] * len(headers)) + "|"
    rows = [_row(b, agg, kind) for b in BENCHMARKS]
    return "\n".join([head, sep, *rows])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", required=True, type=Path)
    p.add_argument("--doc", required=True, type=Path)
    args = p.parse_args()

    agg = _aggregate(args.bench_dir)
    text = args.doc.read_text()

    blocks = {
        "ROUND9_SUCCESS_TABLE": _build_table(BASELINE_HEADERS, agg, "success"),
        "ROUND9_CALLS_TABLE": _build_table(SHORT_HEADERS, agg, "calls"),
        "ROUND9_WALL_TABLE": _build_table(SHORT_HEADERS, agg, "wall"),
    }
    for tag, body in blocks.items():
        pat = re.compile(
            r"<!-- " + tag + r"_START -->.*?<!-- " + tag + r"_END -->",
            re.DOTALL,
        )
        repl = f"<!-- {tag}_START -->\n\n{body}\n\n<!-- {tag}_END -->"
        text, n = pat.subn(repl, text)
        if not n:
            raise RuntimeError(f"no {tag} block found in {args.doc}")
    args.doc.write_text(text)
    print(f"updated {args.doc}")


if __name__ == "__main__":
    main()
