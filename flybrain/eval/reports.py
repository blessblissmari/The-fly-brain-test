"""Final-report builder used by ``flybrain-py report`` (PLAN.md §615).

Stitches the per-benchmark comparison tables and a couple of cherry-
picked traces into a single Markdown document the operator can drop
into ``docs/final_report.md``.

The intent is to have one command produce a report skeleton with all
the numbers filled in — humans still write the discussion sections,
but the metric-driven tables and links never go stale.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from flybrain.eval.metrics import AggregateMetrics
from flybrain.eval.tables import markdown_table

_REPORT_TEMPLATE = """# FlyBrain Optimizer — research report

## 1. Setup

* Suite: {suite_name}
* Methods compared: {num_methods}
* Benchmarks: {benchmark_list}
* Total tasks evaluated: {total_tasks}

## 2. Headline comparison (across all benchmarks)

{overall_table}

## 3. Per-benchmark breakdown

{per_benchmark_section}

## 4. Cherry-picked traces

{trace_section}

## 5. Discussion (TODO)

* Where does FlyBrain prior + RL beat manual / fully-connected?
* Cost / quality trade-offs?
* Failure modes (`failed_component` from the verifier)?
* Bridging gaps to the README §17 hypothesis.
"""


@dataclass(slots=True)
class ReportInputs:
    """Everything needed to render the final report."""

    suite_name: str
    overall: list[AggregateMetrics]
    """One row per method, aggregated across all benchmarks."""

    per_benchmark: dict[str, list[AggregateMetrics]]
    """benchmark id → one row per method on that benchmark."""

    trace_paths: list[Path]
    """Cherry-picked trace files (max 3 by convention)."""


def _per_benchmark_section(per_benchmark: dict[str, list[AggregateMetrics]]) -> str:
    if not per_benchmark:
        return "_(no per-benchmark breakdown available)_\n"
    parts: list[str] = []
    for name in sorted(per_benchmark):
        parts.append(f"### {name}\n\n{markdown_table(per_benchmark[name])}\n")
    return "\n".join(parts)


def _trace_section(trace_paths: Iterable[Path]) -> str:
    paths = list(trace_paths)
    if not paths:
        return "_(no cherry-picked traces selected)_\n"
    return "\n".join(f"* `{p}`" for p in paths) + "\n"


def render_report(inputs: ReportInputs) -> str:
    """Render `inputs` to a Markdown report string."""
    benchmarks = sorted(inputs.per_benchmark)
    total_tasks = sum(r.num_tasks for r in inputs.overall)
    return _REPORT_TEMPLATE.format(
        suite_name=inputs.suite_name,
        num_methods=len({r.name for r in inputs.overall}),
        benchmark_list=", ".join(benchmarks) if benchmarks else "—",
        total_tasks=total_tasks,
        overall_table=markdown_table(inputs.overall),
        per_benchmark_section=_per_benchmark_section(inputs.per_benchmark),
        trace_section=_trace_section(inputs.trace_paths),
    )


def write_report(inputs: ReportInputs, path: Path) -> Path:
    """Render `inputs` and write the result to ``path`` (UTF-8)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(inputs), encoding="utf-8")
    return path


__all__ = [
    "ReportInputs",
    "render_report",
    "write_report",
]
