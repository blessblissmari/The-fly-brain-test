"""Unit tests for round-9 :class:`WatchdogCalibration`.

Covers the calibration math (percentile rounding, floor, fallback for
small samples) and the wiring through
``FinalizerWatchdogController.from_bench_dirs``. See
``flybrain/controller/watchdog_calibrator.py`` and round-9 §3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flybrain.controller.watchdog_calibrator import (
    DEFAULT_FALLBACK_FORCE,
    DEFAULT_FALLBACK_STALL,
    DEFAULT_FORCE_FLOOR,
    DEFAULT_PERCENTILE,
    WatchdogCalibration,
    _percentile_int,
)


def _write_trace(
    root: Path,
    baseline: str,
    benchmark: str,
    task_id: str,
    *,
    task_type: str,
    llm_calls: int,
    passed: bool = True,
) -> None:
    """Write a minimal trace.json that the calibrator can parse."""
    target = root / baseline / benchmark
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "totals": {"llm_calls": llm_calls},
        "verification": {"passed": passed},
        "steps": [],
        "metadata": {},
        "final_answer": "",
    }
    (target / f"{task_id}.trace.json").write_text(json.dumps(payload))


class TestPercentileInt:
    def test_single_sample_returns_self(self) -> None:
        assert _percentile_int([7], 0.9) == 7

    def test_p90_of_uniform_run(self) -> None:
        # 11 evenly spaced values [0..10]: P90 = 9.0 exactly.
        samples = list(range(11))
        assert _percentile_int(samples, 0.9) == 9

    def test_rounds_up(self) -> None:
        # Two values 10 and 30: P90 = 28.0 exactly. Slightly off
        # (e.g. P89) would give a fractional value that must round up.
        assert _percentile_int([10, 30], 0.9) == 28
        assert _percentile_int([10, 30], 0.89) == 28  # ceil(27.8)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            _percentile_int([], 0.9)


class TestWatchdogCalibrationFromBenchDirs:
    def test_calibrates_per_task_type_p90(self, tmp_path: Path) -> None:
        # coding: 11 samples, P90 = 26
        for i, n in enumerate([5, 8, 10, 12, 14, 18, 20, 22, 24, 26, 28]):
            _write_trace(
                tmp_path,
                "manual_graph",
                "humaneval",
                f"task_{i:03d}",
                task_type="coding",
                llm_calls=n,
            )
        cal = WatchdogCalibration.from_bench_dirs([tmp_path])
        assert cal.percentile == DEFAULT_PERCENTILE
        assert cal.n_samples_per_task == {"coding": 11}
        # P90 of [5..28 step 2] = 26 (rank 9.0 → idx 9 = 26).
        assert cal.force_after["coding"] == 26
        # stall_ratio=0.25 → ceil(26 * 0.25) = 7.
        assert cal.stall_after["coding"] == 7

    def test_skips_failed_traces(self, tmp_path: Path) -> None:
        # 5 passed traces with calls=10..30 + 5 failed traces
        # with calls=100 each — failed must NOT inflate the budget.
        for i, n in enumerate([10, 15, 20, 25, 30]):
            _write_trace(
                tmp_path,
                "manual_graph",
                "humaneval",
                f"pass_{i}",
                task_type="coding",
                llm_calls=n,
                passed=True,
            )
        for i in range(5):
            _write_trace(
                tmp_path,
                "manual_graph",
                "humaneval",
                f"fail_{i}",
                task_type="coding",
                llm_calls=100,
                passed=False,
            )
        cal = WatchdogCalibration.from_bench_dirs([tmp_path])
        assert cal.n_samples_per_task == {"coding": 5}
        # P90 of [10,15,20,25,30] = 28 (interp + ceil).
        assert cal.force_after["coding"] == 28

    def test_small_sample_falls_back(self, tmp_path: Path) -> None:
        # 2 samples for tool_use (< default min_samples=3) — must fall
        # back to fallback_force=12 instead of P90=3.
        for i, n in enumerate([3, 3]):
            _write_trace(
                tmp_path,
                "manual_graph",
                "synthetic_routing",
                f"tt_{i}",
                task_type="tool_use",
                llm_calls=n,
            )
        cal = WatchdogCalibration.from_bench_dirs([tmp_path])
        assert cal.n_samples_per_task == {"tool_use": 2}
        assert cal.force_after["tool_use"] == DEFAULT_FALLBACK_FORCE
        assert cal.stall_after["tool_use"] == DEFAULT_FALLBACK_STALL

    def test_floor_clamps_low_calibration(self, tmp_path: Path) -> None:
        # 5 successful samples, all with very few llm_calls. Calibrator
        # would return P90=3 but the floor must clamp to 8.
        for i in range(5):
            _write_trace(
                tmp_path,
                "manual_graph",
                "synthetic_routing",
                f"x_{i}",
                task_type="tool_use",
                llm_calls=3,
            )
        cal = WatchdogCalibration.from_bench_dirs([tmp_path])
        assert cal.force_after["tool_use"] == DEFAULT_FORCE_FLOOR
        # stall_ratio=0.25, ceil(8 * 0.25) = 2.
        assert cal.stall_after["tool_use"] == 2

    def test_multiple_dirs_aggregate(self, tmp_path: Path) -> None:
        # Two bench dirs, each with 3 samples for coding — should
        # aggregate to n=6.
        a = tmp_path / "round7"
        b = tmp_path / "round8"
        for i, n in enumerate([10, 15, 20]):
            _write_trace(a, "manual_graph", "humaneval", f"a_{i}", task_type="coding", llm_calls=n)
        for i, n in enumerate([25, 28, 30]):
            _write_trace(b, "manual_graph", "humaneval", f"b_{i}", task_type="coding", llm_calls=n)
        cal = WatchdogCalibration.from_bench_dirs([a, b])
        assert cal.n_samples_per_task == {"coding": 6}

    def test_missing_dir_silent(self, tmp_path: Path) -> None:
        # Calibrator must not crash on missing dirs; should produce
        # an empty mapping so the watchdog can fall back.
        cal = WatchdogCalibration.from_bench_dirs([tmp_path / "does_not_exist"])
        assert cal.force_after == {}
        assert cal.stall_after == {}
        assert cal.n_samples_per_task == {}

    def test_invalid_percentile_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            WatchdogCalibration.from_bench_dirs([tmp_path], percentile=1.5)
        with pytest.raises(ValueError):
            WatchdogCalibration.from_bench_dirs([tmp_path], percentile=0.0)


class TestFinalizerWatchdogFromBenchDirs:
    def test_factory_wires_calibration_into_watchdog(self, tmp_path: Path) -> None:
        from flybrain.controller.finalizer_watchdog import FinalizerWatchdogController

        # Provide enough samples to calibrate coding.
        for i, n in enumerate([20, 22, 24, 26, 28]):
            _write_trace(
                tmp_path,
                "manual_graph",
                "humaneval",
                f"c_{i}",
                task_type="coding",
                llm_calls=n,
            )

        class _Stub:
            def select_action(self, state):  # pragma: no cover - never used
                return {"kind": "wait"}

        wd = FinalizerWatchdogController.from_bench_dirs(
            _Stub(),  # type: ignore[arg-type]
            [str(tmp_path)],
        )
        assert isinstance(wd.force_after, dict)
        assert "coding" in wd.force_after
        # P90 of [20,22,24,26,28] = ceil(27.2) = 28 → above floor.
        assert wd.force_after["coding"] == 28
        assert isinstance(wd.stall_after, dict)
        assert wd.stall_after["coding"] == 7  # ceil(28 * 0.25)

    def test_factory_uses_int_fallback_when_no_data(self, tmp_path: Path) -> None:
        from flybrain.controller.finalizer_watchdog import FinalizerWatchdogController

        class _Stub:
            def select_action(self, state):  # pragma: no cover - never used
                return {"kind": "wait"}

        wd = FinalizerWatchdogController.from_bench_dirs(
            _Stub(),  # type: ignore[arg-type]
            [str(tmp_path / "nonexistent")],
            fallback_force=11,
            fallback_stall=2,
        )
        # Empty calibration → factory falls back to int defaults so the
        # watchdog stays usable in fresh CI environments.
        assert wd.force_after == 11
        assert wd.stall_after == 2
