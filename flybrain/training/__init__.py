"""Training entrypoints — Phase 6 (sim pretrain), Phase 7 (IL),
Phase 8 (RL/bandit).

Phase 6 ships:

* `simulation_pretrain` — supervised pretrain on synthetic
  ``(state, optimal_action)`` pairs.

Phase 7 ships:

* `imitation_train` — supervised cloning from real expert traces
  collected with `scripts/collect_expert_traces.py`.
* `expert_dataset` helpers — load / replay trace JSONs and emit
  ``ImitationExample``\\s.
"""

from __future__ import annotations

from flybrain.training.expert_dataset import (
    ImitationExample,
    TraceFile,
    collect_examples,
    iter_traces,
    load_trace,
    trace_to_examples,
)
from flybrain.training.graph_ssl import (
    GraphSSLConfig,
    GraphSSLResult,
    apply_to_embedder,
    graph_ssl_pretrain,
    load_checkpoint,
    save_checkpoint,
)
from flybrain.training.imitation import (
    ImitationConfig,
    ImitationResult,
    imitation_train,
)
from flybrain.training.simulation_pretrain import (
    PretrainConfig,
    PretrainResult,
    expert_dataset,
    simulation_pretrain,
)

__all__ = [
    "GraphSSLConfig",
    "GraphSSLResult",
    "ImitationConfig",
    "ImitationExample",
    "ImitationResult",
    "PretrainConfig",
    "PretrainResult",
    "TraceFile",
    "apply_to_embedder",
    "collect_examples",
    "expert_dataset",
    "graph_ssl_pretrain",
    "imitation_train",
    "iter_traces",
    "load_checkpoint",
    "load_trace",
    "save_checkpoint",
    "simulation_pretrain",
    "trace_to_examples",
]
