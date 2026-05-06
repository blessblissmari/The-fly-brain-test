"""Round-12 — LoRA-style adapter over a frozen FlyBrain GNN controller.

Round-13 (paid YandexGPT) closed the bench saying:
``flybrain + watchdog v3`` reaches 0.975 success at 1.48 ₽/task —
2.5 pp behind the hand-coded ``manual_graph`` (1.000) and 17 %
cheaper than raw GNN (1.78 ₽/task). Round-12 asks: can a tiny
*adapter* layered on top of the frozen ``sim_pretrain_gnn_v6``
checkpoint close the remaining 2.5 pp without retraining the whole
network?

Design (LoRA — Hu et al. 2021 §4): freeze every parameter the round-7
checkpoint already learned, then add a low-rank residual ``B(A x)``
on top of the action-kind logits only. The kind head is the locus of
the round-7 structural failure (controller never picks
``terminate``/Finalizer) — round-8/9 watchdog patched it with hard
rules; round-12 patches it with a soft, learnable correction trained
from ``manual_graph`` demonstrations.

Per-task budget targets:

* Rank ``r = 4`` ⇒ ``|A| = hidden_dim × r = 32 × 4 = 128`` and
  ``|B| = r × NUM_KINDS = 4 × 9 = 36`` ⇒ **164 trainable
  parameters** for the canonical ``hidden_dim=32`` config used in
  every round-7+ baseline. That is ~0.4 % of the parameters in the
  base controller (~40 K) — small enough that the adapter cannot
  reproduce the base policy from scratch and has to stay close to
  the frozen prior.
* Init: ``A ~ N(0, 0.02)``, ``B = 0`` so the adapted controller
  starts byte-identical to the frozen one. Any deviation is the
  result of training signal alone.
* Scaling: ``alpha`` defaults to 1.0; can be tuned at inference
  without retraining.

The :class:`FlyBrainGNNLoRAController` subclass wires the adapter
into :class:`FlyBrainGNNController` and exposes
:meth:`freeze_base` so the imitation loop only updates the adapter
weights. Because the adapter only touches ``kind_logits``, every
other head (agent, edge, value, aux) keeps producing exactly the
same outputs as the frozen base — the adapter is a strictly *additive*
residual on the action-type prior.

Saving / loading is done via ``state_dict`` slicing — see
:func:`save_lora_adapter` and :func:`load_lora_adapter` below.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from flybrain.controller.action_space import NUM_KINDS
from flybrain.controller.gnn_controller import FlyBrainGNNController
from flybrain.controller.heads import HeadOutputs
from flybrain.embeddings.state import ControllerState

DEFAULT_LORA_RANK = 4
DEFAULT_LORA_ALPHA = 1.0
DEFAULT_LORA_DROPOUT = 0.0


class LoRAKindAdapter(nn.Module):
    """Low-rank residual on top of a fixed kind-logits head.

    ``output(state_vec) = alpha * B(dropout(A(state_vec)))``

    With ``B`` initialised to zero the adapter starts as a no-op so
    plugging it into a frozen controller does not change the policy
    until training has had a chance to update its weights.
    """

    def __init__(
        self,
        in_dim: int,
        num_kinds: int = NUM_KINDS,
        *,
        rank: int = DEFAULT_LORA_RANK,
        alpha: float = DEFAULT_LORA_ALPHA,
        dropout: float = DEFAULT_LORA_DROPOUT,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank!r}")
        self.in_dim = int(in_dim)
        self.num_kinds = int(num_kinds)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.A = nn.Linear(self.in_dim, self.rank, bias=False)
        self.B = nn.Linear(self.rank, self.num_kinds, bias=False)
        self.drop: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    @property
    def num_parameters(self) -> int:
        return self.in_dim * self.rank + self.rank * self.num_kinds

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.B(self.drop(self.A(state_vec)))


class FlyBrainGNNLoRAController(FlyBrainGNNController):
    """:class:`FlyBrainGNNController` plus a :class:`LoRAKindAdapter`.

    Behaves identically to the base controller until
    :meth:`LoRAKindAdapter.B` learns a non-zero correction. The
    adapter only modifies ``kind_logits``; every other head remains
    untouched.
    """

    name = "flybrain-gnn-lora"

    def __init__(
        self,
        *,
        lora_rank: int = DEFAULT_LORA_RANK,
        lora_alpha: float = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        hidden_dim = int(kwargs.get("hidden_dim", 128))
        self.lora_kind = LoRAKindAdapter(
            in_dim=hidden_dim,
            num_kinds=NUM_KINDS,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

    def forward(self, controller_state: ControllerState) -> HeadOutputs:
        state_vec, agent_vecs = self.encoder.encode(controller_state)
        state_vec, agent_vecs = self._combine(state_vec, agent_vecs, controller_state)
        base = self.heads(state_vec, agent_vecs)
        adapted_kind = base.kind_logits + self.lora_kind(state_vec)
        return HeadOutputs(
            kind_logits=adapted_kind,
            agent_logits=base.agent_logits,
            edge_from_logits=base.edge_from_logits,
            edge_to_logits=base.edge_to_logits,
            edge_scalar=base.edge_scalar,
            value=base.value,
            aux_verifier=base.aux_verifier,
        )

    def freeze_base(self) -> None:
        """Disable ``requires_grad`` on every base-controller parameter,
        leaving only the LoRA adapter trainable. Idempotent.
        """
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith("lora_"))

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return the list of parameters with ``requires_grad=True`` —
        useful when wiring an optimiser around just the adapter.
        """
        return [p for p in self.parameters() if p.requires_grad]


def save_lora_adapter(
    controller: FlyBrainGNNLoRAController,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Persist ONLY the adapter weights (not the frozen base) so the
    file stays small (~10 KiB) and can be cleanly reapplied to any
    base checkpoint."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "lora_kind": controller.lora_kind.state_dict(),
        "config": {
            "in_dim": controller.lora_kind.in_dim,
            "num_kinds": controller.lora_kind.num_kinds,
            "rank": controller.lora_kind.rank,
            "alpha": controller.lora_kind.alpha,
        },
        "metadata": dict(metadata or {}),
    }
    torch.save(state, out)
    return out


def load_lora_adapter(
    controller: FlyBrainGNNLoRAController,
    path: str | Path,
) -> dict[str, Any]:
    """Load adapter weights into ``controller.lora_kind``. Returns the
    metadata dict so callers can introspect training cfg / round id.
    """
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "lora_kind" not in blob:
        raise ValueError(f"file {path!r} is not a LoRA adapter checkpoint")
    cfg = blob.get("config", {})
    if cfg:
        if int(cfg.get("rank", controller.lora_kind.rank)) != controller.lora_kind.rank:
            raise ValueError(
                f"rank mismatch: ckpt rank={cfg.get('rank')!r}, "
                f"controller rank={controller.lora_kind.rank!r}"
            )
        if int(cfg.get("in_dim", controller.lora_kind.in_dim)) != controller.lora_kind.in_dim:
            raise ValueError(
                f"in_dim mismatch: ckpt in_dim={cfg.get('in_dim')!r}, "
                f"controller in_dim={controller.lora_kind.in_dim!r}"
            )
    controller.lora_kind.load_state_dict(blob["lora_kind"])
    return dict(blob.get("metadata") or {})


__all__ = [
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_DROPOUT",
    "DEFAULT_LORA_RANK",
    "FlyBrainGNNLoRAController",
    "LoRAKindAdapter",
    "load_lora_adapter",
    "save_lora_adapter",
]
