"""Round-12 unit tests for ``flybrain.training.lora_adapter``.

Covers:

* ``LoRAKindAdapter``: parameter count, zero-init residual, deterministic
  forward shape.
* ``FlyBrainGNNLoRAController``: ``freeze_base()`` leaves only adapter
  weights trainable; ``forward()`` matches the base controller when
  the adapter is at zero-init.
* Save / load round-trip: ``save_lora_adapter`` writes a file the
  matching ``load_lora_adapter`` can ingest; rank / in_dim mismatch is
  rejected loudly (cannot silently mis-apply weights to a different
  geometry).
"""

from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")

from flybrain.agents.specs import MINIMAL_15  # noqa: E402
from flybrain.controller.action_space import NUM_KINDS  # noqa: E402
from flybrain.embeddings import (  # noqa: E402
    AgentEmbedder,
    AgentGraphEmbedder,
    ControllerStateBuilder,
    FlyGraphEmbedder,
    MockEmbeddingClient,
    TaskEmbedder,
    TraceEmbedder,
)
from flybrain.training.lora_adapter import (  # noqa: E402
    DEFAULT_LORA_RANK,
    FlyBrainGNNLoRAController,
    LoRAKindAdapter,
    load_lora_adapter,
    save_lora_adapter,
)


def _build_builder() -> ControllerStateBuilder:
    client = MockEmbeddingClient(output_dim=32)
    agent_emb = AgentEmbedder(client)
    asyncio.run(agent_emb.precompute(MINIMAL_15))
    return ControllerStateBuilder(
        task=TaskEmbedder(client),
        agents=agent_emb,
        trace=TraceEmbedder(client),
        fly=FlyGraphEmbedder(dim=8),
        agent_graph=AgentGraphEmbedder(in_dim=32, hidden_dim=16, out_dim=32),
    )


def _build_lora_controller(*, rank: int = DEFAULT_LORA_RANK) -> FlyBrainGNNLoRAController:
    return FlyBrainGNNLoRAController(
        builder=_build_builder(),
        task_dim=32,
        agent_dim=32,
        graph_dim=32,
        trace_dim=32 + 13,
        fly_dim=8,
        produced_dim=6,
        hidden_dim=32,
        lora_rank=rank,
    )


def test_lora_kind_adapter_param_count_matches_formula() -> None:
    adapter = LoRAKindAdapter(in_dim=32, num_kinds=NUM_KINDS, rank=4)
    # |A| = in_dim * rank + |B| = rank * num_kinds (no biases).
    assert adapter.num_parameters == 32 * 4 + 4 * NUM_KINDS
    n_grad = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    assert n_grad == adapter.num_parameters


def test_lora_kind_adapter_starts_at_zero_residual() -> None:
    """B is zero-initialised so the adapter is a no-op until trained."""
    adapter = LoRAKindAdapter(in_dim=32, rank=4)
    x = torch.randn(32)
    out = adapter(x)
    assert out.shape == (NUM_KINDS,)
    assert torch.allclose(out, torch.zeros(NUM_KINDS))


def test_lora_adapter_rejects_rank_zero() -> None:
    with pytest.raises(ValueError):
        LoRAKindAdapter(in_dim=32, rank=0)


def test_freeze_base_leaves_only_adapter_trainable() -> None:
    ctrl = _build_lora_controller(rank=4)
    ctrl.freeze_base()
    trainable = ctrl.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    assert n_trainable == ctrl.lora_kind.num_parameters
    # Sanity: every base parameter must report requires_grad=False.
    for name, p in ctrl.named_parameters():
        if name.startswith("lora_"):
            assert p.requires_grad is True, name
        else:
            assert p.requires_grad is False, name


def test_zero_init_lora_matches_base_kind_logits() -> None:
    """With B=0 the LoRA controller's kind_logits must equal what the
    underlying GNN heads produce. Validates the additive residual is
    semantically correct."""
    from flybrain.runtime.state import RuntimeState

    ctrl = _build_lora_controller(rank=4)
    rs = RuntimeState(
        task_id="unit/test",
        task_type="coding",
        prompt="def add(a, b):",
        step_id=0,
        available_agents=[a.name for a in MINIMAL_15],
        pending_inbox={},
        last_active_agent=None,
    )
    cs = ctrl.builder.from_runtime_sync(rs)
    # Zero-init residual ⇒ kind_logits must equal heads.kind(state_vec).
    state_vec, agent_vecs = ctrl.encoder.encode(cs)
    state_vec, agent_vecs = ctrl._combine(state_vec, agent_vecs, cs)
    base_kind_logits = ctrl.heads.kind(state_vec)
    out = ctrl(cs)
    assert torch.allclose(out.kind_logits, base_kind_logits, atol=1e-7)


def test_save_load_round_trip(tmp_path) -> None:
    ctrl = _build_lora_controller(rank=4)
    # Set a deterministic non-zero state in the adapter so the round-trip
    # is non-trivial.
    with torch.no_grad():
        ctrl.lora_kind.A.weight.fill_(0.123)
        ctrl.lora_kind.B.weight.fill_(0.456)
    out_path = save_lora_adapter(ctrl, tmp_path / "lora.pt", metadata={"round": 12})
    assert out_path.exists()

    ctrl2 = _build_lora_controller(rank=4)
    md = load_lora_adapter(ctrl2, out_path)
    assert md == {"round": 12}
    assert torch.allclose(ctrl2.lora_kind.A.weight, ctrl.lora_kind.A.weight)
    assert torch.allclose(ctrl2.lora_kind.B.weight, ctrl.lora_kind.B.weight)


def test_load_rejects_rank_mismatch(tmp_path) -> None:
    ctrl_r4 = _build_lora_controller(rank=4)
    out_path = save_lora_adapter(ctrl_r4, tmp_path / "lora.pt")

    ctrl_r8 = _build_lora_controller(rank=8)
    with pytest.raises(ValueError, match="rank mismatch"):
        load_lora_adapter(ctrl_r8, out_path)


def test_load_rejects_garbage_file(tmp_path) -> None:
    bad = tmp_path / "garbage.pt"
    torch.save({"hello": "world"}, bad)
    ctrl = _build_lora_controller(rank=4)
    with pytest.raises(ValueError, match="not a LoRA adapter checkpoint"):
        load_lora_adapter(ctrl, bad)
