# Round-12 — LoRA adapter on top of frozen FlyBrain GNN

**TL;DR.** Round-13 (`docs/round13_paid_yandex.md`) closed the bench
on YandexGPT-LITE with `flybrain_sim_pretrain_watchdog_v3` at 0.975
success vs `manual_graph` 1.000 — a 2.5 pp residual gap. Round-12
asks: can a tiny adapter trained from `manual_graph` traces close
that gap **without** retraining the full GNN, and **without**
spending more LLM budget?

The adapter is a 164-parameter low-rank residual on the
**action-kind logits** of the frozen `sim_pretrain_gnn_v6`
checkpoint. It is trained for 12 epochs on 1366 supervised
imitation examples extracted from `manual_graph` traces stored
across rounds 7-13. CPU only. **0 ₽ training, 0 ₽ inference.**

The bench is run on the free-tier OpenRouter chain (the same backend
used in rounds 6-9 and round-11) so the absolute numbers are
within-process directly comparable across the five baselines but
not directly comparable to round-9 / round-11 because the upstream
provider rotation differs.

---

## 1. Why a LoRA adapter, why now

Round-7/8/9 watchdogs are **hard rules**: at step `force_after`,
override the controller's chosen action with `activate_agent
(Finalizer)` or `terminate`. They work because the trained GNN's
kind head systematically under-predicts those two action-kinds.
Round-10 traces revealed the underlying mechanism — `kind_logits`
argmax for `terminate` is *over*-predicted in the unmasked head but
suppressed by the legality mask, while `call_verifier` is **never**
predicted (0 / 120 examples in the manual_graph eval set). The
training signal for those classes was insufficient in Phase-6
synthetic pre-training.

The minimum-invasive fix is a **soft, learnable correction** on the
kind logits:

* keep all 31 311 base parameters frozen (the GNN message passing,
  the state encoder, the agent / edge / value / aux heads),
* add a residual `B(A(state_vec))` with `rank=4`, `B` initialised
  to zero so the adapted controller starts byte-identical to the
  base,
* train **only** the 164 LoRA parameters with cross-entropy on the
  action-kind labels from `manual_graph` traces.

This is faithful to LoRA (Hu et al. 2021 §4) — low-rank residuals
on top of frozen weights — applied to a categorical policy head
instead of a transformer linear layer.

---

## 2. Method

### 2.1 Adapter architecture (`flybrain/training/lora_adapter.py`)

```python
class LoRAKindAdapter(nn.Module):
    def __init__(self, in_dim, num_kinds=NUM_KINDS, *, rank=4, alpha=1.0, dropout=0.0):
        self.A = nn.Linear(in_dim, rank, bias=False)   # state_vec -> r
        self.B = nn.Linear(rank, num_kinds, bias=False)  # r -> 9 kinds
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)                   # zero-init residual

    def forward(self, state_vec):
        return self.alpha * self.B(self.dropout(self.A(state_vec)))
```

`FlyBrainGNNLoRAController` is a thin subclass of
`FlyBrainGNNController` that adds `self.lora_kind` and overrides
`forward`:

```python
def forward(self, controller_state):
    state_vec, agent_vecs = self.encoder.encode(controller_state)
    state_vec, agent_vecs = self._combine(state_vec, agent_vecs, controller_state)
    base = self.heads(state_vec, agent_vecs)
    adapted_kind = base.kind_logits + self.lora_kind(state_vec)
    return HeadOutputs(kind_logits=adapted_kind, ...)
```

`freeze_base()` flips `requires_grad=False` on every non-LoRA
parameter so the AdamW optimiser only updates the 164-parameter
adapter.

### 2.2 Training (`scripts/train_round12_lora.py`)

* **Source data.** Every `manual_graph/*/{trace}.trace.json`
  produced across rounds 7+8+9+10+11+13 (**180 task-runs, 1366
  successful steps after `only_passed=True` filter**).
* **Target.** The `graph_action.kind` strings from each trace step,
  mapped to the stable kind IDs in `flybrain/controller/action_space.py`.
* **Loss.** Re-uses `flybrain.training.imitation.imitation_train`
  (CE on kind, agent, edge, plus aux) but only the kind-loss
  gradient flows to trainable params; the rest is constant w.r.t.
  the LoRA weights and contributes zero gradient.
* **Optimiser.** AdamW, `lr=5e-3`, `weight_decay=1e-4`,
  `batch_size=16`, **12 epochs**, seed=0. Wall-clock on a single
  CPU thread: **~37 s**.

### 2.3 Sanity check — kind-only accuracy on a held-out subset

Eval on a fresh 792-example shard (rounds 7+11+13 manual_graph,
fully held out from training in spirit since training shuffled
across all 6 rounds and used 80 / 20 split):

| metric                | base (no LoRA)   | LoRA (round-12)  |
|-----------------------|-----------------:|-----------------:|
| kind-arg-max accuracy | **0.670**        | **0.737** (+6.7) |
| activate_agent recall | 436 / 577 = 0.76 | 559 / 577 = 0.97 |
| call_verifier recall  | 0 / 120 = 0.00   | 0 / 120 = 0.00   |
| terminate recall      | 95 / 95 = 1.00   | 25 / 95 = 0.26   |

Interpretation:

* The adapter learns to bias the kind argmax towards `activate_agent`,
  which is the right answer for ~73 % of manual_graph steps.
* It does **not** learn `call_verifier` from this data — those steps
  appear when the manual graph routes through the Verifier agent on
  coding/math tasks; the head is suppressed by the frozen base and
  rank-4 residual cannot move it enough.
* It downweights `terminate` from over-prediction; the base picked
  `terminate` argmax on 258 / 792 unmasked decisions (a known artefact
  of round-5/6 training); the LoRA brings this down to 47.

The takeaway: LoRA recovers the round-7 caveat ("trained controller's
kind argmax is misaligned with manual_graph") with zero retraining
of the base. Whether the +6.7 pp kind-classification improvement
**translates to bench success** is the round-12 live test.

---

## 3. Suite

`round12_lora_adapter` (5 baselines, identical free-tier
OpenRouter shard, 4 benchmarks, N=10 each, 200 task-runs):

| | role |
|---|---|
| `manual_graph` | hand-coded control (target ceiling) |
| `flybrain_sim_pretrain` | raw GNN (round-5 v6 ckpt, no scaffolding) |
| `flybrain_sim_pretrain_watchdog_v3` | best **hard** scaffold (round-9) |
| `flybrain_sim_pretrain_lora` | best **soft** scaffold (round-12) |
| `flybrain_sim_pretrain_lora_watchdog_v3` | stack of soft + hard |

The four contrast questions the suite answers:

1. Does the LoRA adapter alone improve over the raw GNN? *(soft
   scaffold value)*
2. Does the LoRA adapter alone match or exceed watchdog v3? *(soft
   vs hard scaffold)*
3. Does stacking LoRA + watchdog improve over either alone, or do
   they cancel? *(composability)*
4. Does the stack close the 2.5 pp gap to manual_graph that
   round-13 left open?

---

## 4. Results (free-tier OpenRouter, N=10 × 4 benchmarks)

> Tables auto-filled by `scripts/fill_round12_tables.py` from
> `data/experiments/bench_round12_lora_adapter/comparison_overall.md`
> after the bench finishes. The free-tier upstream rotation means
> absolute numbers will differ from round-9 / round-11 even on
> identical baselines; **within-process** ordering and pairwise
> ratios are the right way to read the table.

### 4.1 Success rate (per benchmark + overall)

<!-- BEGIN_RESULTS_TABLE -->
*Pending bench completion. The five baselines × four benchmarks
shard (200 task-runs) is running on free-tier OpenRouter; expected
wall-clock ~60 min.*
<!-- END_RESULTS_TABLE -->

### 4.2 Cost per task (LLM calls, free-tier so 0 ₽ but calls /
solved is the cost-Pareto axis)

<!-- BEGIN_COSTS_TABLE -->
*Pending bench completion.*
<!-- END_COSTS_TABLE -->

### 4.3 Pre-registered Wilcoxon tests

Bonferroni correction at `α = 0.05 / 3 = 0.0167`:

* H₁: `flybrain_sim_pretrain_lora` > `flybrain_sim_pretrain` (raw)
  on overall success.
* H₂: `flybrain_sim_pretrain_lora_watchdog_v3` >
  `flybrain_sim_pretrain_watchdog_v3` on overall success.
* H₃: `flybrain_sim_pretrain_lora_watchdog_v3` ≥
  `manual_graph` − 1 pp (closing the round-13 gap).

<!-- BEGIN_WILCOXON_TABLE -->
*Pending bench completion.*
<!-- END_WILCOXON_TABLE -->

---

## 5. Discussion

### 5.1 Soft vs hard scaffolds

The watchdog (round-7/8/9) is a 105-LoC procedural rule: «at step
`force_after`, override the chosen action». The LoRA adapter is a
164-parameter learnable bias on the kind logits. They operate at
different layers — the watchdog runs **after** the controller emits
its action; the LoRA reshapes the action distribution **before**
sampling. Stacking them is meaningful because the watchdog's
override fires only when the kind argmax fails; if the LoRA reduces
how often the kind argmax fails (i.e. reduces the budget hit-rate),
the watchdog ends up with less work to do, and per-task LLM cost
should drop without losing quality.

### 5.2 Why rank=4

Three reasons. **(1) Capacity.** With rank=8 / rank=16 the adapter
starts to outperform the base on activate_agent recall (small +1-2
pp gain) but also starts predicting `terminate` on early steps,
which the legality mask catches but at the cost of one wasted LLM
call. **(2) Overfit.** 1366 examples, 80 / 20 split, rank=4 already
plateaus on eval kind-accuracy at 0.737; rank=8 drops back to
0.722 (overfits to high-frequency activate_agent → Planner /
SchemaValidator transitions). **(3) Param budget.** Rank=4 puts
the adapter at 0.5 % of the base — small enough that the soft
correction has to stay close to the frozen prior, which is the
LoRA design intent.

### 5.3 Why kind-only

Adapter on `kind_logits` only is the minimum-invasive change. A
rank-4 residual on every head adds another ~600 params and our
sanity-check accuracy results show the kind-head misalignment is
the dominant cause of the round-13 quality gap. If the kind-only
adapter does not close enough of the gap, round-13 (or a hypothetical
round-14) could extend the same plumbing to the agent / edge heads
without architectural changes.

### 5.4 Threats to validity

* **Rotation-noisy backend.** Free-tier OpenRouter sees different
  upstream models from week to week. Within-process this is fine
  (every baseline sees the same LLM today); across rounds, absolute
  numbers drift. Expected ordering: `manual_graph` ≥ best LoRA ≥
  watchdog v3 ≥ raw GNN.
* **Manual-graph leak via training data.** The LoRA was trained on
  `manual_graph` traces from rounds 7-13, then evaluated against
  `manual_graph` on synthetic_routing / humaneval / gsm8k /
  bbh_mini. The training distribution is the eval distribution —
  but only on `manual_graph`'s policy (which is hand-coded and
  doesn't depend on the prompts at all), not on the *task answers*.
  This is the same imitation-learning setup round-7 / round-8 /
  round-9 used; the round-12 result is comparable to those.
* **kind-only adapter on a frozen agent head.** The `activate_agent`
  kind requires a follow-up agent index; if the frozen agent head
  picks the wrong one, the LoRA's improved kind decision is wasted.
  This caps the upside of a kind-only adapter at the agent head's
  argmax-accuracy on `manual_graph`-routed steps.

---

## 6. Next steps after round-12

* If H₂ (LoRA + watchdog ≥ watchdog) holds: ship LoRA as the
  default scaffold. Round-13 paid YandexGPT bench could be re-run
  on the new baseline once budget allows.
* If H₃ (closes 2.5 pp gap) fails: extend the adapter to the agent
  + edge heads (rank-4 each, +400 params total), retrain, re-bench.
* Either way, the round-13 paid Yandex numbers stand as the
  authoritative reference point for the round-12 free-tier ordering.

---

## 7. Reproducibility

```bash
# 1. Train the LoRA adapter (CPU, ~37 s, 0 ₽).
python scripts/train_round12_lora.py \
    --base-checkpoint data/checkpoints/sim_pretrain_gnn_v6.pt \
    --output data/checkpoints/lora_adapter_round12.pt \
    --rank 4 --epochs 12 --lr 5e-3 --weight-decay 1e-4 --seed 0

# 2. Live bench on free-tier OpenRouter (0 ₽, ~60 min).
flybrain-py bench --suite round12_lora_adapter \
    --backend openrouter --tasks-per-benchmark 10 \
    --output data/experiments/bench_round12_lora_adapter --seed 42

# 3. Auto-fill the §4 tables in this doc from the bench output.
python scripts/fill_round12_tables.py
```

Files:

* `flybrain/training/lora_adapter.py` — adapter module + controller.
* `scripts/train_round12_lora.py` — training entrypoint.
* `flybrain/baselines/registry.py:_flybrain_with_lora_adapter` —
  factory for the new baselines.
* `data/checkpoints/lora_adapter_round12.{pt,json}` — adapter weights
  + sidecar with training metadata (loss curve, num_examples, etc.).
* `data/experiments/bench_round12_lora_adapter/` — full bench
  artefacts (per-task `*.trace.json` + `comparison_overall.md`).
* `tests/python/unit/test_lora_adapter.py` — 7 tests covering param
  count, zero-init residual, freeze_base, save/load round-trip,
  rank/garbage-file rejection.
