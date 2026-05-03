# Round 2 progress (2026-05-02 ~14:30 UTC)

This note captures the round-2 work that builds on the PR #13 round-1
artefacts. It is a checkpoint, not a final report — `docs/final_report.md`
remains the canonical write-up of the round-1 N=50 headline.

## Inputs

* **Round-1 baseline** (committed in `f1dde66`, `d7c6afc`):
  * `data/checkpoints/imitation_gnn.pt` — 12 epochs over 150 lite traces.
  * `data/checkpoints/rl_gnn.pt` — 10 epochs REINFORCE warm-start from
    imitation, entropy ≈ 0.22.
  * `data/experiments/bench_yandex_2026_05_02_n50/` — 412.52 ₽,
    seed=0.

## Round-2 changes

### 1. v3 expert traces (`data/traces/expert/v3_pro/` — gitignored)

200 expert traces collected against the YandexGPT pipeline (lite tier as
served — Pro path was honoured by the script but Yandex backend cached
many prompts so effective cost ≈ lite). 196 / 200 (98%) passed the
verifier; 95.88 ₽ for the run.

Combined with the round-1 v2 lite traces (150 traces) into a 350-trace
training set at `data/traces/expert/v3_combined/`.

### 2. Round-2 trained checkpoints

* **`data/checkpoints/imitation_gnn_v2.pt`** — 16 epochs over the
  combined 350-trace set, 1838 (state, action) pairs total. Held-out
  accuracy 1.000 (vs. 1.000 in round 1). Wall: 34 min CPU.
* **`data/checkpoints/rl_gnn_v2.pt`** — 12 epochs REINFORCE warm-start
  from `imitation_gnn_v2.pt`. Returns flat at 1.289 (all traces pass
  verifier ⇒ no signal differential), entropy collapses 0.28 → 0.0003 over
  12 epochs. The v2 RL checkpoint therefore behaves as a deterministic
  copy of the v2 imitation policy.
* **`data/checkpoints/rl_gnn_v2_ppo.pt`** — 6 PPO iterations × 4 epochs.
  Returns flat at 1.289, losses NaN (constant-advantage PPO failure
  mode). **Not used** for downstream evaluation.

The v1 round-1 checkpoints are preserved verbatim as
`data/checkpoints/{imitation,rl}_gnn_v1.pt` for A/B comparison.

### 3. Live YandexGPT N=50 round-2 bench

Suite `full_min`, seed=1 (round 1 was seed=0), v2 imitation as
`imitation_gnn.pt` + v1 sim_pretrain + v1 rl. 596.72 ₽, 585 task-runs.

Output: `data/experiments/bench_yandex_2026_05_02_n50_v2/`.

| Method                    | success | verifier | cost/task ₽ |
|---------------------------|--------:|---------:|------------:|
| `degree_preserving`       |   0.969 |    0.995 |       0.523 |
| `manual_graph`            |   0.954 |    0.993 |       1.52  |
| `fully_connected`         |   0.938 |    0.991 |       2.04  |
| `flybrain_imitation` (v2) |   0.385 |    0.905 |       0.938 |
| `flybrain_sim_pretrain`   |   0.308 |    0.894 |       1.23  |
| `random_sparse`           |   0.246 |    0.808 |       0.659 |
| `flybrain_rl` (v2)        |   0.169 |    0.864 |       2.24  |
| `learned_router_no_prior` |   0.046 |    0.462 |       0.034 |
| `flybrain_prior_untrained`|   0.000 |    0.728 |       0.000 |

Per-benchmark headline (5/5 each on bbh_mini / gsm8k / humaneval):

* `flybrain_sim_pretrain`: 5/5 / 5/5 / 5/5 — **matches manual_graph**.
* `flybrain_imitation` (v2): 5/5 / 5/5 / 4/5 — small humaneval regression
  vs. round 1.
* `flybrain_rl`: 5/5 / 3/5 / 0/5 — RL collapses on humaneval.

`synthetic_routing` (50/50): manual 0.940, degree-preserving 0.960,
sim_pretrain 0.10, imitation_v2 0.22, rl 0.06. **Trained controllers
remain ~70 pp below static graphs on the synthetic_routing benchmark.**

### 4. Public-benchmark fixtures expanded 5 → 30

`data/benchmarks/fixtures/{humaneval,gsm8k,bbh_mini}.jsonl` now contain
30 tasks each (was 5). Sources:

* HumanEval: full 164-task release from
  `https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz`
  — first 25 unseen rows appended after the existing 5 fixtures so
  cached LLM responses still match.
* GSM8K: from the official `test.jsonl` (1319 tasks).
* BBH-mini: 25 examples drawn round-robin from 6 BBH subtasks
  (logical_deduction_three_objects, logical_deduction_five_objects,
  boolean_expressions, navigate, causal_judgement, web_of_lies).

This closes HANDOFF.md §4.c "fixtures cap at 5". A live N=30 bench on
the expanded fixtures was attempted but interrupted by an environment
restart and will be re-run in round 3.

## Cost so far

| line item                                 | cost ₽   |
|-------------------------------------------|---------:|
| round 1 v1 traces (60)                    |   28.25  |
| round 1 v2 traces (150)                   |   72.70  |
| round 1 pilot N=10 bench                  |  176.55  |
| round 1 N=50 headline bench (seed=0)      |  412.52  |
| round 2 v3 pro traces (200)               |   95.88  |
| round 2 N=50 v2 bench (seed=1)            |  596.72  |
| **total project-to-date**                 | **1382.62** |

User-allotted budget: 2000 ₽ initial + 2000 ₽ extension + 3000 ₽
round-2 = 7000 ₽ cumulative — but only 3000 ₽ remains on balance, so
~1620 ₽ headroom for round-3 work.

## What round 3 should attack

The synthetic_routing gap is **architectural**, not data-bound:

1. Pro-trace expansion (60 → 350 examples) yielded ≤ 2 pp imitation
   improvement.
2. PPO failed (constant-reward NaN losses).
3. REINFORCE collapsed entropy on uniform-positive trace set.

The next levers are:

* Larger sim_pretrain (≥ 120 epochs × `n_per_type=192`) + step penalty
  in `RewardConfig` to discourage 12-step routes on synthetic_routing.
* GNN with 2× hidden width (registry uses 32; bump to 64). Will require
  re-training all three Phase 6/7/8 stages and a checkpoint format
  bump.
* On-policy PPO using a custom reward that breaks the "everything
  passes" plateau — e.g. reward = `success - 0.05 * num_steps` so the
  controller is forced to find shorter routes.

These were scoped out of round 2 because they require a registry
re-pin to width 64, which would invalidate the round-1 / round-2
checkpoints.
