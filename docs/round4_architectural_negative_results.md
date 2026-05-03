% Round 4 — architectural negative results on `synthetic_routing`
% 2026-05-03

## TL;DR

Round 3 left one publishable open question: **why do trained
controllers (`flybrain_sim_pretrain`, `flybrain_imitation`,
`flybrain_rl`) score 13–27 % on the `synthetic_routing` benchmark
when static graphs (`manual_graph`, `degree_preserving`,
`fully_connected`) score 83–100 %?**

Round 4 is a focused, hypothesis-driven attack on this gap, all on
the *real* FlyWire 783 prior (`data/flybrain/fly_graph_64.fbg`,
provenance `'source': 'zenodo_csv'`, K=64 Louvain, Q≈0.6800).

| # | Hypothesis tested | Live experiment | Result for trained imitation | Verdict |
|---|---|---|---|---|
| H1 | "Trained controllers fail because `max_steps=32` lets them over-explore" | `exp5_step_ablation_max6_live` (`max_steps=6`) | 6 / 30 (vs. 8 / 30 at `max_steps=32`) | **rejected** — manual_graph also collapses (8 / 30) |
| H2 | "Trained controllers fail because of distribution shift between training (`max_steps=12`) and inference (`max_steps=32`)" | `exp5b_step_ablation_max12_live` | 6 / 30 | **rejected** — no improvement; manual_graph at 25 / 30 |
| H3 | "Imitation traces never showed the controller a *passing* `synthetic_routing` policy" | `exp6_imitation_v4_synthetic_live` (v4 = v3 + 25 passing manual_graph traces, `--only-passed`) | 8 / 30 | **null** — no change vs. v3 |
| H4 | "v4's 25 added traces are too few — up-weight 5×" | `exp7_imitation_v5_synthetic_live` (v5 = v3 + 5×25 dups, 26 % synthetic-routing share) | 7 / 30 | **null** — slight regression; held-out acc drops 0.906 → 0.598 |

All four experiments together replicate the **architectural-not-data**
conclusion that round-2 ablations (`exp4_training_ablation_live`)
had already pointed at: more imitation data, more steps, or fewer
steps do not move trained controllers off the 13–27 % plateau on
`synthetic_routing`.

## Cost ledger (round 4)

| Experiment | Tasks | Calls | Cost ₽ |
|---|---:|---:|---:|
| `exp5_step_ablation_max6_live` (9 × 30) | 270 | 1135 | 161.96 |
| `exp5b_step_ablation_max12_live` (9 × 30) | 270 | 1982 | 272.41 |
| `exp6_imitation_v4_synthetic_live` (3 × 30) | 90 | 1473 | 199.00 |
| `exp7_imitation_v5_synthetic_live` (1 × 30) | 30 | 740 | 128.85 |
| **Round 4 total** | **660** | **5330** | **762.22** |

Project total: **3417.12 ₽ + 762.22 ₽ = 4179.34 ₽** out of the
9 500 ₽ Ministry-of-Science authorisation (44 % utilisation).

## Why H1 fails (`max_steps`)

```
                     | manual_graph | flybrain_imitation | flybrain_sim_pretrain |
max_steps=6          |   8 / 30     |     6 / 30         |     2 / 30            |
max_steps=12         |  25 / 30     |     6 / 30         |     4 / 30            |
max_steps=32 (R3)    |  25 / 30     |     8 / 30         |     4 / 30            |
```

* Static `manual_graph` *needs* ≥12 steps because each of its
  agents (Planner → Coder → TestRunner → Debugger → Verifier →
  Finalizer) emits an LLM call and the verifier multi-step trace
  occupies several of those.
* Trained controllers do **not** become more effective with more
  budget. Whatever the policy is, more iterations of it do not
  converge on success.
* This rejects the "trained controllers waste budget on
  unproductive agents" hypothesis: even at the *exact* training
  cap (`max_steps=12`) they score the same as at runtime
  (`max_steps=32`).

## Why H3/H4 fail (targeted distillation)

Looking at action sequences from the round-3 `bench_yandex_2026_
05_03_n30_full/flybrain_sim_pretrain/synthetic_routing/` traces:

```
sim_pretrain pattern (most common, count=3):
  Planner, Coder, TestRunner, Debugger, Debugger,
  SchemaValidator × 7

sim_pretrain pattern (count=2):
  Planner, Retriever × 3, Planner × 7
```

The trained controller emits the **prefix** of an OPTIMAL_ROUTES
sequence correctly (Planner → Coder → TestRunner → Debugger), then
falls into a fixed-point loop on a single agent (`SchemaValidator`,
`Planner`, etc.) until the step budget runs out. It never emits
`{"kind": "terminate"}`, even though `SyntheticMAS` training data
*does* include terminate actions at the end of each route.

H3 (v4) added 25 *passing* traces from `manual_graph` running on
real `synthetic_routing` tasks. The intent: show the controller a
working teacher policy on the actual benchmark distribution.

H4 (v5) up-weighted those 25 traces 5× to 125 / 475 = 26 % of
training mix, hoping a bigger gradient signal would shift the
policy.

Neither moved the live numbers (8 / 30 → 8 / 30 → 7 / 30). Held-out
accuracy on the v5 dataset *dropped* from 0.906 (v4) to 0.598 (v5)
— a clean signal that the controller cannot fit the duplicated
distribution while keeping the v3 4-task-family fit.

## Implications for the publication

1. **Round-3 headline stands.** `flybrain_imitation` matches
   `manual_graph` on `gsm8k` (30 / 30) and is within variance on
   `bbh_mini` (29 / 30 vs. 30 / 30) on the canonical N=30 expanded
   fixtures, at cost 1.52 ₽/task vs. 2.46 ₽/task — strictly cheaper
   than the operator's hand-written graph.

2. **`synthetic_routing` is an architectural problem.** The four
   round-4 experiments are *negative results* but they are the
   *correct* negative results: they rule out the cheap data-side
   and step-budget-side fixes. What remains is genuine architecture
   work that does not fit in the 9 500 ₽ Ministry budget:

   * GNN width 32 → 64 (registry re-pin + retrain all of Phase 6 / 7
     / 8).
   * `step_penalty` in `RewardConfig` (currently the RL trainer has
     no explicit step cost, so REINFORCE's optimal policy can be
     "loop one agent forever").
   * On-policy PPO with the production verifier as reward — replaces
     the trace-cloning REINFORCE that collapses to constant return
     under our 98 %-pass-rate trace distribution.
   * Or a fundamentally different controller class (RNN with
     explicit termination head; transformer routing; learned
     halting policy).

3. **Round-4 negative results are themselves publishable.** The
   action-sequence analysis ("trained controller emits OPTIMAL
   prefix then loops on one agent") is a clean falsifiable claim,
   and the cost-vs.-improvement plateau across four interventions
   is the kind of evidence reviewers expect.

## Replicate

```
# H1 / H2: step ablation
flybrain-py bench --suite full_min --backend yandex \
    --tasks-per-benchmark 30 --benchmarks synthetic_routing \
    --max-steps {6, 12} --budget-rub 350 --parallelism 4 --seed 2 \
    --output data/experiments/exp5{,b}_step_ablation_max{6,12}_live

# H3 (imitation v4 = v3 + 25 manual_graph passing traces)
python scripts/run_imitation.py --controller gnn \
    --traces data/traces/expert/v4_combined --warm-from \
    data/checkpoints/sim_pretrain_gnn.pt --epochs 16 ... \
    --output data/checkpoints/imitation_gnn_v4.pt

FLYBRAIN_BASELINE_IMITATION=data/checkpoints/imitation_gnn_v4.pt \
    flybrain-py bench --suite full_min --backend yandex \
    --tasks-per-benchmark 30 --benchmarks synthetic_routing \
    --only flybrain_imitation manual_graph degree_preserving ...

# H4 (imitation v5 = v3 + 5×25 upweighted)
# same with --traces data/traces/expert/v5_combined_upweighted
```
