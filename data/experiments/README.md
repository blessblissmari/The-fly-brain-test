# Ablation experiments — README §18

Mock-LLM smoke runs of the four §18 experiments that PR #2 stopped at
"caркас готов, но не прогнан":

| Experiment | Suite | Output | Notes |
|---|---|---|---|
| 2 | `embedding_ablation` (5 levels) | `exp2_embedding_ablation/` | LearnedRouter with 0–5 of {task, agent, trace, graph, fly} embeddings unmasked. |
| 3 | `verifier_ablation` (4 levels) | `exp3_verifier_ablation/` | FlyBrain prior with `MASConfig.verification_mode` ∈ {`off`, `final`, `step`, `full`}. |
| 4 | `full_min` | `docs/final_report.md` §2-3 | Training progression (untrained → sim → imitation → RL); the missing graph-SSL row is added by the next stacked PR. |
| 5 | `full_min` | `docs/final_report.md` §8 Exp 5 | Generalisation = finetune on `min_set` (10–20%) and eval on full suite. |

Reproduce:

```bash
flybrain-py bench --suite embedding_ablation --backend mock --tasks-per-benchmark 5 --output data/experiments/exp2_embedding_ablation
flybrain-py bench --suite verifier_ablation  --backend mock --tasks-per-benchmark 5 --output data/experiments/exp3_verifier_ablation
```

Each run writes the canonical `comparison_overall.{md,json,csv}` plus per-baseline trace folders (excluded from this directory to keep the repo small — set `--output` to a path outside the repo to keep them).

The mock backend is intentional: ablations are about *relative* signal between rows, and a deterministic mock LLM removes per-call noise that would otherwise drown out the small effect sizes at this n. Live YandexGPT runs of the same suites cost about the same as `full_min` (~120 ₽ each) and can be invoked by swapping `--backend mock` for `--backend yandex`.
