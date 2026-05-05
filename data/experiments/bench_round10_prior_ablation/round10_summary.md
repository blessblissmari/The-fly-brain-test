# Round 10 — connectome-prior ablation analysis

Source: `data/experiments/bench_round10_prior_ablation` (N_total=200 task-runs across 5 baselines).

## Success-rate table (mean, 95 % bootstrap CI)

| Baseline | bbh_mini | gsm8k | humaneval | synthetic_routing | _overall |
|---|---:|---:|---:|---:|---:|
| `manual_graph` | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.900 (0.70-1.00) | 0.900 (0.70-1.00) | 0.950 (0.88-1.00) |
| `flybrain_sim_pretrain` | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.800 (0.50-1.00) | 0.300 (0.00-0.60) | 0.775 (0.65-0.90) |
| `er_prior_sim_pretrain` | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.800 (0.50-1.00) | 0.200 (0.00-0.50) | 0.750 (0.62-0.88) |
| `shuffled_fly_sim_pretrain` | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.800 (0.50-1.00) | 0.300 (0.10-0.60) | 0.775 (0.65-0.90) |
| `reverse_fly_sim_pretrain` | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.900 (0.70-1.00) | 0.300 (0.00-0.60) | 0.800 (0.68-0.93) |

## Pre-registered Wilcoxon paired tests

| Comparison | n_pairs | mean(flybrain) | mean(null) | diff | W+ | p (two-sided, uncorrected) | p (Bonf-3) |
|---|---:|---:|---:|---:|---:|---:|---:|
| flybrain vs `er_prior_sim_pretrain` | 40 | 0.775 | 0.750 | +0.025 | 4.0 | 0.789 | 1.000 |
| flybrain vs `shuffled_fly_sim_pretrain` | 40 | 0.775 | 0.775 | +0.000 | 0.0 | 1.000 | 1.000 |
| flybrain vs `reverse_fly_sim_pretrain` | 40 | 0.775 | 0.800 | -0.025 | 2.0 | 0.789 | 1.000 |

## Verdict

**H-null supported (controller insensitive to prior):**

Observed pattern: flybrain (0.775) ≈ shuffled (0.775) ≈ reverse (0.800) ≈ er (0.750).

Directionality finding: direction immaterial (flybrain 0.775 vs reverse 0.800; Δ=-0.025).
