# Round 11 — null-prior × watchdog v2 cross-bench analysis

Source: `data/experiments/bench_round11_priors_watchdog` (N_total=200 task-runs across 5 baselines).

## Success-rate table (mean, 95 % bootstrap CI)

| Baseline | bbh_mini | gsm8k | humaneval | synthetic_routing | _overall |
|---|---:|---:|---:|---:|---:|
| `manual_graph` | 0.900 (0.70-1.00) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.975 (0.93-1.00) |
| `flybrain_sim_pretrain_watchdog_v2` | 0.900 (0.70-1.00) | 1.000 (1.00-1.00) | 1.000 (1.00-1.00) | 0.800 (0.50-1.00) | 0.925 (0.85-1.00) |
| `er_prior_watchdog_v2` | 0.900 (0.70-1.00) | 1.000 (1.00-1.00) | 0.200 (0.00-0.50) | 0.900 (0.70-1.00) | 0.750 (0.60-0.88) |
| `shuffled_fly_watchdog_v2` | 0.900 (0.70-1.00) | 1.000 (1.00-1.00) | 0.900 (0.70-1.00) | 0.800 (0.50-1.00) | 0.900 (0.80-0.97) |
| `reverse_fly_watchdog_v2` | 0.900 (0.70-1.00) | 1.000 (1.00-1.00) | 0.900 (0.70-1.00) | 0.800 (0.50-1.00) | 0.900 (0.80-0.97) |

## Pre-registered Wilcoxon paired tests

| Comparison | n_pairs | mean(real_fly+wd2) | mean(null+wd2) | diff | W+ | p (two-sided, uncorrected) | p (Bonf-3) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flybrain_sim_pretrain_watchdog_v2` vs `er_prior_watchdog_v2` | 40 | 0.925 | 0.750 | +0.175 | 40.0 | 0.044 | 0.132 |
| `flybrain_sim_pretrain_watchdog_v2` vs `shuffled_fly_watchdog_v2` | 40 | 0.925 | 0.900 | +0.025 | 1.0 | 1.000 | 1.000 |
| `flybrain_sim_pretrain_watchdog_v2` vs `reverse_fly_watchdog_v2` | 40 | 0.925 | 0.900 | +0.025 | 1.0 | 1.000 | 1.000 |

## Verdict

**Mixed / partial support:** real-fly+wd2 separates from *some* but not *all* null priors. Either biology helps selectively (e.g. on humaneval but not synthetic_routing) or noise at N=10 obscures a small but real effect. Re-running at N=30 (round-13 paid) is the natural follow-up.

Observed pattern: manual (0.975) | real_fly+wd2 (0.925) > er+wd2 (0.750); real_fly+wd2 (0.925) ≈ shuffled+wd2 (0.900); real_fly+wd2 (0.925) ≈ reverse+wd2 (0.900).

Directionality finding: direction immaterial under watchdog scaffold (real_fly+wd2 0.925 vs reverse+wd2 0.900; Δ=+0.025).
