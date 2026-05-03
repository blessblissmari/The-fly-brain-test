| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | bbh_mini | 30 | 1.00 | 1.00 | 2826 | 10.57 | 7193 | 2.60 | 2.60 |
| fully_connected | bbh_mini | 30 | 1.00 | 1.00 | 3455 | 10.60 | 6473 | 2.97 | 2.97 |
| random_sparse | bbh_mini | 30 | 0.300 | 0.868 | 1759 | 7.43 | 4024 | 1.12 | 3.74 |
| degree_preserving | bbh_mini | 30 | 1.00 | 1.00 | 834 | 3.20 | 231 | 0.589 | 0.589 |
| learned_router_no_prior | bbh_mini | 30 | 0.000 | 0.283 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | bbh_mini | 30 | 0.000 | 0.683 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | bbh_mini | 30 | 0.967 | 0.995 | 3049 | 11.67 | 4672 | 2.12 | 2.20 |
| flybrain_imitation | bbh_mini | 30 | 0.967 | 0.995 | 3038 | 12.00 | 3511 | 2.34 | 2.42 |
| flybrain_rl | bbh_mini | 30 | 0.633 | 0.945 | 3215 | 12.00 | 1284 | 3.28 | 5.18 |
