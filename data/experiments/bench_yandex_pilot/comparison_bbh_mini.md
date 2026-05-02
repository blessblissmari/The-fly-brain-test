| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | bbh_mini | 5 | 1.00 | 1.00 | 591 | 3.00 | 2841 | 0.402 | 0.402 |
| fully_connected | bbh_mini | 5 | 1.00 | 1.00 | 744 | 3.00 | 1675 | 0.463 | 0.463 |
| random_sparse | bbh_mini | 5 | 0.600 | 0.940 | 1700 | 7.60 | 7042 | 0.879 | 1.47 |
| degree_preserving | bbh_mini | 5 | 1.00 | 1.00 | 591 | 3.00 | 0.000 | 0.402 | 0.402 |
| learned_router_no_prior | bbh_mini | 5 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | bbh_mini | 5 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | bbh_mini | 5 | 1.00 | 1.00 | 1972 | 10.20 | 3707 | 1.13 | 1.13 |
| flybrain_imitation | bbh_mini | 5 | 1.00 | 1.00 | 2256 | 12.00 | 2860 | 2.08 | 2.08 |
| flybrain_rl | bbh_mini | 5 | 1.00 | 1.00 | 2370 | 12.00 | 3345 | 1.99 | 1.99 |
