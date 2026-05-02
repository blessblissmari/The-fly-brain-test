| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | bbh_mini | 3 | 1.00 | 1.00 | 1787 | 8.33 | 5831 | 1.63 | 1.63 |
| fully_connected | bbh_mini | 3 | 1.00 | 1.00 | 2124 | 7.67 | 4489 | 1.67 | 1.67 |
| random_sparse | bbh_mini | 3 | 0.333 | 0.900 | 1032 | 5.33 | 2625 | 0.760 | 2.28 |
| degree_preserving | bbh_mini | 3 | 1.00 | 1.00 | 553 | 3.00 | 0.000 | 0.379 | 0.379 |
| learned_router_no_prior | bbh_mini | 3 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | bbh_mini | 3 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | bbh_mini | 3 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_imitation | bbh_mini | 3 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_rl | bbh_mini | 3 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
