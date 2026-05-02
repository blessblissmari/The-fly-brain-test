| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | bbh_mini | 5 | 1.00 | 1.00 | 1930 | 9.60 | 6410 | 1.75 | 1.75 |
| fully_connected | bbh_mini | 5 | 1.00 | 1.00 | 2440 | 9.40 | 5459 | 2.08 | 2.08 |
| random_sparse | bbh_mini | 5 | 0.000 | 0.850 | 1564 | 9.00 | 4165 | 0.966 | ∞ |
| degree_preserving | bbh_mini | 5 | 1.00 | 1.00 | 586 | 3.00 | 3.20 | 0.403 | 0.403 |
| learned_router_no_prior | bbh_mini | 5 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | bbh_mini | 5 | 0.000 | 0.600 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | bbh_mini | 5 | 0.000 | 0.600 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_imitation | bbh_mini | 5 | 0.000 | 0.600 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_rl | bbh_mini | 5 | 0.000 | 0.600 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
