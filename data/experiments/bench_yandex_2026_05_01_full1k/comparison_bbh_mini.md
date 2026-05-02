| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | bbh_mini | 5 | 1.00 | 1.00 | 2009 | 9.40 | 6477 | 1.83 | 1.83 |
| fully_connected | bbh_mini | 5 | 1.00 | 1.00 | 2659 | 10.00 | 5716 | 2.28 | 2.28 |
| random_sparse | bbh_mini | 5 | 0.200 | 0.750 | 978 | 4.80 | 3546 | 0.497 | 2.48 |
| degree_preserving | bbh_mini | 5 | 1.00 | 1.00 | 593 | 3.00 | 0.000 | 0.400 | 0.400 |
| learned_router_no_prior | bbh_mini | 5 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | bbh_mini | 5 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | bbh_mini | 5 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_imitation | bbh_mini | 5 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_rl | bbh_mini | 5 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
