| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | humaneval | 3 | 1.00 | 1.00 | 2212 | 9.67 | 10594 | 1.76 | 1.76 |
| fully_connected | humaneval | 3 | 1.00 | 1.00 | 3247 | 10.00 | 8318 | 2.38 | 2.38 |
| random_sparse | humaneval | 3 | 0.000 | 0.700 | 1175 | 5.00 | 5118 | 0.510 | ∞ |
| degree_preserving | humaneval | 3 | 1.00 | 1.00 | 2237 | 8.33 | 3764 | 1.49 | 1.49 |
| learned_router_no_prior | humaneval | 3 | 0.000 | 0.383 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | humaneval | 3 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | humaneval | 3 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_imitation | humaneval | 3 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_rl | humaneval | 3 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
