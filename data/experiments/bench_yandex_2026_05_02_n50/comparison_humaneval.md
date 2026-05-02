| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | humaneval | 5 | 1.00 | 1.00 | 1114 | 4.00 | 7537 | 0.637 | 0.637 |
| fully_connected | humaneval | 5 | 1.00 | 1.00 | 1564 | 4.00 | 5597 | 0.817 | 0.817 |
| random_sparse | humaneval | 5 | 0.000 | 0.790 | 1546 | 6.20 | 7754 | 0.945 | ∞ |
| degree_preserving | humaneval | 5 | 1.00 | 1.00 | 1236 | 4.00 | 0.000 | 0.686 | 0.686 |
| learned_router_no_prior | humaneval | 5 | 0.000 | 0.510 | 587 | 2.60 | 768 | 0.235 | ∞ |
| flybrain_prior_untrained | humaneval | 5 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | humaneval | 5 | 1.00 | 1.00 | 3292 | 12.00 | 7175 | 1.51 | 1.51 |
| flybrain_imitation | humaneval | 5 | 0.600 | 0.940 | 1763 | 7.20 | 5360 | 1.39 | 2.32 |
| flybrain_rl | humaneval | 5 | 0.200 | 0.820 | 3009 | 12.00 | 2516 | 3.13 | 15.66 |
