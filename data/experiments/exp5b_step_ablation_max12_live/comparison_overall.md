| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | _overall | 30 | 0.833 | 0.975 | 1847 | 10.97 | 8999 | 1.63 | 1.96 |
| fully_connected | _overall | 30 | 1.00 | 1.00 | 2560 | 10.77 | 8829 | 2.17 | 2.17 |
| random_sparse | _overall | 30 | 0.267 | 0.795 | 1172 | 8.17 | 5726 | 0.760 | 2.85 |
| degree_preserving | _overall | 30 | 0.967 | 0.995 | 810 | 4.20 | 867 | 0.531 | 0.549 |
| learned_router_no_prior | _overall | 30 | 0.100 | 0.640 | 46.00 | 0.267 | 426 | 0.018 | 0.184 |
| flybrain_prior_untrained | _overall | 30 | 0.000 | 0.735 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | _overall | 30 | 0.133 | 0.870 | 1997 | 12.00 | 5349 | 1.12 | 8.41 |
| flybrain_imitation | _overall | 30 | 0.200 | 0.880 | 1339 | 7.70 | 2458 | 0.777 | 3.88 |
| flybrain_rl | _overall | 30 | 0.067 | 0.855 | 1930 | 12.00 | 1436 | 2.07 | 31.08 |
