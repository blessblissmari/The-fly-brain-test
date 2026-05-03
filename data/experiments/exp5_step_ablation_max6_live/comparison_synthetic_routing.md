| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | synthetic_routing | 30 | 0.267 | 0.890 | 952 | 5.93 | 5358 | 0.977 | 3.67 |
| fully_connected | synthetic_routing | 30 | 0.333 | 0.900 | 1374 | 5.87 | 5080 | 1.20 | 3.61 |
| random_sparse | synthetic_routing | 30 | 0.033 | 0.735 | 512 | 3.87 | 2843 | 0.293 | 8.80 |
| degree_preserving | synthetic_routing | 30 | 0.833 | 0.975 | 898 | 4.90 | 2469 | 0.703 | 0.844 |
| learned_router_no_prior | synthetic_routing | 30 | 0.033 | 0.487 | 77.10 | 0.267 | 375 | 0.031 | 0.925 |
| flybrain_prior_untrained | synthetic_routing | 30 | 0.000 | 0.735 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | synthetic_routing | 30 | 0.067 | 0.855 | 1184 | 6.00 | 5110 | 0.625 | 9.38 |
| flybrain_imitation | synthetic_routing | 30 | 0.200 | 0.875 | 957 | 5.00 | 2731 | 0.526 | 2.63 |
| flybrain_rl | synthetic_routing | 30 | 0.000 | 0.815 | 945 | 6.00 | 594 | 1.04 | ∞ |
