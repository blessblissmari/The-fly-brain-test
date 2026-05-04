| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | synthetic_routing | 10 | 1.00 | 1.00 | 621 | 3.30 | 4868 | 0.365 | 0.365 |
| fully_connected | synthetic_routing | 10 | 1.00 | 1.00 | 924 | 3.30 | 3800 | 0.486 | 0.486 |
| random_sparse | synthetic_routing | 10 | 0.300 | 0.785 | 865 | 5.10 | 4945 | 0.474 | 1.58 |
| degree_preserving | synthetic_routing | 10 | 1.00 | 1.00 | 658 | 3.30 | 0.000 | 0.380 | 0.380 |
| learned_router_no_prior | synthetic_routing | 10 | 0.000 | 0.645 | 33.20 | 0.100 | 336 | 0.013 | ∞ |
| flybrain_prior_untrained | synthetic_routing | 10 | 0.000 | 0.730 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | synthetic_routing | 10 | 0.500 | 0.925 | 2185 | 12.00 | 6375 | 1.18 | 2.37 |
| flybrain_imitation | synthetic_routing | 10 | 0.500 | 0.910 | 1721 | 11.10 | 3478 | 1.20 | 2.41 |
| flybrain_rl | synthetic_routing | 10 | 0.000 | 0.850 | 2062 | 12.00 | 4974 | 1.64 | ∞ |
