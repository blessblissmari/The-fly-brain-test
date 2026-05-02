| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | synthetic_routing | 50 | 0.920 | 0.988 | 537 | 3.26 | 3665 | 0.335 | 0.364 |
| fully_connected | synthetic_routing | 50 | 0.960 | 0.994 | 862 | 3.26 | 3153 | 0.465 | 0.484 |
| random_sparse | synthetic_routing | 50 | 0.440 | 0.866 | 979 | 5.76 | 5665 | 0.597 | 1.36 |
| degree_preserving | synthetic_routing | 50 | 0.920 | 0.988 | 564 | 3.26 | 0.000 | 0.346 | 0.376 |
| learned_router_no_prior | synthetic_routing | 50 | 0.040 | 0.519 | 215 | 1.00 | 591 | 0.086 | 2.15 |
| flybrain_prior_untrained | synthetic_routing | 50 | 0.000 | 0.736 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | synthetic_routing | 50 | 0.280 | 0.889 | 2130 | 11.82 | 5856 | 1.11 | 3.98 |
| flybrain_imitation | synthetic_routing | 50 | 0.220 | 0.862 | 1741 | 11.34 | 4124 | 0.963 | 4.38 |
| flybrain_rl | synthetic_routing | 50 | 0.120 | 0.856 | 1967 | 12.00 | 2986 | 2.01 | 16.71 |
