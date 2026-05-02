| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | synthetic_routing | 3 | 1.00 | 1.00 | 1342 | 8.00 | 6981 | 1.12 | 1.12 |
| fully_connected | synthetic_routing | 3 | 1.00 | 1.00 | 2140 | 9.33 | 7767 | 1.66 | 1.66 |
| random_sparse | synthetic_routing | 3 | 0.333 | 0.850 | 1323 | 9.33 | 6207 | 0.971 | 2.91 |
| degree_preserving | synthetic_routing | 3 | 1.00 | 1.00 | 559 | 3.33 | 0.000 | 0.336 | 0.336 |
| learned_router_no_prior | synthetic_routing | 3 | 0.000 | 0.617 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | synthetic_routing | 3 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | synthetic_routing | 3 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_imitation | synthetic_routing | 3 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_rl | synthetic_routing | 3 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
