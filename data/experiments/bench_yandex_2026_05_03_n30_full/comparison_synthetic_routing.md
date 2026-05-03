| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | synthetic_routing | 30 | 0.833 | 0.975 | 1858 | 11.03 | 8550 | 1.65 | 1.99 |
| fully_connected | synthetic_routing | 30 | 0.933 | 0.990 | 2566 | 10.57 | 8255 | 2.15 | 2.30 |
| random_sparse | synthetic_routing | 30 | 0.133 | 0.770 | 1032 | 7.53 | 4026 | 0.656 | 4.92 |
| degree_preserving | synthetic_routing | 30 | 1.00 | 1.00 | 812 | 4.20 | 889 | 0.533 | 0.533 |
| learned_router_no_prior | synthetic_routing | 30 | 0.033 | 0.532 | 8.70 | 0.067 | 104 | 0.003 | 0.104 |
| flybrain_prior_untrained | synthetic_routing | 30 | 0.000 | 0.735 | 0.000 | 0.000 | 0.033 | 0.000 | ∞ |
| flybrain_sim_pretrain | synthetic_routing | 30 | 0.133 | 0.870 | 2017 | 12.00 | 5335 | 1.15 | 8.59 |
| flybrain_imitation | synthetic_routing | 30 | 0.267 | 0.890 | 1296 | 7.37 | 2318 | 0.758 | 2.84 |
| flybrain_rl | synthetic_routing | 30 | 0.067 | 0.855 | 1952 | 12.00 | 1525 | 2.10 | 31.44 |
