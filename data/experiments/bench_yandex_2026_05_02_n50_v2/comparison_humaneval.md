| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | humaneval | 5 | 1.00 | 1.00 | 2701 | 10.60 | 12457 | 2.27 | 2.27 |
| fully_connected | humaneval | 5 | 1.00 | 1.00 | 3855 | 11.20 | 12007 | 3.02 | 3.02 |
| random_sparse | humaneval | 5 | 0.000 | 0.570 | 776 | 2.80 | 2779 | 0.358 | ∞ |
| degree_preserving | humaneval | 5 | 1.00 | 1.00 | 2287 | 6.80 | 3706 | 1.38 | 1.38 |
| learned_router_no_prior | humaneval | 5 | 0.000 | 0.480 | 40.00 | 0.200 | 271 | 0.016 | ∞ |
| flybrain_prior_untrained | humaneval | 5 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | humaneval | 5 | 1.00 | 1.00 | 3266 | 12.00 | 6430 | 1.50 | 1.50 |
| flybrain_imitation | humaneval | 5 | 0.800 | 0.970 | 1620 | 4.80 | 3121 | 0.840 | 1.05 |
| flybrain_rl | humaneval | 5 | 0.000 | 0.790 | 2886 | 12.00 | 1015 | 3.23 | ∞ |
