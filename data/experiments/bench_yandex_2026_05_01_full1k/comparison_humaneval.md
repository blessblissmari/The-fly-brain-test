| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | humaneval | 5 | 1.00 | 1.00 | 2896 | 11.40 | 12812 | 2.54 | 2.54 |
| fully_connected | humaneval | 5 | 1.00 | 1.00 | 3884 | 11.20 | 9410 | 3.17 | 3.17 |
| random_sparse | humaneval | 5 | 0.000 | 0.730 | 2105 | 9.00 | 7267 | 1.15 | ∞ |
| degree_preserving | humaneval | 5 | 1.00 | 1.00 | 2974 | 9.60 | 5415 | 2.06 | 2.06 |
| learned_router_no_prior | humaneval | 5 | 0.000 | 0.530 | 40.00 | 0.200 | 231 | 0.016 | ∞ |
| flybrain_prior_untrained | humaneval | 5 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | humaneval | 5 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_imitation | humaneval | 5 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_rl | humaneval | 5 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
