| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | bbh_mini | 30 | 1.00 | 1.00 | 4591 | 17.40 | 9417 | 3.90 | 3.90 |
| fully_connected | bbh_mini | 30 | 1.00 | 1.00 | 6434 | 19.50 | 10662 | 5.22 | 5.22 |
| random_sparse | bbh_mini | 30 | 0.367 | 0.883 | 5039 | 20.10 | 9671 | 3.16 | 8.62 |
| degree_preserving | bbh_mini | 30 | 1.00 | 1.00 | 843 | 3.20 | 193 | 0.594 | 0.594 |
| learned_router_no_prior | bbh_mini | 30 | 0.000 | 0.417 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | bbh_mini | 30 | 0.000 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | bbh_mini | 30 | 0.967 | 0.995 | 7194 | 29.17 | 4534 | 4.75 | 4.92 |
| flybrain_imitation | bbh_mini | 30 | 0.967 | 0.995 | 8099 | 32.00 | 6937 | 6.68 | 6.91 |
| flybrain_rl | bbh_mini | 30 | 0.967 | 0.995 | 8097 | 32.00 | 4483 | 6.70 | 6.94 |
