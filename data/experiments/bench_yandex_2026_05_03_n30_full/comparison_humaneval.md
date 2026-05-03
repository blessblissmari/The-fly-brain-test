| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | humaneval | 30 | 0.967 | 0.995 | 3627 | 11.90 | 12380 | 3.18 | 3.29 |
| fully_connected | humaneval | 30 | 1.00 | 1.00 | 4949 | 11.90 | 12562 | 4.07 | 4.07 |
| random_sparse | humaneval | 30 | 0.000 | 0.682 | 2288 | 7.40 | 6795 | 1.39 | ∞ |
| degree_preserving | humaneval | 30 | 0.967 | 0.995 | 4061 | 10.53 | 9319 | 2.68 | 2.77 |
| learned_router_no_prior | humaneval | 30 | 0.000 | 0.410 | 44.27 | 0.100 | 412 | 0.018 | ∞ |
| flybrain_prior_untrained | humaneval | 30 | 0.000 | 0.550 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | humaneval | 30 | 0.700 | 0.955 | 3942 | 12.00 | 8214 | 1.99 | 2.85 |
| flybrain_imitation | humaneval | 30 | 0.733 | 0.960 | 2155 | 6.17 | 3738 | 1.27 | 1.73 |
| flybrain_rl | humaneval | 30 | 0.033 | 0.805 | 3464 | 12.00 | 2196 | 3.73 | 112 |
