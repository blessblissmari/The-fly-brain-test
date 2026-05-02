| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | _overall | 65 | 0.938 | 0.991 | 601 | 3.28 | 4009 | 0.372 | 0.397 |
| fully_connected | _overall | 65 | 0.969 | 0.995 | 905 | 3.28 | 3145 | 0.494 | 0.509 |
| random_sparse | _overall | 65 | 0.431 | 0.862 | 1035 | 5.60 | 5664 | 0.658 | 1.53 |
| degree_preserving | _overall | 65 | 0.938 | 0.991 | 632 | 3.28 | 0.000 | 0.384 | 0.410 |
| learned_router_no_prior | _overall | 65 | 0.031 | 0.481 | 211 | 0.969 | 513 | 0.084 | 2.74 |
| flybrain_prior_untrained | _overall | 65 | 0.000 | 0.720 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | _overall | 65 | 0.446 | 0.915 | 2219 | 11.57 | 5779 | 1.17 | 2.62 |
| flybrain_imitation | _overall | 65 | 0.369 | 0.889 | 1806 | 11.12 | 3920 | 1.03 | 2.80 |
| flybrain_rl | _overall | 65 | 0.231 | 0.871 | 2126 | 12.00 | 2906 | 2.15 | 9.32 |
