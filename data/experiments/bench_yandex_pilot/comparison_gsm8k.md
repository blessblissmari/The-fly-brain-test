| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | gsm8k | 5 | 1.00 | 1.00 | 759 | 3.00 | 4800 | 0.461 | 0.461 |
| fully_connected | gsm8k | 5 | 1.00 | 1.00 | 865 | 3.00 | 2348 | 0.504 | 0.504 |
| random_sparse | gsm8k | 5 | 0.800 | 0.970 | 974 | 3.60 | 4275 | 0.527 | 0.659 |
| degree_preserving | gsm8k | 5 | 1.00 | 1.00 | 759 | 3.00 | 0.000 | 0.461 | 0.461 |
| learned_router_no_prior | gsm8k | 5 | 0.000 | 0.450 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | gsm8k | 5 | 1.00 | 1.00 | 1799 | 8.00 | 3992 | 1.13 | 1.13 |
| flybrain_imitation | gsm8k | 5 | 1.00 | 1.00 | 2390 | 12.00 | 3961 | 1.98 | 1.98 |
| flybrain_rl | gsm8k | 5 | 1.00 | 1.00 | 2623 | 12.00 | 5196 | 2.04 | 2.04 |
