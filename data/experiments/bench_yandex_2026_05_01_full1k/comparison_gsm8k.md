| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | gsm8k | 5 | 1.00 | 1.00 | 2586 | 10.60 | 11228 | 2.15 | 2.15 |
| fully_connected | gsm8k | 5 | 1.00 | 1.00 | 3137 | 10.20 | 7822 | 2.57 | 2.57 |
| random_sparse | gsm8k | 5 | 1.00 | 1.00 | 1630 | 6.80 | 5845 | 0.925 | 0.925 |
| degree_preserving | gsm8k | 5 | 1.00 | 1.00 | 754 | 3.00 | 318 | 0.462 | 0.462 |
| learned_router_no_prior | gsm8k | 5 | 0.000 | 0.450 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_imitation | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_rl | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
