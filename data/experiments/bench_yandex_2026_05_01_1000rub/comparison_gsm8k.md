| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | gsm8k | 5 | 1.00 | 1.00 | 2468 | 10.20 | 10517 | 2.04 | 2.04 |
| fully_connected | gsm8k | 5 | 1.00 | 1.00 | 2742 | 9.20 | 6845 | 2.23 | 2.23 |
| random_sparse | gsm8k | 5 | 1.00 | 1.00 | 2339 | 8.40 | 8660 | 1.10 | 1.10 |
| degree_preserving | gsm8k | 5 | 1.00 | 1.00 | 847 | 3.40 | 297 | 0.534 | 0.534 |
| learned_router_no_prior | gsm8k | 5 | 0.000 | 0.350 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_imitation | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_rl | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
