| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | gsm8k | 3 | 1.00 | 1.00 | 2002 | 8.00 | 11402 | 1.36 | 1.36 |
| fully_connected | gsm8k | 3 | 1.00 | 1.00 | 2439 | 7.67 | 7904 | 1.84 | 1.84 |
| random_sparse | gsm8k | 3 | 1.00 | 1.00 | 1118 | 5.33 | 5183 | 0.712 | 0.712 |
| degree_preserving | gsm8k | 3 | 1.00 | 1.00 | 717 | 3.00 | 0.000 | 0.432 | 0.432 |
| learned_router_no_prior | gsm8k | 3 | 0.000 | 0.350 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | gsm8k | 3 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | gsm8k | 3 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_imitation | gsm8k | 3 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_rl | gsm8k | 3 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
