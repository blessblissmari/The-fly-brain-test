| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | gsm8k | 30 | 1.00 | 1.00 | 2873 | 10.87 | 12182 | 2.40 | 2.40 |
| fully_connected | gsm8k | 30 | 1.00 | 1.00 | 3487 | 10.07 | 8978 | 2.83 | 2.83 |
| random_sparse | gsm8k | 30 | 0.933 | 0.973 | 1845 | 7.50 | 6584 | 1.09 | 1.17 |
| degree_preserving | gsm8k | 30 | 1.00 | 1.00 | 867 | 3.07 | 55.97 | 0.524 | 0.524 |
| learned_router_no_prior | gsm8k | 30 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | gsm8k | 30 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | gsm8k | 30 | 1.00 | 1.00 | 2838 | 11.67 | 6831 | 1.73 | 1.73 |
| flybrain_imitation | gsm8k | 30 | 1.00 | 1.00 | 2556 | 11.97 | 2354 | 1.72 | 1.72 |
| flybrain_rl | gsm8k | 30 | 0.767 | 0.965 | 2800 | 12.00 | 1771 | 2.90 | 3.78 |
