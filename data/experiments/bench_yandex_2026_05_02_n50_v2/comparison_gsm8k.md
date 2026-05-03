| Method | Benchmark | Tasks | Success | Verifier | Tokens/task | Calls/task | Latency (ms) | Cost/task ₽ | Cost/solved ₽ |
|---|---|---|---|---|---|---|---|---|---|
| manual_graph | gsm8k | 5 | 1.00 | 1.00 | 2582 | 10.20 | 11460 | 2.01 | 2.01 |
| fully_connected | gsm8k | 5 | 1.00 | 1.00 | 2578 | 6.80 | 7977 | 1.91 | 1.91 |
| random_sparse | gsm8k | 5 | 1.00 | 1.00 | 1534 | 6.60 | 5891 | 1.04 | 1.04 |
| degree_preserving | gsm8k | 5 | 1.00 | 1.00 | 753 | 3.00 | 0.000 | 0.462 | 0.462 |
| learned_router_no_prior | gsm8k | 5 | 0.000 | 0.350 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_prior_untrained | gsm8k | 5 | 0.000 | 0.850 | 0.000 | 0.000 | 0.000 | 0.000 | ∞ |
| flybrain_sim_pretrain | gsm8k | 5 | 1.00 | 1.00 | 2714 | 12.00 | 6811 | 1.65 | 1.65 |
| flybrain_imitation | gsm8k | 5 | 1.00 | 1.00 | 2547 | 12.00 | 1810 | 1.55 | 1.55 |
| flybrain_rl | gsm8k | 5 | 0.600 | 0.940 | 2630 | 12.00 | 1020 | 2.94 | 4.90 |
