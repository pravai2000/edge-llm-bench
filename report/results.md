| Device | Power mode | Model | Quant | Size (GB) | Prefill t/s | Decode t/s | Mean W | J/token* | PPL |
|---|---|---|---|---|---|---|---|---|---|
| mac-metal | default | llama-3.2-1b | f16 | 2.47 | 2847.2 | 96.1 | 28.35 | 0.295 | 14.34 |
| mac-metal | default | llama-3.2-1b | q8_0 | 1.31 | 2717.1 | 165.7 | 32.16 | 0.194 | 14.35 |
| mac-metal | default | llama-3.2-1b | q4_k_m | 0.80 | 2603.3 | 224.3 | 29.98 | 0.134 | 14.89 |
| mac-metal | default | llama-3.2-1b | q3_k_m | 0.68 | 2499.6 | 166.4 | 32.83 | 0.197 | 16.91 |

*J/token = mean power over the bench window / decode tokens-per-second (approximate).
