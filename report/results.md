| Device | Power mode | Model | Quant | Size (GB) | Prefill t/s | Decode t/s | Eff. GB/s** | Mean W | J/token* | PPL |
|---|---|---|---|---|---|---|---|---|---|---|
| mac-metal | default | llama-3.1-8b | q8_0 | 8.53 | 378.6 | 29.4 | 251 | 32.89 | 1.119 | 7.12 |
| mac-metal | default | llama-3.1-8b | q4_k_m | 4.91 | 363.7 | 43.0 | 212 | 31.39 | 0.729 | 7.30 |
| mac-metal | default | llama-3.1-8b | q3_k_m | 4.01 | 341.8 | 28.9 | 116 | 35.23 | 1.218 | 7.64 |
| mac-metal | default | llama-3.2-1b | f16 | 2.47 | 2847.2 | 96.1 | 238 | 28.35 | 0.295 | 14.34 |
| mac-metal | default | llama-3.2-1b | q8_0 | 1.31 | 2717.1 | 165.7 | 218 | 32.16 | 0.194 | 14.35 |
| mac-metal | default | llama-3.2-1b | q4_k_m | 0.80 | 2603.3 | 224.3 | 179 | 29.98 | 0.134 | 14.89 |
| mac-metal | default | llama-3.2-1b | q3_k_m | 0.68 | 2499.6 | 166.4 | 114 | 32.83 | 0.197 | 16.91 |
| mac-metal | default | llama-3.2-3b | f16 | 6.43 | 983.9 | 38.5 | 247 | 37.08 | 0.964 | 10.57 |
| mac-metal | default | llama-3.2-3b | q8_0 | 3.41 | 931.5 | 67.7 | 231 | 32.01 | 0.473 | 10.58 |
| mac-metal | default | llama-3.2-3b | q4_k_m | 2.01 | 895.8 | 96.3 | 194 | 30.16 | 0.313 | 10.87 |
| mac-metal | default | llama-3.2-3b | q3_k_m | 1.68 | 864.7 | 69.6 | 117 | 33.90 | 0.487 | 11.81 |

*J/token = mean power over the bench window / decode tokens-per-second (approximate).
**Eff. GB/s = decode tokens/s x model size: the weight bytes streamed per second.
