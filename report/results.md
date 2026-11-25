| Device | Power mode | Model | Quant | Size (GB) | Prefill t/s | Decode t/s | Run σ (n)*** | Eff. GB/s** | Mean W | J/token* | PPL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-metal | default | llama-3.1-8b | q8_0 | 8.53 | 376.6 | 29.0 | 0.5 (2) | 248 | 32.78 | 1.130 | 7.12 |
| mac-metal | default | llama-3.1-8b | q4_k_m | 4.91 | 361.8 | 42.1 | 1.3 (2) | 207 | 31.55 | 0.749 | 7.30 |
| mac-metal | default | llama-3.1-8b | q3_k_m | 4.01 | 341.2 | 29.4 | 0.6 (2) | 118 | 35.98 | 1.225 | 7.64 |
| mac-metal | default | llama-3.2-1b | f16 | 2.47 | 2847.2 | 96.1 | - (1) | 238 | 28.35 | 0.295 | 14.34 |
| mac-metal | default | llama-3.2-1b | q8_0 | 1.31 | 2717.1 | 165.7 | - (1) | 218 | 32.16 | 0.194 | 14.35 |
| mac-metal | default | llama-3.2-1b | q4_k_m | 0.80 | 2603.3 | 224.3 | - (1) | 179 | 29.98 | 0.134 | 14.89 |
| mac-metal | default | llama-3.2-1b | q4_k_m_imat | 0.80 | 2488.3 | 213.1 | - (1) | 170 | 33.04 | 0.155 | 14.80 |
| mac-metal | default | llama-3.2-1b | iq4_xs_imat | 0.74 | 2548.4 | 200.2 | - (1) | 147 | 33.24 | 0.166 | 14.99 |
| mac-metal | default | llama-3.2-1b | q3_k_m | 0.68 | 2499.6 | 166.4 | - (1) | 114 | 32.83 | 0.197 | 16.91 |
| mac-metal | default | llama-3.2-1b | q3_k_m_imat | 0.68 | 2389.0 | 158.9 | - (1) | 108 | 35.84 | 0.226 | 16.06 |
| mac-metal | default | llama-3.2-3b | f16 | 6.43 | 983.9 | 38.5 | - (1) | 247 | 37.08 | 0.964 | 10.57 |
| mac-metal | default | llama-3.2-3b | q8_0 | 3.41 | 931.5 | 67.7 | - (1) | 231 | 32.01 | 0.473 | 10.58 |
| mac-metal | default | llama-3.2-3b | q4_k_m | 2.01 | 895.8 | 96.3 | - (1) | 194 | 30.16 | 0.313 | 10.87 |
| mac-metal | default | llama-3.2-3b | q3_k_m | 1.68 | 864.7 | 69.6 | - (1) | 117 | 33.90 | 0.487 | 11.81 |

*J/token = mean power over the bench window / decode tokens-per-second (approximate).
**Eff. GB/s = decode tokens/s x model size: the weight bytes streamed per second.
***Run σ (n) = standard deviation of decode tokens/s across n repeated runs; metrics are means over runs.
