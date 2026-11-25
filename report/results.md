| Device | Power mode | Model | Quant | Size (GB) | Prefill t/s | Decode t/s | Run σ (n)*** | Eff. GB/s** | Mean W | J/token* | PPL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-metal | default | llama-3.1-8b | q8_0 | 8.53 | 376.6 | 29.0 | 0.5 (2) | 248 | 32.78 | 1.130 | 7.12 |
| mac-metal | default | llama-3.1-8b | q4_k_m | 4.91 | 361.8 | 42.1 | 1.3 (2) | 207 | 31.55 | 0.749 | 7.30 |
| mac-metal | default | llama-3.1-8b | q3_k_m | 4.01 | 341.2 | 29.4 | 0.6 (2) | 118 | 35.98 | 1.225 | 7.64 |
| mac-metal | default | llama-3.2-1b | f16 | 2.47 | 2793.2 | 94.0 | 3.1 (2) | 232 | 30.55 | 0.326 | 14.34 |
| mac-metal | default | llama-3.2-1b | q8_0 | 1.31 | 2664.9 | 161.3 | 6.3 (2) | 212 | 33.60 | 0.209 | 14.35 |
| mac-metal | default | llama-3.2-1b | q4_k_m | 0.80 | 2556.6 | 219.4 | 6.9 (2) | 176 | 31.36 | 0.143 | 14.89 |
| mac-metal | default | llama-3.2-1b | q4_k_m_imat | 0.80 | 2499.3 | 214.5 | 1.2 (3) | 172 | 33.09 | 0.154 | 14.80 |
| mac-metal | default | llama-3.2-1b | iq4_xs_imat | 0.74 | 2557.5 | 202.0 | 1.6 (3) | 148 | 34.14 | 0.169 | 14.99 |
| mac-metal | default | llama-3.2-1b | q3_k_m | 0.68 | 2379.9 | 158.0 | 11.9 (2) | 108 | 35.29 | 0.225 | 16.91 |
| mac-metal | default | llama-3.2-1b | q3_k_m_imat | 0.68 | 2401.3 | 159.8 | 0.8 (3) | 109 | 37.19 | 0.233 | 16.06 |
| mac-metal | default | llama-3.2-1b-mlx | mlx_8bit | 1.31 | 2461.6 | 148.7 | - (1) | 195 | 17.84 | 0.120 | - |
| mac-metal | default | llama-3.2-1b-mlx | mlx_4bit | 0.70 | 2505.6 | 243.5 | - (1) | 169 | 17.84 | 0.073 | - |
| mac-metal | default | llama-3.2-3b | f16 | 6.43 | 958.5 | 36.7 | 1.6 (3) | 236 | 35.10 | 0.957 | 10.57 |
| mac-metal | default | llama-3.2-3b | q8_0 | 3.41 | 911.4 | 64.9 | 2.5 (3) | 222 | 33.13 | 0.511 | 10.58 |
| mac-metal | default | llama-3.2-3b | q4_k_m | 2.01 | 873.4 | 93.4 | 3.1 (3) | 188 | 32.29 | 0.346 | 10.87 |
| mac-metal | default | llama-3.2-3b | q3_k_m | 1.68 | 843.9 | 67.9 | 1.4 (3) | 114 | 36.03 | 0.531 | 11.81 |
| mac-metal | default | llama-3.2-3b-mlx | mlx_8bit | 3.41 | 798.7 | 59.5 | - (1) | 203 | 19.97 | 0.336 | - |
| mac-metal | default | llama-3.2-3b-mlx | mlx_4bit | 1.81 | 862.0 | 105.0 | - (1) | 190 | 21.75 | 0.207 | - |

*J/token = mean power over the bench window / decode tokens-per-second (approximate).
**Eff. GB/s = decode tokens/s x model size: the weight bytes streamed per second.
***Run σ (n) = standard deviation of decode tokens/s across n repeated runs; metrics are means over runs.
