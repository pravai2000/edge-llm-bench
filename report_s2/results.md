| Device | Power mode | Model | Quant | Size (GB) | Prefill t/s | Decode t/s | Run σ (n)*** | Eff. GB/s** | Mean W | CPU W | GPU W | J/token* | PPL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mac-metal | default | llama-3.2-1b | f16 | 2.47 | 2526.1 | 83.5 | 4.5 (2) | 206 | 19.88 | 3.3 | 16.6 | 0.239 | 14.34 |
| mac-metal | default | llama-3.2-1b | q8_0 | 1.31 | 2431.9 | 143.2 | 6.7 (2) | 188 | 22.88 | 3.3 | 19.6 | 0.160 | 14.35 |
| mac-metal | default | llama-3.2-1b | q4_k_m | 0.80 | 2320.1 | 197.7 | 7.1 (2) | 158 | 23.29 | 3.2 | 19.8 | 0.118 | 14.89 |
| mac-metal | default | llama-3.2-1b | q4_k_m_imat | 0.80 | 2335.4 | 200.0 | 10.3 (2) | 160 | 24.11 | 3.1 | 21.0 | 0.121 | 14.80 |
| mac-metal | default | llama-3.2-1b | iq4_xs_imat | 0.74 | 2382.5 | 190.8 | 9.4 (2) | 140 | 23.17 | 2.8 | 20.4 | 0.122 | 14.99 |
| mac-metal | default | llama-3.2-1b | q3_k_m | 0.68 | 2227.4 | 148.1 | 4.7 (2) | 101 | 24.00 | 3.0 | 20.9 | 0.162 | 16.91 |
| mac-metal | default | llama-3.2-1b | q3_k_m_imat | 0.68 | 2238.2 | 149.9 | 7.2 (2) | 102 | 24.06 | 2.8 | 21.2 | 0.161 | 16.06 |
| mac-metal | default | llama-3.2-1b-mlx | mlx_8bit | 1.31 | 2481.4 | 149.6 | 0.2 (2) | 196 | 18.85 | 3.1 | 15.7 | 0.126 | 14.36 |
| mac-metal | default | llama-3.2-1b-mlx | mlx_4bit | 0.70 | 2521.2 | 247.0 | 0.0 (2) | 172 | 19.37 | 3.8 | 15.5 | 0.078 | 17.27 |
| mac-metal | default | llama-3.2-3b | f16 | 6.43 | 882.8 | 32.7 | 2.2 (2) | 210 | 20.93 | 3.2 | 17.5 | 0.643 | 10.57 |
| mac-metal | default | llama-3.2-3b | q8_0 | 3.41 | 840.0 | 57.4 | 4.7 (2) | 196 | 23.59 | 3.1 | 20.3 | 0.413 | 10.58 |
| mac-metal | default | llama-3.2-3b | q4_k_m | 2.01 | 851.5 | 90.3 | 5.5 (2) | 182 | 22.42 | 1.9 | 20.1 | 0.250 | 10.87 |
| mac-metal | default | llama-3.2-3b | q3_k_m | 1.68 | 821.5 | 66.3 | 3.5 (2) | 111 | 24.11 | 1.8 | 21.8 | 0.366 | 11.81 |
| mac-metal | default | llama-3.2-3b-mlx | mlx_8bit | 3.41 | 813.1 | 60.1 | 0.3 (2) | 205 | 20.23 | 3.4 | 16.8 | 0.337 | 10.62 |
| mac-metal | default | llama-3.2-3b-mlx | mlx_4bit | 1.81 | 827.9 | 101.7 | 5.0 (2) | 184 | 23.50 | 3.6 | 19.9 | 0.232 | 11.48 |

*J/token = mean power over the bench window / decode tokens-per-second (approximate).
**Eff. GB/s = decode tokens/s x model size: the weight bytes streamed per second.
***Run σ (n) = standard deviation of decode tokens/s across n repeated runs; metrics are means over runs.
