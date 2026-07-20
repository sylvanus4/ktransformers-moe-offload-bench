# 진짜 ktransformers INT4 AMX 커널 재측정 (2026-07-21)

하드웨어: RunPod H100 + Intel Xeon Platinum 8480+ (Sapphire Rapids, amx_bf16/amx_tile), 2TB RAM.
kt_kernel 0.6.3.post1 소스 빌드(main d1a3ed8), torch 2.9.1+cu128. 모델 geometry = DeepSeek-V3
(hidden 7168, moe_inter 2048, topk 8, 58 MoE layers). decode = qlen=1 per-token, MoE-only(CPU).

## Decode tok/s (MoE-only, V3 geometry, 60 threads = 튜닝값)
| 커널 | per-layer µs | V3 decode tok/s (MoE-only) |
|---|---|---|
| AMXInt4_MOE   | 1386.5 | **12.44** |
| AMXInt8_MOE   | 2886.5 | 5.97 |
| AMXBF16_MOE   | 5365.8 | 3.21 |
| AVX2BF16_MOE  | 5863.2 | 2.94 |

INT4 vs BF16 = 3.87x · INT4 vs AVX2 = 4.23x

## 핵심
- 초판 프록시(llama.cpp 1.2, kt_kernel BF16 3.8, AVX2 2.9)는 위 표의 BF16/AVX2 줄과 일치.
  → 나는 미최적화(BF16) 경로를 쟀고 INT4 AMX 커널을 빠뜨렸다. 사용자 의심이 정확했다.
- 올바른 INT4 커널 = 12.44 tok/s (MoE-only, 60스레드). ktransformers 공식 14~16(4090+듀얼Xeon,
  GPU 오버랩 포함)의 이웃. 112스레드는 6.6으로 하락(qlen=1 메모리바운드 = 스레드 과다 시 NUMA 오버헤드).
- baseline SOSP25 4.68과 내 BF16/AVX2(2.9~3.2)가 같은 미최적화 대역.
