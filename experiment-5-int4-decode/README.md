# Experiment 5 — the correction: real INT4 AMX decode

Hardware: RunPod H100 + Intel Xeon Platinum 8480+ (Sapphire Rapids, amx_bf16/amx_tile), 2TB RAM.
Real kt-kernel 0.6.3 built from source (ktransformers main), torch 2.9.1+cu128.

We reproduce our own earlier mistake and fix it. The first run measured BF16 kernels with random
weights (INT4 path effectively off) and extrapolated per-layer MoE latency. Here we take the proper
`AMXInt4_MOE(config)` + `load_weights_task()` path (kt-kernel quantizes fp32 -> INT4 internally) and
measure decode (qlen=1) per-token latency on DeepSeek-V3 geometry (hidden 7168, moe_inter 2048,
topk 8), scaled by 58 MoE layers.

## Result (60 threads, tuned for qlen=1)

| Kernel | decode tok/s (V3 geom, MoE-only) |
|---|---|
| AMXInt4_MOE  | 12.4 |
| AMXInt8_MOE  | 6.0 |
| AMXBF16_MOE  | 3.2 |
| AVX2BF16_MOE | 2.9 |

INT4 vs BF16 = 3.9x, INT4 vs AVX2 = 4.2x.

Notes:
- 112 threads was *slower* (6.6) — qlen=1 is memory-bandwidth bound, so oversubscribing threads adds
  NUMA sync cost. 60 threads is the tuned point.
- This is MoE-only on CPU. In the real serving pipeline attention + shared expert overlap on the GPU,
  landing near ktransformers' published ~14-16 tok/s (4090 + dual Xeon 6454S).
- `decode_int4.json` here is the 112-thread run object; the 60-thread headline numbers are in SUMMARY.md.

Run: `python3 decode_int4_bench.py` (needs a source-built kt-kernel with AMX on a Sapphire/Emerald Rapids Xeon).
