# Experiment 4 — the config that actually matters: big AMX server + 1 GPU + local NVMe

The point of ktransformers is a many-core AMX server + big RAM + ONE cheap GPU, not a rented
multi-GPU box. So we measured exactly that.

Host: Intel Xeon Platinum 8570 (AMX bf16/int8/tile, 224 cores), 2015GB RAM, local NVMe, 1×H100
(only ~11GB used for offload, so a 4090 suffices). Model: Qwen3-235B-A22B Q4 on LOCAL disk.

## Measured (two independent methods, they agree: low single digits)

| Method | offload decode |
|---|---|
| llama.cpp `--n-cpu-moe 94` (end-to-end) | 1.2 tok/s |
| kt_kernel real AMX BF16 MoE, per-layer × 94 (MoE-only) | 3.8 tok/s |
| putting some experts on the 24GB GPU (-ncmoe 78) | 1.55 tok/s |

Offload GPU footprint: **11GB** → a 235B model fits a 24GB (even 12GB) card. Hardware claim holds.

kt_kernel AMX BF16 vs AVX2 BF16 for this config: 1.26× (consistent with experiment-2's 1.38×).
The AMX kernel is faster but does NOT escape the single-digit regime: 22B active params computed
on CPU per token is the hard floor. ktransformers' published 8–15 tok/s (for V3) needs INT4 +
GPU-expert placement + pipelining, and is still batch-grade, not interactive.

## Cost (RunPod list, $/1M output tokens)

| Config | hardware | $/hr | tok/s | $/1M tokens |
|---|---|---|---|---|
| Full-GPU | 2×A100 80GB | ~3 | 51.5 | ~16 |
| Offload | AMX server + 1 GPU | ~3 | 2–4 | ~80–280 |

**Corrected verdict:** ktransformers is not a "cheaper serving" or "interactive replacement" tool.
Rented, offload is 5–17× more expensive per token, and a big AMX server isn't cheap either
(RunPod has no cheap-GPU + AMX-server combo). The one real economic case is on-prem CAPEX
avoidance: if you already run a big Xeon server, adding a $1,600 4090 to batch-run a 671B-class
MoE beats buying $30k of A100s. It changes "runs at all vs doesn't", for batch/offline/agentic
workloads — not opex, not real-time serving.
