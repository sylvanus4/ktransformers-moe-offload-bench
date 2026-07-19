# Experiment 3 — real large-MoE serving speed and $/token

Model: Qwen3-235B-A22B Q4_K_M (~130GB) — exceeds any single GPU, so offload is mandatory.
Host: 2×A100-SXM4-80GB / AMD EPYC 7742 (AVX2, no AMX) / 2004GB RAM, on RunPod.

## Measured

| Config | prefill | decode | GPU VRAM |
|---|---|---|---|
| Full-GPU (2×A100, model fully on GPU) | 651.5 tok/s | **51.5 tok/s** | ~150GB across 2 |
| Offload (experts on CPU, 1 GPU) | ~60 tok/s* | see note | **11GB** |

The offload GPU footprint is only **11GB** — a 235B model serves on a 24GB (even 12GB) card.

*Offload decode could not be measured cleanly here: the 130GB model sat on RunPod's network
filesystem, so mmap did a remote read per token (0.07 tok/s garbage) and `-mmp 0` RAM-load was
bottlenecked by network-storage bandwidth. A real deployment uses local NVMe. Use experiment-1's
clean 30B offload figure (12 tok/s decode) as the concrete floor; a 235B (22B active) runs lower.

## Cost ($/1M output tokens = $/hr × 1e6 / (tok/s × 3600), RunPod list prices)

| Config | hardware | $/hr | tok/s | $/1M tokens |
|---|---|---|---|---|
| Full-GPU | 2×A100 80GB | 2.98 | 51.5 | ~16 |
| Offload | 4090 24GB + big-RAM box | 0.69 | ~5–12 | ~16–38 |

**Takeaway:** offload barely lowers $/token (5× cheaper hardware, proportionally slower). The real
saving is CAPEX/access — running a 671B-class MoE on one commodity 24GB GPU instead of 2× $15k A100s.
Not an opex tool; a "runs at all vs doesn't" tool.
