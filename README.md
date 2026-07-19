# ktransformers "28×" — a reproduction benchmark

We rented GPUs on RunPod for ~$5 and measured the viral ktransformers claims for ourselves:
a 24GB card replacing a $400K rack, and up to 28× speedup for MoE inference. The trick is real.
The headline numbers rest on three hidden preconditions. This repo is the full reproduction harness.

Full write-up (Korean): https://thakicloud.github.io/ko/llmops/ktransformers-moe-offload-28x-validation/

## TL;DR findings

| Measurement | Result |
|---|---|
| Expert-offload mechanism (experts on CPU, attention on GPU) vs pure CPU | **1.62×** decode |
| Full-GPU vs the offload mechanism (when the model fits in VRAM) | **22×** faster |
| AMX kernel vs AVX2 kernel (same BF16, DeepSeek-V3-scale MoE) | **1.38×** |

The "28×" is a **system comparison** (full ktransformers stack on AMX+GPU vs llama.cpp CPU-only),
not a single kernel multiple. It decomposes into modest factors: GPU handling attention (biggest
lever), the AMX expert kernel (~1.4×), and INT-quantization. They only multiply into double digits
when the model exceeds VRAM, the CPU has AMX, and the baseline is pure-CPU llama.cpp.

"24GB in one card" moves memory rather than removing it: DeepSeek-V3 still needs ~380GB of CPU DRAM.

## What's here

- `experiment-1-mechanism/` — RunPod RTX 4090 + AMD Ryzen 9 7950X (no AMX), 188GB RAM.
  Uses llama.cpp `--n-cpu-moe` (the same expert-offload trick) on Qwen3-30B-A3B Q4.
  - `remote_bench.sh`, `corrected_bench.sh` — the benchmark cells (full-GPU / mechanism / CPU-only)
  - `rebuild_cuda.sh` — build llama.cpp with CUDA (sm_89)
  - `hf_naive.py`, `kt_run.py` — HF device_map baseline and a ktransformers invocation stub
- `experiment-2-amx-kernel/` — RunPod H100 + Intel Xeon Platinum 8470 (Sapphire Rapids, AMX), 1TB RAM.
  Compares `AMXBF16_MOE` vs `AVX2BF16_MOE` in `kt_kernel` on identical BF16 weights.
  - `amx_bench.py` — the AMX-vs-AVX2 kernel benchmark
  - `inspect_backends.py` — enumerate the kernel classes each backend exposes
- `results/` — raw JSON from the runs
- `RESULTS.md` — the detailed lab notes (Korean)

## Reproduce

Experiment 1 (any 24GB GPU + CPU with enough RAM):
```bash
# on the pod
bash rebuild_cuda.sh          # build llama.cpp with CUDA
bash corrected_bench.sh       # runs full-GPU / mechanism (-ncmoe 48) / CPU-only, writes results/
```

Experiment 2 (needs an Intel Sapphire Rapids+ host with AMX):
```bash
pip install kt_kernel
python3 amx_bench.py          # AMXBF16_MOE vs AVX2BF16_MOE, DeepSeek-V3-scale MoE
```

## Caveats

The `kt_run.py` end-to-end ktransformers path is a stub. ktransformers v0.6.x moved to a
`kt-kernel` + `balance_serve` serving stack; the pip `ktransformers` package is an SFT shim,
so a real end-to-end run needs the full source build. We measured the mechanism via llama.cpp's
equivalent and the AMX advantage via `kt_kernel` directly, which is cleaner for isolating each factor.

Numbers are from single runs on rented hardware; treat them as directional, not authoritative.

## Credit

Builds on [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) (Apache-2.0,
Tsinghua MADSYS lab). This repo only measures and decomposes their published claims.

## License

Apache-2.0.
