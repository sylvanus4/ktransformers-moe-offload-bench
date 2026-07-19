#!/usr/bin/env bash
set -uo pipefail
W=/workspace; L=$W/logs; mkdir -p $L
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/cuda_build.log; }
say "CUDA build start (sm_89) -> build_cuda"
cmake -S $W/llama.cpp -B $W/llama.cpp/build_cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF >>$L/cuda_build.log 2>&1
cmake --build $W/llama.cpp/build_cuda -j32 --target llama-bench llama-cli >>$L/cuda_build.log 2>&1
if ldd $W/llama.cpp/build_cuda/bin/llama-bench 2>/dev/null | grep -qi cudart; then
  say "CUDA_BUILD_OK"
else
  say "CUDA_BUILD_FAIL"; tail -8 $L/cuda_build.log; exit 1
fi
mkdir -p $W/llama.cpp/build/bin
ln -sf $W/llama.cpp/build_cuda/bin/llama-bench $W/llama.cpp/build/bin/llama-bench
ln -sf $W/llama.cpp/build_cuda/bin/llama-cli   $W/llama.cpp/build/bin/llama-cli
say "rerun bench2 (GPU-enabled)"
bash $W/bench2.sh
say "ALL_DONE"
