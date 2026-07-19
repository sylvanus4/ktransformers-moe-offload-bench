#!/usr/bin/env bash
# 진짜 구성: AMX 서버 CPU(Xeon 8570) + 대용량 RAM + GPU 1장 + 로컬 NVMe.
# Qwen3-235B-A22B Q4 오프로드(experts=CPU/AMX, attention=GPU) 실제 decode tok/s.
set -uo pipefail
W=/root; L=$W/logs; R=$W/results; mkdir -p $L $R
export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/real.log; }

say "apt cmake"; apt-get update -qq >>$L/setup.log 2>&1; apt-get install -y -qq cmake >>$L/setup.log 2>&1
say "build llama.cpp CUDA sm_90"
[ -d $W/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp $W/llama.cpp >>$L/setup.log 2>&1
if ! ldd $W/llama.cpp/build/bin/llama-bench 2>/dev/null | grep -qi cudart; then
  cmake -S $W/llama.cpp -B $W/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DLLAMA_CURL=OFF >>$L/setup.log 2>&1
  cmake --build $W/llama.cpp/build -j64 --target llama-bench llama-cli >>$L/setup.log 2>&1
fi
ldd $W/llama.cpp/build/bin/llama-bench 2>/dev/null | grep -qi cudart && say "llama.cpp CUDA OK" || { say "BUILD FAIL"; exit 1; }

say "pip hf_transfer"; pip -q install hf_transfer "huggingface_hub[cli]" >>$L/setup.log 2>&1
say "download Qwen3-235B-A22B Q4_K_M to LOCAL /root/m (~130GB)"
hf download unsloth/Qwen3-235B-A22B-GGUF --include "*Q4_K_M*" --local-dir $W/m/q235 >>$L/dl.log 2>&1 && say "dl done" || { say "dl FAIL"; exit 1; }
G=$(find $W/m/q235 -name "*Q4_K_M*00001*.gguf" | head -1); [ -z "$G" ] && G=$(find $W/m/q235 -name "*Q4_K_M*.gguf"|sort|head -1)
say "gguf: $G  (local disk)"
LB=$W/llama.cpp/build/bin/llama-bench
NL=94

# 오프로드 decode 측정 (로컬 디스크 → 네트워크FS 병목 없음, AMX 커널)
bench(){ # tag  ncmoe
  say "OFFLOAD -ncmoe $2 (experts→CPU/AMX)"
  CUDA_VISIBLE_DEVICES=0 $LB -m "$G" -ngl 99 -ncmoe $2 -p 128 -n 64 -r 2 -o json 2>>$L/real.log > $R/off_$1.json
  python3 -c "import json;d=json.load(open('$R/off_$1.json'));print('$1:',{('pp' if r['n_gen']==0 else 'tg'):round(r['avg_ts'],2) for r in d})" | tee -a $L/real.log
}
# 전 experts CPU = GPU ~11GB (4090 최소). 그리고 일부 experts를 GPU에 = 24GB 활용.
bench full_cpu 94
bench part_24g 78   # 일부 층 experts를 GPU로(24GB급 활용) — 더 빠를 것

# VRAM 캡처 (full_cpu가 4090에 들어가나)
( for i in $(seq 1 200); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done > $L/vram.raw ) & VP=$!
CUDA_VISIBLE_DEVICES=0 $W/llama.cpp/build/bin/llama-cli -m "$G" -ngl 99 -ncmoe 94 -p "hello" -n 8 -no-cnv 2>>$L/real.log >/dev/null
kill $VP 2>/dev/null
echo "FULLCPU_VRAM_MB=$(sort -n $L/vram.raw|tail -1)" > $R/vram.txt; cat $R/vram.txt | tee -a $L/real.log

say "REAL_DONE"
