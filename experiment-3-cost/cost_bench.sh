#!/usr/bin/env bash
# 실사용 답: 24GB 초과 대형 MoE(Qwen3-235B-A22B Q4)의 실제 서빙 tok/s + 비용.
# full-GPU(2×A100) vs offload(1 GPU + experts=CPU RAM, 24GB 캡). 동일 모델/하드.
set -uo pipefail
W=/workspace; L=$W/logs; R=$W/results; mkdir -p $L $R
export PATH=/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc
export HF_HOME=$W/hf HF_HUB_ENABLE_HF_TRANSFER=1
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/cost.log; }

say "apt cmake"; apt-get update -qq >>$L/setup.log 2>&1; apt-get install -y -qq cmake >>$L/setup.log 2>&1
say "build llama.cpp CUDA sm_80"
[ -d $W/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp $W/llama.cpp >>$L/setup.log 2>&1
if ! ldd $W/llama.cpp/build/bin/llama-bench 2>/dev/null | grep -qi cudart; then
  cmake -S $W/llama.cpp -B $W/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF >>$L/setup.log 2>&1
  cmake --build $W/llama.cpp/build -j64 --target llama-bench llama-cli >>$L/setup.log 2>&1
fi
ldd $W/llama.cpp/build/bin/llama-bench 2>/dev/null | grep -qi cudart && say "llama.cpp CUDA OK" || { say "BUILD FAIL"; exit 1; }

say "pip hf_transfer"; pip -q install hf_transfer "huggingface_hub[cli]" >>$L/setup.log 2>&1
say "download Qwen3-235B-A22B Q4_K_M (~130GB)"
hf download unsloth/Qwen3-235B-A22B-GGUF --include "*Q4_K_M*" --local-dir $W/m/q235 >>$L/dl.log 2>&1 && say "dl done" || say "dl FAIL"
G=$(ls $W/m/q235/**/*Q4_K_M*00001* $W/m/q235/*Q4_K_M*00001* 2>/dev/null | head -1)
[ -z "$G" ] && G=$(find $W/m/q235 -name "*Q4_K_M*.gguf" | sort | head -1)
say "gguf head: $G"
LB=$W/llama.cpp/build/bin/llama-bench
CLI=$W/llama.cpp/build/bin/llama-cli

# n_layers 확인
NL=$(python3 -c "import json,glob;f=glob.glob('$W/m/q235/**/config.json',recursive=True);print(json.load(open(f[0]))['num_hidden_layers'] if f else 94)" 2>/dev/null || echo 94)
say "n_layers=$NL"

# --- full-GPU: 2×A100, 모델 통째 ---
say "FULL-GPU (2×A100, -ngl 99)"
$LB -m "$G" -ngl 99 -p 512 -n 128 -r 2 -o json 2>>$L/cost.log > $R/full_2xa100.json
# --- offload: 1 GPU + 전 experts CPU (24GB급 재현) ---
say "OFFLOAD (1 GPU + -ncmoe $NL, experts=CPU)"
CUDA_VISIBLE_DEVICES=0 $LB -m "$G" -ngl 99 -ncmoe $NL -p 512 -n 128 -r 2 -o json 2>>$L/cost.log > $R/offload_1gpu.json

# --- VRAM 캡처(offload가 24GB에 들어가나) ---
say "VRAM offload capture"
( for i in $(seq 1 240); do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 0.5; done > $L/vram.raw ) & VP=$!
CUDA_VISIBLE_DEVICES=0 $CLI -m "$G" -ngl 99 -ncmoe $NL -p "In one paragraph, explain expert offloading for MoE serving." -n 64 -no-cnv --temp 0 2>>$L/cost.log > $R/offload_gen.txt
kill $VP 2>/dev/null
OFF_VRAM=$(awk -F', ' '$1==0{print $2}' $L/vram.raw | sort -n | tail -1)
say "offload GPU0 peak VRAM=${OFF_VRAM}MB"

# --- 집계 + 비용 ---
python3 - > $R/cost_summary.json <<PY
import json,os
R="$W/results"
def tps(f):
  try:
    d=json.load(open(f)); o={"backend":d[0].get("backends")}
    for r in d:
      if r.get("n_prompt",0)>0 and r.get("n_gen",0)==0: o["prefill_tok_s"]=round(r["avg_ts"],1)
      if r.get("n_gen",0)>0: o["decode_tok_s"]=round(r["avg_ts"],1)
    return o
  except Exception as e: return {"err":str(e)[:80]}
full=tps(f"{R}/full_2xa100.json"); off=tps(f"{R}/offload_1gpu.json")
# RunPod 실가 (secure, 2026-07): A100 SXM ~\$1.49/GPU-hr, 4090 ~\$0.69/hr
PRICE_FULL=2*1.49   # 2×A100
PRICE_OFF=0.69      # 24GB급 1장(4090) + 대용량 RAM 박스 (오프로드가 필요로 하는 실제 하드)
def per_mtok(price_hr,toks): return round(price_hr*1e6/(toks*3600),3) if toks else None
S={"model":"Qwen3-235B-A22B Q4_K_M (~130GB, 24GB/80GB 단일 GPU 초과)",
   "host":"2×A100-SXM4-80GB / AMD EPYC 7742(AVX2, no AMX) / 2004GB RAM",
   "full_gpu_2xa100":full,"offload_1gpu_experts_cpu":off,
   "offload_gpu0_peak_vram_mb":${OFF_VRAM:-0},
   "pricing_usd_per_hr":{"full_2xa100":round(PRICE_FULL,2),"offload_24gb_box":PRICE_OFF},
   "cost_per_1M_tokens_usd":{
      "full_2xa100":per_mtok(PRICE_FULL,full.get("decode_tok_s")),
      "offload_24gb_box":per_mtok(PRICE_OFF,off.get("decode_tok_s"))},
   "note":"offload은 AVX2(AMX 없음) 하한. AMX Xeon이면 experts ~1.4x 더 빠름."}
try:
  S["hardware_cost_ratio_full_over_offload"]=round(PRICE_FULL/PRICE_OFF,1)
  S["speed_ratio_full_over_offload"]=round(full["decode_tok_s"]/off["decode_tok_s"],1)
  cf=S["cost_per_1M_tokens_usd"]["full_2xa100"]; co=S["cost_per_1M_tokens_usd"]["offload_24gb_box"]
  S["per_token_cost_ratio_full_over_offload"]=round(cf/co,2)
except: pass
print(json.dumps(S,ensure_ascii=False,indent=2))
PY
say "COST_DONE"; cat $R/cost_summary.json
