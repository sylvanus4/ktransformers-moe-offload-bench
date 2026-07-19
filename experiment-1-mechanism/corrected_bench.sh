#!/usr/bin/env bash
# 수정 메커니즘 셀: llama-bench는 --cpu-moe 미지원 → -ncmoe N 사용. Qwen3=48 layers.
set -uo pipefail
W=/workspace; R=$W/results; L=$W/logs
LB=$W/llama.cpp/build/bin/llama-bench
CLI=$W/llama.cpp/build/bin/llama-cli
G=$(ls $W/m/qwen3-gguf/*.gguf | head -1)
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $L/corrected.log; }

# A2 mechanism = experts of all 48 layers on CPU, attention/rest on GPU (ktransformers trick)
say "A2 mechanism -ncmoe 48"
$LB -m "$G" -ngl 99 -ncmoe 48 -p 512 -n 128 -r 2 -o json 2>>$L/corrected.log > $R/A2_mechanism.json
# A2b partial offload (half) for scaling picture
say "A2b partial -ncmoe 24"
$LB -m "$G" -ngl 99 -ncmoe 24 -p 512 -n 128 -r 2 -o json 2>>$L/corrected.log > $R/A2b_partial.json

# VRAM delta: full-GPU vs mechanism (H1 = experts off GPU -> big VRAM drop)
vram_cap(){ local tag="$1"; shift
  ( for i in $(seq 1 120); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done > $L/vram_$tag.raw ) & local P=$!
  timeout 180 $CLI -m "$G" "$@" -p "Explain the MoE VRAM vs CPU DRAM tradeoff in 4 sentences." -n 48 -no-cnv --temp 0 2>>$L/corrected.log > $R/gen_$tag.txt
  kill $P 2>/dev/null; local peak=$(sort -n $L/vram_$tag.raw 2>/dev/null | tail -1)
  echo "{\"tag\":\"$tag\",\"peak_vram_mb\":${peak:-0}}" > $R/vram_$tag.json; say "$tag peak_vram=${peak}MB"
}
vram_cap fullgpu -ngl 99
vram_cap mechanism -ngl 99 -ncmoe 48

# aggregate
python3 - > $R/summary.json <<PY
import json,os
R="/workspace/results"
def tps(f):
  try:
    d=json.load(open(f)); o={"backend":d[0].get("backends")}
    for r in d:
      if r.get("n_prompt",0)>0 and r.get("n_gen",0)==0: o["prefill_tok_s"]=round(r["avg_ts"],1)
      if r.get("n_gen",0)>0: o["decode_tok_s"]=round(r["avg_ts"],1)
    return o
  except Exception as e: return {"err":str(e)[:60]}
S={"env":json.load(open(f"{R}/env.json"))}
S["A1_fullgpu"]=tps(f"{R}/A1_fullgpu.json")
S["A2_mechanism_ncmoe48"]=tps(f"{R}/A2_mechanism.json")
S["A2b_partial_ncmoe24"]=tps(f"{R}/A2b_partial.json")
S["A3_cpu_only"]=tps(f"{R}/A3_cpu.json")
for t in ["fullgpu","mechanism"]:
  p=f"{R}/vram_{t}.json"
  if os.path.exists(p): S.setdefault("vram_mb",{})[t]=json.load(open(p))["peak_vram_mb"]
try:
  m=S["A2_mechanism_ncmoe48"]["decode_tok_s"]; c=S["A3_cpu_only"]["decode_tok_s"]
  S["H2_mechanism_speedup_vs_cpu"]=round(m/c,2)
except: pass
try:
  gen=open(f"{R}/gen_fullgpu.txt").read().strip(); gen2=open(f"{R}/gen_mechanism.txt").read().strip()
  S["H4_outputs_identical"]=(gen[:200]==gen2[:200])
except: pass
print(json.dumps(S,ensure_ascii=False,indent=2))
PY
say "CORRECTED DONE"
cat $R/summary.json
