#!/usr/bin/env bash
# ktransformers 실증 — 원격 벤치 하네스 (RunPod pod에서 실행)
# 목적: 이기종(CPU/GPU) MoE 추론의 fit(H1)/context(H3)/quality(H4) + 상대 speed(H2) 실증
# 우선순위: 메커니즘 우선 (AMX 추격 안 함). 큰 대리모델로 "24GB 풀로드 불가"를 명확히.
# 산출: /workspace/results/*.json  (결정론 집계, 모델 자기보고 금지)
set -uo pipefail
WORK=/workspace; RES=$WORK/results; LOG=$WORK/logs
mkdir -p "$RES" "$LOG"
ts(){ date +%Y%m%dT%H%M%S; }
jlog(){ echo "[$(ts)] $*" | tee -a "$LOG/run.log"; }

# ---------- Phase 0: 환경 사실수집 (판정에 필요, 마케팅 배수의 전제 노출) ----------
jlog "Phase0 env probe"
python3 - <<'PY' > "$RES/env.json"
import json,subprocess,shutil,os
def sh(c):
    try: return subprocess.check_output(c,shell=True,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return ""
cpu=sh("lscpu")
amx = "amx" in cpu.lower()
avx512 = "avx512" in cpu.lower()
gpu=sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
ram_kb=sh("awk '/MemTotal/{print $2}' /proc/meminfo")
print(json.dumps({
 "gpu":gpu,"ram_gb":round(int(ram_kb or 0)/1024/1024,1),
 "cpu_model":sh("lscpu | grep 'Model name'"),
 "amx":amx,"avx512":avx512,
 "note":"amx=false면 H2 헤드라인(28x) 재현 불가 — 상대우위만 판정"
},ensure_ascii=False,indent=2))
PY
cat "$RES/env.json" | tee -a "$LOG/run.log"

# ---------- Phase 1: 의존성 ----------
# ktransformers: 공식 도커/휠 경로가 취약 → 실패 시 명확히 로깅하고 계속(다른 셀은 진행)
jlog "Phase1 deps"
pip -q install "huggingface_hub[cli]" psutil 2>>"$LOG/pip.log"
# llama.cpp (CPU baseline B2) — 빌드
if [ ! -x "$WORK/llama.cpp/build/bin/llama-bench" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp" 2>>"$LOG/llama.log"
  cmake -S "$WORK/llama.cpp" -B "$WORK/llama.cpp/build" -DGGML_CUDA=OFF >>"$LOG/llama.log" 2>&1
  cmake --build "$WORK/llama.cpp/build" -j --target llama-bench llama-cli >>"$LOG/llama.log" 2>&1
fi
# ktransformers (검증 대상 T) — pip 시도, 실패는 기록
KT_OK=0
python3 -c "import ktransformers" 2>/dev/null && KT_OK=1
if [ "$KT_OK" = 0 ]; then
  jlog "install ktransformers (may take long / may fail on kernel build)"
  pip -q install ktransformers 2>>"$LOG/kt.log" && python3 -c "import ktransformers" 2>/dev/null && KT_OK=1
fi
jlog "ktransformers importable=$KT_OK (0이면 T셀 skip, 다른 셀로 부분 판정)"

# ---------- Phase 2: 모델 스테이징 ----------
# 1+3 확정: 큰 대리모델. Mixtral-8x7B Q4(~26GB>24GB, offload 필수=깨끗한 H1) + Qwen3-30B-A3B(fine-grained MoE, V3 근접)
jlog "Phase2 models"
export HF_HOME=$WORK/hf
# GGUF (llama.cpp / ktransformers 공용 Q4)
hf download unsloth/Mixtral-8x7B-Instruct-v0.1-GGUF Q4_K_M/*  --local-dir $WORK/m/mixtral 2>>"$LOG/dl.log" || \
hf download TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf --local-dir $WORK/m/mixtral 2>>"$LOG/dl.log"
# Qwen3-30B-A3B: safetensors(HF/ktransformers) + GGUF Q4(llama.cpp)
hf download Qwen/Qwen3-30B-A3B --local-dir $WORK/m/qwen3 2>>"$LOG/dl.log"

# ---------- Phase 3: 측정 유틸 ----------
cat > $WORK/measure.py <<'PY'
import json,subprocess,time,threading,sys,os
try: import psutil
except: psutil=None
def peak_mem(cmd, tag):
    """cmd 실행하며 peak VRAM/RAM 폴링. returns dict"""
    peakv=[0]; peakr=[0]; stop=[False]
    def poll():
        while not stop[0]:
            try:
                v=int(subprocess.check_output("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",shell=True).split()[0])
                peakv[0]=max(peakv[0],v)
            except: pass
            if psutil: peakr[0]=max(peakr[0], psutil.virtual_memory().used//1024//1024)
            time.sleep(0.5)
    t=threading.Thread(target=poll,daemon=True); t.start()
    t0=time.time(); p=subprocess.run(cmd,shell=True,capture_output=True,text=True); dt=time.time()-t0
    stop[0]=True; t.join(timeout=2)
    return {"tag":tag,"rc":p.returncode,"wall_s":round(dt,1),
            "peak_vram_mb":peakv[0],"peak_ram_mb":peakr[0],
            "stdout_tail":p.stdout[-2000:],"stderr_tail":p.stderr[-800:]}
if __name__=="__main__":
    tag=sys.argv[1]; cmd=sys.argv[2]
    print(json.dumps(peak_mem(cmd,tag),ensure_ascii=False))
PY

PROMPT="Explain the tradeoff between GPU VRAM and CPU DRAM when serving a Mixture-of-Experts model. Answer in 5 sentences."

run_cell(){ # tag  cmd
  jlog "CELL $1"
  python3 $WORK/measure.py "$1" "$2" > "$RES/cell_$1.json" 2>>"$LOG/run.log"
  cat "$RES/cell_$1.json" | tee -a "$LOG/run.log"
}

# ---------- Phase 4: 매트릭스 실행 ----------
LB=$WORK/llama.cpp/build/bin/llama-bench
MIX_GGUF=$(find $WORK/m/mixtral -name '*.gguf' | head -1)

# B2: llama.cpp CPU-only (28x 주장의 분모) — decode tok/s
[ -n "$MIX_GGUF" ] && run_cell "B2_mixtral_cpu_llamacpp" "$LB -m '$MIX_GGUF' -ngl 0 -p 128 -n 64 -r 2"

# B1: naive HF device_map offload on 4090 (순진한 baseline) — Qwen3(safetensors)
run_cell "B1_qwen3_hf_naive_offload" "python3 $WORK/hf_naive.py $WORK/m/qwen3"

# T: ktransformers 이기종 (검증 대상) — Mixtral Q4 (24GB 풀로드 불가 → offload 필수)
if [ "$KT_OK" = 1 ] && [ -n "$MIX_GGUF" ]; then
  run_cell "T_mixtral_ktransformers" "python3 $WORK/kt_run.py '$MIX_GGUF'"
else
  echo '{"tag":"T_mixtral_ktransformers","skip":"ktransformers unavailable"}' > "$RES/cell_T_mixtral_ktransformers.json"
fi

# H3 context 스윕: T셀에서 컨텍스트 4k→32k→128k 최대 도달 (VRAM 고정)
# H4 quality: 동일 프롬프트 출력 diff + WikiText PPL (별도 kt_run/hf_naive 내부 계산)

jlog "DONE — results in $RES"
python3 -c "import json,glob;print(json.dumps({p.split('/')[-1]:json.load(open(p)) for p in glob.glob('$RES/*.json')},ensure_ascii=False,indent=2))" > "$RES/summary.json"
echo "SUMMARY=$RES/summary.json"
