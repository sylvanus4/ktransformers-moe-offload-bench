#!/usr/bin/env bash
# offload decode 유효 측정: --no-mmap로 133GB를 RAM에 적재(네트워크FS mmap 병목 제거).
W=/workspace; R=$W/results; L=$W/logs; mkdir -p $R $L
G=/workspace/m/q235/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf
LB=$W/llama.cpp/build/bin/llama-bench
echo "[$(date +%H:%M:%S)] no-mmap decode bench start (load 133GB to RAM)" > $L/nomm.log
CUDA_VISIBLE_DEVICES=0 $LB -m "$G" -ngl 99 -ncmoe 94 --no-mmap -p 64 -n 32 -r 2 -o json 2>>$L/nomm.log > $R/offload_nomm.json
echo "[$(date +%H:%M:%S)] NOMM_DONE" >> $L/nomm.log
python3 -c "import json;d=json.load(open('$R/offload_nomm.json'));[print(('prefill' if r['n_gen']==0 else 'decode'),round(r['avg_ts'],2),'tok/s') for r in d]" | tee -a $L/nomm.log
