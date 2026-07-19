#!/usr/bin/env python3
# 실제 ktransformers 커널(kt_kernel AMX INT8/BF16)로 Qwen3-235B-A22B decode 추정.
# llama.cpp --n-cpu-moe는 AMX INT 타일을 안 써서 과소평가 → 여기서 진짜 커널로 측정.
# per-layer MoE forward(qlen=1) latency × 94 layers = per-token MoE latency → decode tok/s.
import os, sys, time, json, statistics
os.environ.setdefault("KT_KERNEL_CPU_VARIANT", "amx")
import torch, kt_kernel
ext = kt_kernel.kt_kernel_ext

# Qwen3-235B-A22B
EXPERT_NUM=128; HIDDEN=4096; INTER=1536; TOPK=8; LAYERS=94
THREADS=int(os.environ.get("KT_THREADS","112")); MAXLEN=4096
WARMUP,ITERS=5,30
CPUInfer=ext.CPUInfer(THREADS)
p2l=torch.tensor(range(EXPERT_NUM),dtype=torch.int64).contiguous()

def build(cls):
    gate=torch.randn((EXPERT_NUM,HIDDEN,INTER),dtype=torch.bfloat16).contiguous()
    up  =torch.randn((EXPERT_NUM,HIDDEN,INTER),dtype=torch.bfloat16).contiguous()
    down=torch.randn((EXPERT_NUM,INTER,HIDDEN),dtype=torch.bfloat16).contiguous()
    cfg=ext.moe.MOEConfig(EXPERT_NUM,TOPK,HIDDEN,INTER,0)
    cfg.max_len=MAXLEN; cfg.gate_proj=gate.data_ptr(); cfg.up_proj=up.data_ptr(); cfg.down_proj=down.data_ptr()
    cfg.gate_scale=0; cfg.pool=CPUInfer.backend_
    moe=getattr(ext.moe,cls)(cfg)
    CPUInfer.submit(moe.load_weights_task(p2l.data_ptr())); CPUInfer.sync()
    try: CPUInfer.submit(moe.warm_up_task()); CPUInfer.sync()
    except: pass
    return moe,(gate,up,down)

def per_layer_tps(cls):
    moe,_r=build(cls); q=1
    bsz=torch.tensor([q]).contiguous()
    eids=torch.stack([torch.randperm(EXPERT_NUM)[:TOPK] for _ in range(q)]).contiguous()
    w=torch.rand((q,TOPK),dtype=torch.float32).contiguous()
    x=(torch.randn((q,HIDDEN),dtype=torch.bfloat16)/100).contiguous()
    out=torch.empty((q,HIDDEN),dtype=torch.bfloat16).contiguous()
    def one():
        CPUInfer.submit(moe.forward_task(bsz.data_ptr(),TOPK,eids.data_ptr(),w.data_ptr(),x.data_ptr(),out.data_ptr(),False)); CPUInfer.sync()
    for _ in range(WARMUP): one()
    ts=[]
    for _ in range(ITERS):
        t=time.perf_counter(); one(); ts.append(time.perf_counter()-t)
    lat=statistics.mean(ts)  # sec per layer per token
    return lat

R={"model":"Qwen3-235B-A22B","dims":{"experts":EXPERT_NUM,"hidden":HIDDEN,"inter":INTER,"topk":TOPK,"layers":LAYERS},"threads":THREADS,"kernels":{}}
for cls in ["AMXInt8_MOE","AMXBF16_MOE","AVX2BF16_MOE"]:
    try:
        lat=per_layer_tps(cls)
        tok_latency=lat*LAYERS  # sec/token (MoE only, all layers)
        R["kernels"][cls]={"per_layer_ms":round(lat*1000,2),"per_token_moe_ms":round(tok_latency*1000,1),"decode_tok_s_moe_bound":round(1.0/tok_latency,2)}
        print(cls,R["kernels"][cls],flush=True)
    except Exception as e:
        print(cls,"ERR",str(e)[:80]); R["kernels"][cls]={"err":str(e)[:80]}
# 실제 ktransformers는 INT8 커널이 서빙 기본 → 그게 대표치
try:
    i8=R["kernels"]["AMXInt8_MOE"]["decode_tok_s_moe_bound"]; av=R["kernels"]["AVX2BF16_MOE"]["decode_tok_s_moe_bound"]
    R["amx_int8_vs_avx2_bf16"]=round(i8/av,2)
except: pass
open("/root/results/kt_real_235b.json","w").write(json.dumps(R,ensure_ascii=False,indent=2))
print("KT_REAL_DONE", json.dumps(R.get("amx_int8_vs_avx2_bf16")))
