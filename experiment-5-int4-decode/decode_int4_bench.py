#!/usr/bin/env python3
# Correct decode (qlen=1) benchmark of AMX INT4 vs INT8 vs BF16 vs AVX2 MoE kernels,
# using kt_kernel's proper quantized-weight setup (AMXInt4_MOE quantizes fp32 internally).
# Fixes the earlier flaw (BF16 + gate_scale=0 broke the INT4 path).
# DeepSeek-V3 MoE geometry: hidden 7168, moe_inter 2048, topk 8, 58 MoE layers.
import os, sys, time, json, statistics
import torch, kt_kernel
ext = kt_kernel.kt_kernel_ext

HIDDEN=7168; INTER=2048; TOPK=8; EXPERT_NUM=32; MAXLEN=1024
QLEN=1; WARMUP=400; ITERS=800
V3_MOE_LAYERS=58   # DeepSeek-V3: 58 MoE layers (first 3 dense of 61)
THREADS=int(os.environ.get("KT_THREADS","60"))

wc=ext.WorkerPoolConfig()
wc.subpool_count=2; wc.subpool_numa_map=[0,1]; wc.subpool_thread_count=[THREADS//2,THREADS//2]
CPUInfer=ext.CPUInfer(wc)

def build(cls):
    g=torch.randn((EXPERT_NUM,INTER,HIDDEN),dtype=torch.float32).contiguous()
    u=torch.randn((EXPERT_NUM,INTER,HIDDEN),dtype=torch.float32).contiguous()
    d=torch.randn((EXPERT_NUM,HIDDEN,INTER),dtype=torch.float32).contiguous()
    cfg=ext.moe.MOEConfig(EXPERT_NUM,TOPK,HIDDEN,INTER,0)
    cfg.max_len=MAXLEN; cfg.gate_proj=g.data_ptr(); cfg.up_proj=u.data_ptr(); cfg.down_proj=d.data_ptr()
    cfg.pool=CPUInfer.backend_
    moe=getattr(ext.moe,cls)(cfg)
    CPUInfer.submit(moe.load_weights_task()); CPUInfer.sync()
    return moe,(g,u,d)

def bench(cls):
    moe,_r=build(cls)
    eids=torch.rand(QLEN,EXPERT_NUM).argsort(dim=-1)[:,:TOPK].reshape(QLEN*TOPK).contiguous()
    w=torch.rand((QLEN,TOPK),dtype=torch.float32).contiguous()
    x=torch.randn((QLEN,HIDDEN),dtype=torch.bfloat16).contiguous()
    o=torch.empty((QLEN,HIDDEN),dtype=torch.bfloat16).contiguous()
    bsz=torch.tensor([QLEN])
    def one():
        CPUInfer.submit(moe.forward_task(bsz.data_ptr(),TOPK,eids.data_ptr(),w.data_ptr(),x.data_ptr(),o.data_ptr(),False))
        CPUInfer.sync()
    for _ in range(WARMUP): one()
    t=time.perf_counter()
    for _ in range(ITERS): one()
    lat=(time.perf_counter()-t)/ITERS   # sec per MoE-layer per token
    return lat

R={"geom":{"hidden":HIDDEN,"inter":INTER,"topk":TOPK,"expert_num":EXPERT_NUM,"qlen":QLEN,"moe_layers_v3":V3_MOE_LAYERS},
   "threads":THREADS,"kernels":{}}
for cls in ["AMXInt4_MOE","AMXInt8_MOE","AMXBF16_MOE","AVX2BF16_MOE"]:
    try:
        lat=bench(cls)
        tok_lat=lat*V3_MOE_LAYERS           # sec/token if all 58 MoE layers on CPU
        R["kernels"][cls]={"per_layer_us":round(lat*1e6,1),
                           "v3_moe_only_decode_tok_s":round(1.0/tok_lat,2)}
        print(cls,R["kernels"][cls],flush=True)
    except Exception as e:
        R["kernels"][cls]={"err":str(e)[:100]}; print(cls,"ERR",str(e)[:100],flush=True)
try:
    i4=R["kernels"]["AMXInt4_MOE"]["per_layer_us"]; bf=R["kernels"]["AMXBF16_MOE"]["per_layer_us"]
    av=R["kernels"]["AVX2BF16_MOE"]["per_layer_us"]
    R["int4_vs_bf16_speedup"]=round(bf/i4,2)
    R["int4_vs_avx2_speedup"]=round(av/i4,2)
except: pass
os.makedirs("/dev/shm/kt/out",exist_ok=True)
open("/dev/shm/kt/out/decode_int4.json","w").write(json.dumps(R,ensure_ascii=False,indent=2))
print("DECODE_DONE",json.dumps({k:R["kernels"][k].get("v3_moe_only_decode_tok_s") for k in R["kernels"]}))
print("INT4_vs_BF16",R.get("int4_vs_bf16_speedup"),"INT4_vs_AVX2",R.get("int4_vs_avx2_speedup"))
