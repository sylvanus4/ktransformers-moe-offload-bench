#!/usr/bin/env python3
# AMX vs AVX2 MoE 커널 순수 비교 (동일 BF16 가중치, DeepSeek-V3 스케일).
# ktransformers 28x 헤드라인의 근거 = AMX 커널. 이 벤치가 그 커널 자체의 배수를 측정.
import os, sys, time, json, statistics
os.environ.setdefault("KT_KERNEL_CPU_VARIANT", "amx")  # amx .so는 AMX+AVX2 클래스 모두 포함
import torch
import kt_kernel
ext = kt_kernel.kt_kernel_ext

EXPERT_NUM = 256          # DeepSeek-V3 scale
HIDDEN = 7168
INTER = 2048
TOPK = 8
MAXLEN = 4096
THREADS = int(os.environ.get("KT_THREADS", "104"))
WARMUP, ITERS = 5, 30
CPUInfer = ext.CPUInfer(THREADS)
p2l = torch.tensor(range(EXPERT_NUM), dtype=torch.int64).contiguous()

def build(cls_name):
    gate = torch.randn((EXPERT_NUM, HIDDEN, INTER), dtype=torch.bfloat16).contiguous()
    up   = torch.randn((EXPERT_NUM, HIDDEN, INTER), dtype=torch.bfloat16).contiguous()
    down = torch.randn((EXPERT_NUM, INTER, HIDDEN), dtype=torch.bfloat16).contiguous()
    cfg = ext.moe.MOEConfig(EXPERT_NUM, TOPK, HIDDEN, INTER, 0)
    cfg.max_len = MAXLEN
    cfg.gate_proj = gate.data_ptr(); cfg.up_proj = up.data_ptr(); cfg.down_proj = down.data_ptr()
    cfg.gate_scale = 0; cfg.pool = CPUInfer.backend_
    moe = getattr(ext.moe, cls_name)(cfg)
    CPUInfer.submit(moe.load_weights_task(p2l.data_ptr())); CPUInfer.sync()
    try:
        CPUInfer.submit(moe.warm_up_task()); CPUInfer.sync()
    except Exception: pass
    return moe, (gate, up, down)   # keep refs alive

def bench(cls_name, qlen):
    moe, _refs = build(cls_name)
    bsz = torch.tensor([qlen]).contiguous()
    eids = torch.stack([torch.randperm(EXPERT_NUM)[:TOPK] for _ in range(qlen)]).contiguous()
    w = torch.rand((qlen, TOPK), dtype=torch.float32).contiguous()
    x = (torch.randn((qlen, HIDDEN), dtype=torch.bfloat16)/100).contiguous()
    out = torch.empty((qlen, HIDDEN), dtype=torch.bfloat16).contiguous()
    def one():
        CPUInfer.submit(moe.forward_task(bsz.data_ptr(), TOPK, eids.data_ptr(), w.data_ptr(),
                        x.data_ptr(), out.data_ptr(), False)); CPUInfer.sync()
    for _ in range(WARMUP): one()
    ts=[]
    for _ in range(ITERS):
        t=time.perf_counter(); one(); ts.append((time.perf_counter()-t)*1000)
    avg=statistics.mean(ts)
    return {"class":cls_name,"qlen":qlen,"avg_ms":round(avg,3),"tok_s":round(qlen/(avg/1000),1)}

RES={"config":{"experts":EXPERT_NUM,"hidden":HIDDEN,"inter":INTER,"topk":TOPK,"threads":THREADS},"runs":[]}
for qlen in [1, 128]:
    for cls in ["AMXBF16_MOE","AVX2BF16_MOE"]:
        r=bench(cls,qlen); RES["runs"].append(r); print(r, flush=True)
# quantized serving path (ktransformers 실제 모드): AMX INT8 참고치
for qlen in [1,128]:
    try:
        r=bench("AMXInt8_MOE",qlen); r["note"]="amx-int8-serving"; RES["runs"].append(r); print(r,flush=True)
    except Exception as e: print("int8 skip",e)
# 배수 계산
def find(cls,q):
    return next((x["tok_s"] for x in RES["runs"] if x["class"]==cls and x["qlen"]==q), None)
RES["amx_vs_avx2_bf16"]={f"qlen{q}":round(find("AMXBF16_MOE",q)/find("AVX2BF16_MOE",q),2) for q in [1,128]}
open("/workspace/results/amx_bench.json","w").write(json.dumps(RES,ensure_ascii=False,indent=2))
print("SPEEDUP amx/avx2 bf16:", RES["amx_vs_avx2_bf16"])
print("AMX_BENCH_DONE")
