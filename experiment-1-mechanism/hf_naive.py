#!/usr/bin/env python3
# B1 baseline: HF transformers device_map=auto (순진한 CPU offload) on 24GB GPU.
# decode tok/s + peak mem + 출력(=H4 diff용) 산출. 결정론.
import sys,time,json,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
mp=sys.argv[1]
tok=AutoTokenizer.from_pretrained(mp,trust_remote_code=True)
t0=time.time()
model=AutoModelForCausalLM.from_pretrained(
    mp,torch_dtype=torch.bfloat16,device_map="auto",
    trust_remote_code=True,
    max_memory={0:"22GiB","cpu":"200GiB"},   # 24GB에 안 들어가면 CPU로 넘김
)
load_s=time.time()-t0
p="Explain the VRAM vs CPU DRAM tradeoff for serving a MoE model. 5 sentences."
ids=tok(p,return_tensors="pt").to("cuda")
torch.cuda.synchronize(); s=time.time()
out=model.generate(**ids,max_new_tokens=64,do_sample=False)
torch.cuda.synchronize(); gen_s=time.time()-s
ntok=out.shape[-1]-ids.input_ids.shape[-1]
txt=tok.decode(out[0,ids.input_ids.shape[-1]:],skip_special_tokens=True)
print(json.dumps({"load_s":round(load_s,1),"decode_tok_s":round(ntok/gen_s,2),
 "gen_tokens":int(ntok),"output":txt},ensure_ascii=False))
