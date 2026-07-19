#!/usr/bin/env python3
# T셀: ktransformers 이기종 추론 (검증 대상).
# ⚠️ ktransformers 호출부는 버전 민감 — pod에서 라이브 조정 필요(문서화된 취약점).
# 실제 경로: ktransformers는 HF config(model_path) + 양자화 가중치(gguf_path) + injection YAML로
# attention/shared expert/KV=GPU, routed expert=CPU 로 배치한다.
# 참조 엔트리: python -m ktransformers.local_chat --model_path <hf> --gguf_path <gguf_dir> --cpu_infer <N>
import sys,time,json,subprocess,os
gguf=sys.argv[1]                      # Mixtral Q4 gguf dir/file
hf_cfg=os.environ.get("KT_HF","/workspace/m/mixtral_hf")  # config+tokenizer (별도 다운로드 필요)
# 비대화형 1-shot: local_chat에 prompt를 파이프. tok/s는 ktransformers가 stderr로 출력 → 파싱.
cmd=(f"python -m ktransformers.local_chat --model_path {hf_cfg} "
     f"--gguf_path {os.path.dirname(gguf)} --cpu_infer {os.cpu_count()} --max_new_tokens 64")
t0=time.time()
p=subprocess.run(cmd,shell=True,input="Explain the VRAM vs CPU DRAM tradeoff for a MoE model. 5 sentences.\n",
                 capture_output=True,text=True,timeout=1800)
dt=time.time()-t0
# ktransformers 로그에서 'prefill'/'decode' tok/s 정규식 추출 (버전별 상이 → 원문 tail 보존)
import re
def grab(pat,s):
    m=re.search(pat,s); return float(m.group(1)) if m else None
blob=p.stdout+p.stderr
print(json.dumps({
 "wall_s":round(dt,1),"rc":p.returncode,
 "prefill_tok_s":grab(r"prefill.*?([\d.]+)\s*tok",blob),
 "decode_tok_s":grab(r"decode.*?([\d.]+)\s*tok",blob),
 "stdout_tail":p.stdout[-1500:],"stderr_tail":p.stderr[-800:]
},ensure_ascii=False))
