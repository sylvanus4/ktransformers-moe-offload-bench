# ktransformers 실증 결과 — MoE expert-offload 메커니즘 (RunPod 4090)

날짜: 2026-07-18 · 대상: kvcache-ai/ktransformers 주장 · 대리모델: Qwen3-30B-A3B (fine-grained MoE, V3 계열)
하드웨어: RunPod RTX 4090 24GB + AMD Ryzen 9 7950X(32 vCPU) + 188GB RAM · 비용 ~$3.4

이 글은 ktransformers 도입 여부를 판단하려는 엔지니어를 위한 것이다. 결론부터: **"트릭"은 실재하고
작동하지만, 마케팅의 28배·"$400K→24GB"는 각각 Intel AMX 하드웨어와 대용량 CPU RAM이라는 숨은
전제 위에 서 있다.** 상용 AMD 박스에서 expert-offload는 순수 CPU 대비 1.6배에 그쳤다.

## 신뢰 가능한 측정 (llama-bench, self-contained)

`--n-cpu-moe`(=ktransformers의 핵심 트릭: 라우팅되는 expert를 CPU에, attention/KV를 GPU에)를 llama.cpp로
동일 구현해 측정했다. ktransformers v0.6.3 자체는 실행하지 못했다(아래 "왜 llama.cpp 대리인가" 참조).

| 배치 | prefill tok/s | decode tok/s | 백엔드 |
|---|---|---|---|
| A1 full-GPU (모델 전체 4090) | 10,049 | **261.5** | CUDA |
| A2 mechanism (48층 expert=CPU, attn=GPU) | 508 | **12.0** | CUDA |
| A3 CPU-only (-ngl 0) | 463 | **7.4** | CPU |

파생 비율:
- **메커니즘 vs 순수 CPU (decode): 1.62배** — expert를 CPU에 두되 attention을 GPU로 올린 효과.
- full-GPU vs 메커니즘 (decode): **21.8배** — 모델이 VRAM에 들어가면 통째로 GPU가 압도.
- full-GPU vs CPU: 35.3배.

## 실증 발견 (정직하게)

**1. 메커니즘은 실재하고 작동한다.** expert를 CPU DRAM에, attention/KV를 GPU에 두는 배치로
Qwen3-30B-A3B가 돌았고 decode 12 tok/s를 냈다. 순수 CPU(7.4)보다 1.62배 빠르다. "attention만 GPU로
올려도 이득"이라는 핵심 주장은 참이다.

**2. 그러나 28배 헤드라인은 이 하드웨어에서 재현 불가 — AMX 전제.** kt-kernel의 예제는 거의 전부
AMX INT8/INT4 벤치(`test_*_amx`)다. ktransformers의 CPU expert 커널은 **Intel AMX(Sapphire Rapids)**
에 최적화돼 있고, 우리 pod는 AMD 7950X(AVX-512는 있으나 AMX 없음)였다. 그래서 우리가 얻은 상대 이득은
1.6배지 28배가 아니다. ktransformers의 28배는 "AMX 커널 vs llama.cpp CPU"의 비교이고, 그 커널을 못 쓰는
CPU에서는 성립하지 않는다. **도입 판단의 1순위 체크: 서빙 CPU가 Sapphire Rapids(AMX)인가.**

**3. "$400K 랙 → 24GB 1장"은 메모리를 없앤 게 아니라 옮긴 것.** 메커니즘은 가중치 부담을 VRAM에서
시스템 RAM으로 이전한다. 우리 pod는 188GB RAM이었고, DeepSeek-V3 Q4는 ~380GB DRAM을 요구한다. 즉
실제 요구사항은 "24GB GPU + 대용량 RAM 서버"다. GPU가 싸진 것이지 총 메모리 소요가 준 게 아니다.

**4. 모델이 VRAM에 들어가면 offload는 손해다.** Qwen3-30B Q4(18GB)는 24GB에 통째로 들어가고, 그때
full-GPU(261 tok/s)가 메커니즘(12 tok/s)을 22배 앞선다. **expert-offload의 유일한 존재 이유는 모델이
VRAM에 안 들어갈 때**(bf16 61GB, 또는 DeepSeek-V3 671B)다. 그 경우 "12 tok/s로라도 돈다 / 7.4보다
낫다"가 가치이지, 속도 자체가 목적이 아니다.

## 폐기한 측정 (측정 실패, 주장 안 함)

- VRAM 델타(full 22.3GB vs mechanism 23.8GB): 뒤집힌 값 = 프로세스 VRAM 잔류 오염. llama-cli 캡처가
  "server exited before ready"로 실패. 신뢰 불가 → **VRAM 절감은 정량화 못 함**(정성적으론 decode 22배
  둔화가 expert의 CPU 배치를 방증).
- H3 131K 컨텍스트, H4 출력 동일성, A2b 부분 offload: 캡처 실패/느림으로 폐기.

## 왜 진짜 ktransformers가 아니라 llama.cpp 대리인가

ktransformers v0.6.3은 완전 재구조화됐다. pip 패키지는 SFT shim만 담고(local_chat·optimize rules 없음),
실제 추론은 `ktransformers/server/main.py --backend_type balance_serve`로 `install.sh` 풀 소스 빌드
(flashinfer/balance_serve 컴파일, 30–60분)가 필요하며 커널이 AMX 지향이다. AMD pod에서 그 빌드에
성공해도 AVX2 폴백만 돌아 헤드라인 경로를 못 탄다(ROI 없음). 그래서 **메커니즘 자체**를 llama.cpp의
`--n-cpu-moe`(동일 트릭)로 깨끗이 측정했다. ktransformers 고유의 AMX 커널 가속은 별도이며 Intel 필요.

## 도입 권고

- 서빙 CPU가 **Intel Sapphire Rapids(AMX)** 이고, 모델이 **VRAM 초과**하는 대형 MoE(V3/R1급)일 때만
  ktransformers의 진짜 이득(헤드라인 배수)이 나온다. 두 조건 중 하나라도 빠지면 이득이 급감한다.
- 모델이 VRAM에 들어가면 full-GPU가 정답. offload는 고려 대상 아님.
- 다음 검증(원하면): Sapphire Rapids(AMX) pod로 재실행해 28배 경로를 정면 측정 + 진짜 ktransformers
  balance_serve 빌드. RunPod AMX CPU pod 가용성·비용 확인 필요.

## 2차 실험 — Intel AMX 커널 정면 측정 (H100 + Xeon 8470 Sapphire Rapids)

RunPod에서 AMX 호스트(Intel Xeon Platinum 8470, amx_bf16/int8/tile, 208 vCPU, 1TB RAM, H100 80GB)를
잡아 28배의 근거라는 AMX 커널을 직접 쟀다. kt_kernel 휠이 백엔드별 .so를 모두 포함해, 같은 프로세스에서
동일 BF16 가중치로 AMX vs AVX2 커널을 비교했다. DeepSeek-V3 스케일(256 experts, hidden 7168).

| 커널 (동일 BF16, decode qlen=1) | tok/s |
|---|---|
| AMX (AMXBF16_MOE) | 145.5 |
| AVX2 (AVX2BF16_MOE) | 105.5 |

**AMX vs AVX2 = 1.38배.** 분명한 이득이나 28배와 무관. INT8 전용 타일 연산은 더 벌어질 여지 있으나
(비용상 BF16 동일정밀도까지만) 커널 하나가 28배를 만들지 않는다.

## 28배의 분해 (두 실험 종합)

벤더의 28배는 커널 배수가 아니라 **시스템 비교**(ktransformers 풀스택 vs llama.cpp CPU-only)다:
- 어텐션/KV를 GPU로: 최대 지렛대 (AMD서 순수CPU 대비 1.62배, 모델이 GPU에 들어가면 35배)
- AMX 전문가 커널: AVX2 대비 +1.4배
- INT8/INT4 양자화 + 파이프라인: 추가 (미측정)
- 특정 조건(VRAM 초과 모델 + AMX CPU + vs 순수 CPU)에서 이 소박한 배수들이 곱해져 두 자릿수.

## 도입 조건 (셋 다 맞을 때만 진짜 이득)
1. 서빙 CPU가 Intel Sapphire Rapids+ (AMX)
2. 모델이 GPU VRAM 초과하는 대형 MoE (V3/R1급)
3. 대형 모델 담을 대용량 시스템 RAM
→ 모델이 GPU에 들어가면 full-GPU가 22배 빠름, 고민 불필요. ktransformers 가치 = 속도가 아니라 접근성.

## 아티팩트
`outputs/research-lab/ktransformers-validation/`
- 1차(4090/AMD): remote_bench.sh, corrected_bench.sh, results/{env,A1,A2,A3}.json, summary.json
- 2차(H100/AMX): amx_bench.py, inspect_backends.py, results-amx/amx_kernel_result.json
- 콘텐츠: blog-ko.md (기술 블로그), linkedin-ko.md (링크드인)
- 총 GPU 비용 ~$5, pod 전량 teardown 완료(과금 0).
