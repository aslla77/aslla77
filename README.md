# aslla77

LLM 에이전트 · 데이터 분석

통계 판정과 이상치 탐지는 결정론적 스크립트가 맡고, 모델은 서술만 담당하는 구조로 만듭니다.
숫자가 이상하면 그대로 넘기지 않고, 결과를 검증하는 절차를 먼저 설계합니다.

---

## Agent Skills

Claude Code · Codex · Cursor 등 Agent Skills 클라이언트에서 동작하는 스킬.

| 저장소 | 내용 |
|---|---|
| [**adaptive-upskill**](https://github.com/aslla77/adaptive-upskill) | 가르치기 전에 학습자가 실제로 무엇을 할 수 있는지부터 측정한다. 적응형 진단, 선수지식을 고려한 학습 계획, 근거 기반 숙달도 추적. 모든 채점은 모델이 아니라 결정론적 스크립트가 한다 |
| [**company-scout**](https://github.com/aslla77/company-scout) | 회사명 하나로 공개자료만 써서 리서치 보고서를 만든다. 모든 사실 주장에 출처와 등급(A/B/C)을 강제 부착하고, 검증되지 않은 것은 가설로 명시한다 |

## AX 인재전쟁 해커톤 — 예선 상위 10%

기업이 제시한 공개·검증 가능한 문제를 푸는 Codex 플러그인 2종.
통계와 이상치 판정은 파이썬이, 서술형 리포트는 LLM이 담당하는 2부 구조.

| 저장소 | 내용 |
|---|---|
| [**myrealtrip-opportunity-report**](https://github.com/aslla77/myrealtrip-opportunity-report) | 13개국 입국통계에서 YoY로 계절성을 제거한 뒤 양방향 시차상관과 순열검정으로 '한국을 앞선 리드국'을 검증. 점유율 게이트와 대권거리 보정으로 근접성 착시를 걷어내고 기회 목적지를 랭킹 |
| [**musinsa-fit-review-analyzer**](https://github.com/aslla77/musinsa-fit-review-analyzer) | 상품 리뷰에서 '정상 체형인데 핏 불만'인 이상치만 선별해 부위별 결함 패턴을 진단. 핏 무관 사유(색감·배송) 격리, 과잉탐지 방지, 표본 부족 시 판정 보류 |

## App Store

Agent Skills로 기획 → 리서치 → 검증 → 구현 → 리뷰 → 릴리즈 게이트 워크플로우와
역할별 서브에이전트 팀을 설계해 혼자 만들고 출시. 7개 언어 현지화, 인앱결제 적용.

- [**Islet — Todo List**](https://apps.apple.com/kr/app/islet-todo-list/id6792624887) — 다이나믹 아일랜드·잠금화면 할 일
- [**Daily Relic**](https://apps.apple.com/kr/app/daily-relic/id6791167446) — 유틸리티

## 기록

| 저장소 | 내용 |
|---|---|
| [100-projects](https://github.com/aslla77/100-projects) | 프로젝트 개발 일지 |

---

## 기술

**언어·분석** Python (NumPy, pandas, SciPy, Matplotlib, NetworkX) · SQL · Julia · R
**데이터 수집** Selenium · BeautifulSoup · 공공데이터 API · REST API 연동
**통계·모델링** 시계열 분석 · 파워 스펙트럼 · 순열검정 · 시차상관 · IQR/정규분포 이상치 처리 · Monte Carlo
**LLM·자동화** Claude Code (Agent Skills, 서브에이전트 오케스트레이션) · Gemini API
**앱·인프라** Flutter · Swift/SwiftUI · AWS Lambda · Cloudflare Pages · Supabase · Linux · Git
