# aslla77

혼자 기획부터 출시·운영까지 하는 앱 스튜디오 · LLM 에이전트 · 데이터 분석

리서치 → 기획 → 레드팀 → 구현 → 리뷰 → QA → 보안 → 릴리즈를 게이트로 나누고,
역할별 Claude Code 서브에이전트 팀이 각 단계를 맡는 구조로 앱을 만듭니다.
숫자가 걸린 판정은 결정론적 스크립트가, 서술은 모델이 맡습니다.

---

## Apps

iOS · Android 양쪽 스토어에 출시해 운영 중인 앱. 최대 20개 언어 현지화.

| 앱 | 한 줄 | iOS | Android |
|---|---|---|---|
| **언제했지** | 할 일 대신 마지막으로 한 날짜를 기록해 지난 날수를 보여준다 | [App Store](https://apps.apple.com/app/id6801866888) | [Google Play](https://play.google.com/store/apps/details?id=com.asllasoft.when_did_i) |
| **뭐하지** | 할 일을 다이나믹 아일랜드·잠금화면·Apple Watch에 띄운다 | [App Store](https://apps.apple.com/app/id6792624887) | [Google Play](https://play.google.com/store/apps/details?id=com.asllasoft.whatdoido.android) |
| **누구지** | 전화번호부를 한 명씩 카드로 넘기며 "이 사람 누구지?"를 판단해 정리한다 | [App Store](https://apps.apple.com/app/id6832419020) | [Google Play](https://play.google.com/store/apps/details?id=com.asllasoft.whoareyou) |
| **바로앨범** | 촬영 전에 앨범을 고르면 찍는 즉시 그 앨범에 저장된다 | [App Store](https://apps.apple.com/app/id6803097904) | [Google Play](https://play.google.com/store/apps/details?id=com.asllasoft.album_saver) |
| **쉼호흡** | 정한 시각에 종이 울리면 5분간 느린 호흡을 목소리로 안내한다 | [App Store](https://apps.apple.com/app/id6809217718) | [Google Play](https://play.google.com/store/apps/details?id=com.asllasoft.stillbell) |
| **Daily Relic** | 매일 박물관 3D 유물 한 점을 해설과 함께 감상한다 | [App Store](https://apps.apple.com/app/id6791167446) | [Google Play](https://play.google.com/store/apps/details?id=kr.dailyrelic.daily_relic) |
| **무럭무럭** | 물주기와 사진을 기록하면 기기 안에서 성장 타임랩스가 만들어진다 | [App Store](https://apps.apple.com/app/id6797058555) | [Google Play](https://play.google.com/store/apps/details?id=com.asllasoft.plantlog) |

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

## 기록

| 저장소 | 내용 |
|---|---|
| [**100-projects**](https://github.com/aslla77/100-projects) | 프로젝트 100개를 목표로 만드는 과정의 개발 일지. 무엇을 왜 만들었고, 무엇을 접었고, 무엇을 배웠는지 |
