# Phase 1 — GitHub Copilot Trial Guide

> 1~2주 trial 후 gap 발견 → instructions 보강 → Phase 2 (Entra ID 전환) 결정.

## 사전 준비 체크리스트

### 1. 개인 GitHub Copilot Pro 구독
- 회사 정책으로 Entra ID Copilot 받기 전이므로 **개인 GitHub 계정** 사용
- 구독: https://github.com/settings/copilot
- Copilot Pro ($10/월) 또는 Pro+ ($39/월, agent 사용량 더 많음) — agent mode 많이 쓰면 Pro+ 권장

### 2. VS Code + Extensions 설치
필요한 것:
- VS Code (latest)
- **GitHub Copilot** extension (`GitHub.copilot`)
- **GitHub Copilot Chat** extension (`GitHub.copilot-chat`)

확인:
- VS Code 우측 하단 상태바에 Copilot 아이콘
- Cmd/Ctrl+Shift+I 또는 Cmd/Ctrl+I로 Chat 열기

### 3. Workspace 열기
```bash
cd /home/user/SERS-AI
code .
```

### 4. Copilot Agent Mode 활성화 확인
- Chat panel 상단의 mode selector에 **"Agent"** 옵션 있는지
- Settings → `chat.agent.enabled: true` (`.vscode/settings.json`에 이미 설정됨)

### 5. MCP 서버 설치 검증
첫 실행 시 npx로 자동 다운로드:
```bash
# 테스트 (한 번만 실행해보기)
npx -y @modelcontextprotocol/server-filesystem --help
npx -y @upstash/context7-mcp --help
```

설치 안 되면:
```bash
npm install -g @modelcontextprotocol/server-filesystem @upstash/context7-mcp
```

VS Code 재시작 후 Chat에서 MCP 서버가 인식되는지 확인 (Agent mode 도구 목록).

### 6. 모델 선택
Chat 우측 model selector에서 선택:
- **Claude Sonnet 4.5** (코딩 일반) — 추천
- **Claude Opus 4.7** (복잡한 reasoning, 1M context) — 큰 작업 시
- **GPT-5** (빠른 코드 완성)
- **Gemini 2.5 Pro** (백업)

---

## Trial 시나리오 (1주차)

다음 5가지를 시도하면서 gap 측정.

### 시나리오 1: Instructions 자동 적용 검증
- `models/`, `scripts/analysis/`, `scripts/pipeline/` 안의 .py 파일 편집
- Chat에 "clinical 데이터를 spectral 데이터와 merge해줘" 질문
- **검증**: `clinical-utils.instructions.md`가 자동 적용되어 `merge_clinical_features()` 사용 권유?

### 시나리오 2: Dashboard 작성
- `workspace/solum-dashboard/` 안에 새 dashboard HTML 작성 요청
- **검증**: 흰색 배경 + sync_dashboard.py 실행 권유?

### 시나리오 3: SQL 작성
- 새 .sql 파일 생성 + 쿼리 작성 요청
- **검증**: 곡예 금지 + 학습 친화적 주석 포함?

### 시나리오 4: 모드 prompt 호출
- Chat에 `/ceo-report` 또는 prompt 파일 attach
- "STK-V2 결과로 1장짜리 보고 자료 만들어줘"
- **검증**: 5가지 체크리스트 자체 점검 보고?

### 시나리오 5: Vault 검색 (MCP 검증)
- "최근 hospital confound 발견 내용 알려줘"
- **검증**: Agent가 `obsidian-vault` MCP로 `Findings/2026-04-09__hospital-confound.md` 찾는지?

---

## Gap 추적

trial 동안 발견한 부족한 점 기록:
- `01_Projects/SERS-AI/00_Status/copilot-trial-log.md` (vault에 노트 작성)
- 매일 1회 정리 권장

추적 양식:
```markdown
## YYYY-MM-DD
### 잘 동작
- ...
### 부족
- [ ] X 시나리오에서 Y가 안 됨 → instructions 보강 필요?
### 대안
- ...
```

---

## Phase 2 진입 조건

Entra ID 받기 전이라도 1~2주 trial 후 다음 점검:
1. **자동 메모리 손실 감수 가능?** — 매 세션 컨텍스트 직접 챙기는 것이 부담스럽지 않은지
2. **Skill 손실 영향?** — pptx, docx, xlsx 자동화 손실. 필요한 부분은 직접 Python 스크립트로 작성.
3. **Hooks 부재 영향?** — 세션 시작 자동 모드 확인 등이 사라짐
4. **Sub-agent orchestration 손실?** — 복잡한 다단계 작업이 더 어려워지는지

trial 후 결정:
- 만족 → Phase 2 (Entra ID 받으면 계정 갈아끼움)
- 불만족 → Cursor 검토 (단, 회사 정책 충돌 가능성 검증 필요)
- 부분 만족 → Claude Code + Copilot 병행 (단기적, 권장은 아님)

---

## Reference Files

- `.github/copilot-instructions.md` — TIER1 (모든 chat 자동 첨부)
- `.github/instructions/*.md` — TIER2 (path-scoped)
- `.github/prompts/*.md` — PROMPT (수동 호출)
- `.github/COPILOT_DRAFTS_README.md` — Source mapping (어떤 memory에서 왔는지)
- `.vscode/settings.json` — Copilot 활성화 설정
- `.vscode/mcp.json` — MCP 서버 (Obsidian, Context7, workspace 4개)
