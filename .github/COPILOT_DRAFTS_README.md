# Copilot Drafts — SERS-AI Migration

Phase 1 (Claude Code → GitHub Copilot 이전) 대비 instructions/prompts 초안.
임시 위치 — Phase 1 시작 시 SERS-AI repo의 `.github/`로 이전.

## 구조

```
copilot_drafts/
├── copilot-instructions.md           # TIER1 — 모든 chat에 자동 첨부
├── instructions/                      # TIER2 — applyTo glob 자동 적용
│   ├── clinical-utils.instructions.md   # models/clinical_utils.py 사용 강제
│   ├── dashboard.instructions.md        # workspace/solum-dashboard/ 위치 + 흰색 배경
│   ├── figure-scripts.instructions.md   # figure 생성 시 CLI 기록
│   └── sql.instructions.md              # SQL 학습자 모드 + 곡예 금지
├── prompts/                           # PROMPT — 사용자 명시적 호출
│   ├── ceo-report.prompt.md             # 대표님 보고 5가지 체크
│   ├── chairman-report.prompt.md        # 회장님 1~2장 비기술적
│   ├── timeline-update.prompt.md        # Master Timeline 양식 유지
│   └── research-outputs.prompt.md       # vault 저장 + 경쟁사 양식
└── README.md                          # (이 파일)
```

## Phase 1 이전 대상

SERS-AI repo의 `.github/`:
```
SERS-AI/
└── .github/
    ├── copilot-instructions.md   ← copilot-instructions.md
    ├── instructions/              ← instructions/* (4개)
    └── prompts/                   ← prompts/* (4개)
```

## VS Code Copilot 활성화 (settings.json)

```json
{
  "github.copilot.chat.codeGeneration.useInstructionFiles": true,
  "chat.promptFiles": true,
  "chat.agent.enabled": true
}
```

## Source mapping (어떤 Claude memory에서 왔는지)

| Draft file | Source memory + extra |
|------|---------------|
| `copilot-instructions.md` | feedback_peak_data_source, feedback_vscode_file_guide, project_pan_composition, reference_obsidian_vault, project_hospital_confound (1줄), project_qc_issue (1줄), `~/CLAUDE.md` (CSV/comparison/BLC/sync) |
| `instructions/clinical-utils.instructions.md` | feedback_clinical_utils |
| `instructions/dashboard.instructions.md` | feedback_dashboard_location + feedback_dashboard_white_bg |
| `instructions/figure-scripts.instructions.md` | feedback_figure_regeneration_cli |
| `instructions/sql.instructions.md` | feedback_sql_realistic + user_sql_learning |
| `prompts/ceo-report.prompt.md` | user_ceo_persona + `~/.claude/rules/ceo-report.md` |
| `prompts/chairman-report.prompt.md` | user_clevel_audience |
| `prompts/timeline-update.prompt.md` | feedback_timeline_update |
| `prompts/research-outputs.prompt.md` | feedback_research_output_location + reference_competitors |

## TODO (Phase 1 trial 시점에 검토)

- [ ] VS Code + Copilot extension 셋업 + Agent Mode 활성화
- [ ] `.vscode/mcp.json` MCP 서버 셋업
  - [ ] mcp-obsidian (vault SSOT 접근)
  - [ ] context7 (라이브러리 문서)
  - [ ] supabase (DB 접근)
- [ ] 첫 1~2주 trial — gap 발견 시 instructions 보강
- [ ] Custom slash commands (/ceo-report, /chairman, /timeline-update, /research-outputs) 등록
- [ ] Phase 2 Entra ID 전환 시 .github/ 자산은 그대로 유지 (계정만 갈아끼움)

## Trial 평가 항목

1. SERS-AI 코딩 작업에서 instructions 자동 적용 빈도/정확성
2. 모드 prompt files 호출 편의성 (Claude의 자동 trigger 대비 손실)
3. Obsidian vault 검색 정확도 (mcp-obsidian)
4. CEO/회장님 모드의 결과물 품질 (sample 1건씩 시도)
