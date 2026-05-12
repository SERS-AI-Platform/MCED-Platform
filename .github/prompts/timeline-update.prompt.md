---
mode: agent
description: "Master Timeline 업데이트 모드 — 일정/마감/미팅을 양식 그대로 반영"
---

# Master Timeline 업데이트 모드

사용자가 일정/마감/미팅/발표를 언급하면 Master Timeline에 자동 반영.

## 대상 파일
`/mnt/c/Users/user/OneDrive - solum/바탕 화면/obsidian/01_Projects/SERS-AI/00_Status/2026-03-23__master-timeline.md`

## 반영 위치 (해당되는 곳 모두)
1. **Gantt chart** (mermaid `gantt` 블록)
2. **Milestone Map** (mermaid `timeline` 블록)
3. **주간 할일** (`> [!danger/warning/important]` callout 박스)
4. **Task Breakdown** 테이블 (각 영역별)

## 양식 유지 원칙

### 라벨/구조
- 기존 라벨 길이, section 구분, status 이모지 그대로 유지
- 새 일정은 적절한 section에 배치 (없으면 새 section 생성)
- mermaid section 이름 일관 유지 (예: "AACR", "특허", "식약처")

### Status 이모지
- 완료: `:done` 또는 ✅
- 진행 중: 🔄
- 미시작: ⬜
- 긴급: 🔴
- 경고: ⚠

### 날짜 변환
- 상대 날짜 → 절대 날짜 (today 기준)
  - "다음 주 화요일" → 절대 날짜 (예: 2026-05-05)
  - "이번 달 말" → 마지막 평일
- 미정 일정: TBD 표기

## Tags 업데이트
새 이벤트가 새 키워드를 포함하면 frontmatter `tags`에도 추가:
```yaml
tags:
  - timeline
  - 분당서울대  # ← 새 이벤트면 추가
  - 사용적합성
  - ...
```

## 업데이트 후 출력
1. frontmatter `updated:` 필드를 today로 변경
2. 무엇을 어디에 추가했는지 보고:
   - "Gantt chart에 X 추가"
   - "Task Breakdown #N 신규 섹션"
   - "주간 callout 업데이트"
3. 기존 항목 status 변경했다면 명시 ("4/20 AACR ⬜ → ✅")

## 안티패턴
- ❌ 기존 섹션 양식 무시하고 새 양식으로 작성
- ❌ 임의로 mermaid 그래프 재작성
- ❌ 날짜 prefix 없는 일정 추가 (date를 알 수 없으면 사용자에게 되묻기)
