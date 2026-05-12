---
mode: agent
description: "리서치 산출물 작성 모드 — Obsidian vault 저장 + 경쟁사 프로파일 양식"
---

# 리서치 산출물 작성 모드

리서치 보고서, 문헌 리뷰, 경쟁사 프로파일, 시장 분석 등 분석 결과물 작성 시.

## 1. 저장 위치 (필수)

vault 경로: `/mnt/c/Users/user/OneDrive - solum/바탕 화면/obsidian/`

| 산출물 종류 | 저장 위치 |
|---|---|
| SERS-AI 리서치 (workflow, methods 등) | `01_Projects/SERS-AI/References/` |
| 일반 문헌 리뷰 | `03_Resources/<주제별 폴더>/` |
| 경쟁사 프로파일 | `03_Resources/Competitors/` |
| 시장 리서치 | `01_Projects/SERS-AI/References/` 또는 `03_Resources/` |

**`docs/`나 repo 내부에 저장 금지** — Obsidian vault만.

## 2. 파일명 규칙
`YYYY-MM-DD__제목.md` (kebab-case)
- 예: `2026-04-29__exopert-patent-analysis.md`

## 3. 경쟁사 프로파일 양식

frontmatter 필수:
```yaml
---
created: YYYY-MM-DD
last_verified: YYYY-MM-DD
tags: [competitor, ...]
category: competitor-profile
status: active|archived
---
```

섹션 순서 (양식 통일):
1. 회사 정보 (본사, 설립, 직원, 자금)
2. 기술 플랫폼
3. 규제 현황 (FDA, MFDS, CE, PMDA 등)
4. 임상 데이터 (peer-reviewed 논문, clinical trial 등록 여부)
5. 타깃 커버리지 (어떤 암종/적용 영역)
6. SERS-AI 관점 시사점 (FTO 위험, 차별화 포인트)
7. 소스 (HTTP 200 검증 후 기재)
8. 업데이트 예정 이벤트

## 4. 소스 검증 (필수)

모든 외부 링크는 HTTP 200 확인 후 기재:
```bash
curl -o /dev/null -s -w "%{http_code}" <URL>
```

200 외 응답이면 소스 제외 또는 archive.org Wayback Machine으로 대체.

## 5. 기존 양식 참조
- `03_Resources/Competitors/2026-04-17__competitor-TOBY.md` — TOBY, Inc. (소변 VOC GC-MS + AI, FDA BDD 2건)

## 6. 작업 완료 시 보고
- 저장 위치 (전체 경로)
- 검증된 소스 개수
- 양식 통일 여부 (✅/⚠)
- 백링크 추가한 곳 (master-timeline, project board 등)
