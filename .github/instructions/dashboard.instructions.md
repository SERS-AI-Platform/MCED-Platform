---
applyTo: "workspace/solum-dashboard/**/*.html,**/*Dashboard*.html,**/*dashboard*.html"
---

# Dashboard 생성/수정 규칙

## 1. 배포 위치

대시보드 HTML 파일은 반드시 `/home/user/workspace/solum-dashboard/`에 배포할 것.
- `SERS-AI/dashboard/`는 프로젝트 내부 참고용
- 실제 배포/공유는 `workspace/solum-dashboard/` 쪽

새 대시보드 생성 시 `workspace/solum-dashboard/`에 직접 작성하거나, 다른 곳에 만든 경우 반드시 복사.

## 2. 배경 색상

**항상 흰색 배경**으로 통일. dark mode 변형 사용 금지:
- `dark:bg-gray-900`, `dark:text-white` 등 `dark:` 접두사 Tailwind 클래스 사용 금지
- 배경색은 `white` 또는 warm/light 계열만

## 3. 메타 변경 후 동기화

대시보드 이름/설명/카테고리 변경 시 반드시 실행 (Supabase + index.html FALLBACK 동기화):

```bash
cd /home/user/workspace/solum-dashboard && python sync_dashboard.py \
  --url "파일명.html" \
  --name "표시 이름" \
  --description "설명" \
  --icon "아이콘(1-2자)" \
  --icon-color "blue|purple|green|amber" \
  --category "research|executive|clinical|lab" \
  --tags "태그1,태그2"
```

내용만 수정하고 메타데이터(이름/설명) 미변경 시는 실행 불필요.
`--list` 옵션으로 현재 등록 상태 확인.

## 4. C-Level 간소화 버전
간소화 버전 작성 시: 한국어, 2장 이내, 비기술적. 파일명: `*_CLevel_Dashboard.html`.
