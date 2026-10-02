# ops — 측정 계획 대비 진행 현황

측정 계획표(엑셀, DAY 시트)와 실제 측정(`measurement` 스키마)을 이어 진행률을 본다. Teams 채널의 Power BI 탭이 목적지.
`measurement`는 읽기 전용이라 사용자 결정(2026-09-29)으로 `ops` 스키마에 둔다.

| 객체 | 내용 |
|---|---|
| `ops.plan_versions` | 적재한 계획표 파일(MD5로 구분, `is_current` 한 벌) |
| `ops.measurement_plan` | 계획표 한 칸 = 검체 한 행 (DAY·Lot·Paper·군·셀 위치) |
| `ops.clinical_exclusions` | 계획표의 `제외사유_목록_v8` 시트 (환자 단위 사유 — **대시보드로 내보내지 않음**) |
| `ops.v_plan_progress` | 현재 계획 × 실제 측정 × QC. 사유 텍스트 없음 — Power BI는 이 뷰만 읽는다 |

`v_plan_progress`의 정의
- `measured`: 환원제 lot **BCCP0922**로 측정된 기록이 있음 (변경 전 4~7월 측정은 완료로 세지 않음)
- `qc_status`: 미측정 / QC 미계산(`experiment.sample_qc`에 없음) / QC 탈락(2b) / QC 통과
- `excluded`: 제외사유 목록에 있음. 계획표 빨간 글씨는 칸 일부만 칠한 경우가 많아 쓰지 않는다
- 군(`cohort_group`)은 검체 자신의 진단(`diagnosis_id = sample_id`) — 다중암 환자 4명의 중복 방지
- `exclusion_category`: 제외 분류. `포함` 또는 제외사유 목록 플래그 조합(`타암이력`, `요로감염`, `치료후채취`, `다중암`, `Drop`, 여러 개면 `+`로 연결, 플래그 없이 노란표시만 있으면 `기타`). 자유 텍스트 사유는 넣지 않음
- `sample_status`: `제외` / `측정 완료` / `미측정` — 검체 하나가 셋 중 하나
- `plan_scope`: 전체 계획 = `현재 계획`(현재 계획표의 칸 전부) + `이전 계획`(`--previous`로 넣은 이전 계획표에만 있는 칸). 제외 여부는 둘 다 현재 계획표의 제외 목록 기준
  - 2026-10-02: 0907 계획표는 0728 계획 DAY 1을 잰 뒤 남은 검체로 다시 짠 것(그룹마다 번호가 이어짐). 0728 계획표(`…전체매칭1722.xlsx`)를 이전 계획으로 넣어 0907에 없는 252칸을 합침 → 전체 2,748칸, 그중 8/19~9/7 BCCP0922 측정 236

## 적재 (계획표가 바뀔 때마다)

```bash
source scripts/db/pghost.sh
psql -d aecd_platform -f scripts/db/ops/01_measurement_plan.sql      # 처음 한 번 (재실행 안전)
python scripts/db/ops/load_measurement_plan.py --file "<계획표.xlsx>" --dry-run
python scripts/db/ops/load_measurement_plan.py --file "<계획표.xlsx>"
# 이전 계획표: 계획 칸만 넣고 현재 계획은 그대로 (제외사유 시트 형식이 달라도 됨)
python scripts/db/ops/load_measurement_plan.py --file "<이전 계획표.xlsx>" --previous
```

같은 파일을 다시 넣으면 그 버전을 다시 현재로 표시만 한다. 새 파일(MD5 다름)은 새 버전이 되고 이전 버전은 보관된다.
측정 데이터가 DB에 새로 적재되면 뷰가 자동으로 반영하므로 계획표 재적재는 필요 없다.

## Power BI

1. Power BI Desktop(Windows) → 데이터 가져오기 → **PostgreSQL 데이터베이스** → 서버 `localhost`, 데이터베이스 `aecd_platform`, 가져오기(Import) → `ops.v_plan_progress` 선택. 또는 `aecd_plan_progress.pbids`를 열면 서버·DB가 채워진 연결 창이 바로 뜬다(인증: 데이터베이스 탭, `solumhc`). 쿼리 이름은 `v_plan_progress`로 바꾼다(아래 DAX가 이 이름을 씀).
2. 측정값(DAX):
   ```
   계획 = COUNTROWS(v_plan_progress)
   제외 = CALCULATE([계획], v_plan_progress[excluded] = TRUE())
   측정 대상 = [계획] - [제외]
   측정 완료 = CALCULATE([계획], v_plan_progress[measured] = TRUE(), v_plan_progress[excluded] = FALSE())
   QC 통과 = CALCULATE([계획], v_plan_progress[qc_status] = "QC 통과", v_plan_progress[excluded] = FALSE())
   진행률 = DIVIDE([측정 완료], [측정 대상])
   남은 계획 진행률 = CALCULATE([진행률], v_plan_progress[plan_scope] = "현재 계획")
   ```
   `진행률`은 전체(현재 + 이전 계획) 기준이다. 표·차트에서 `plan_scope`를 범례로 쓰면 두 묶음을 나눠 볼 수 있다.
3. 게시 → 작업 영역. DB가 이 PC에 있으므로 예약 새로 고침에는 **온프레미스 데이터 게이트웨이**(개인 모드로 충분, 가져오기 모드 지원)가 이 PC에 설치·실행돼 있어야 한다. PC가 꺼져 있으면 새로 고침 실패.
4. Teams 채널 → 탭 추가(+) → **Power BI** → 게시한 보고서 선택. 보는 사람에게 해당 작업 영역·보고서 권한이 있어야 한다.

Supabase는 쓰지 않는다(회사 정책, 2026-09-02).
