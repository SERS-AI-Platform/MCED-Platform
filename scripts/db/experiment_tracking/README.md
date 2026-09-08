# experiment_tracking — aecd_platform `experiment` 도메인

논문 출처 전처리 방법과 그 벤치마크 결과를 `aecd_platform` 안에 기록하는
스키마. Preprocessing Lab 프로젝트(`SERS-AI/.claude/skills/preprocessing-lab/`)가
사용한다.

## 파일

| 파일 | 용도 |
|---|---|
| `01_schema.sql` | `experiment` 스키마 DDL. 최초 한 번만 실행 |
| `02_import_registry.sql` | 위 JSON을 SQL로 내보낸 것. **자격증명 없이 DBeaver에서 바로 실행 가능** |
| `import_registry.py` | `logs/experiment_registry.json`(55건)을 `experiment.runs`로 적재. 멱등, 재실행 가능 (PG* 환경변수 필요) |
| `03_design_guide.sql` | `docs/ml/preprocessing_design_guide.md`의 참조 24편 + method 17개를 등록. stage 어휘도 8단계로 확장 |
| `99_verify.sql` | 검증 쿼리 모음. 실행 파일이 아니라 한 블록씩 복사해서 확인용 |

## 적용 상태

**2026-09-02 적용 완료** — `aecd_platform`(PostgreSQL 15.15, 윈도우 호스트)에
`experiment` 스키마 5개 테이블 생성 + `02_import_registry.sql`로 실험 이력
55건 적재. FK(`run_measurements.measurement_id → measurement.measurements`)와
`UNIQUE NULLS NOT DISTINCT` 제약 검증 완료.

### WSL에서 접속할 때 (2026-09-02 조치 완료)

DBeaver(윈도우 앱)의 `localhost`와 WSL의 `localhost`는 **서로 다른 서버**다.
WSL 안에는 PostgreSQL **16**이 5432에 떠 있고, 실제 `aecd_platform`은 윈도우
쪽 PostgreSQL **15.15**에 있다. 그래서 `.env`의 `PGHOST=localhost`로는 엉뚱한
서버에 붙어 인증 실패한다.

**즉시 동작하는 방법** — 스크립트 실행 전에:

```bash
source scripts/db/pghost.sh    # 후보 호스트에 실제 인증을 시도해 맞는 쪽을 PGHOST로 export
```

게이트웨이 IP는 WSL 재시작 시 바뀌므로 `.env`에 하드코딩하지 않는다.
`pghost.sh`는 인증 성공 여부로 고르기 때문에 아래 mirrored 적용 전/후 모두
올바르게 동작한다.

**근본 해결 (적용됨, 재시작 대기 중)** — `C:\Users\user\.wslconfig`에
`networkingMode=mirrored`를 추가했다 (백업: `.wslconfig.bak-20260902`).
적용되면 WSL 안에서도 `localhost:5432`가 윈도우 PostgreSQL을 가리킨다.

발효시키려면 **아래 두 단계를 순서대로** 해야 한다:

```bash
# 1) WSL 내부 PG16을 5433으로 비켜준다 (mirrored 모드에서 5432 충돌 방지)
sudo sed -i 's/^port = 5432/port = 5433/' /etc/postgresql/16/main/postgresql.conf
sudo service postgresql restart
```
```powershell
# 2) PowerShell에서 WSL 재시작
wsl --shutdown
```

1번을 건너뛰면 재시작 후 WSL의 PostgreSQL 16이 포트 충돌로 기동에 실패할 수
있다 (윈도우 쪽 15.15가 이미 5432를 점유). 되돌리려면 `.wslconfig`에서
`networkingMode=mirrored` 줄을 지우고 포트를 5432로 복원한다.

## 적용
## 적용

```bash
# 1) 스키마 생성
psql -d aecd_platform -f 01_schema.sql

# 2) 기존 실험 이력 55건 적재 — 둘 중 하나
psql -d aecd_platform -f 02_import_registry.sql          # SQL만으로

# 3) 전처리 설계 가이드의 논문/method 등록
psql -d aecd_platform -f 03_design_guide.sql
# 또는
python scripts/db/experiment_tracking/import_registry.py --dry-run
python scripts/db/experiment_tracking/import_registry.py
```

DBeaver를 쓴다면 `01_schema.sql` → `02_import_registry.sql` 순서로 SQL 편집기에
붙여넣고 실행하면 된다 (두 파일 모두 DB명 가드 포함).

레지스트리에 새 실험이 append되면 `02_import_registry.sql`을 다시 생성하거나
`import_registry.py`를 재실행한다 — 둘 다 `ON CONFLICT`로 멱등이다.

스크립트 안에 `current_database() <> 'aecd_platform'`이면 중단하는 가드가
있어 다른 DB에 실수로 실행되지 않는다. 적용 후 `99_verify.sql`의 A 블록
(A1~A4)을 실행해 테이블·FK·제약이 의도대로 걸렸는지, 그리고 기존
`master`/`clinical`/`measurement` 테이블 수가 그대로인지 확인할 것.

롤백: `DROP SCHEMA experiment CASCADE;` (이 도메인만 제거)

## 테이블 구성

`experiment.runs`가 **일반 실험 이력 본체**다. `logs/experiment_registry.json`의
14개 필드(phase/hypothesis/cancer_types/result_summary/artifacts_dir 등)가 그대로
컬럼으로 있고, 과거 실험 55건이 여기 들어간다. 전처리 실험은 그 특수 케이스로,
`method_id`(→ `preprocessing_methods`)와 전처리 전용 컬럼
(`config_snapshot`, `data_query_filters`, QC 카운트)을 추가로 채운다.
일반 실험은 그 컬럼들이 NULL이다.

## 설계 요지

- `master` / `clinical` / `measurement`는 **읽기 전용** — 이 스키마는 FK로
  참조만 하고 `CREATE`/`ALTER`하지 않는다.
- `experiment.run_measurements`가 `measurement.measurements`에 **진짜 FK**로
  걸린다. 실험이 존재하지 않는 measurement를 참조하면 DB가 거부한다 —
  별도 DB에 ID만 복사해두는 방식으로는 얻을 수 없는 보장.
- `result_summary`는 산문(예: "Det AUC 0.793, Id F1 0.356")이며 **절대
  `run_metrics`로 파싱하지 않는다**. 산문에서 구조화된 지표를 뽑아내는 건
  숫자를 지어내는 것과 같다 — `run_metrics`는 실제 구조화된 지표를 산출한
  실행만 채운다.
- `audit_status`는 사람 검수 게이트. `approved`가 아닌 방법은 비교표에
  올리지 않는다 (`@sers-preprocessing-comparator` 규칙).

## 스키마 변경 시

`02_*.sql`, `03_*.sql` 처럼 번호를 올려 **새 파일로 추가**한다
(`scripts/db/clinical_unified/`와 같은 컨벤션). `01_schema.sql`을 직접
수정하면 이미 적용된 DB와 파일이 어긋난다.

## 쓰기 경로

`src/sers/preprocessing_lab/db.py`의 `ExperimentTracker`가 유일한 쓰기
지점이다. `src/sers/aecd_api/repository.py`는 읽기 전용이므로 혼동하지 말 것.
