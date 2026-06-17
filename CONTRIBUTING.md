# 기여 가이드 (CONTRIBUTING)

SERS-AI / MCED-Platform 협업 규칙입니다. **처음 합류하셨다면 이 문서부터** 읽어주세요.

## 1. 개발 환경 셋업

```bash
# 1) 저장소 클론
git clone https://github.com/SERS-AI-Platform/MCED-Platform.git
cd MCED-Platform

# 2) conda 환경 생성 (권장)
conda env create -f config/environment.yml
conda activate sers-analysis

# 3) 패키지 설치 (편집 모드 + 개발 도구)
pip install -e ".[all]"

# 4) (선택) 고정 버전으로 재현 설치 — 동일 환경 보장
#    requirements.lock 은 의존성을 정확한 버전으로 핀 고정한 파일입니다.
pip install -r requirements.lock

# 5) pre-commit 훅 설치 (커밋 시 자동 린트/포맷/메시지 검사)
pre-commit install
```

## 2. 작업 흐름 (git) — 모든 변경은 PR로

`main` 브랜치에는 **직접 push할 수 없습니다**(브랜치 보호). 항상 곁가지(브랜치)에서 작업하고 PR(Pull Request)로 합칩니다.

```bash
git checkout main && git pull            # 최신 main 받기
git checkout -b feat/my-change           # 작업 브랜치 생성
# ... 코드 작업 ...
git add -p                               # 변경을 골라서 스테이징
git commit -m "feat(scope): 설명"        # 커밋 (메시지 규칙은 3번 참고)
git push -u origin feat/my-change        # 브랜치 push
gh pr create                             # PR 생성 (또는 GitHub 웹에서)
```

- **브랜치 이름**: `feat/...`, `fix/...`, `docs/...`, `chore/...`
- PR을 올리면 CI(core import smoke + lint + scoped mypy + tests/coverage)가 자동 실행됩니다.
- 검토 후 GitHub PR 페이지에서 **"Merge pull request"** 클릭. (현재 리뷰 승인 0건 설정 — 소규모 팀이라 작성자가 직접 병합 가능)

## 3. 커밋 메시지 규칙 (Conventional Commits)

형식: `<type>(<scope>): <설명>` — pre-commit 훅이 강제합니다.

| type | 용도 |
|------|------|
| `feat` | 기능 추가 |
| `fix` | 버그 수정 |
| `docs` | 문서 |
| `refactor` | 동작 변화 없는 구조 개선 |
| `test` | 테스트 |
| `chore` | 잡무(설정, 빌드 파일 등) |
| `perf` / `build` / `ci` | 성능 / 빌드 / CI |

예: `feat(qc): add saturation gate for medical data`

## 4. 코드 품질

```bash
ruff check src/ tests/ scripts/quality/                               # 린트 검사
ruff format src/ tests/                                               # 자동 포맷
mypy                                                                  # 현재 CI-gated 타입 표면 검사
pytest --cov=sers --cov-report=term-missing --cov-report=xml --cov-fail-under=35
python scripts/quality/coverage_by_process.py coverage.xml            # 과정별 커버리지 요약
pytest -m "not slow"                                                  # 빠른 테스트만 (실데이터 파일 불필요)
```

현재 CI는 Python 3.10/3.11/3.12에서 ruff, mypy, pytest+coverage, 과정별 coverage summary를 실행합니다.
mypy는 `pyproject.toml`에 명시한 CLI/config/scoring 표면부터 게이트로 사용하며, 전체 `src/` 타입 검사는 별도 마이그레이션 과제입니다.
신규/수정 코드는 위 검사를 통과하도록 작성해 주세요.

## 5. 데이터 · 임상 규칙 (필독)

- **원시 데이터/결과물은 git에 올리지 않습니다** — `.gitignore`로 제외됨 (`data/`, `results/`, `*.db`, 발표 산출물 등).
- **환자 식별정보(PII) 금지**: 코드/SQL/노트북에 주민번호·환자명·병원 식별자 등을 **절대 하드코딩하지 마세요.**
- **비밀키/토큰 금지**: `.env`, `credentials/`, `secrets/`는 커밋 금지(이미 제외됨). 키가 필요하면 팀에 문의.
- **데이터 사전**: 컬럼 정의는 `data/clinical_data/*_column_detail.xlsx` 참조. (코드(YAML)화 진행 예정)
- **임상 프로토콜**: `docs/clinical_use/` 참조.
- CSV 저장 시 한글 깨짐 방지를 위해 `encoding="utf-8-sig"` 사용.

## 6. 도움말 / 참고 문서

- 프로젝트 구조: `docs/PROJECT_STRUCTURE.md`
- 개발 워크플로우: `docs/MODEL_WORKFLOW.md`
- 용어 표준: `docs/TERMINOLOGY_STANDARD.md`
- 막히면 GitHub 이슈를 열거나 팀에 문의하세요.
