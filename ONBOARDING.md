# SERS-AI 팀원 온보딩 가이드

> **합류를 환영합니다! 이 문서부터 읽으세요.**
> - 개발 환경·git 규칙의 상세 → [`CONTRIBUTING.md`](CONTRIBUTING.md)
> - 도메인 규칙의 **원본(SSOT)** → [`.github/copilot-instructions.md`](.github/copilot-instructions.md) 와 Obsidian vault `01_Projects/SERS-AI/`
> - 아래 수치·규칙은 요약본이며, 최신/정확한 값은 위 원본을 확인하세요.

---

## 1. 프로젝트 한눈에

**SERS-AI** (저장소: `MCED-Platform`)는 소변 대사체의 **표면증강라만분광(SERS)** 신호로 **다중 암을 스크리닝**하는 파이프라인입니다. (SOLUM Healthcare)

- **2단계 분류기**
  1. **Cancer Screening** — 암 vs 비암
  2. **Cancer Type ID** — 암종(subtype) 구분
  - ⚠️ "Stage 1 / Stage 2" 라는 표현은 **쓰지 않습니다** (위 명칭 사용)
- 한국 여러 병원에서 수집한 11+ 암종/대조군, 2종 라만 장비(Thermo, Medical)
- **규제 대상 의료기기(MFDS) + 특허 포트폴리오** 보유 → 데이터·표현에 항상 신중

## 2. 저장소 둘러보기

| 위치 | 내용 |
|------|------|
| `src/sers/` | 코어 라이브러리 (I/O → 신호처리 → 전처리 → QC → 분석 → 시각화) |
| `models/` | 학습/평가 코드 + production 아티팩트 |
| `scripts/` | `pipeline · analysis · deployment · db · qc_validation` 실행 스크립트 |
| `tests/` | 단위 테스트 |
| `docs/` | 프로젝트 구조·워크플로우·용어·실험 맥락 등 상세 문서 |
| `publications/aacr/` | 학회 figure 코드 |
| `config/` | `config.yaml`, `environment.yml` |

- 상세 구조: `docs/PROJECT_STRUCTURE.md`
- 모델 워크플로우: `docs/MODEL_WORKFLOW.md`
- 용어 표준: `docs/TERMINOLOGY_STANDARD.md`

## 3. 개발 환경 셋업 (요약)

상세는 [`CONTRIBUTING.md`](CONTRIBUTING.md). 핵심만:

```bash
git clone https://github.com/SERS-AI-Platform/MCED-Platform.git
cd MCED-Platform
conda env create -f config/environment.yml && conda activate sers-analysis
pip install -e ".[all]"          # 재현 설치를 원하면: pip install -r requirements.lock
pre-commit install               # 커밋 시 자동 린트/포맷/메시지 검사
pytest -m "not slow"             # 셋업 확인 — 통과하면 정상
```

## 4. 우리가 일하는 방식 (협업 규칙) ⭐

**`main`에 직접 push할 수 없습니다.** 모든 변경은 PR(Pull Request)을 거칩니다.

```bash
git checkout main && git pull           # 최신 main
git checkout -b feat/내작업              # 브랜치: feat/ fix/ docs/ chore/
# ... 작업 ...
git commit -m "feat(scope): 설명"        # Conventional Commits (pre-commit이 강제)
git push -u origin feat/내작업
gh pr create                            # 또는 GitHub 웹에서
```

- PR을 올리면 **CI(린트 + 테스트)가 자동 실행** → **통과해야 병합 가능**
- 검토 후 PR 페이지에서 **"Merge pull request"** (현재 리뷰 승인 0건 — 작성자가 직접 병합 가능)
- PR마다 자동 체크리스트가 뜹니다 — 특히 **PII·비밀키 미포함**을 확인하세요

## 5. 🔴 꼭 알아야 할 도메인 규칙 (모르면 결과가 왜곡됩니다)

> 원본·최신은 [`.github/copilot-instructions.md`](.github/copilot-instructions.md). 아래는 요약입니다.

1. **Hospital confound (병원 교란) — 외부 보고 시 필수 주의**
   Cancer Screening(암 vs 대조군) 성능은 **병원 교란 가능성**이 있습니다 (대조군과 암 샘플의 병원 출처가 다름). 외부 보고/대시보드에서 cross-hospital 일반화를 말할 땐 **반드시 disclaimer를 명시**하세요. (Cancer Type ID는 대체로 honest biology) — 근거: vault `Findings/2026-04-09__hospital-confound.md`
2. **PAN 구성**: PAN = CPAN(충북대) + YPAN(연세) = 100명. **SPAN(삼성 post-op)은 PAN에 절대 포함 금지** (스크리닝 부적합)
3. **감별 피크 출처**: 우리 모델/데이터(LR 계수·SHAP·figure)에서 나온 것만 사용. 문헌 피크는 보조 해석용. 데이터 없는 암종은 "데이터 미보유"로 표기
4. **Run 비교 4규칙**: ① aggregation(mean/medoid) ② cancer set ③ non-cancer set ④ sample count — **4개가 모두 같아야 비교 가능**. (예: 8-class와 5-class 결과는 직접 비교 불가)
5. **QC 처리**: `results/processed_spectra.csv` 재생성 시 `--skip-qc` 사용 (QC는 inference 단계에만 적용)
6. **용어**: "Cancer Screening" / "Cancer Type ID" 사용 (Stage 1/2 금지)

## 6. 데이터 · 보안 규칙 (필수)

- **PII 금지**: 코드/SQL/노트북에 환자명·주민번호·병원 식별자 등을 **하드코딩하지 마세요**
- **원시 데이터·결과물은 git에 올리지 않습니다** (`.gitignore`로 제외 — `data/`, `results/`, `*.db`, 발표 산출물 등)
- **비밀키 금지**: `.env`, `credentials/`, `secrets/` (이미 제외됨). 키가 필요하면 팀에 문의
- **CSV 저장 시 `encoding='utf-8-sig'`** (Windows/Excel 한글 깨짐 방지)

## 7. 자주 쓰는 명령

```bash
pytest                       # 전체 테스트
pytest -m "not slow"         # 빠른 테스트 (실데이터 불필요)
ruff check src/ tests/       # 린트
ruff format src/ tests/      # 자동 포맷
python main.py               # 전처리 파이프라인
```

> 데이터/대시보드 변경 후 **동기화 명령**(`sync_data.py`, `sync_dashboard.py`)은 `.github/copilot-instructions.md` §6 참조.

## 8. 역할별 시작점

- **ML / SW 엔지니어**: `src/sers/` 구조 파악 → `docs/MODEL_WORKFLOW.md` → 테스트 실행 → 작은 `fix/` PR로 워크플로우 체험
- **데이터 / 임상 담당**: `data/clinical_data/*_column_detail.xlsx`(데이터 사전) → `docs/clinical_use/` → 위 §5 도메인 규칙 숙지

## 9. 막히면 어디를 보나

| 궁금한 것 | 보는 곳 |
|-----------|---------|
| 프로젝트 구조·워크플로우 | `docs/` |
| 도메인 규칙·실험 맥락 | `.github/copilot-instructions.md`, vault `01_Projects/SERS-AI/` |
| 기여 방법·커밋 규칙 | `CONTRIBUTING.md` |
| 코드 소유자(누구에게 물을지) | `.github/CODEOWNERS` |
| 그래도 막힐 때 | GitHub 이슈 등록 또는 팀 문의 |

---

## ✅ 첫 주 체크리스트

- [ ] 개발 환경 셋업 + `pytest -m "not slow"` 통과 확인
- [ ] 이 문서 + `CONTRIBUTING.md` + `.github/copilot-instructions.md` 정독
- [ ] 작은 `docs/` 또는 `fix/` PR을 하나 올려 **PR → CI → 병합** 흐름 체험
- [ ] §5 도메인 규칙 숙지 (특히 **hospital confound**)
- [ ] 본인 역할 영역(`src/sers/` 또는 `data/clinical_data/`) 둘러보기
