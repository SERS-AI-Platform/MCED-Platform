# SERS-AI Project Structure

> Last updated: 2026-03-31

```
SERS-AI/
│
├── main.py                        # 메인 전처리 파이프라인 오케스트레이터
├── pyproject.toml                 # 패키지 메타데이터 (sers-analysis v0.1.0)
├── README.md
│
├── src/sers/                      # ── 코어 라이브러리 ──
│   ├── config.py                  #   설정 관리, QC 임계값, 경로
│   ├── io.py                      #   스펙트럼 파일 읽기/파싱
│   ├── preprocessing.py           #   신호처리 (smooth, baseline, normalize)
│   ├── signal.py                  #   Savitzky-Golay, SNV, 리샘플링
│   ├── validation.py              #   데이터 검증 (NaN, 중복, 구조 체크)
│   ├── analysis.py                #   데이터셋 분석, 그룹 통계
│   ├── scoring.py                 #   스코어링 유틸리티
│   ├── cli.py                     #   CLI 엔트리포인트 (sers 명령)
│   ├── logging_config.py          #   로깅 설정
│   ├── qc/                        #   3단계 QC (intensity gate, RSD, correlation)
│   ├── visualization/             #   시각화 (spectra, ROC, SHAP, confusion matrix)
│   ├── ingest_ypan.py             #   Y-Pancreatic 데이터 수집
│   ├── reingest_staging.py        # LEGACY 비관리형 row-order 재현 전용
│   └── update_lun_dates.py        #   Lung 날짜 업데이트
│
├── models/                        # ── 모델 학습/평가 ──
│   ├── model.py                   #   ResNet18-1D 2단계 계층 모델
│   ├── train.py                   #   학습 파이프라인 (5-fold CV, MLflow→SQLite)
│   ├── test.py                    #   테스트셋 평가
│   ├── train_ensemble.py          #   앙상블 학습
│   ├── train_multichannel.py      #   멀티채널 학습
│   ├── tune_torch.py              #   하이퍼파라미터 튜닝
│   ├── run_train_val_test.py      #   60/20/20 분할 학습
│   ├── run_multimodal.py          #   퓨전 모델 (SERS + clinical)
│   ├── run_clinical_analysis.py   #   임상 서브그룹 분석
│   ├── run_confounding_analysis.py #  교란변수 평가
│   ├── run_step4_normalization.py #   정규화 실험
│   ├── run_pan_improvement.py     #   췌장암 개선 추적
│   ├── pancreatic_experiment.py   #   췌장암 특화 실험
│   ├── build_production_model.py  #   프로덕션 모델 패키징
│   ├── learning_curve_comparison.py # 학습 곡선 분석
│   ├── plot_overall_comparison.py #   모델 간 비교 플롯
│   ├── production/                #   프로덕션 아티팩트 (joblib, manifest)
│   ├── results/                   #   학습 결과 (predictions, metrics)
│   └── manifest.json              #   학습 실행 메타데이터
│
├── scripts/                       # ── 스크립트 (카테고리별) ──
│   ├── pipeline/                  #   전처리/QC 파이프라인
│   │   ├── run_qc_preprocess.py   #     전체 QC→전처리 파이프라인
│   │   ├── run_qc_threshold_experiment.py # QC 임계값 실험
│   │   ├── pancreatic_preprocess.py #    췌장암 전처리
│   │   ├── standardize_clinical_data.py # 임상 데이터 표준화
│   │   └── standardize_span_clinical.py # SPAN 임상 표준화
│   │
│   ├── analysis/                  #   분석/비교 스크립트
│   │   ├── analyze_equipment.py   #     5기기 스펙트럼 재현성 분석
│   │   ├── analyze_equipment_qc.py #    AECD 표준 QC 분석
│   │   ├── analyze_raw_grid.py    #     Grid 원시 분석
│   │   ├── compare_baseline_methods.py # Rolling Min vs ALS 비교
│   │   ├── compare_baseline_phaseQ.py # Phase Q 기준 baseline 비교
│   │   ├── compare_grid_performance.py # Grid 성능 비교
│   │   ├── create_rds.py          #     R RDS 파일 생성
│   │   ├── explore_data.py        #     데이터 탐색
│   │   ├── export_group_spectra.py #    그룹별 스펙트럼 내보내기
│   │   ├── extract_dashboard_spectra.py # 대시보드용 데이터 추출
│   │   ├── generate_pancreatic_reports.py # 췌장암 리포트
│   │   ├── generate_qc_preprocessing_examples.py # QC 시각화
│   │   ├── plot_spectral_interpretation_7cancer.py # 7암 스펙트럼 해석
│   │   ├── run_cpan_binary_experiment.py # CPAN 이진분류 실험
│   │   ├── backfill_experiment_registry.py # MLflow 레지스트리
│   │   └── test_transform.py      #     변환 탐색
│   │
│   ├── deployment/                #   배포/추론
│   │   ├── sers_predict.py        #     CLI 추론 (production 모델 로드)
│   │   ├── sers_webapp.py         #     비임상 연구용 FastAPI 웹앱
│   │   └── sers_api.py            #     비임상 연구용 REST API
│   │
│   ├── db/                        #   데이터베이스
│   │   ├── db_create.py           #     DB 스키마 초기화
│   │   ├── upload_to_postgres.py  #     PostgreSQL 업로드
│   │   └── upload_to_supabase.py  #     Supabase 업로드
│   │
│   └── qc_validation/             #   QC 검증 (4단계)
│       ├── 01_literature_standards.py
│       ├── 02_statistical_optimization.py
│       ├── 03_clinical_impact.py
│       └── 04_generate_report.py
│
├── config/                        # ── 설정 ──
│   ├── config.yaml                #   마스터 설정 (데이터셋, QC, 전처리)
│   ├── nanostructure.yaml         #   나노구조 파라미터
│   ├── environment.yml            #   Conda 환경
│   └── environment.lock.yml       #   잠금된 의존성
│
├── data/                          # ── 데이터 (git-ignored) ──
│   ├── raw_data/                  #   원시 스펙트럼 (14개 폴더, 476MB)
│   ├── raw_data_medical/          #   의료기기 데이터 (Nanoscope, 472MB)
│   ├── equipment_test_data/       #   장비 재현성 데이터
│   ├── clinical_data/             #   임상 메타데이터 (.xlsm)
│   ├── processed/                 #   전처리 완료 스펙트럼
│   └── Metabolite analysis_Thermo/ # Thermo 대사체 분석
│
├── results/                       # ── 실험 결과 (git-ignored) ──
│   ├── training/                  #   학습 결과 (fold_predictions 등)
│   ├── figures/                   #   분석 figure
│   ├── baseline_comparison/       #   baseline 비교 결과
│   ├── equipment_analysis/        #   장비 분석 결과
│   ├── qc_validation/             #   QC 검증 결과
│   ├── pancreatic/                #   췌장암 실험
│   └── ...                        #   기타 실험별 폴더
│
├── publications/                  # ── 출판물 ──
│   └── aacr/                      #   AACR 포스터 figure 생성
│       ├── src/                   #     figure 생성 코드 (Python + R)
│       ├── figures/               #     생성된 figure
│       └── data/                  #     포스터용 데이터
│
├── metabolite_profiling/          # ── 대사체 프로파일링 (서브프로젝트) ──
│   ├── analyze_metabolites.py     #   대사체 정량
│   ├── cross_reference_peaks.py   #   피크 주석
│   ├── Thermo_Metabolite_Dashboard.html # Thermo 대시보드
│   ├── experiments/               #   5단계 체계적 분석
│   ├── data/                      #   대사체 데이터 + Thermo JSON
│   ├── figures/                   #   분석 figure
│   └── poster_figures/            #   포스터 figure
│
├── docs/                          # ── 문서 ──
│   ├── README.md                  #   문서 인덱스
│   ├── api/                       #   API, CLI, 사용 예제
│   ├── architecture/              #   프로젝트·모델·SaMD 구조
│   ├── ml/                        #   실험, MLOps, explainability
│   ├── clinical/                  #   임상 사용·평가 문서
│   ├── compliance/                #   QMS, SBOM, 용어·거버넌스
│   ├── sharepoint/                #   SharePoint 연계 작업 문서
│   ├── assets/                    #   문서 이미지 및 첨부 자산
│   └── CHANGELOG.md               #   변경 이력
│
├── figures/                       # 프로젝트 공용 figure
├── logs/                          # 실행 로그
├── notebooks/                     # Jupyter 탐색 노트북
├── tests/                         # 테스트
├── infra/                         # Docker, 배포 설정
├── .github/workflows/             # CI/CD
└── mlflow.db                      # MLflow 실험 추적 (SQLite, 자동 생성)
```

## 엔트리 포인트

| 용도 | 명령 |
|------|------|
| 전처리 + QC 파이프라인 | `sers preprocess` |
| Active STK-V2/uSERS-Net 학습 | `sers train stacking` |
| Baseline/legacy 모델 학습 | `sers train resnet18`, `sers train xgboost` |
| 모델 평가 | `sers test` |
| 프로덕션 추론 (CLI) | `python scripts/deployment/sers_predict.py` |
| 비임상 연구용 웹앱 추론 | `python scripts/deployment/sers_webapp.py` |
| 직접 QC 전처리 | `python scripts/pipeline/run_qc_preprocess.py` |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` |

## MLflow 실험 추적

- 백엔드: SQLite (`mlflow.db`, 프로젝트 루트에 자동 생성)
- 사용 파일: legacy wrapper 경로(`models/legacy/scripts/train.py`) 또는 `sers train ...` (--no-mlflow 플래그로 비활성화 가능)
- UI 조회: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
- 직접 조회: `sqlite3 mlflow.db "SELECT * FROM runs ORDER BY start_time DESC LIMIT 10;"`

## 스크립트 실행 시 참고

모든 scripts/ 하위 파일은 `PROJECT_ROOT`를 자동 계산하여 `src/sers` 및 `config/`를 import합니다.
프로젝트 루트에서 실행:

```bash
cd /home/user/SERS-AI
python scripts/analysis/analyze_equipment.py
python scripts/pipeline/run_qc_preprocess.py
```
