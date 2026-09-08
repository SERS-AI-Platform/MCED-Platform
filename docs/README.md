# SERS-AI Documentation

이 디렉터리는 저장소에서 함께 버전 관리해야 하는 기술·임상·규정 문서를 주제별로 정리한다.

## 문서 분류

| 경로 | 내용 |
| --- | --- |
| [`api/`](api/) | API 레퍼런스, CLI 사용법, 실행 예제 |
| [`architecture/`](architecture/) | 프로젝트 구조, 모델 워크플로우, SaMD 및 데이터베이스 설계 |
| [`ml/`](ml/) | 실험 기록, 결과, MLOps, explainability, calibration |
| [`clinical/`](clinical/) | 임상 사용 설명서, 사용적합성 평가, 추적성 문서 |
| [`compliance/`](compliance/) | QMS, SBOM, 개발 자산 거버넌스, 용어 표준 |
| [`sharepoint/`](sharepoint/) | SharePoint 게시·연계용 작업 문서와 버전 기록 |
| [`assets/`](assets/) | 문서에서 참조하는 이미지와 기타 정적 자산 |
| [`CHANGELOG.md`](CHANGELOG.md) | 저장소 변경 이력 |

## 주요 시작점

- 프로젝트 구조: [`architecture/PROJECT_STRUCTURE.md`](architecture/PROJECT_STRUCTURE.md)
- 모델 워크플로우: [`architecture/MODEL_WORKFLOW.md`](architecture/MODEL_WORKFLOW.md)
- CLI: [`api/CLI.md`](api/CLI.md)
- 실험 맥락: [`ml/EXPERIMENT_CONTEXT.md`](ml/EXPERIMENT_CONTEXT.md)
- 임상 사용 문서: [`clinical/README.md`](clinical/README.md)
- QMS 문서 목록: [`compliance/qms/README.md`](compliance/qms/README.md)
- 용어 표준: [`compliance/TERMINOLOGY_STANDARD.md`](compliance/TERMINOLOGY_STANDARD.md)

## 저장 위치 원칙

- 저장소 동작과 함께 변경되어야 하는 재현 가능한 문서만 `docs/`에 둔다.
- 특허 작업 자료는 C 드라이브 Obsidian vault의 `07_Patent/SERS-AI_repository/`에서 관리한다.
- 대용량 연구 산출물과 publication 작업물은 저장소 문서와 분리한다.
- 문서를 이동할 때 저장소 내부 링크와 생성 스크립트의 출력 경로도 함께 갱신한다.
- 폐기 여부가 승인된 문서만 추후 `archive/`로 이동한다.
