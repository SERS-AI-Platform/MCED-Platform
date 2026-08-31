# SERS Clinical Webapp — Windows .exe 빌드 가이드

병원 PC에 배포하기 위한 standalone Windows 실행 파일 빌드 방법입니다.

## 사전 준비 (Windows PC에서)

### 1. Python 설치
- Python 3.10 이상: https://www.python.org/downloads/
- 설치 시 **"Add Python to PATH"** 체크 필수

### 2. 프로젝트 코드 복사
WSL에서 작업 중인 SERS-AI 폴더 전체를 Windows로 복사:
```
C:\SERS-AI\
```

특히 다음 디렉토리는 반드시 포함:
- `scripts/deployment/` — 웹앱 코드
- `artifacts/usersnet/current/` — uSERS-Net/STK-V2 모델
- `artifacts/baselines/lr-fusion/v1.0.0/` — LR fallback 모델
- `src/sers/` — 전처리 라이브러리
- `scripts/training/` — uSERS-Net production builder

### 3. 아이콘 파일 (선택)
프로그램에 사용할 아이콘을 준비:
- 형식: `.ico` (256x256 권장)
- 위치: `scripts/deployment/static/icon.ico`
- 무료 변환 도구: https://convertio.co/png-ico/

아이콘이 없으면 기본 PyInstaller 아이콘이 사용됩니다.

---

## 빌드 방법

### 자동 빌드 (권장)
1. Windows에서 **명령 프롬프트** 열기 (cmd)
2. 프로젝트 디렉토리로 이동:
   ```
   cd C:\SERS-AI
   ```
3. 빌드 스크립트 실행:
   ```
   scripts\deployment\build_windows.bat
   ```
4. 5~10분 후 빌드 완료

### 수동 빌드
```cmd
pip install pyinstaller fastapi uvicorn jinja2 python-multipart joblib scikit-learn scipy xgboost numpy pandas

pyinstaller --clean scripts\deployment\sers_clinical.spec
```

---

## 빌드 결과

```
dist\
└── SERS_Clinical\
    ├── SERS_Clinical.exe          ← 실행 파일 (이것을 더블클릭)
    ├── _internal\                 ← 의존 라이브러리
    │   ├── artifacts\
    │   │   ├── usersnet\
    │   │   └── baselines\
    │   ├── scripts\
    │   ├── src\
    │   └── ... (Python DLL 등)
    └── ...
```

**전체 폴더 크기**: 약 500MB ~ 1GB (sklearn, scipy, xgboost 포함)

---

## 사용 방법

### 처음 실행
1. `dist\SERS_Clinical\SERS_Clinical.exe` 더블클릭
2. 콘솔 창이 뜨면서 서버 시작 메시지 표시
3. 자동으로 기본 브라우저(Chrome/Edge)에서 `http://127.0.0.1:8080` 열림
4. 로그인 화면에서:
   - **admin** / `admin123` (관리자)
   - **doctor1** / `admin123` (의사)
   - 또는 회원가입

### 종료
- 콘솔 창을 닫거나 `Ctrl+C` 누름

### 바탕화면 바로가기 만들기
1. `SERS_Clinical.exe` 우클릭
2. **보내기 → 바탕 화면(바로 가기 만들기)**
3. 바탕화면에 아이콘 생성됨
4. 우클릭 → 속성 → **아이콘 변경**으로 커스텀 아이콘 적용 가능

### 시작 메뉴에 등록
1. 바로가기 파일을 다음 폴더로 복사:
   ```
   %APPDATA%\Microsoft\Windows\Start Menu\Programs\
   ```
2. 시작 메뉴에서 "SERS"로 검색하면 나타남

---

## 배포 (다른 PC로)

1. `dist\SERS_Clinical\` 폴더 전체를 USB 또는 네트워크 드라이브로 복사
2. 대상 PC의 원하는 위치에 붙여넣기 (예: `C:\Program Files\SERS_Clinical\`)
3. `SERS_Clinical.exe` 더블클릭하여 실행
4. **Python 설치 불필요** — 모든 의존성이 .exe에 포함됨

---

## 데이터 저장 위치

- **DB**: `SERS_Clinical.exe`와 같은 폴더에 `clinical_data.db` 자동 생성
- **감사 로그**: 같은 DB에 audit_log 테이블로 저장
- **PDF 보고서**: 메모리에서 생성 후 브라우저로 다운로드 (디스크 저장 안 함)

⚠ **주의**: PC를 옮기면 DB도 함께 옮겨야 환자 기록이 유지됩니다.

---

## 문제 해결

### "ImportError: No module named ..."
- `sers_clinical.spec`의 `hiddenimports`에 누락된 모듈 추가 후 재빌드

### "Model not found"
- `artifacts/usersnet/current/`와 `artifacts/baselines/lr-fusion/v1.0.0/`이 프로젝트 루트에 있는지 확인

### 콘솔 창 숨기기 (production용)
- `sers_clinical.spec`에서 `console=True` → `console=False`로 변경 후 재빌드
- 단, 디버그용으로는 console=True 권장

### 방화벽 경고
- Windows 방화벽이 localhost 8080 포트 차단 시 "허용" 클릭
- 외부 네트워크는 차단 (127.0.0.1만 사용하므로 안전)

---

## 빌드 환경 정보

| 항목 | 권장 |
|------|------|
| OS | Windows 10/11 64-bit |
| Python | 3.10 ~ 3.12 |
| RAM | 8GB+ (빌드 시) |
| Disk | 5GB+ (빌드 임시 파일 + 결과물) |
| PyInstaller | 6.0+ |
