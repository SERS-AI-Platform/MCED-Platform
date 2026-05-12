"""
SERS Clinical Webapp - Internationalization (한국어/English)
"""

from starlette.requests import Request

STRINGS = {
    "ko": {
        # App
        "app_title": "SERS 암 선별검사 시스템",
        "app_subtitle": "소변 SERS 기반 다중 암 스크리닝",
        "version": "v1.0",
        "company": "SOLUM Healthcare",

        # Login
        "login": "로그인",
        "username": "사용자 ID",
        "password": "비밀번호",
        "login_submit": "로그인",
        "login_error": "사용자 ID 또는 비밀번호가 올바르지 않습니다.",
        "logout": "로그아웃",
        "register": "회원가입",
        "register_title": "계정 등록",
        "display_name": "이름",
        "role": "역할",
        "role_technician": "검사실 기사",
        "role_clinician": "임상의",
        "password_confirm": "비밀번호 확인",
        "register_submit": "등록",
        "register_success": "계정이 등록되었습니다. 로그인해주세요.",
        "register_error_exists": "이미 사용 중인 사용자 ID입니다.",
        "register_error_password": "비밀번호가 일치하지 않습니다.",
        "register_error_fields": "모든 필수 항목을 입력해주세요.",
        "have_account": "이미 계정이 있으신가요?",
        "no_account": "계정이 없으신가요?",

        # Decision standard
        "mode_title": "분석 기준",
        "mode_screening": "선별검사",
        "mode_screening_desc": "민감도 우선 — 목표 민감도 ≥ 95%",
        "mode_balanced": "표준 판정 기준",
        "mode_balanced_desc": "민감도/특이도 균형점 (Youden's J)",
        "mode_confirmatory": "확진검사",
        "mode_confirmatory_desc": "특이도 우선 — 목표 특이도 ≥ 95%",
        "analysis_standard": "분석 기준",
        "standard_decision_rule": "표준 판정 기준",
        "decision_threshold": "판정 기준값",
        "sensitivity": "민감도",
        "specificity": "특이도",
        "select": "선택",

        # Patient
        "patient_title": "환자 정보 입력",
        "patient_id": "환자번호",
        "age": "나이",
        "sex": "성별",
        "male": "남성",
        "female": "여성",
        "bmi": "체질량지수 (BMI)",
        "bmi_optional": "선택 — 미입력 시 중앙값(24.0) 대체",
        "bio_constraint_male": "전립선암(PRO) 분석 가능, 난소암(OVA) 분석 제외",
        "bio_constraint_female": "난소암(OVA) 분석 가능, 전립선암(PRO) 분석 제외",
        "model_fusion": "SERS + 임상정보 융합 모델",
        "model_sers_only": "SERS 단독 모델",
        "model_stacking_v2": "STK-V2 앙상블 모델",
        "next": "다음",

        # Upload
        "upload_title": "스펙트럼 업로드",
        "upload_hint": "CSV 파일을 드래그하거나 클릭하여 업로드",
        "upload_format": "파형수(Wavenumber) + 강도(Intensity) 2열 CSV 형식",
        "upload_recommend": "환자당 반복 측정 5개 권장 (1-20개 허용)",
        "start_analysis": "분석 시작",
        "confirm_analysis": "환자 {patient_id}의 스펙트럼 {n}개를 표준 판정 기준으로 분석하시겠습니까?",

        # QC
        "qc_title": "품질 관리 확인",
        "qc_pass": "통과",
        "qc_fail": "실패",
        "qc_summary": "{passed}/{total}개 스펙트럼 QC 통과",
        "qc_all_fail": "모든 스펙트럼이 품질 기준을 충족하지 않습니다. 재측정이 필요합니다.",
        "qc_intensity": "강도 기준",
        "qc_correlation": "반복 상관 계수",
        "view_results": "결과 확인",

        # Results
        "results_title": "분석 결과",
        "positive": "추가 확인 권고",
        "negative": "기준 미만",
        "screening_index": "SSI 점수",
        "estimated_type": "최상위 추정 암종",
        "confidence": "분류 확률",
        "confidence_high": "높은 분류 확률",
        "confidence_medium": "중간 분류 확률",
        "confidence_low": "낮은 분류 확률",
        "type_probabilities": "암종 분류 확률",
        "replicate_details": "반복 측정 상세",
        "disclaimer": "본 결과는 선별검사 목적의 모델 기반 참고 지표입니다. SSI 점수와 암종 분류 확률은 최종 진단이나 강한/약한 양성 등급을 의미하지 않으며, 최종 판단은 임상 전문의의 평가에 따릅니다.",

        # Cancer types
        "cancer_PRO": "전립선암",
        "cancer_BRE": "유방암",
        "cancer_LUN": "폐암",
        "cancer_CRC": "대장암",
        "cancer_PAN": "췌장암",
        "cancer_CPAN": "췌장암",
        "cancer_OVA": "난소암",
        "cancer_BLC": "방광암",

        # Report
        "generate_report": "보고서 생성",
        "report_title": "SERS 암 선별검사 보고서",
        "report_confirm": "보고서를 생성하시겠습니까? 생성된 보고서는 감사 추적에 기록됩니다.",
        "report_id": "보고서 ID",
        "operator": "검사자",
        "test_date": "검사 일시",
        "patient_info": "환자 정보",
        "test_conditions": "검사 조건",
        "qc_summary_label": "품질 관리 요약",
        "test_result": "검사 결과",

        # Navigation
        "new_patient": "새 환자",
        "step_patient": "환자정보",
        "step_upload": "업로드",
        "step_qc": "QC",
        "step_results": "결과",
        "step_report": "보고서",

        # Audit
        "audit_title": "감사 추적 로그",
        "audit_time": "시각",
        "audit_user": "사용자",
        "audit_action": "행위",
        "audit_detail": "상세",
    },

    "en": {
        "app_title": "SERS Cancer Screening System",
        "app_subtitle": "Urine SERS-based Multi-cancer Screening",
        "version": "v1.0",
        "company": "SOLUM Healthcare",

        "login": "Login",
        "username": "User ID",
        "password": "Password",
        "login_submit": "Log In",
        "login_error": "Invalid user ID or password.",
        "logout": "Logout",
        "register": "Register",
        "register_title": "Create Account",
        "display_name": "Display Name",
        "role": "Role",
        "role_technician": "Lab Technician",
        "role_clinician": "Clinician",
        "password_confirm": "Confirm Password",
        "register_submit": "Register",
        "register_success": "Account created. Please log in.",
        "register_error_exists": "User ID already exists.",
        "register_error_password": "Passwords do not match.",
        "register_error_fields": "Please fill in all required fields.",
        "have_account": "Already have an account?",
        "no_account": "Don't have an account?",

        "mode_title": "Analysis Standard",
        "mode_screening": "Screening",
        "mode_screening_desc": "Sensitivity-first — Target Sensitivity ≥ 95%",
        "mode_balanced": "Standard Decision Rule",
        "mode_balanced_desc": "Balanced sensitivity/specificity (Youden's J)",
        "mode_confirmatory": "Confirmatory",
        "mode_confirmatory_desc": "Specificity-first — Target Specificity ≥ 95%",
        "analysis_standard": "Analysis Standard",
        "standard_decision_rule": "Standard Decision Rule",
        "decision_threshold": "Decision Threshold",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "select": "Select",

        "patient_title": "Patient Information",
        "patient_id": "Patient ID",
        "age": "Age",
        "sex": "Sex",
        "male": "Male",
        "female": "Female",
        "bmi": "BMI",
        "bmi_optional": "Optional — median (24.0) used if omitted",
        "bio_constraint_male": "Prostate (PRO) included, Ovarian (OVA) excluded",
        "bio_constraint_female": "Ovarian (OVA) included, Prostate (PRO) excluded",
        "model_fusion": "SERS + Clinical Fusion Model",
        "model_sers_only": "SERS-only Model",
        "model_stacking_v2": "STK-V2 Ensemble Model",
        "next": "Next",

        "upload_title": "Upload Spectra",
        "upload_hint": "Drag & drop CSV files or click to browse",
        "upload_format": "Wavenumber + Intensity 2-column CSV format",
        "upload_recommend": "5 replicates per patient recommended (1-20 allowed)",
        "start_analysis": "Start Analysis",
        "confirm_analysis": "Analyze {n} spectra for patient {patient_id} using the standard decision rule?",

        "qc_title": "Quality Control Review",
        "qc_pass": "Pass",
        "qc_fail": "Fail",
        "qc_summary": "{passed}/{total} spectra passed QC",
        "qc_all_fail": "All spectra failed quality criteria. Re-measurement is required.",
        "qc_intensity": "Intensity Gate",
        "qc_correlation": "Replicate Correlation",
        "view_results": "View Results",

        "results_title": "Analysis Results",
        "positive": "Further Evaluation Recommended",
        "negative": "Below Decision Threshold",
        "screening_index": "SSI Score",
        "estimated_type": "Top Estimated Cancer Type",
        "confidence": "Classification Probability",
        "confidence_high": "High classification probability",
        "confidence_medium": "Moderate classification probability",
        "confidence_low": "Low classification probability",
        "type_probabilities": "Cancer Type Classification Probabilities",
        "replicate_details": "Replicate Details",
        "disclaimer": "This result is a model-based screening reference. The SSI score and cancer type probabilities are not a final diagnosis and do not represent strong/weak positive grades. Final interpretation should be made by a clinical specialist.",

        "cancer_PRO": "Prostate",
        "cancer_BRE": "Breast",
        "cancer_LUN": "Lung",
        "cancer_CRC": "Colorectal",
        "cancer_PAN": "Pancreatic",
        "cancer_CPAN": "Pancreatic",
        "cancer_OVA": "Ovarian",
        "cancer_BLC": "Bladder",

        "generate_report": "Generate Report",
        "report_title": "SERS Cancer Screening Report",
        "report_confirm": "Generate report? The report will be recorded in the audit trail.",
        "report_id": "Report ID",
        "operator": "Operator",
        "test_date": "Test Date",
        "patient_info": "Patient Information",
        "test_conditions": "Test Conditions",
        "qc_summary_label": "QC Summary",
        "test_result": "Test Result",

        "new_patient": "New Patient",
        "step_patient": "Patient",
        "step_upload": "Upload",
        "step_qc": "QC",
        "step_results": "Results",
        "step_report": "Report",

        "audit_title": "Audit Trail",
        "audit_time": "Time",
        "audit_user": "User",
        "audit_action": "Action",
        "audit_detail": "Detail",
    },
}

DEFAULT_LANG = "ko"
LANG_COOKIE = "sers_lang"


def get_lang(request: Request) -> str:
    return request.cookies.get(LANG_COOKIE, DEFAULT_LANG)


def t(request: Request, key: str, **kwargs) -> str:
    """Translate a key to the current language."""
    lang = get_lang(request)
    strings = STRINGS.get(lang, STRINGS[DEFAULT_LANG])
    text = strings.get(key, STRINGS[DEFAULT_LANG].get(key, key))
    if kwargs:
        text = text.format(**kwargs)
    return text


def get_strings(request: Request) -> dict:
    """Get all strings for the current language (for template rendering)."""
    lang = get_lang(request)
    return STRINGS.get(lang, STRINGS[DEFAULT_LANG])
