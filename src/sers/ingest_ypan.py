"""
Ingest YPAN (SMCXD04) clinical data into standardized format.
Source: /home/user/workspace/SMCXD04/SMCXD04_CRF_data.xlsx (60 patients)
  - 치료 경험이 없는 췌장암: 30
  - 췌장낭종: 10
  - 건강한 자: 10
  - 기타 양성 질환: 6
  - 만성 췌장염: 4

Also copy SERS dataset to SERS-AI.
"""

import shutil

import numpy as np
import pandas as pd

crf = pd.read_excel('/home/user/workspace/SMCXD04/SMCXD04_CRF_data.xlsx')
print(f"CRF: {len(crf)} rows")

# ============================================================
# 1. Build standardized clinical table
# ============================================================
rows = []
for i, r in crf.iterrows():
    screening = r['스크리닝번호']  # e.g. SMCXD2-001
    num = int(screening.split('-')[1])
    pid = f"YPAN {num}"

    # Disease group mapping
    disease = r['대상자 질환']
    if disease == '치료 경험이 없는 췌장암':
        group = 'YPAN'
    elif disease == '췌장낭종':
        group = 'YPAN_CYST'
    elif disease == '건강한 자':
        group = 'YPAN_NOR'
    elif disease == '만성 췌장염':
        group = 'YPAN_CP'
    elif disease == '기타 양성 질환':
        group = 'YPAN_BENIGN'
    else:
        group = 'YPAN_OTHER'

    # Sex
    sex = 'M' if r['성별'] == '남성' else 'F'

    # Height, Weight, BMI
    h = r['신장(cm)']
    w = r['체중(kg)']
    bmi = w / (h / 100) ** 2 if pd.notna(h) and pd.notna(w) and h > 0 else np.nan

    # Smoking: 비 흡연=0, 과거 흡연=1, 현재 흡연=2
    smoke_map = {'비 흡연': 0, '과거 흡연': 1, '현재 흡연': 2}
    smoking = smoke_map.get(r['흡연력'], np.nan)

    # Drinking: 비 음주=0, 과거 음주=1, 현재 음주=2
    drink_map = {'비 음주': 0, '과거 음주': 1, '현재 음주': 2}
    drinking = drink_map.get(r['음주력'], np.nan)

    # Dates
    sample_date = r.get('공복 뇨검체 수집일', np.nan)
    if pd.notna(sample_date):
        try:
            sample_date = pd.to_datetime(sample_date).strftime('%Y-%m-%d')
        except:
            sample_date = np.nan

    diagnosis_date = r.get('췌장암 진단일', np.nan)
    if pd.isna(diagnosis_date):
        diagnosis_date = r.get('질환 진단일', np.nan)
    if pd.notna(diagnosis_date):
        try:
            diagnosis_date = pd.to_datetime(diagnosis_date).strftime('%Y-%m-%d')
        except:
            diagnosis_date = np.nan

    surgery_date = r.get('수술일', np.nan)
    if pd.notna(surgery_date):
        try:
            surgery_date = pd.to_datetime(surgery_date).strftime('%Y-%m-%d')
        except:
            surgery_date = np.nan

    # Stage
    stage = r.get('Overall Stage', np.nan)
    t_stage = r.get('T Stage', np.nan)
    n_stage = r.get('N Stage', np.nan)
    m_stage = r.get('M Stage', np.nan)

    # Convert T/N/M to string format
    if pd.notna(t_stage):
        t_stage = f"T{int(t_stage)}" if isinstance(t_stage, (int, float)) else str(t_stage)
    if pd.notna(n_stage):
        n_stage = f"N{n_stage}" if not str(n_stage).startswith('N') else str(n_stage)
        if n_stage == 'NUnknown':
            n_stage = 'NX'
    if pd.notna(m_stage):
        m_stage = f"M{int(m_stage)}" if isinstance(m_stage, (int, float)) else str(m_stage)

    # Pathology
    pathology = r.get('Pathologic Type', np.nan)
    diagnosis_text = r.get('췌장암 진단명', np.nan)
    if pd.isna(diagnosis_text):
        diagnosis_text = disease

    # Past history
    past_history = r.get('병력 상세', np.nan)

    # Metastasis
    meta = r.get('Metastasis', np.nan)
    if meta == '예':
        metastasis = 'Yes'
    elif meta == '아니오':
        metastasis = 'No'
    else:
        metastasis = np.nan

    # Treatment
    surgery_yn = r.get('수술 시행여부', np.nan)
    chemo_yn = r.get('항암 시행여부', np.nan)
    surgery_name = r.get('수술명', np.nan)

    treatment_parts = []
    if surgery_yn == '예':
        treatment_parts.append('수술')
    if chemo_yn == '예':
        treatment_parts.append('항암')
    treatment = ', '.join(treatment_parts) if treatment_parts else np.nan

    # Fasting
    fasting_str = r.get('공복비공복 대상자', np.nan)
    fasting = np.nan
    if pd.notna(fasting_str):
        if '공복' in str(fasting_str):
            fasting = 'Y'

    # Lab values
    cea = r.get('CEA_결과', np.nan)
    ca19_9 = r.get('CA19-9_결과', np.nan)
    ast = r.get('SGOT_결과', np.nan)
    alt = r.get('SGPT_결과', np.nan)
    total_bilirubin = r.get('Total_bilirubin_결과', np.nan)

    # Urine analysis
    ua_ph = r.get('pH', np.nan)
    ua_sg = r.get('Specific_gravity', np.nan)
    ua_protein = r.get('Protein_Albumin', np.nan)
    ua_blood = r.get('Blood_Heme', np.nan)

    row = {
        'patient_id': pid,
        'disease_group': group,
        'source_file': 'SMCXD04_CRF_data.xlsx',
        'age': r['연령'],
        'sex': sex,
        'height_cm': h,
        'weight_kg': w,
        'bmi': round(bmi, 1) if pd.notna(bmi) else np.nan,
        'bp_systolic': np.nan,
        'bp_diastolic': np.nan,
        'smoking_status': smoking,
        'drinking_status': drinking,
        'past_history': past_history,
        'diagnosis': diagnosis_text,
        'diagnosis_date': diagnosis_date,
        'sample_date': sample_date,
        'surgery_date': surgery_date,
        'pathology': pathology,
        'stage': stage,
        'tnm': np.nan,
        't_stage': t_stage,
        'n_stage': n_stage,
        'm_stage': m_stage,
        'metastasis': metastasis,
        'treatment': treatment,
        'fasting': fasting,
        'surgery_name': surgery_name,
        'chemo_date': np.nan,
        'treatment_detail': np.nan,
        'sample_timing': np.nan,
        'wbc': np.nan, 'rbc': np.nan, 'hb': np.nan, 'hct': np.nan,
        'platelet': np.nan, 'neutrophil_pct': np.nan, 'lymphocyte_pct': np.nan,
        'ast': ast, 'alt': alt,
        'alp': np.nan, 'ggt': np.nan, 'bun': np.nan, 'creatinine': np.nan,
        'uric_acid': np.nan, 'glucose': np.nan,
        'total_protein': np.nan, 'albumin': np.nan,
        'total_bilirubin': total_bilirubin,
        'ldh': np.nan, 'calcium': np.nan, 'hs_crp': np.nan,
        'sodium': np.nan, 'potassium': np.nan, 'chloride': np.nan,
        'total_cholesterol': np.nan, 'triglyceride': np.nan,
        'hdl_c': np.nan, 'ldl_c': np.nan, 'hba1c': np.nan,
        'afp': np.nan,
        'cea': cea, 'ca19_9': ca19_9,
        'psa': np.nan,
        'ua_sg': ua_sg, 'ua_ph': ua_ph,
        'ua_protein': ua_protein, 'ua_glucose': np.nan,
        'ua_blood': ua_blood,
    }
    rows.append(row)

ypan_df = pd.DataFrame(rows)

# Ensure numeric columns
numeric_cols = ['cea', 'ca19_9', 'ast', 'alt', 'total_bilirubin', 'ua_ph', 'ua_sg']
for col in numeric_cols:
    ypan_df[col] = pd.to_numeric(ypan_df[col], errors='coerce')

# ============================================================
# 2. Save YPAN standardized
# ============================================================
# Save individual group files
for g in ypan_df['disease_group'].unique():
    sub = ypan_df[ypan_df['disease_group'] == g]
    sub.to_csv(f'data/clinical_data/standardized/{g}_clinical_standardized.csv',
               index=False, encoding='utf-8-sig')
    print(f"  Saved {g}: {len(sub)} rows")

# Append to all_clinical_standardized
all_clin = pd.read_csv('data/clinical_data/standardized/all_clinical_standardized.csv', encoding='utf-8-sig')
print(f"\nBefore: {len(all_clin)} rows, groups: {all_clin['disease_group'].nunique()}")

# Remove existing YPAN groups if any
ypan_groups = ypan_df['disease_group'].unique()
all_clin = all_clin[~all_clin['disease_group'].isin(ypan_groups)]

all_clin = pd.concat([all_clin, ypan_df], ignore_index=True)
all_clin.to_csv('data/clinical_data/standardized/all_clinical_standardized.csv',
                index=False, encoding='utf-8-sig')
print(f"After: {len(all_clin)} rows, groups: {all_clin['disease_group'].nunique()}")
for g in sorted(all_clin['disease_group'].unique()):
    print(f"  {g}: {len(all_clin[all_clin['disease_group']==g])}")

# ============================================================
# 3. Copy SERS dataset and CRF to SERS-AI
# ============================================================
dest_dir = 'data/clinical_data/10. 췌장암/YPAN'
shutil.copy2('/home/user/workspace/SMCXD04/SMCXD04_CRF_data.xlsx', dest_dir)
shutil.copy2('/home/user/workspace/SMCXD04/SMCXD04_SERS_dataset.csv', dest_dir)
print(f"\nCopied CRF + SERS dataset to {dest_dir}")

# ============================================================
# 4. Summary
# ============================================================
print("\n=== YPAN INGESTION SUMMARY ===")
for g in ypan_df['disease_group'].unique():
    sub = ypan_df[ypan_df['disease_group'] == g]
    print(f"\n{g} (n={len(sub)}):")
    print(f"  Age: {sub['age'].mean():.1f} ± {sub['age'].std():.1f}")
    print(f"  Sex: M={( sub['sex']=='M').sum()}, F={(sub['sex']=='F').sum()}")
    print(f"  BMI: {sub['bmi'].mean():.1f} ± {sub['bmi'].std():.1f}")
    stage = sub['stage'].dropna()
    if len(stage) > 0:
        print(f"  Stage: {stage.value_counts().to_dict()}")
    print(f"  sample_date: {sub['sample_date'].notna().sum()}/{len(sub)}")
    print(f"  diagnosis_date: {sub['diagnosis_date'].notna().sum()}/{len(sub)}")
    print(f"  CEA: {sub['cea'].notna().sum()}/{len(sub)}")
    print(f"  CA19-9: {sub['ca19_9'].notna().sum()}/{len(sub)}")
