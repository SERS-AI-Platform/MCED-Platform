"""
Update LUN standardized clinical data with missing dates.

Mapping:
  LUN file 1 (30명, LUN 1~30):
    - 자원접수일 → sample_date (already ingested)
    - 진단일 → diagnosis_date (already ingested)
    - surgery_date: N/A (no column)

  LUN file 2 (170명, LUN 31~200):
    - 수술일 → surgery_date (already ingested)
    - sample_date: leave empty (no column)
    - diagnosis_date: leave empty (조직검사 결과 진단일 is pathology date, not cancer diagnosis)

  LUN file 3 (100명, LUN 201~300):
    - 검사일자 (진단검사 sheet) → diagnosis_date
    - 검체수집일 (진단검사 sheet) → sample_date
    - surgery_date: N/A (no column)
"""

import pandas as pd
import numpy as np

# Load current standardized
lun = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/LUN_clinical_standardized.csv')
print(f"LUN standardized: {len(lun)} rows")
print(f"Before update:")
print(f"  sample_date:    {lun['sample_date'].notna().sum()}/300")
print(f"  diagnosis_date: {lun['diagnosis_date'].notna().sum()}/300")
print(f"  surgery_date:   {lun['surgery_date'].notna().sum()}/300")

# Extract patient number from patient_id (e.g., "LUN 201" -> 201)
lun['_num'] = lun['patient_id'].str.extract(r'(\d+)').astype(int)

# ── LUN file 3: 진단검사 sheet (patients 201-300) ──
lun3_diag = pd.read_excel(
    '/home/user/SERS-AI/data/clinical_data/4. 폐암/SMCXD06_폐암 3.xlsx',
    sheet_name='진단검사'
)
lun3_demo = pd.read_excel(
    '/home/user/SERS-AI/data/clinical_data/4. 폐암/SMCXD06_폐암 3.xlsx',
    sheet_name='인구학적정보 및 암 관련 정보'
)

# 검체수집일 → sample_date, 검사일자 → diagnosis_date
# Also get 암 진단일 from demographic sheet as more appropriate diagnosis_date
def parse_date(val):
    """Convert various date formats to YYYY-MM-DD string."""
    if pd.isna(val) or str(val).strip() in ('', '.', 'x', 'X'):
        return np.nan
    val = str(val).strip().replace('.', '')
    if len(val) == 8 and val.isdigit():
        return f"{val[:4]}-{val[4:6]}-{val[6:8]}"
    try:
        ts = pd.to_datetime(val)
        return ts.strftime('%Y-%m-%d')
    except:
        return np.nan

# Map by order: LUN 201-300
for i in range(100):
    pat_num = 201 + i
    mask = lun['_num'] == pat_num

    # sample_date from 검체수집일
    sample_val = parse_date(lun3_diag.iloc[i]['검체수집일'])
    if pd.notna(sample_val):
        lun.loc[mask, 'sample_date'] = sample_val

    # diagnosis_date from 암 진단일 (인구학적정보 sheet)
    diag_val = parse_date(lun3_demo.iloc[i]['암 진단일'])
    if pd.notna(diag_val):
        lun.loc[mask, 'diagnosis_date'] = diag_val

# ── Verify ──
print(f"\nAfter update:")
print(f"  sample_date:    {lun['sample_date'].notna().sum()}/300")
print(f"  diagnosis_date: {lun['diagnosis_date'].notna().sum()}/300")
print(f"  surgery_date:   {lun['surgery_date'].notna().sum()}/300")

# Breakdown by file
for start, end, label in [(1, 30, 'File1'), (31, 200, 'File2'), (201, 300, 'File3')]:
    sub = lun[(lun['_num'] >= start) & (lun['_num'] <= end)]
    sd = sub['sample_date'].notna().sum()
    dd = sub['diagnosis_date'].notna().sum()
    od = sub['surgery_date'].notna().sum()
    print(f"  {label} (LUN {start}-{end}, n={len(sub)}): sample={sd}, diagnosis={dd}, surgery={od}")

# Drop helper column and save
lun.drop(columns=['_num'], inplace=True)
lun.to_csv('/home/user/SERS-AI/data/clinical_data/standardized/LUN_clinical_standardized.csv', index=False)
print("\nSaved: LUN_clinical_standardized.csv")

# ── Also update all_clinical_standardized.csv ──
all_clin = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/all_clinical_standardized.csv')
# Replace LUN rows
all_clin = all_clin[all_clin['disease_group'] != 'LUN']
lun_for_merge = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/LUN_clinical_standardized.csv')
all_clin = pd.concat([all_clin, lun_for_merge], ignore_index=True)
all_clin.to_csv('/home/user/SERS-AI/data/clinical_data/standardized/all_clinical_standardized.csv', index=False)
print("Saved: all_clinical_standardized.csv")
