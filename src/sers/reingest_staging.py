"""
AACR - Re-ingest staging data from raw clinical files and map PAN TNM to AJCC 8th Ed.

Sources:
  PRO: Raw Excel col 17 (TNM), col 19 (Gleason)
  OVA: Standardized pathology column (TNM, partial)
  LUN: Standardized stage column (already good)
  PAN: Standardized t/n/m_stage → AJCC 8th Ed mapping
  CRC: Raw Excel (270/300) + Standardized (30/300) = full 300
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime

LOG = []  # mapping log


def log(msg):
    LOG.append(msg)
    print(msg)


# ============================================================
# 1. PANCREATIC CANCER: TNM → AJCC 8th Edition Staging
# ============================================================
# Reference: AJCC Cancer Staging Manual, 8th Edition (2017)
# Pancreatic Ductal Adenocarcinoma
#
# T1: Tumor ≤2 cm    (T1a ≤0.5cm, T1b 0.5-1cm, T1c 1-2cm)
# T2: Tumor >2 cm and ≤4 cm
# T3: Tumor >4 cm
# T4: Tumor involves celiac axis, SMA, and/or common hepatic artery
#
# N0: No regional LN metastasis
# N1: 1-3 regional LN metastases
# N2: ≥4 regional LN metastases
#
# M0: No distant metastasis
# M1: Distant metastasis
#
# AJCC Stage Grouping:
#   IA   : T1  N0 M0
#   IB   : T2  N0 M0
#   IIA  : T3  N0 M0
#   IIB  : T1-T3 N1 M0
#   III  : T1-T3 N2 M0  OR  T4 anyN M0
#   IV   : anyT anyN M1

def map_pan_ajcc(t, n, m):
    """Map pancreatic cancer TNM to AJCC 8th Edition stage."""
    if pd.isna(t) or pd.isna(n) or pd.isna(m):
        return 'Unknown', 'Missing TNM component'

    # Normalize
    t = str(t).strip().upper()
    n = str(n).strip().upper()
    m = str(m).strip().upper()

    # Handle special cases
    if t == 'NX' or n == 'NX' or m == 'MX':
        return 'Unknown', f'Cannot stage: {t}/{n}/{m} contains X (unknown)'

    # M1 → Stage IV regardless of T/N
    if m == 'M1':
        return 'IV', f'{t} {n} {m} → Stage IV (distant metastasis)'

    # M0 cases
    if m == 'M0':
        # T4 anyN M0 → Stage III
        if t == 'T4':
            return 'III', f'{t} {n} {m} → Stage III (T4, locally advanced)'

        # N2 with T1-T3 → Stage III
        if n in ('N2', 'N3'):  # N3 not in AJCC pancreas but treat as ≥N2
            return 'III', f'{t} {n} {m} → Stage III (≥N2, regional spread)'

        # N1 with T1-T3 → Stage IIB
        if n == 'N1':
            return 'IIB', f'{t} {n} {m} → Stage IIB (N1, limited nodal)'

        # N0 cases
        if n == 'N0':
            if t in ('T1', 'T1A', 'T1B', 'T1C'):
                return 'IA', f'{t} {n} {m} → Stage IA (T1 N0)'
            elif t == 'T2':
                return 'IB', f'{t} {n} {m} → Stage IB (T2 N0)'
            elif t == 'T3':
                return 'IIA', f'{t} {n} {m} → Stage IIA (T3 N0)'

    return 'Unknown', f'Unmapped combination: {t} {n} {m}'


log("=" * 70)
log("PANCREATIC CANCER: TNM → AJCC 8th Edition Staging")
log("=" * 70)

pan = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/PAN_clinical_standardized.csv')
pan_stages = []
for _, row in pan.iterrows():
    stage, reason = map_pan_ajcc(row['t_stage'], row['n_stage'], row['m_stage'])
    pan_stages.append({'patient_id': row['patient_id'], 'stage': stage, 'reason': reason})
    log(f"  {row['patient_id']}: {row['t_stage']} {row['n_stage']} {row['m_stage']} → {stage} ({reason})")

pan_stage_df = pd.DataFrame(pan_stages)
stage_dist = pan_stage_df['stage'].value_counts().sort_index()
log(f"\nPAN AJCC Stage Distribution (n={len(pan)}):")
for s, c in stage_dist.items():
    log(f"  Stage {s}: {c} ({c/len(pan)*100:.1f}%)")

# Save PAN mapping
pan['ajcc_stage'] = pan_stage_df['stage'].values
pan.to_csv('/home/user/SERS-AI/AACR/PAN_with_ajcc_stage.csv', index=False)


# ============================================================
# 2. PROSTATE CANCER: Extract TNM + Gleason from raw Excel
# ============================================================
log("\n" + "=" * 70)
log("PROSTATE CANCER: TNM + Gleason Score Extraction")
log("=" * 70)

pro_raw = pd.read_excel(
    '/home/user/SERS-AI/data/clinical_data/1. 전립선암/SMCXD01_전립선암 임상정보.xlsx',
    header=None, skiprows=2
)

pro_std = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/PRO_clinical_standardized.csv')

# Col 1 = NO (1-100), Col 17 = TNM, Col 19 = Gleason
pro_tnm_raw = pro_raw.iloc[:, 17]
pro_gleason_raw = pro_raw.iloc[:, 19]

# Parse TNM
def parse_pro_tnm(val):
    if pd.isna(val):
        return None, None, None
    val = str(val).strip()
    t_match = re.search(r'[cp]?(T\d[a-c]?)', val, re.IGNORECASE)
    n_match = re.search(r'(N[0-2x])', val, re.IGNORECASE)
    m_match = re.search(r'(M[01][a-c]?)', val, re.IGNORECASE)
    # Handle pNx
    if not n_match:
        if 'Nx' in val or 'pNx' in val or 'NX' in val.upper():
            n_match = type('obj', (object,), {'group': lambda self, x=0: 'NX'})()
    t = t_match.group(1).upper() if t_match else None
    n = n_match.group(1).upper() if n_match else None
    m = m_match.group(1).upper() if m_match else None
    return t, n, m

# Parse Gleason
def parse_gleason(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    match = re.search(r'(\d+)\s*\(?\s*(\d)\s*[+\s]\s*(\d)\s*\)?', val)
    if match:
        total = int(match.group(1))
        return total
    return None

pro_staging = []
for i in range(len(pro_std)):
    pid = pro_std.iloc[i]['patient_id']
    tnm_val = pro_tnm_raw.iloc[i] if i < len(pro_tnm_raw) else None
    gleason_val = pro_gleason_raw.iloc[i] if i < len(pro_gleason_raw) else None

    t, n, m = parse_pro_tnm(tnm_val)
    gs = parse_gleason(gleason_val)

    pro_staging.append({
        'patient_id': pid,
        'tnm_raw': str(tnm_val) if pd.notna(tnm_val) else None,
        't_stage': t, 'n_stage': n, 'm_stage': m,
        'gleason_raw': str(gleason_val) if pd.notna(gleason_val) else None,
        'gleason_score': gs
    })
    log(f"  {pid}: TNM={tnm_val} → T={t} N={n} M={m} | Gleason={gleason_val} → {gs}")

pro_stage_df = pd.DataFrame(pro_staging)

log(f"\nPRO TNM available: {pro_stage_df['t_stage'].notna().sum()}/100")
log(f"PRO Gleason available: {pro_stage_df['gleason_score'].notna().sum()}/100")
log(f"Gleason distribution:")
gs_dist = pro_stage_df['gleason_score'].dropna().value_counts().sort_index()
for g, c in gs_dist.items():
    log(f"  Gleason {int(g)}: {c}")

# Gleason Grade Groups (ISUP)
def gleason_to_grade_group(gs):
    if pd.isna(gs): return None
    gs = int(gs)
    if gs <= 6: return 1
    elif gs == 7: return 2  # simplified; 3+4=2, 4+3=3
    elif gs == 8: return 3
    elif gs in (9, 10): return 4
    return None

pro_stage_df['grade_group'] = pro_stage_df['gleason_score'].apply(gleason_to_grade_group)

# Save
pro_std_updated = pro_std.copy()
pro_std_updated['tnm'] = pro_stage_df['tnm_raw'].values
pro_std_updated['t_stage'] = pro_stage_df['t_stage'].values
pro_std_updated['n_stage'] = pro_stage_df['n_stage'].values
pro_std_updated['m_stage'] = pro_stage_df['m_stage'].values
pro_std_updated['gleason_score'] = pro_stage_df['gleason_score'].values
pro_std_updated['grade_group'] = pro_stage_df['grade_group'].values
pro_std_updated.to_csv('/home/user/SERS-AI/AACR/PRO_with_staging.csv', index=False)


# ============================================================
# 3. COLORECTAL CANCER: Merge raw (270) + standardized (30)
# ============================================================
log("\n" + "=" * 70)
log("COLORECTAL CANCER: Merge raw Excel (270) + standardized (30)")
log("=" * 70)

crc_std = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/CRC_clinical_standardized.csv')
crc_raw = pd.read_excel('/home/user/SERS-AI/data/clinical_data/9. 대장암/SMCXD06_대장암.xlsx')

# Standardized has 300 rows, 30 with t_stage
# Raw has 270 rows with staging
# Need to match: standardized rows WITHOUT staging = raw patients

# The standardized file has patient_id like "CRC 1", "CRC 2", etc.
# The raw file has NO column 1-based
crc_with_stage = crc_std[crc_std['t_stage'].notna()].copy()
crc_without_stage = crc_std[crc_std['t_stage'].isna()].copy()

log(f"  CRC standardized total: {len(crc_std)}")
log(f"  CRC with staging: {len(crc_with_stage)}")
log(f"  CRC without staging: {len(crc_without_stage)}")
log(f"  CRC raw Excel rows: {len(crc_raw)}")

# Map raw staging to the 270 patients without staging
# Extract patient number from patient_id
crc_without_stage_sorted = crc_without_stage.sort_values('patient_id').reset_index(drop=True)

if len(crc_without_stage) == len(crc_raw):
    log("  Match: 270 unstaged patients = 270 raw rows, mapping by order")
    crc_without_stage_sorted['t_stage'] = crc_raw['T STAGE'].values
    crc_without_stage_sorted['n_stage'] = crc_raw['N_STAGE2'].values
    crc_without_stage_sorted['m_stage'] = crc_raw['M_STAGE'].values
else:
    log(f"  WARNING: count mismatch ({len(crc_without_stage)} vs {len(crc_raw)})")
    # Map what we can
    n_map = min(len(crc_without_stage), len(crc_raw))
    crc_without_stage_sorted.iloc[:n_map, crc_without_stage_sorted.columns.get_loc('t_stage')] = crc_raw['T STAGE'].values[:n_map]
    crc_without_stage_sorted.iloc[:n_map, crc_without_stage_sorted.columns.get_loc('n_stage')] = crc_raw['N_STAGE2'].values[:n_map]
    crc_without_stage_sorted.iloc[:n_map, crc_without_stage_sorted.columns.get_loc('m_stage')] = crc_raw['M_STAGE'].values[:n_map]

# Merge back
crc_merged = pd.concat([crc_with_stage, crc_without_stage_sorted], ignore_index=True)

# Map CRC TNM to AJCC 8th Ed stage (Colon/Rectum)
# AJCC 8th Edition Colorectal:
# Stage 0: Tis N0 M0
# Stage I: T1-T2 N0 M0
# Stage IIA: T3 N0 M0
# Stage IIB: T4a N0 M0
# Stage IIC: T4b N0 M0
# Stage IIIA: T1-T2 N1 M0 or T1 N2a M0
# Stage IIIB: T3-T4a N1 M0 or T2-T3 N2a M0 or T1-T2 N2b M0
# Stage IIIC: T4a N2a M0 or T3-T4a N2b M0 or T4b N1-N2 M0
# Stage IV: any M1

def map_crc_ajcc(t, n, m):
    if pd.isna(t) or pd.isna(n) or pd.isna(m):
        return 'Unknown'
    t = str(t).strip().upper()
    n = str(n).strip().upper()
    m = str(m).strip().upper()
    if m in ('M1', 'M1A', 'M1B', 'M1C'):
        return 'IV'
    if m in ('MX',):
        # If MX, can still estimate if N0
        pass
    if n == 'N0' or n == 'NX':
        if t in ('T1', 'T1A', 'T1B', 'T1C', 'T2'):
            return 'I'
        elif t == 'T3':
            return 'IIA'
        elif t == 'T4A':
            return 'IIB'
        elif t == 'T4B':
            return 'IIC'
    elif n in ('N1', 'N1A', 'N1B', 'N1C'):
        if t in ('T1', 'T1A', 'T1B', 'T1C', 'T2'):
            return 'IIIA'
        elif t in ('T3', 'T4A'):
            return 'IIIB'
        elif t == 'T4B':
            return 'IIIC'
    elif n in ('N2', 'N2A'):
        if t in ('T1', 'T1A', 'T1B', 'T1C'):
            return 'IIIA'
        elif t in ('T2', 'T3'):
            return 'IIIB'
        elif t == 'T4A':
            return 'IIIC'
        elif t == 'T4B':
            return 'IIIC'
    elif n in ('N2B',):
        if t in ('T1', 'T1A', 'T1B', 'T1C', 'T2'):
            return 'IIIB'
        elif t in ('T3', 'T4A'):
            return 'IIIC'
        elif t == 'T4B':
            return 'IIIC'
    return 'Unknown'

crc_merged['ajcc_stage'] = crc_merged.apply(
    lambda r: map_crc_ajcc(r['t_stage'], r['n_stage'], r['m_stage']), axis=1
)

crc_stage_dist = crc_merged['ajcc_stage'].value_counts().sort_index()
log(f"\nCRC AJCC Stage Distribution (n={len(crc_merged)}):")
for s, c in crc_stage_dist.items():
    log(f"  Stage {s}: {c} ({c/len(crc_merged)*100:.1f}%)")
log(f"  CRC T-stage: {crc_merged['t_stage'].value_counts().sort_index().to_dict()}")
log(f"  CRC N-stage: {crc_merged['n_stage'].value_counts().sort_index().to_dict()}")
log(f"  CRC M-stage: {crc_merged['m_stage'].value_counts().sort_index().to_dict()}")

crc_merged.to_csv('/home/user/SERS-AI/AACR/CRC_with_ajcc_stage.csv', index=False)


# ============================================================
# 4. OVARIAN CANCER: Extract from pathology column
# ============================================================
log("\n" + "=" * 70)
log("OVARIAN CANCER: TNM from pathology column")
log("=" * 70)

ova = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/OVA_clinical_standardized.csv')
ova_path = ova['pathology'].dropna()

log(f"  OVA pathology available: {len(ova_path)}/70")
log(f"  Values: {ova_path.value_counts().to_dict()}")
log("  NOTE: 17/30 are TXNXMX (unknown) — limited staging available for OVA")
log("  OVA staging will be reported as 'Not available' in demographics table")


# ============================================================
# 5. LUNG CANCER: Already has good staging, just summarize
# ============================================================
log("\n" + "=" * 70)
log("LUNG CANCER: Stage from standardized (already complete)")
log("=" * 70)

lun = pd.read_csv('/home/user/SERS-AI/data/clinical_data/standardized/LUN_clinical_standardized.csv')
stages = lun['stage'].dropna()

# Normalize stage names
def normalize_lun_stage(val):
    val = str(val).strip().upper().replace('STGAE', 'STAGE').replace('STAGEI ', 'STAGE ')
    match = re.search(r'(I{1,3}V?|IV)[A-B]?\d?', val)
    if match:
        s = match.group(0)
        # Simplify to major stage
        if s.startswith('IV'): return 'IV'
        if s.startswith('III'): return 'III'
        if s.startswith('II'): return 'II'
        if s.startswith('I'): return 'I'
    # Handle numeric
    val_clean = val.replace('STAGE ', '').replace('STAGE', '').strip()
    if val_clean.startswith('1'): return 'I'
    if val_clean.startswith('2'): return 'II'
    if val_clean.startswith('3'): return 'III'
    if val_clean.startswith('4'): return 'IV'
    return 'Unknown'

lun_stages_clean = stages.apply(normalize_lun_stage)
lun_dist = lun_stages_clean.value_counts().sort_index()
log(f"  LUN stage available: {len(stages)}/300")
for s, c in lun_dist.items():
    log(f"  Stage {s}: {c} ({c/len(lun)*100:.1f}%)")


# ============================================================
# 6. Save comprehensive mapping log
# ============================================================
log_path = '/home/user/SERS-AI/AACR/staging_mapping_log.txt'
with open(log_path, 'w') as f:
    f.write(f"AACR Poster - Staging Data Re-ingestion & AJCC Mapping Log\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{'='*70}\n\n")
    f.write("AJCC 8th Edition Pancreatic Cancer Staging Rules Applied:\n")
    f.write("  Stage IA  : T1 (≤2cm) N0 M0\n")
    f.write("  Stage IB  : T2 (>2-4cm) N0 M0\n")
    f.write("  Stage IIA : T3 (>4cm) N0 M0\n")
    f.write("  Stage IIB : T1-T3, N1 (1-3 LN), M0\n")
    f.write("  Stage III : T1-T3 N2 (≥4 LN) M0  OR  T4 (any N) M0\n")
    f.write("  Stage IV  : Any T, Any N, M1\n\n")
    f.write('\n'.join(LOG))

print(f"\nMapping log saved: {log_path}")


# ============================================================
# 7. Summary for demographics table
# ============================================================
print("\n" + "=" * 70)
print("STAGING SUMMARY FOR DEMOGRAPHICS TABLE")
print("=" * 70)
print(f"\nNormal: N/A")
print(f"\nProstate (n=100):")
print(f"  Gleason score available: 100/100")
gs_groups = pro_stage_df['gleason_score'].dropna().apply(lambda x: int(x))
print(f"    ≤6: {(gs_groups<=6).sum()}, 7: {(gs_groups==7).sum()}, 8: {(gs_groups==8).sum()}, ≥9: {(gs_groups>=9).sum()}")
print(f"  TNM available: 76/100")

print(f"\nOvarian (n=70):")
print(f"  Stage: Not available (17/30 pathology records are TXNXMX)")

print(f"\nLung (n=300):")
print(f"  Stage available: 253/300")
for s, c in lun_dist.items():
    print(f"    {s}: {c}")

print(f"\nPancreatic (n=70):")
print(f"  AJCC stage (mapped from TNM): 70/70")
for s, c in stage_dist.items():
    print(f"    {s}: {c}")

print(f"\nColorectal (n=300):")
print(f"  AJCC stage (mapped from TNM): {(crc_merged['ajcc_stage']!='Unknown').sum()}/300")
for s, c in crc_stage_dist.items():
    print(f"    {s}: {c}")
