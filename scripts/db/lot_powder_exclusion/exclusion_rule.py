"""임상표(전체환자_임상정보_정규화_vN.xlsx)에서 제외 대상 검체를 판정한다.

판정 기준 (합집합)

  1) 요로감염   ua_nitrite = Positive (LE 동시 양성 여부는 상세 표기용)
  2) 타암 이력  past_history_1~9 에 현재 암종과 다른 암/종양
  3) 치료 후 채취  collection_date 가 surgery_date 또는 chemo_start_date 보다 뒤.
                 단 primary_sample_timing_interpretation 에 pre_treatment_specimen
                 이 붙어 있으면 제외하지 않는다 (2026-09-07 사용자 판단).
  4) 다중암     MULTI_CANCER_PATIENTS 의 환자 (검체 전부 제외)
  5) Drop       group='Drop' (임상표 초록색 행)
  6) 임상표 노란 표시 (행 전체가 노란 행만)

1~2 는 2026-07 0728 파일의 '제외대상 Bold 처리'와 같은 정의고, 3~4 는
2026-09-07 에 사용자 요청으로 추가됐다.

pre_treatment_specimen 태그가 있으면 살리는 이유와 그 한계
  사용자 판단이다. Lot 범위에서 날짜상 치료 후인 77건 중 12건이 이 태그를 달고
  있고, 그중 8건이 이 예외로 살아난다 (PAN_31/41/48/49/50/52/64/66).
  주의: PAN_49/64/66 은 채취일이 항암 시작 3~9개월 뒤이고 수술일은 채취일보다
  뒤다. 선행항암 중 검체를 '수술 전'이라는 뜻으로 pre_treatment 라고 적은 것으로
  보인다. 나머지 5건은 수술 3~7일 후라 성격이 다르다. 기준을 다시 볼 일이 있으면
  이 3건부터 확인할 것.

치료 후 채취를 timing 문자열이 아니라 날짜로 보는 이유
  primary_sample_timing_interpretation 에 post_* 가 붙은 검체는 101개지만, 그중
  6건은 collection_date 와 surgery_date, chemo_start_date 가 전부 같은 날이라
  실제로 치료 후에 받은 검체가 아니다 (pre_treatment_specimen 태그도 같이 붙어
  있다). 반대로 문자열이 pre_surgery/pre_treatment 인데 날짜상 치료 후인 건이
  3건 있다. 그래서 문자열이 아니라 날짜로 가른다 -- 2026-09-07 사용자 판단.
  주의: 날짜 컬럼은 워크북에 문자열('YYYY-MM-DD')과 datetime 이 섞여 저장돼
  있다. parse_date() 로 양쪽을 다 받아야 한다.

노란색을 근거로 따로 두는 이유
  노란색 자체는 사유가 아니라 사람이 손으로 남긴 표시다. 179건 중 173건은 위
  1~4 로도 설명되고, 남는 6건만 노란 표시밖에 근거가 없다 (BNOR_30/79/84,
  BPRO_81, PRO_247 은 셀 1~2개만 노란 값 확인 표시, CRC_294 는 행 전체가 노랗지만
  LE 만 양성). 이 6건을 놓치지 않으려고 근거로 남겨 둔다.

노란색의 정체
  워크북에 색 범례가 없어 2026-09-07에 역산했다. 암 코호트 1,498개 중 노란 행
  144개를 보면 121개가 타암 이력, 17개가 nitrite+LE 동시 양성, 4개가 둘 다여서
  142개(98.6%)가 설명된다. 남은 2개(CRC_69, CRC_294)도 nitrite 또는 LE 한쪽이
  양성이다. 즉 노란색은 '타암 이력'과 '요로감염 의심'을 사람이 손으로 표시한
  것으로 보인다. 다만 하이라이트가 암 파일에만 적용돼 있어(control/lung/breast는
  0건) 노란색만으로는 대조군의 타암 이력을 놓친다. 그래서 2)3)을 함께 쓴다.

LE 단독을 기준에 넣지 않는 이유
  암 코호트에서 nitrite 양성 25건 중 22건이 노란 행인 반면, LE 양성 254건 중
  노란 행은 53건뿐이다. LE 단독은 사람의 판단 기준이 아니었다. 프로젝트의
  scripts/db/aecd_clinical_v7/08_uti_indicators_view.sql 이 nitrite 하나만 쓰는
  근거(LE 단독은 방광암에서 종양 염증과 구분되지 않음)와도 일치한다.
  참고: Lot 범위에서 기준을 nitrite 단독으로 바꿔도 결과는 같고(양성 26건이
  이미 전부 포함), LE 까지 넣으면 209건이 추가된다.
"""
import argparse
import datetime
import re

import openpyxl
from paths import CLINICAL

YELLOW = 'FFFFFF00'
GREEN = 'FF00B050'

CANCER_KEYWORD = re.compile(
    r'cancer|carcinoma|tumor|tumour|neoplasm|lymphoma|leukemia|leukaemia'
    r'|sarcoma|myeloma|malignan', re.I)

# 현재 암종 -> 자기 암으로 볼 표현. 여기 걸리면 '타암'이 아니다.
SAME_ORGAN = {
    'prostate': ['prostate'],
    'bladder': ['bladder'],
    'breast': ['breast'],
    'colorectal': ['colon', 'rectal', 'rectum', 'colorectal'],
    'lung': ['lung'],
    'pancreatic': ['pancrea'],
    'ovarian': ['ovar'],
    'OVA': ['ovar'],          # v7 표기. v8에서 'ovarian'으로 바뀌었다.
    'control': [],
}

# 노란 셀이 이만큼 이상이어야 '행 전체 하이라이트'로 본다. 셀 1~2개만 노란 행이
# 몇 개 있는데(BNOR_30/79 weight·height, BPRO_81/BNOR_84 treatment_info,
# PRO_247 diagnosis_date) 몸무게가 없어서 표시해 둔 것이라 제외 사유가 아니다
# (2026-09-07 사용자 확인, 정상 처리).
FULL_ROW_MIN_CELLS = 10

# 다중암 환자. 진단이 두 개라 검체가 두 코호트에 걸쳐 있어 전부 제외한다.
#   19022041 -> BLC_247, PRO_60
#   19276017 -> CRC_186, PAN_78
MULTI_CANCER_PATIENTS = {'19022041', '19276017'}


def parse_date(v):
    """워크북의 날짜 셀. 문자열('YYYY-MM-DD')과 datetime 이 섞여 있다."""
    if isinstance(v, datetime.datetime):
        return v.date()
    if isinstance(v, datetime.date):
        return v
    if isinstance(v, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', v.strip()):
        return datetime.date.fromisoformat(v.strip())
    return None


def collected_after_treatment(vals, ix):
    """채취일이 수술일 또는 항암 시작일보다 뒤면 True.

    단 timing 에 pre_treatment_specimen 이 붙어 있으면 False (위 주석 참고).
    """
    timing = str(vals[ix['primary_sample_timing_interpretation']] or '')
    if 'pre_treatment_specimen' in timing:
        return False
    coll = parse_date(vals[ix['collection_date']])
    if coll is None:
        return False
    for key in ('surgery_date', 'chemo_start_date'):
        d = parse_date(vals[ix[key]])
        if d is not None and coll > d:
            return True
    return False


def norm(label):
    """solum_label 정규화. 워크북에 'PRO_ 60' 처럼 공백이 섞여 있다."""
    return re.sub(r'\s+', '', str(label)) if label is not None else ''


def load(path=CLINICAL):
    """(컬럼인덱스, [(행 값 리스트, 노란칸수, 초록여부)]) 반환."""
    wb = openpyxl.load_workbook(path)
    ws = wb['table']
    header = [c.value for c in ws[1]]
    ix = {h: i for i, h in enumerate(header) if h}
    rows = []
    for row in ws.iter_rows(min_row=2):
        vals = [c.value for c in row]
        if not vals[ix['solum_label']]:
            continue
        yellow = green = 0
        for c in row:
            try:
                rgb = c.fill.fgColor.rgb if c.fill and c.fill.fgColor else None
            except Exception:
                rgb = None
            if not isinstance(rgb, str):
                continue
            if rgb.upper() == YELLOW:
                yellow += 1
            elif rgb.upper() == GREEN:
                green += 1
        rows.append((vals, yellow, green > 0))
    wb.close()
    return ix, rows


def other_cancer_history(vals, ix):
    """현재 암종과 다른 암/종양 이력 목록."""
    ctype = str(vals[ix['cancer_type']] or '').strip()
    same = SAME_ORGAN.get(ctype, [])
    out = []
    for k in range(1, 10):
        v = vals[ix['past_history_%d' % k]]
        if v in (None, '', '-'):
            continue
        v = str(v).strip()
        if not CANCER_KEYWORD.search(v):
            continue
        # 부분문자열이 아니라 단어 경계로 본다. 'Gallbladder cancer' 안에
        # 'bladder'가 들어 있어 방광암 환자의 담낭암 이력이 자기 암으로 잘못
        # 걸렸다 (BLC_16, BLC_88, BLC_147).
        if any(re.search(r'(?<![a-z])' + re.escape(s), v.lower()) for s in same):
            continue
        out.append(v)
    return out


def evaluate(ix, rows):
    """{정규화 라벨: {yellow, others, both, drop, excl}}"""
    res = {}
    for vals, n_yellow, green in rows:
        nitrite = str(vals[ix['ua_nitrite']] or '').strip().lower()
        le = str(vals[ix['ua_leukocyte esterase']] or '').strip().lower()
        others = other_cancer_history(vals, ix)
        yellow = n_yellow >= FULL_ROW_MIN_CELLS
        # 요로감염은 특이도가 높은 nitrite 하나로 가른다. LE 동시 양성은 상세
        # 표기용이고 별도 기준이 아니다 -- 08_uti_indicators_view.sql 과 같은 근거.
        uti = nitrite == 'positive'
        both = uti and le == 'positive'
        post = collected_after_treatment(vals, ix)
        multi = str(vals[ix['patient_code']] or '').strip() in MULTI_CANCER_PATIENTS
        drop = green or str(vals[ix['group']] or '').strip().lower() == 'drop'
        res[norm(vals[ix['solum_label']])] = dict(
            yellow=yellow, n_yellow=n_yellow, others=others, uti=uti, both=both,
            post=post, multi=multi, drop=drop,
            excl=bool(uti or others or post or multi or drop or yellow))
    return res


def basis_text(r):
    parts = []
    if r['uti']:
        parts.append('요로감염(Nitrite + Leukocyte esterase 동시 Positive)'
                     if r['both'] else '요로감염(Nitrite Positive)')
    if r['others']:
        parts.append('타암 이력(' + ', '.join(r['others']) + ')')
    if r['post']:
        parts.append('치료 후 채취(collection_date > surgery/chemo)')
    if r['multi']:
        parts.append('다중암 환자')
    if r['drop']:
        parts.append("group='Drop' (임상표 초록 표시)")
    if not parts and r['yellow']:
        parts.append(f'임상표 노란 표시 (사유 미상, 노란 셀 {r["n_yellow"]}개)')
    return '; '.join(parts)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='임상표 제외 판정 요약')
    ap.add_argument('--clinical', default=CLINICAL)
    a = ap.parse_args()
    ix, rows = load(a.clinical)
    res = evaluate(ix, rows)
    print(f'{a.clinical}')
    print(f'  라벨 {len(res)}'
          f' | 요로감염 {sum(v["uti"] for v in res.values())}'
          f' | 타암 {sum(bool(v["others"]) for v in res.values())}'
          f' | 치료후채취 {sum(v["post"] for v in res.values())}'
          f' | 다중암 {sum(v["multi"] for v in res.values())}'
          f' | Drop {sum(v["drop"] for v in res.values())}'
          f' | 노란표시 {sum(v["yellow"] for v in res.values())}'
          f' | 제외 합집합 {sum(v["excl"] for v in res.values())}')
