"""제외 규칙이 이전 수작업 결과(0728 파일의 '제외대상 Bold 처리')를 재현하는지 확인.

새 임상표 버전으로 넘어갈 때 규칙이 조용히 달라지지 않았는지 보는 회귀 검사다.
0907 워크북에 남아 있는 '제외사유_목록' 시트의 solum_label 83개를 정답으로 쓴다
(그 시트의 Lot_Cell 열은 어긋나 있지만 라벨은 유효하다 -- lot_map.py 주석 참고).

  python scripts/db/lot_powder_exclusion/validate_rule.py

기대값 (2026-09-07 기준, --clinical 을 v7 로 두었을 때):
  못 잡은 기존 제외 라벨 0 / yellow 50 대 50 / 타암 81 대 81
"""
import argparse

import openpyxl
from exclusion_rule import evaluate, load, norm
from paths import CLINICAL_PREV, LOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clinical', default=CLINICAL_PREV,
                    help='이전 목록을 만들 때 쓴 임상표 (기본 v7)')
    ap.add_argument('--lot', default=LOT)
    a = ap.parse_args()

    ix, rows = load(a.clinical)
    res = evaluate(ix, rows)

    wb = openpyxl.load_workbook(a.lot, read_only=True)
    ws = wb['제외사유_목록']
    data = list(ws.iter_rows(values_only=True))
    jx = {h: i for i, h in enumerate(data[0]) if h}
    old = {}
    for r in data[1:]:
        if not r[jx['Lot_Sheet']]:
            continue
        old[norm(r[jx['solum_label']])] = dict(
            yellow=str(r[jx['Clinical_Yellow_Highlighted']] or '').strip() == 'Y',
            others=str(r[jx['Other_Cancer_History_Match']] or '').strip() == 'Y',
            both=str(r[jx['Nitrite_Leukocyte_Both_Positive']] or '').strip() == 'Y')
    wb.close()

    print(f'기존 목록 고유 라벨 {len(old)}')
    absent = [label for label in old if label not in res]
    if absent:
        print(f'  임상표에 없는 라벨: {absent}')
    missed = [label for label in old if label in res and not res[label]['excl']]
    print(f'  못 잡은 기존 제외 라벨: {len(missed)} {missed[:10]}')

    for axis in ('yellow', 'others', 'both'):
        mine = {label for label in old if label in res and bool(res[label][axis])}
        theirs = {label for label, v in old.items() if v[axis]}
        print(f'  {axis:7s} 기존 {len(theirs):3d} / 내규칙 {len(mine):3d}'
              f' | 기존만 {sorted(theirs - mine)} | 내것만 {sorted(mine - theirs)}')


if __name__ == '__main__':
    main()
