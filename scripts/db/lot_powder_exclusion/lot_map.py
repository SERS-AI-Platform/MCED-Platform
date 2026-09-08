"""Lot_Powder 워크북의 DAY 시트 셀 <-> 임상표 solum_label 매핑.

DAY 시트 구조
  2행이 그룹 머리글(C~R열), 3~32행이 데이터. 한 셀에 검체 번호가 쉼표로 여러 개
  들어 있다 (예: K3 = '27,28,29,30').

매핑 근거
  - 같은 이름 접두사가 임상표에 있는 그룹은 그대로 간다 (NOR, HBP, DIA, H. D.,
    YNOR, BNOR, PRO, BPRO, YPAN, BRE, OVA, LUN, CRC, BLC).
  - CPAN -> PAN 은 0907 워크북의 기존 '제외사유_목록' 행으로 확인된다
    (CPAN 1 -> PAN_1 등).
  - KPAN 은 재료연에서 받은 검체라 임상표에 대응 라벨이 없다. 임상 기준으로
    제외 판정을 할 수 없으므로 매핑에서 빼 두었고, 표시 대상이 아니다.
    번호만으로는 PAN(1~120)과 SPAN(1~126) 어느 쪽과도 구분되지 않으니
    (CPAN 과 53개 번호가 겹친다) 임의로 되살리지 말 것.

주의: 워크북에 들어 있는 '제외사유_목록' 시트의 Lot_Cell 열은 0907 DAY 시트와
어긋나 있다 (목록은 K3 = 'CPAN 1' 이라 하지만 0907의 K3 은 '27,28,29,30').
중간 버전을 기준으로 만들어진 것이라 셀 주소는 믿으면 안 되고, solum_label 열만
유효하다. 이 스크립트가 만드는 '제외사유_목록_v8' 의 셀 주소는 0907 기준이다.
"""
import re

import openpyxl
from paths import LOT

DAYS = ['DAY 1', 'DAY 2', 'DAY 3', 'DAY 4', 'DAY 5', 'DAY 6']
DATA_MIN_ROW, DATA_MAX_ROW = 3, 32
DATA_MIN_COL, DATA_MAX_COL = 3, 18          # C~R

GROUP2PREFIX = {
    'NOR': 'NOR', 'HBP': 'HBP', 'DIA': 'DIA', 'H. D.': 'H. D.',
    'YNOR': 'YNOR', 'BNOR': 'BNOR',
    'PRO': 'PRO', 'BPRO': 'BPRO',
    'CPAN': 'PAN', 'YPAN': 'YPAN',
    # 'KPAN': 재료연 검체 -- 임상정보 없음, 판정 대상 아님 (위 주석 참고)
    'BRE': 'BRE', 'OVA': 'OVA', 'LUN': 'LUN', 'CRC': 'CRC', 'BLC': 'BLC',
}


def norm(label):
    return re.sub(r'\s+', '', str(label)) if label is not None else ''


def scan(path=LOT):
    """[(sheet, coord, group, number, label)] -- DAY 시트의 모든 검체 번호."""
    wb = openpyxl.load_workbook(path)
    hits = []
    for name in DAYS:
        ws = wb[name]
        header = {c.column: c.value for c in ws[2] if c.value}
        for row in ws.iter_rows(min_row=DATA_MIN_ROW, max_row=DATA_MAX_ROW):
            for cell in row:
                group = header.get(cell.column)
                if group not in GROUP2PREFIX or cell.value in (None, ''):
                    continue
                for token in str(cell.value).split(','):
                    t = token.strip()
                    if t.isdigit():
                        hits.append((name, cell.coordinate, group, int(t),
                                     norm(f'{GROUP2PREFIX[group]}_{t}')))
    wb.close()
    return hits


if __name__ == '__main__':
    from collections import Counter
    hits = scan()
    print(f'Lot 셀 토큰 {len(hits)} | 고유 라벨 {len({h[4] for h in hits})}')
    print('그룹별:', dict(Counter(h[2] for h in hits)))
