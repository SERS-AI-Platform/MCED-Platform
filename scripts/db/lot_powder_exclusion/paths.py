"""공통 경로/기본 파일명.

Windows 쪽 Downloads에 원본 워크북이 있어서 WSL 경로로 잡는다. 다른 위치에
두고 싶으면 각 스크립트의 --clinical / --lot / --out 옵션으로 덮어쓴다.
"""
from pathlib import Path

DOWNLOADS = Path('/mnt/c/Users/user/Downloads')

CLINICAL = DOWNLOADS / '전체환자_임상정보_정규화_v8.xlsx'
LOT = DOWNLOADS / '검체 측정 Lot_Powder_0907_ver 2.xlsx'
OUT = DOWNLOADS / '검체 측정 Lot_Powder_0907_ver 2_v8임상기준_제외대상_빨간색처리.xlsx'

# 이전 버전 (규칙 검증용 - validate_against_previous.py 가 쓴다)
CLINICAL_PREV = DOWNLOADS / '전체환자_임상정보_정규화_v7.xlsx'
