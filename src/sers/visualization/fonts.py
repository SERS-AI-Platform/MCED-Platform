"""한글 라벨이 들어간 그림을 위한 matplotlib 폰트 설정.

WHY: 기본 sans-serif에는 한글 글리프가 없어서 라벨이 두부(□)로 렌더링된다.
매 스크립트에서 rcParams를 손대는 대신 여기 한 곳에서 처리한다.

폰트가 없으면 조용히 넘어가지 않고 False를 돌려준다 -- 두부가 찍힌 그림을
성공으로 착각하는 것보다 호출부가 영문 라벨로 내려가는 편이 낫다.

설치 (sudo 없이):
    pip download koreanize-matplotlib -d /tmp/kf --no-deps
    unzip -o /tmp/kf/koreanize_matplotlib-*.whl -d /tmp/kf_x
    cp /tmp/kf_x/koreanize_matplotlib/fonts/NanumGothic*.ttf ~/.local/share/fonts/
    fc-cache -f ~/.local/share/fonts
"""

from __future__ import annotations

from typing import Final

import matplotlib
import matplotlib.font_manager as fm

PREFERRED_FONTS: Final = ("NanumGothic", "NanumSquare", "Malgun Gothic", "AppleGothic")


def available_korean_font() -> str | None:
    """설치된 한글 폰트 이름을 우선순위대로 찾아 돌려준다."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in PREFERRED_FONTS:
        if name in installed:
            return name
    return None


def use_korean_font(size: int | None = None) -> bool:
    """matplotlib 기본 폰트를 한글 폰트로 바꾼다. 성공하면 True.

    unicode_minus를 끄는 이유: 나눔고딕에는 U+2212(MINUS SIGN) 글리프가 없어서
    음수 축 눈금이 두부가 된다. ASCII 하이픈으로 대체한다.
    """
    font = available_korean_font()
    if font is None:
        return False
    matplotlib.rcParams["font.family"] = font
    matplotlib.rcParams["axes.unicode_minus"] = False
    if size is not None:
        matplotlib.rcParams["font.size"] = size
    return True
