from __future__ import annotations


def html_document(markdown_body: str) -> str:
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>기존 액상 SERS와 신규 powder SERS 비교</title>
<style>
body{{font-family:'NanumSquare','Noto Sans KR',sans-serif;max-width:1080px;margin:0 auto;padding:36px;color:#252525;line-height:1.65;background:#fff}}
h1,h2{{line-height:1.3}} h2{{margin-top:2.2em;border-bottom:1px solid #ddd;padding-bottom:.3em}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}} th,td{{border-bottom:1px solid #ddd;padding:8px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
img{{max-width:100%;height:auto;margin:16px 0 28px}} code{{background:#f4f4f4;padding:2px 5px;border-radius:3px}}
@media(max-width:700px){{body{{padding:18px}}table{{font-size:12px;display:block;overflow-x:auto}}}}
</style></head><body>{markdown_body}</body></html>"""
