#!/usr/bin/env bash
# AECD Platform Data API(read-only)를 로컬에서 기동한다.
#
# WHY: notebooks/aecd_api_*.ipynb 와 scripts/analysis/aecd_api_*.py 는
#      http://127.0.0.1:8000 의 이 API에서 스펙트럼을 읽는다. 서버가 떠 있지
#      않으면 노트북 첫 실행 셀이 httpx ConnectError(WinError 10061 —
#      "대상 컴퓨터에서 연결을 거부")로 죽는다. 이 스크립트가 없어서
#      기동 방법이 매번 재발견 대상이었다 (2026-09-02).
#
# 네트워킹: .wslconfig 가 networkingMode=mirrored 이므로 WSL에서 띄운
#      0.0.0.0:8000 을 Windows 쪽 Jupyter가 http://127.0.0.1:8000 으로
#      그대로 접근한다 (2026-09-02 windows curl.exe 로 200 확인).
#
# 사용법:
#   scripts/deployment/serve_aecd_api.sh            # 포그라운드 (Ctrl+C 종료)
#   scripts/deployment/serve_aecd_api.sh --port 8001
#   nohup scripts/deployment/serve_aecd_api.sh > logs/aecd_api.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# .env(PG 접속정보) + .env.aecd-api(AECD_API_KEY)를 환경으로 올린다.
set -a
[ -f "$REPO_ROOT/.env" ] && . "$REPO_ROOT/.env"
[ -f "$REPO_ROOT/.env.aecd-api" ] && . "$REPO_ROOT/.env.aecd-api"
set +a

# mirrored 네트워킹에서는 localhost가 Windows PostgreSQL(aecd_platform)을 가리킨다.
# pghost.sh 가 실제 인증을 시도해 올바른 호스트를 고른다.
# shellcheck source=scripts/db/pghost.sh
source "$REPO_ROOT/scripts/db/pghost.sh" --quiet

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

HOST="0.0.0.0"
PORT="8000"
while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "알 수 없는 옵션: $1" >&2; exit 2 ;;
    esac
done

if [ -z "${AECD_API_KEY:-}" ]; then
    echo "경고: AECD_API_KEY가 없습니다. API가 인증 없이 열립니다." >&2
fi

echo "AECD API 기동: http://127.0.0.1:${PORT}  (DB: ${PGUSER:-postgres}@${PGHOST}/${PGDATABASE:-aecd_platform})"
exec python -m uvicorn sers.aecd_api.app:app --host "$HOST" --port "$PORT"
