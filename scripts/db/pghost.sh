#!/usr/bin/env bash
# WSL에서 Windows 호스트의 PostgreSQL(aecd_platform)을 찾아 PGHOST를 설정한다.
#
# WHY: WSL의 localhost와 Windows의 localhost는 서로 다른 서버다. WSL 안에도
#      PostgreSQL 16이 5432에 떠 있어서, PGHOST=localhost로는 엉뚱한 서버에
#      붙어 인증 실패한다 (2026-09-02 확인). Windows 쪽은 PostgreSQL 15.15에
#      실제 aecd_platform이 있다.
#
# 이 스크립트는 후보 호스트에 실제로 인증을 시도해서 성공하는 쪽을 고르므로,
# .wslconfig의 mirrored 네트워킹 적용 전(게이트웨이 IP 필요)과 적용 후
# (localhost로 동작) 양쪽에서 모두 올바르게 동작한다. 게이트웨이 IP는 WSL
# 재시작 시 바뀌므로 하드코딩하지 않는다.
#
# 사용법:
#   source scripts/db/pghost.sh          # PGHOST를 export
#   source scripts/db/pghost.sh --quiet  # 메시지 없이

set -a; [ -f "$(dirname "${BASH_SOURCE[0]}")/../../.env" ] && . "$(dirname "${BASH_SOURCE[0]}")/../../.env"; set +a

_pghost_quiet=0
[ "$1" = "--quiet" ] && _pghost_quiet=1

_pghost_try() {
    PGHOST="$1" PGCONNECT_TIMEOUT=3 psql -d "${PGDATABASE:-aecd_platform}" \
        -tAc "select 1" >/dev/null 2>&1
}

_pghost_gateway="$(ip route show default 2>/dev/null | awk '{print $3}' | head -1)"

for _candidate in localhost "$_pghost_gateway"; do
    [ -z "$_candidate" ] && continue
    if _pghost_try "$_candidate"; then
        export PGHOST="$_candidate"
        [ "$_pghost_quiet" -eq 0 ] && echo "PGHOST=$_candidate (aecd_platform 인증 성공)"
        unset _pghost_try _pghost_gateway _pghost_quiet _candidate
        return 0 2>/dev/null || exit 0
    fi
done

echo "경고: localhost / $_pghost_gateway 어느 쪽에서도 ${PGDATABASE:-aecd_platform}에 인증하지 못했습니다." >&2
echo "      .env의 PGUSER/PGPASSWORD를 확인하세요." >&2
unset _pghost_try _pghost_gateway _pghost_quiet _candidate
return 1 2>/dev/null || exit 1
