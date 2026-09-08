#!/usr/bin/env bash
# Load 전체환자_임상정보_정규화_v7.xlsx into aecd_platform (master.* + clinical.*).
#
# Usage:
#   scripts/db/aecd_clinical_v7/run_load.sh --dry-run   # everything, but roll back step 04
#   scripts/db/aecd_clinical_v7/run_load.sh             # commit
#
# Steps 01-03 are safe to repeat: 01 is idempotent, 02 replaces its own batch,
# 03 only reads. Step 04 is the one that writes clinical data.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
CSV="${STAGING_CSV:-$ROOT/data/processed/aecd_platform_ingest/전체환자_임상정보_정규화_v7_staging.csv}"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# shellcheck source=/dev/null
source "$ROOT/scripts/db/pghost.sh" --quiet

PSQL=(psql -d aecd_platform -v ON_ERROR_STOP=1)

if [ ! -f "$CSV" ]; then
    echo "staging CSV not found: $CSV" >&2
    echo "run: python scripts/db/aecd_clinical_v7/build_staging_csv.py" >&2
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="$ROOT/data/processed/aecd_platform_ingest/backup_master_clinical_ingest_$STAMP.sql"
echo "== backup -> $BACKUP"
pg_dump -d aecd_platform -n master -n clinical -n ingest -f "$BACKUP"

echo "== 01 sites"
"${PSQL[@]}" -f "$HERE/01_sites.sql"

echo "== 02 staging"
COPY_SQL="$(mktemp)"
sed "s#__STAGING_CSV__#$CSV#" "$HERE/02_load_staging.sql" > "$COPY_SQL"
"${PSQL[@]}" -f "$COPY_SQL"
rm -f "$COPY_SQL"

echo "== 03 preflight"
"${PSQL[@]}" -f "$HERE/03_preflight.sql"

echo "== 04 master + clinical"
if [ "$DRY_RUN" -eq 1 ]; then
    TMP="$(mktemp)"
    sed 's/^COMMIT;$/ROLLBACK;/' "$HERE/04_load_master_clinical.sql" > "$TMP"
    "${PSQL[@]}" -f "$TMP"
    rm -f "$TMP"
    echo "== dry run: step 04 rolled back"
else
    "${PSQL[@]}" -f "$HERE/04_load_master_clinical.sql"
    echo "== 05 verify"
    "${PSQL[@]}" -f "$HERE/05_verify.sql"
fi
