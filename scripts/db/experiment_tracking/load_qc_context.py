"""Fill experiment.point_qc / sample_qc / measurement_context from the BM-13 outputs (idempotent).

Source: SERS-AI-ci-tiered/results/qc_fail_factors_1038/ (points.csv, samples.csv), written by
scripts/analysis/qc_fail_factors.py. Needs 05_qc_and_measurement_context.sql applied first.

    source scripts/db/pghost.sh
    python scripts/db/experiment_tracking/load_qc_context.py [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

SRC = Path("/home/user/SERS-AI-ci-tiered/results/qc_fail_factors_1038")
QC_VERSION = "stage1+2a_ver1_allpoints_20260929"
RULE_2B = "121점은 36점 서브샘플 후, 2단계a 뒤 25점 미만이면 검체 제외 (paired_analysis_data.MIN_POINTS)"
PHASE = "BM-13"
REAGENT_LOT = "BCCP0922"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pts = pd.read_csv(SRC / "points.csv", encoding="utf-8-sig")
    smp = pd.read_csv(SRC / "samples.csv", encoding="utf-8-sig")

    with psycopg2.connect(dbname="aecd_platform") as conn, conn.cursor() as cur:
        cur.execute("SELECT run_id FROM experiment.runs WHERE phase = %s", (PHASE,))
        run_id = cur.fetchone()[0]
        # 같은 검체가 변경 전(4~5월)·7월에도 측정돼 있으므로 BM-13 코호트인 환원제 lot 측정분만 잇는다
        cur.execute("""SELECT m.sample_id, m.point_no, m.measurement_id, m.measurement_run_id
                       FROM measurement.measurements m
                       JOIN measurement.runs r ON r.measurement_run_id = m.measurement_run_id
                       JOIN measurement.reagent_lots rl ON rl.reagent_lot_id = r.reagent_lot_id
                       WHERE m.sample_id = ANY(%s) AND m.measurement_role = 'clinical' AND rl.lot_number = %s""",
                    (smp.sample_id.astype(int).tolist(), REAGENT_LOT))
        mid = pd.DataFrame(cur.fetchall(), columns=["sample_id", "point_no", "measurement_id", "measurement_run_id"])
        cur.execute("""SELECT sl.lot_number || '-' || u.unit_number, u.strip_unit_id
                       FROM measurement.strip_units u JOIN measurement.strip_lots sl ON sl.strip_lot_id = u.strip_lot_id""")
        unit_id = dict(cur.fetchall())

        # 같은 검체가 두 run에 있으면 지점 번호만으로 이을 수 없다 — 실제로는 검체당 run 1개
        runs_per_sample = mid.groupby("sample_id").measurement_run_id.nunique()
        assert runs_per_sample.max() == 1, runs_per_sample[runs_per_sample > 1]
        p = pts.merge(mid, on=["sample_id", "point_no"], how="left", validate="one_to_one")
        assert p.measurement_id.notna().all(), p[p.measurement_id.isna()].head()

        point_rows = [(int(r.measurement_id), QC_VERSION, not bool(r.stage1_fail),
                       (r.stage1_reason or None) if isinstance(r.stage1_reason, str) and r.stage1_reason else None,
                       None if bool(r.stage1_fail) else not bool(r.stage2a_fail), run_id)
                      for r in p.itertuples()]
        nfail = pts.groupby("sample_id").fail.sum()
        sample_rows = [(int(r.sample_id), QC_VERSION, int(r.n_points), int(nfail[r.sample_id]),
                        bool(r.pipeline_dropped), RULE_2B, run_id) for r in smp.itertuples()]
        run_of = mid.drop_duplicates("sample_id").set_index("sample_id").measurement_run_id
        ctx_rows = [(int(run_of[r.sample_id]), int(r.sample_id),
                     unit_id.get(r.strip_unit) if isinstance(r.strip_unit, str) else None,
                     r.strip_unit if isinstance(r.strip_unit, str) else None,
                     pd.Timestamp(r.t_end).to_pydatetime(), int(r.order_in_day),
                     "strip_unit: 원본 경로의 SK…-n 폴더 / acquired_at: 검체 첫 지점 CSV 수정 시각 / order_in_day: 같은 측정일 안 순위")
                    for r in smp.itertuples()]
        unmatched = sum(1 for r in ctx_rows if r[3] and r[2] is None)
        print(f"지점 {len(point_rows)} · 검체 {len(sample_rows)} · 측정 맥락 {len(ctx_rows)} "
              f"(판 번호 있음 {sum(1 for r in ctx_rows if r[3])}, 등록 판과 못 이은 것 {unmatched}) · run_id {run_id}", flush=True)
        if args.dry_run:
            print("dry-run: 쓰지 않음")
            conn.rollback()
            return

        execute_values(cur, """INSERT INTO experiment.point_qc
            (measurement_id, qc_version, stage1_pass, stage1_reason, stage2a_pass, run_id) VALUES %s
            ON CONFLICT (measurement_id, qc_version) DO UPDATE SET stage1_pass = EXCLUDED.stage1_pass,
            stage1_reason = EXCLUDED.stage1_reason, stage2a_pass = EXCLUDED.stage2a_pass,
            run_id = EXCLUDED.run_id, computed_at = now()""", point_rows, page_size=5000)
        execute_values(cur, """INSERT INTO experiment.sample_qc
            (sample_id, qc_version, n_points, n_fail, dropped_2b, dropped_2b_rule, run_id) VALUES %s
            ON CONFLICT (sample_id, qc_version) DO UPDATE SET n_points = EXCLUDED.n_points, n_fail = EXCLUDED.n_fail,
            dropped_2b = EXCLUDED.dropped_2b, dropped_2b_rule = EXCLUDED.dropped_2b_rule,
            run_id = EXCLUDED.run_id, computed_at = now()""", sample_rows)
        execute_values(cur, """INSERT INTO experiment.measurement_context
            (measurement_run_id, sample_id, strip_unit_id, strip_unit_label, acquired_at, order_in_day, source) VALUES %s
            ON CONFLICT (measurement_run_id, sample_id) DO UPDATE SET strip_unit_id = EXCLUDED.strip_unit_id,
            strip_unit_label = EXCLUDED.strip_unit_label, acquired_at = EXCLUDED.acquired_at,
            order_in_day = EXCLUDED.order_in_day, source = EXCLUDED.source, updated_at = now()""", ctx_rows)
        conn.commit()

        cur.execute("SELECT count(*), count(*) FILTER (WHERE NOT stage1_pass OR NOT stage2a_pass) FROM experiment.point_qc WHERE qc_version = %s", (QC_VERSION,))
        n, f = cur.fetchone()
        print(f"적재 확인: point_qc {n}행, 탈락 {f} ({f / n:.1%})", flush=True)
        if abs(f / n - pts.fail.mean()) > 1e-9:
            sys.exit("탈락률이 원본 CSV와 다름")


if __name__ == "__main__":
    main()
