"""Literature-reference-linked preprocessing method benchmarking (SERS-AI Preprocessing Lab).

See docs/ml/preprocessing_lab_plugin_guide.md for how to add a new method, and
/home/user/.claude/plans/fluttering-jumping-lagoon.md for the full project design.

Storage (decided 2026-09-02): experiment history lives in the `experiment`
schema of the aecd_platform PostgreSQL database — see
scripts/db/experiment_tracking/01_schema.sql for the DDL and db.py for the
writer. Supabase was the original plan but is banned by company security
policy. Keeping the tracking tables in aecd_platform means run→measurement
lineage is a real enforced foreign key rather than a copied id list.
"""
