-- aecd_platform: allow runs without a recorded measurement date (2026-09-11, user decision)
--
-- Why: retrospective Thermo (raw_data main groups) and Medical (NanoScope Ramcheck A1) spectra
-- have no surviving acquisition date — file timestamps on every copy are copy/upload/sync
-- times, and the files carry no internal timestamp. They are loaded grouped by x-axis grid
-- session (Thermo) or delivered folder (Medical) instead of by day.
--
-- Guard: a NULL date must be declared in notes, so it cannot slip in by accident.
-- Downstream: src/sers/aecd_api/repository.py excludes NULL-date runs from the spectrum
-- endpoint (measured_at is required by the published contract).
--
-- NOTE: the canonical measurement.* DDL is not in this repository; apply the same change
-- wherever it lives, or a fresh schema build will restore NOT NULL.

BEGIN;

ALTER TABLE measurement.runs ALTER COLUMN measurement_date DROP NOT NULL;

ALTER TABLE measurement.runs
    ADD CONSTRAINT runs_unknown_date_declared
    CHECK (measurement_date IS NOT NULL OR notes ~ 'measurement_date=unknown');

COMMENT ON COLUMN measurement.runs.measurement_date IS
    'Acquisition date. NULL only for retrospective runs whose date was not preserved; '
    'such runs must say measurement_date=unknown in notes (runs_unknown_date_declared).';

COMMIT;
