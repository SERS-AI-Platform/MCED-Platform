# SERS Clinical Demo Data

Synthetic upload data for UI, build, and QC testing. These files are not clinical
examples and must not be used for validation.

- `patient_ids.csv`: five demo patient IDs with age, sex, and BMI values.
- `normal/`: five replicate CSV spectra per demo patient.
- `bad/`: intentionally poor-quality spectra for QC and warning checks.

Each spectrum CSV has no header and uses two numeric columns:

1. Raman shift / wavenumber
2. intensity

For QC testing, upload four normal files for one patient plus one file from
`bad/`, especially `LOWINT-001_1.csv` or `NOISY-001_1.csv`.
