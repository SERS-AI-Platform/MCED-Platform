# SERS Clinical Demo Data

Synthetic upload data for UI, build, and QC testing. These files are not clinical
examples and must not be used for validation.

- `patient_ids.csv`: five demo patient IDs with age, sex, and BMI values.
- `normal/`: five replicate CSV spectra per demo patient.
- `bad/`: intentionally poor-quality spectra for QC and warning checks.
- `usability/heldout_pass/`: de-identified heldout-test sets covering all seven
  cancer types plus low and medium SSI examples.
- `usability/qc_remeasurement/`: one patient-level QC Fail set followed by a
  five-file remeasurement Pass set.
- `usability/patient_mapping.csv`: patient-entry values, expected QC/SSI/type,
  source split, and relative upload directory for every usability scenario.

Each spectrum CSV has no header and uses two numeric columns:

1. Raman shift / wavenumber
2. intensity

For QC testing, upload four normal files for one patient plus one file from
`bad/`, especially `LOWINT-001_1.csv` or `NOISY-001_1.csv`.

For the summative usability workflow, use only the patient-level directories
under `usability/` and enter the matching age, sex, and BMI from
`patient_mapping.csv`. The original hospital identifier and the remaining
clinical table are intentionally excluded. These retrospective fixtures are
for UI/QC demonstration only, not clinical validation. Cancer-vs-control
outputs may be hospital-confounded and must not be presented as cross-hospital
generalization evidence.
