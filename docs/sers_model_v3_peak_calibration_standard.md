# SERS Model V3: calibrated evidence model and peak-window standard

This note defines the model change requested after the 2026-06-10 KAO-style
rerun. The previous STK-V2 pipeline is useful as a classifier, but it is not
yet an auditable clinical evidence model.

## 1. Meaning of model probability

The training labels are clinical group labels. Therefore an output such as
`P(cancer)=0.82` must be interpreted as:

`P(training label = cancer | spectrum, preprocessing, cohort distribution)`

It is not an absolute disease probability until the model is calibrated and
validated against an external reference population with known prevalence.

Required outputs:

- calibrated binary cancer probability using the validation split only;
- calibration metrics: Brier score, expected calibration error, calibration
  intercept/slope where applicable;
- fixed operating thresholds chosen on validation data:
  `balanced`, `screening_high_sensitivity`, and `confirmatory_high_specificity`;
- patient-level report that shows probability, threshold class, and supporting
  spectral evidence.

## 2. Peak existence standard

Voigt fitting must not define whether a peak exists. It can be used only after
the peak/window has been defined by data-driven criteria.

Primary peak existence criteria:

1. Smooth the subject-level spectrum with Savitzky-Golay filter.
2. Estimate local noise from the residual between raw processed intensity and
   the smoothed spectrum using robust MAD.
3. A subject-level peak exists when all conditions hold:
   - local maximum in the smoothed spectrum;
   - prominence >= 3 x subject noise sigma;
   - half-prominence width is within the configured physical width range.
4. A group-level peak exists when all conditions hold:
   - group mean local maximum;
   - group mean prominence SNR >= 3;
   - reproducibility >= configured subject fraction within the group;
   - the peak window is defined from group-level half-prominence width, then
     expanded by a fixed margin and clipped to the min/max width bounds.

Secondary disease-discriminating evidence:

- For each accepted group peak, compare the window feature against Control.
- Report Welch t-test p-value, Benjamini-Hochberg q-value, and standardized
  effect size.
- A peak may be biologically present even when it is not statistically
  disease-discriminating.

## 3. Model architecture change

The interpretable model should use explicit peak-window features as the primary
evidence layer:

- peak presence;
- peak height/prominence;
- peak window area;
- selected ratios only after a biochemical or statistical reason is recorded.

Derivative channels such as D1 may remain as auxiliary signals, but they should
not be the main explanation. Any attribution from raw/D1/D2 models must be
aggregated back to registered peak windows.

## 4. Current implementation status

The current codebase now keeps the original STK-V2 run intact, and adds:

- validation-based probability calibration outputs;
- a data-driven peak/window registry with explicit acceptance criteria.

The next model iteration should train a sparse calibrated peak-evidence model
and compare it against STK-V2 on the same locked subject split.
