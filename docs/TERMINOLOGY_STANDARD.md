# Terminology Standard

This project uses one standard decision rule for clinical-style outputs.
Historical threshold options such as `screening`, `balanced`, and `confirmatory`
are model evaluation profiles, not patient-level data attributes.

## Decision Rule

| Concept | Standard term | Avoid |
|---|---|---|
| User-facing cancer vs non-cancer score | SSI score / SSI 점수 | raw probability, risk score |
| Fixed cutoff used for display/reporting | SSI decision threshold / SSI 판정 기준값 | operating mode |
| Score above the standard cutoff | Further evaluation recommended / 추가 확인 권고 | positive, strong positive, cancer diagnosed |
| Score below the standard cutoff | Below decision threshold / 기준 미만 | negative, normal |
| Stage 2 output | Cancer type classification probability / 암종 분류 확률 | confidence, diagnostic confidence |

## Current Standard

The clinical webapp uses SSI as the user-facing score:

- decision profile: `standard_balanced`
- source threshold profile: `balanced`
- user-facing SSI scale: 0-10
- user-facing SSI decision threshold: 4.0
- internal STK-V2 probability threshold: 0.4439
- rationale: Youden's J balance between sensitivity and specificity

The internal probability threshold maps to SSI 4.0. Therefore `0.4439` is not
the displayed SSI cutoff; it is the model probability cutoff used before SSI
score transformation.

The UI and reports should not expose this as a selectable per-patient mode.
The stored patient result should contain the score, threshold, decision rule,
and model version, but not imply that a user selected a different data mode.

## Interpretation

Do not use "strong positive", "weak positive", or similar grading unless a
separate, validated ordinal grading scheme is defined with explicit cutoffs and
performance characteristics. Until then, the only supported patient-level
binary interpretation is whether the SSI score is above or below 4.0.
