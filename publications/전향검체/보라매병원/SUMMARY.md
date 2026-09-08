# 보라매병원 전향검체 SERS 분석 산출물

- 분석 포함: 109명 — Control 21, Biopsy-negative 47, Prostate 41
- Figures: Fig01 preprocessing, Fig02 Clean PRO comparison, Fig03 Screening AUC/CM, Fig04a Screening peaks, Fig04b 3-group peaks, Fig05 Grade Group, Fig06 3-group AUC/CM
- Screening ROC-AUC: 0.714, balanced accuracy: 0.704
- 3-group macro OVR ROC-AUC: 0.811, balanced accuracy: 0.668
- Peak criteria: group mean preprocessed SNV spectra, local maxima, prominence `max(0.08, 10% of group range)`, min distance 18 cm^-1, common if all groups are within +/-12 cm^-1
- Peak clusters: Fig04a Screening common 11, differential 0; Fig04b 3-group common 11, differential 1
- Shaded regions: top 5 differential peak clusters for Fig04a and top 5 for Fig04b
- Peak tables: `tables/fig04a_screening_*`, `tables/fig04b_three_group_*`
- 상세 성능과 95% CI: `PROSTATE_COMPARISON.md`
