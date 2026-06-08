# Figure Index

## 01_accuracy_summary.png

Shows sample-majority accuracy after voting across replicates. `Primary CV` is
the internal positive control: each primary replicate is predicted using the
other primary replicates. Thermo/Handheld/Medical are primary-to-lot-balanced
transfer tests. If same-ID transfer worked, the same-ID bars would stay high
outside `Primary CV`. They do not.

## 02_true_rank_distribution.png

Shows the rank of the correct sample ID among all candidate sample IDs. Rank 1
means the model chose the correct sample. The y-axis is log-scaled because the
lot-balanced ranks are often hundreds of positions away from the correct ID.

## 03_group_same_id_heatmap.png

Shows same-ID accuracy separately by true disease/control group. Each cell also
prints the number of overlapping samples. This identifies whether failure is
global or concentrated in specific groups.

## 04_group_confusion_matrices.png

Shows where each group is predicted after sample-majority voting. This is not
same-ID accuracy; it is a disease/control group-level confusion view, useful for
seeing whether the transfer failure is just identity-level or also group-level.
