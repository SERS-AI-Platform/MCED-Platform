# Experiment Results: Step 1-2 (GPU Baseline & ResNet18 Optimization)

Date: 2026-03-17
Environment: WSL2, NVIDIA GeForce RTX 5070 Ti (16GB), PyTorch 2.10.0+cu128

---

## Step 1: Baseline Re-establishment (GPU)

### Setting

- Aggregation: `none` (all spectra)
- Samples: 6,200
- Cancer types (5): PRO, LUN, CRC, CPAN, OVA
- Non-cancer groups (4): NOR, DIA, HBP, H.D.
- CV: 5-fold StratifiedGroupKFold

### Results

| Model | S1 AUC | S1 Acc | S1 Sens | S1 Spec | S2 AUC | S2 Acc | S2 F1 macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.981** | **0.933** | **0.939** | **0.922** | **0.985** | **0.902** | **0.872** |
| XGBoost | 0.969 | 0.908 | 0.924 | 0.876 | 0.960 | 0.828 | 0.757 |
| ResNet18-1D | 0.958 | 0.895 | 0.901 | 0.881 | 0.930 | 0.771 | 0.710 |

### Comparison with Previous Runs (2026-03-13)

| Model | S2 F1 (prev) | S2 F1 (now) | Delta |
|---|---:|---:|---|
| Logistic Regression | 0.870 | 0.872 | +0.002 (reproduced) |
| XGBoost | 0.776 | 0.757 | -0.019 (within variance) |
| ResNet18-1D | 0.710 | 0.710 | 0.000 (reproduced) |

### Conclusions

- Reproducibility confirmed: all models within expected variance of prior results.
- Ranking unchanged: LR >> XGBoost >> ResNet18.
- GPU speedup: ResNet18 training ~6x faster (9 min CPU -> ~1.5 min/fold GPU).
- Overfitting confirmed: ResNet18 train S1 AUC 0.997 vs val 0.958.

---

## Step 2: ResNet18 Optimization Hypotheses

### Common Setting

Same as Step 1 baseline (6,200 samples, 5 cancers, 4 controls, aggregation=none).

### Hypothesis A: Larger Encoder

**Rationale**: Default slim encoder (32, 64, 128, 256) may lack capacity.

| Parameter | Baseline | Hypothesis A |
|---|---|---|
| ResNet channels | (32, 64, 128, 256) | (64, 128, 256, 512) |
| Total params | 996,774 | 3,909,958 |
| Encoder output dim | 256 | 512 |
| Learning rate | 5e-4 | 3e-4 |
| Dropout | 0.5 | 0.3 |
| Batch size | 32 | 64 |
| Max epochs | 150 | 200 |

**Results**:

| Metric | Baseline | Hypothesis A | Delta |
|---|---:|---:|---|
| S1 AUC | 0.958 | 0.959 | +0.001 |
| S2 Acc | 0.771 | 0.741 | -0.030 |
| S2 F1 macro | 0.710 | 0.677 | **-0.033** |
| S2 AUC | 0.930 | 0.919 | -0.011 |
| Avg epochs (early stop) | ~37 | ~34 | - |

**Verdict**: WORSE. 4x more parameters increased overfitting without generalization gain.

---

### Hypothesis B: Stronger Regularization

**Rationale**: Overfitting gap (train 0.997 vs val 0.958) suggests regularization is insufficient.

| Parameter | Baseline | Hypothesis B |
|---|---|---|
| Dropout | 0.5 | 0.6 |
| Weight decay | 1e-3 | 5e-3 |
| Learning rate | 5e-4 | 1e-3 |
| Focal loss | off | on |
| Class balance beta | None | 0.999 |

**Results**:

| Metric | Baseline | Hypothesis B | Delta |
|---|---:|---:|---|
| S1 AUC | 0.958 | 0.945 | -0.013 |
| S2 Acc | 0.771 | 0.762 | -0.009 |
| S2 F1 macro | 0.710 | 0.707 | **-0.003** |
| S2 AUC | 0.930 | 0.937 | +0.007 |
| Train S1 AUC | 0.997 | 0.960 | -0.037 |
| Overfitting gap | 0.039 | 0.015 | reduced |

**Verdict**: NEUTRAL. Overfitting gap reduced (0.039 -> 0.015), but val performance did not improve. The model simply learned less overall.

---

### Hypothesis C: Stage 2 Loss Reweighting

**Rationale**: Stage 2 (cancer type) may be under-trained because Stage 1 loss dominates.

| Parameter | Baseline | Hypothesis C |
|---|---|---|
| Stage 2 loss weight | 1.0 | 2.0 |
| Focal loss | off | on (gamma=1.0) |

**Results**:

| Metric | Baseline | Hypothesis C | Delta |
|---|---:|---:|---|
| S1 AUC | 0.958 | 0.921 | -0.037 |
| S2 Acc | 0.771 | 0.715 | -0.056 |
| S2 F1 macro | 0.710 | 0.668 | **-0.042** |
| S2 AUC | 0.930 | 0.917 | -0.013 |

**Verdict**: WORSE. Overweighting Stage 2 degraded Stage 1 without improving Stage 2. The two objectives appear coupled through the shared encoder.

---

## Step 2 Summary

| Experiment | S1 AUC | S2 F1 macro | vs Baseline |
|---|---:|---:|---|
| **Baseline ResNet18** | 0.958 | **0.710** | -- |
| A: Larger encoder | 0.959 | 0.677 | -3.3% |
| B: Strong regularization | 0.945 | 0.707 | -0.3% |
| C: Stage2 reweight | 0.921 | 0.668 | -4.2% |
| **LR (reference)** | **0.981** | **0.872** | +16.2% above ResNet18 |

### Key Takeaway

Standard hyperparameter-level tuning of ResNet18 cannot close the 16% F1 gap with Logistic Regression. The bottleneck is not model capacity, regularization, or loss weighting. The bottleneck is the **input representation**: SNV-preprocessed spectra are already linearly separable, giving classical models an inherent advantage.

### Next Steps

- **Step 3**: Change the input representation (raw spectra, multi-channel, derivatives) so DL has a representation learning advantage.
- **Step 5**: Ensemble LR + ResNet18 to extract complementary signal.
