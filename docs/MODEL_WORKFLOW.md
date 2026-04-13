# Model Workflow

## 1. Train a Single Model

Example: XGBoost on a selected class subset.

```powershell
python models/train.py `
  --aggregate none `
  --model xgboost `
  --cancer-types PRO LUN CRC CPAN OVA `
  --non-cancer-groups NOR DIA HBP H.D. `
  -i results/processed_spectra.csv `
  -o models/results/benchmark_all_spectra
```

Output layout:

```text
models/results/benchmark_all_spectra/
  xgboost/
    latest.txt
    v001/
      checkpoints/
      fold_predictions.npz
      training_summary.json
      fold_metrics.csv
      experiment_log.json
```

## 2. Train Multiple Models for Comparison

```powershell
python models/train.py `
  --aggregate none `
  --benchmark-models resnet18 cnn1d xgboost `
  --cancer-types PRO LUN CRC CPAN OVA `
  --non-cancer-groups NOR DIA HBP H.D. `
  -i results/processed_spectra.csv `
  -o models/results/benchmark_all_spectra
```

Comparison table:

```text
models/results/benchmark_all_spectra/benchmark_summary.csv
```

## 3. Evaluate a Model

Passing the model root resolves the latest version automatically.

```powershell
python models/test.py `
  -i models/results/benchmark_all_spectra/xgboost `
  --processed-csv results/processed_spectra.csv `
  --no-shap
```

Default `t-SNE` is 3D. To force 2D:

```powershell
python models/test.py `
  -i models/results/benchmark_all_spectra/xgboost `
  --processed-csv results/processed_spectra.csv `
  --tsne-dim 2 `
  --no-shap
```

## 4. Evaluation Outputs

Important outputs under `evaluation/`:

- `stage1/roc_curve.png`
- `stage1/confusion_matrix.png`
- `stage2/roc_curves_per_type.png`
- `stage2/confusion_matrix.png`
- `stage2/mean_spectra_overlay.png`
- `stage2/mean_spectra_overlay.csv`
- `stage2/by_diagnosis/<class>/mean_spectrum.png`
- `stage2/by_diagnosis/<class>/mean_spectrum.csv`
- `stage2/by_diagnosis/<class>/peak_difference_vs_rest.png`
- `stage2/by_diagnosis/<class>/peak_difference_vs_rest.csv`
- `stage2/by_diagnosis/<class>/feature_importance.png`
- `stage2/by_diagnosis/<class>/mean_abs_shap_spectrum.png`
- `stage2/by_diagnosis/<class>/mean_shap_spectrum.png`

## 5. Hyperparameter Tuning Focus

### ResNet18-1D

Tune first:

- `learning_rate`: `1e-4`, `3e-4`, `5e-4`
- `weight_decay`: `1e-4`, `5e-4`, `1e-3`, `3e-3`
- `dropout_rate`: `0.3`, `0.4`, `0.5`, `0.6`
- `batch_size`: `16`, `32`, `64`
- `stage2_loss_weight`: `1.0`, `1.5`, `2.0`

Tune architecture second:

- `resnet_channels`: `(32,64,128,256)` vs `(64,128,256,512)`
- `head_hidden_dim`: `64` vs `128`

### CNN1D-Shallow

Tune first:

- `learning_rate`: `1e-4`, `3e-4`, `5e-4`, `1e-3`
- `dropout_rate`: `0.2`, `0.3`, `0.4`, `0.5`
- `batch_size`: `32`, `64`, `128`
- `head_hidden_dim`: `64`, `128`, `256`

Because the model is shallow, capacity is often the bottleneck. If `cnn1d` underfits, increase:

- channel count
- kernel diversity
- head hidden dimension

### XGBoost

Tune:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `reg_lambda`

Current defaults are saved into `training_summary.json -> model_params`.
