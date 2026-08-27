from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BLOCK_CHOICES: Final = (3, 6, 9, 12, 15, 18)


@dataclass(frozen=True, slots=True)
class ResNetFit:
    model: torch.nn.Module
    history: tuple[dict[str, float | int | str], ...]
    best_epoch: int
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiScaleResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.10) -> None:
        super().__init__()
        branch_channels = max(8, out_channels // 3)
        branch_groups = 4 if branch_channels % 4 == 0 else 2 if branch_channels % 2 == 0 else 1
        self.branches = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels, branch_channels, kernel_size=kernel, padding=kernel // 2),
                    torch.nn.GroupNorm(branch_groups, branch_channels),
                    torch.nn.GELU(),
                )
                for kernel in (3, 5, 7)
            ]
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.Conv1d(branch_channels * 3, out_channels, kernel_size=1),
            torch.nn.GroupNorm(8 if out_channels >= 8 else 1, out_channels),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.shortcut = torch.nn.Identity() if in_channels == out_channels else torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        branches = [branch(inputs) for branch in self.branches]
        return self.fuse(torch.cat(branches, dim=1)) + self.shortcut(inputs)


class MultiScaleResNet(torch.nn.Module):
    def __init__(self, n_classes: int, n_blocks: int = 3, position_aware: bool = False) -> None:
        super().__init__()
        if n_blocks not in BLOCK_CHOICES:
            raise ValueError(f"n_blocks must be one of {BLOCK_CHOICES}")
        self.position_aware = position_aware
        self.pool_bins = 8 if position_aware else 1
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(2 if position_aware else 1, 32, kernel_size=3, padding=1),
            torch.nn.GroupNorm(8, 32),
            torch.nn.GELU(),
        )
        blocks: list[torch.nn.Module] = []
        stage_channels = (32, 64, 128)
        blocks_per_stage = n_blocks // 3
        in_channels = 32
        for stage_index, out_channels in enumerate(stage_channels):
            for _ in range(blocks_per_stage):
                blocks.append(MultiScaleResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if stage_index < len(stage_channels) - 1:
                blocks.append(torch.nn.MaxPool1d(2))
        self.blocks = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(self.pool_bins),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(128 * self.pool_bins, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stem_input = inputs.unsqueeze(1)
        if self.position_aware:
            coordinate = torch.linspace(-1.0, 1.0, inputs.shape[-1], device=inputs.device, dtype=inputs.dtype)
            coordinate = coordinate.view(1, 1, -1).expand(inputs.shape[0], -1, -1)
            stem_input = torch.cat((stem_input, coordinate), dim=1)
        return self.head(self.blocks(self.stem(stem_input)))


def _auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def _resnet_predict(model: torch.nn.Module, X: np.ndarray, device: str, batch_size: int = 512) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device=device, dtype=torch.float32)
            output.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.vstack(output)


def fit_resnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    n_blocks: int,
    position_aware: bool,
    seed: int,
    epochs: int,
    device: str,
) -> ResNetFit:
    set_seed(seed)
    model = MultiScaleResNet(n_classes=n_classes, n_blocks=n_blocks, position_aware=position_aware).to(device)
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = len(y_train) / np.maximum(n_classes * counts, 1.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.int64))
        ),
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = math.inf
    best_epoch = 1
    patience = 8 if epochs > 12 else max(2, epochs)
    stale = 0
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(batch_y)
            train_count += len(batch_y)
        model.eval()
        with torch.inference_mode():
            val_tensor = torch.from_numpy(X_val.astype(np.float32)).to(device=device)
            val_logits = model(val_tensor)
            val_loss = float(loss_fn(val_logits, torch.from_numpy(y_val).to(device=device)).cpu())
            val_prob = torch.softmax(val_logits, dim=1).cpu().numpy()
        train_prob = _resnet_predict(model, X_train, device, batch_size=1024)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(train_count, 1),
                "validation_loss": val_loss,
                "train_auc": _auc(y_train, train_prob),
                "validation_auc": _auc(y_val, val_prob),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_epoch": best_epoch,
            }
        )
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    for row in history:
        row["best_epoch"] = best_epoch
    return ResNetFit(model=model, history=tuple(history), best_epoch=best_epoch, device=device)


def fit_logistic(X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray, seed: int) -> tuple[BaseEstimator, float]:
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=seed)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"logistic__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups_train)
    return search.best_estimator_, float(search.best_params_["logistic__C"])
