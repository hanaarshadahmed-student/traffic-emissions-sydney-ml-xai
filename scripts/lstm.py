"""
LSTM regressor (ML model #6) -- a recurrent neural network that looks at a
short window of recent history instead of a single row.

How it fits the pipeline
------------------------
Models 1-5 see one row at a time: the features at time t (which already
include hand-made lag/rolling features). The LSTM sees the SAME feature
columns, but for the last `seq_len` time steps at that station
(t - seq_len + 1 ... t), and learns its own way of using that history.

  * Same inputs: the feature set's columns from data/processed/splits/,
    already scaled by 05_train_test_split.py (fit on train only).
  * Same targets, rows and scoring: one prediction per row of each split,
    in the split file's row order, scored by 06_run_models.py exactly like
    every other model.
  * Missing time steps (gaps in the data, or the start of a station's
    record) are filled with zeros and flagged by an extra "available"
    input channel -- the same idea as the `_available` flags on the lag
    features. Rows are never dropped.
  * A window may reach back into an earlier split (e.g. a validation row's
    history comes from the end of train). That's only past INPUTS, never
    future information, so it's how a real forecast would work.
  * Early stopping on the validation split (like XGBoost's early stopping),
    keeping the epoch with the lowest validation loss.
  * Training uses aq_quality_weight as a per-row loss weight, like the
    other models' sample_weight.

Settings live in config/config.yaml under training.models.lstm.params.
Requires PyTorch (`pip install torch`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - only hit without torch installed
    raise ImportError(
        "The LSTM model needs PyTorch. Install it with `pip install torch`, "
        "or set training.models.lstm.enabled: false in config/config.yaml."
    ) from error

TIME_STEP = {"daily": pd.Timedelta(days=1), "hourly": pd.Timedelta(hours=1)}


# ---------------------------------------------------------------------------
# Sequence windows
# ---------------------------------------------------------------------------

@dataclass
class SequenceData:
    """Every row of train/val/test stacked into one feature matrix, plus,
    for each row, the positions of its last `seq_len` time steps."""
    features: torch.Tensor          # (n_rows, n_features) float32
    windows: torch.Tensor           # (n_rows, seq_len) int64, -1 = step missing
    split_rows: dict[str, np.ndarray]  # split -> row positions, in split-file order


def build_sequences(
    frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    grain: str,
    seq_len: int,
) -> SequenceData:
    """frames: {"train": df, "val": df, "test": df} as read from the split
    CSVs (each needs station_id, timestamp and the feature columns)."""
    parts, split_rows, offset = [], {}, 0
    for split, frame in frames.items():
        parts.append(frame[["station_id", "timestamp", *feature_columns]])
        split_rows[split] = np.arange(offset, offset + len(frame))
        offset += len(frame)
    combined = pd.concat(parts, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    lookup = pd.MultiIndex.from_arrays([combined["station_id"].astype(str), combined["timestamp"]])
    if lookup.has_duplicates:
        raise ValueError("duplicate (station_id, timestamp) rows -- can't build sequences")

    step = TIME_STEP[grain]
    windows = np.full((len(combined), seq_len), -1, dtype=np.int64)
    for k in range(seq_len):
        # column seq_len-1 is the current row, column 0 the oldest step
        wanted = pd.MultiIndex.from_arrays(
            [combined["station_id"].astype(str), combined["timestamp"] - k * step]
        )
        windows[:, seq_len - 1 - k] = lookup.get_indexer(wanted)

    features = combined[feature_columns].to_numpy(dtype=np.float32)
    return SequenceData(
        features=torch.from_numpy(features),
        windows=torch.from_numpy(windows),
        split_rows=split_rows,
    )


def _gather(data: SequenceData, rows: np.ndarray) -> torch.Tensor:
    """(batch, seq_len, n_features + 1): features per step plus an
    'available' channel; missing steps are all zeros."""
    window = data.windows[torch.as_tensor(rows)]
    available = (window >= 0).unsqueeze(-1).float()
    steps = data.features[window.clamp(min=0)] * available
    return torch.cat([steps, available], dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _Network(nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            n_inputs, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)  # last step's hidden state


class LSTMRegressor:
    """Trained by 06_run_models.py through fit_sequences()/predict_sequences()
    rather than the tabular fit()/predict() models 1-5 use."""

    needs_sequences = True

    def __init__(
        self,
        seq_len: int | dict = 24,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        max_epochs: int = 30,
        patience: int = 5,
        random_state: int = 42,
        num_threads: int | None = None,
        verbose: bool = True,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state
        self.num_threads = num_threads
        self.verbose = verbose

    def window_length(self, grain: str) -> int:
        """seq_len may be one number or {daily: 14, hourly: 24}."""
        if isinstance(self.seq_len, dict):
            return int(self.seq_len[grain])
        return int(self.seq_len)

    def _predict_scaled(self, data: SequenceData, rows: np.ndarray) -> np.ndarray:
        self.network_.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(rows), 4096):
                out.append(self.network_(_gather(data, rows[start:start + 4096])).numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def fit_sequences(
        self,
        data: SequenceData,
        y_train: pd.Series,
        sample_weight: pd.Series | None,
        y_val: pd.Series,
        val_split: str = "val",
    ) -> "LSTMRegressor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if self.num_threads:
            torch.set_num_threads(int(self.num_threads))

        # standardise the target with TRAIN statistics (undone in predict)
        self.y_mean_ = float(np.mean(y_train))
        self.y_std_ = float(np.std(y_train)) or 1.0
        y_tr = torch.tensor(((np.asarray(y_train) - self.y_mean_) / self.y_std_), dtype=torch.float32)
        y_va = (np.asarray(y_val, dtype=float) - self.y_mean_) / self.y_std_
        weights = torch.tensor(
            np.ones(len(y_tr)) if sample_weight is None else np.asarray(sample_weight), dtype=torch.float32
        )

        train_rows = data.split_rows["train"]
        val_rows = data.split_rows[val_split]
        self.network_ = _Network(data.features.shape[1] + 1, self.hidden_size, self.num_layers, self.dropout)
        optimiser = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        generator = torch.Generator().manual_seed(self.random_state)

        best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
        self.history_ = []
        for epoch in range(1, self.max_epochs + 1):
            self.network_.train()
            order = torch.randperm(len(train_rows), generator=generator).numpy()
            total = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                prediction = self.network_(_gather(data, train_rows[batch]))
                loss = (weights[batch] * (prediction - y_tr[batch]) ** 2).sum() / weights[batch].sum()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += loss.item() * len(batch)

            val_loss = float(np.mean((self._predict_scaled(data, val_rows) - y_va) ** 2))
            self.history_.append({"epoch": epoch, "train_loss": total / len(order), "val_loss": val_loss})
            if self.verbose:
                print(f"      epoch {epoch:2d}  train_loss={total / len(order):.4f}  "
                      f"val_loss={val_loss:.4f}", flush=True)
            if val_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(self.network_.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self.network_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        return self

    def predict_sequences(self, data: SequenceData, split: str) -> np.ndarray:
        scaled = self._predict_scaled(data, data.split_rows[split])
        return scaled * self.y_std_ + self.y_mean_
