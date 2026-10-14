"""
Sequence models: LSTM and GRU, which look at a window of recent time steps at each station instead of a single row, and are scored the same way as every other model. 
Settings live under training.models.lstm / gru in config.yaml; needs PyTorch.
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
    ) from error

TIME_STEP = {"daily": pd.Timedelta(days=1), "hourly": pd.Timedelta(hours=1)}


# ---------------------------------------------------------------------------
# Sequence windows
# ---------------------------------------------------------------------------

@dataclass
class SequenceData:
    features: torch.Tensor          
    windows: torch.Tensor           
    split_rows: dict[str, np.ndarray]  


def build_sequences(
    frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    grain: str,
    seq_len: int,
) -> SequenceData:
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
    window = data.windows[torch.as_tensor(rows)]
    available = (window >= 0).unsqueeze(-1).float()
    steps = data.features[window.clamp(min=0)] * available
    return torch.cat([steps, available], dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMNetwork(nn.Module):
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

        
        self.y_mean_ = float(np.mean(y_train))
        self.y_std_ = float(np.std(y_train)) or 1.0
        y_tr = torch.tensor(((np.asarray(y_train) - self.y_mean_) / self.y_std_), dtype=torch.float32)
        y_va = (np.asarray(y_val, dtype=float) - self.y_mean_) / self.y_std_
        weights = torch.tensor(
            np.ones(len(y_tr)) if sample_weight is None else np.asarray(sample_weight), dtype=torch.float32
        )

        train_rows = data.split_rows["train"]
        val_rows = data.split_rows[val_split]
        self.network_ = LSTMNetwork(data.features.shape[1] + 1, self.hidden_size, self.num_layers, self.dropout)
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


# ---------------------------------------------------------------------------
# GRU -- same windows and training procedure, lighter recurrent cell
# ---------------------------------------------------------------------------

class GRUNetwork(nn.Module):
    def __init__(self, n_inputs: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            n_inputs,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class GRURegressor:
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
        gradient_clip: float | None = 1.0,
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
        self.gradient_clip = gradient_clip
        self.random_state = random_state
        self.num_threads = num_threads
        self.verbose = verbose

    def window_length(self, grain: str) -> int:
        if isinstance(self.seq_len, dict):
            return int(self.seq_len[grain])
        return int(self.seq_len)

    def _predict_scaled(self, data: SequenceData, rows: np.ndarray) -> np.ndarray:
        self.network_.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(rows), 4096):
                batch = _gather(data, rows[start:start + 4096])
                predictions.append(self.network_(batch).numpy())
        return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)

    def fit_sequences(
        self,
        data: SequenceData,
        y_train: pd.Series,
        sample_weight: pd.Series | None,
        y_val: pd.Series,
        val_split: str = "val",
    ) -> "GRURegressor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if self.num_threads:
            torch.set_num_threads(int(self.num_threads))

        self.y_mean_ = float(np.mean(y_train))
        self.y_std_ = float(np.std(y_train)) or 1.0
        y_train_scaled = torch.tensor(
            (np.asarray(y_train) - self.y_mean_) / self.y_std_, dtype=torch.float32
        )
        y_val_scaled = (np.asarray(y_val, dtype=float) - self.y_mean_) / self.y_std_
        weights = torch.tensor(
            np.ones(len(y_train_scaled)) if sample_weight is None else np.asarray(sample_weight),
            dtype=torch.float32,
        )

        train_rows = data.split_rows["train"]
        val_rows = data.split_rows[val_split]
        self.network_ = GRUNetwork(
            data.features.shape[1] + 1,
            self.hidden_size,
            self.num_layers,
            self.dropout,
        )
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        generator = torch.Generator().manual_seed(self.random_state)

        best_loss = np.inf
        best_state = None
        best_epoch = 0
        stale_epochs = 0
        self.history_ = []

        for epoch in range(1, self.max_epochs + 1):
            self.network_.train()
            order = torch.randperm(len(train_rows), generator=generator).numpy()
            weighted_loss = 0.0

            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                predictions = self.network_(_gather(data, train_rows[batch]))
                loss = (
                    weights[batch] * (predictions - y_train_scaled[batch]) ** 2
                ).sum() / weights[batch].sum()
                optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(self.network_.parameters(), self.gradient_clip)
                optimizer.step()
                weighted_loss += loss.item() * len(batch)

            train_loss = weighted_loss / len(order)
            val_loss = float(
                np.mean((self._predict_scaled(data, val_rows) - y_val_scaled) ** 2)
            )
            self.history_.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            if self.verbose:
                print(
                    f"      epoch {epoch:2d}  train_loss={train_loss:.4f}  "
                    f"val_loss={val_loss:.4f}",
                    flush=True,
                )

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_epoch = epoch
                stale_epochs = 0
                best_state = copy.deepcopy(self.network_.state_dict())
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.network_.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        return self

    def predict_sequences(self, data: SequenceData, split: str) -> np.ndarray:
        predictions = self._predict_scaled(data, data.split_rows[split])
        return predictions * self.y_std_ + self.y_mean_
