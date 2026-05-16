# ============================================================
# models/lstm_models.py
# ============================================================

import torch
import torch.nn as nn
from typing import List, Optional
from loguru import logger
from config import config


# ── Models ────────────────────────────────────────────────────

class SimpleLSTM(nn.Module):
    """Proven base: LSTM stack + dropout + linear."""
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 output_size=1, dropout=0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])), None


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 output_size=1, dropout=0.2, **kw):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])), None


class LSTMTransformer(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1,
                 dropout=0.2, **kw):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2,
                            batch_first=True, dropout=dropout)
        enc = nn.TransformerEncoderLayer(hidden_size, nhead=4,
                                          dim_feedforward=128,
                                          dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = self.transformer(out)
        return self.fc(out[:, -1, :]), None


# ── Factory ───────────────────────────────────────────────────

def build_model(model_type: Optional[str] = None) -> nn.Module:
    cfg   = config.model
    mtype = model_type or cfg.model_type

    kwargs = dict(input_size=cfg.input_size, hidden_size=cfg.hidden_sizes[0],
                  num_layers=len(cfg.hidden_sizes), output_size=cfg.output_size,
                  dropout=cfg.dropout)

    if mtype in ("StackedLSTM", "SimpleLSTM"):
        model = SimpleLSTM(**kwargs)
    elif mtype == "BiLSTM":
        model = BiLSTM(**kwargs)
    elif mtype == "Transformer":
        model = LSTMTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model: {mtype}")

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[Model] {mtype} | Params: {n:,}")
    return model


# ── Loss Functions ────────────────────────────────────────────

class DirectionalLoss(nn.Module):
    """
    MSE + directional penalty.
    Penalises predictions that move in the wrong direction vs actual.
    Directly improves Directional Accuracy (DA) metric.
    alpha: weight of direction penalty (0.3 = 30% direction, 70% MSE)
    """
    def __init__(self, alpha: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)

        if pred.size(0) < 2:
            return mse_loss

        # Direction: sign of change from previous prediction/target
        pred_dir   = pred[1:] - pred[:-1]
        actual_dir = target[1:] - target[:-1]

        # Wrong direction = pred_dir and actual_dir have opposite signs
        wrong = torch.clamp(-pred_dir * actual_dir, min=0)
        dir_loss = wrong.mean()

        return mse_loss + self.alpha * dir_loss


def get_loss_fn(name: str) -> nn.Module:
    fns = {
        "MSELoss":        nn.MSELoss(),
        "L1Loss":         nn.L1Loss(),
        "HuberLoss":      nn.HuberLoss(delta=1.0),
        "DirectionalLoss":DirectionalLoss(alpha=0.5),
    }
    if name not in fns:
        logger.warning(f"Unknown loss '{name}', using MSELoss")
        return nn.MSELoss()
    return fns[name]
