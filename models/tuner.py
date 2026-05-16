# ============================================================
# models/tuner.py — Optuna Hyperparameter Optimization
# ============================================================

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from typing import Dict

from config import config
from models.lstm_models import SimpleLSTM, get_loss_fn
from utils.metrics import compute_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparamTuner:
    def __init__(self, data: Dict):
        self.data   = data
        self.cfg    = config.tuning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params: Dict = {}

    def _objective(self, trial: optuna.Trial) -> float:
        lr       = trial.suggest_float("lr",       *self.cfg.lr_range, log=True)
        hidden   = trial.suggest_int  ("hidden",   *self.cfg.hidden_range)
        dropout  = trial.suggest_float("dropout",  *self.cfg.dropout_range)
        batch_sz = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        n_layers = trial.suggest_int  ("n_layers", *self.cfg.layers_range)

        model = SimpleLSTM(
            input_size  = self.data["X_train"].shape[2],
            hidden_size = hidden,
            num_layers  = n_layers,
            output_size = self.data["y_train"].shape[1],
            dropout     = dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = get_loss_fn("MSELoss")

        X_tr = torch.tensor(self.data["X_train"], dtype=torch.float32)
        y_tr = torch.tensor(self.data["y_train"], dtype=torch.float32)
        X_v  = torch.tensor(self.data["X_val"],   dtype=torch.float32)
        y_v  = torch.tensor(self.data["y_val"],   dtype=torch.float32)

        tr_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_sz, shuffle=True)
        v_dl  = DataLoader(TensorDataset(X_v,  y_v),  batch_size=batch_sz)

        best_val = float("inf")
        for epoch in range(30):
            model.train()
            for X_b, y_b in tr_dl:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                pred, _ = model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in v_dl:
                    pred, _ = model(X_b.to(self.device))
                    val_loss += criterion(pred, y_b.to(self.device)).item() * len(X_b)
            val_loss /= len(v_dl.dataset)
            best_val  = min(best_val, val_loss)

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val

    def run(self) -> Dict:
        study = optuna.create_study(
            direction  = "minimize",
            sampler    = TPESampler(seed=42),
            pruner     = MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name = f"lstm_tuning_{self.data['ticker']}",
        )
        study.optimize(self._objective, n_trials=self.cfg.n_trials,
                       timeout=self.cfg.timeout, show_progress_bar=True)
        self.best_params = study.best_params
        logger.success(f"[Tuner] Best params: {self.best_params}")
        logger.success(f"[Tuner] Best value:  {study.best_value:.6f}")
        return self.best_params
    