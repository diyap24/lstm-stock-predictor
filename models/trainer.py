# ============================================================
# models/trainer.py — Fixed return-based evaluation
# ============================================================

import time
import json
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from tqdm import tqdm

from config import config
from models.lstm_models import build_model, get_loss_fn
from utils.metrics import compute_metrics


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = float("inf")
        self.counter    = 0
        self.best_epoch = 0
        self.triggered  = False

    def step(self, val: float, epoch: int) -> bool:
        if val < self.best - self.min_delta:
            self.best = val; self.counter = 0; self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                logger.warning(
                    f"Early stopping @ epoch {epoch} "
                    f"(best={self.best:.6f} @ {self.best_epoch})"
                )
        return self.triggered


class Trainer:
    def __init__(self, data: Dict, model_type: Optional[str] = None,
                 experiment_name: str = "stock_lstm"):
        self.data          = data
        self.cfg_d         = config.data
        self.cfg_m         = config.model
        self.cfg_t         = config.training
        self.ticker        = data["ticker"]
        self.target_scaler = data["scaler"]
        self.close_prices  = data.get("close_prices", None)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"[Trainer] Device: {self.device}")

        self.cfg_m.input_size  = data["X_train"].shape[2]
        self.cfg_m.output_size = data["y_train"].shape[1]
        self.model = build_model(model_type).to(self.device)

        self.criterion     = get_loss_fn(self.cfg_t.loss_fn)
        self.optimizer     = self._build_optimizer()
        self.scheduler     = self._build_scheduler()
        self.history       = {"train_loss": [], "val_loss": [], "lr": []}
        self.best_val_loss = float("inf")
        run_dir = self.cfg_t.runs_dir / f"{self.ticker}_{model_type or self.cfg_m.model_type}"
        self.writer = SummaryWriter(log_dir=str(run_dir))

    def _build_optimizer(self):
        lr, wd = self.cfg_t.learning_rate, self.cfg_t.weight_decay
        if self.cfg_t.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

    def _build_scheduler(self):
        if self.cfg_t.scheduler == "ReduceLROnPlateau":
            return ReduceLROnPlateau(self.optimizer,
                                     patience=self.cfg_t.scheduler_patience,
                                     factor=self.cfg_t.scheduler_factor,
                                     min_lr=1e-7)
        return StepLR(self.optimizer, step_size=50, gamma=0.5)

    def _loader(self, X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=self.cfg_t.batch_size,
                          shuffle=shuffle, num_workers=0)

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(self.device), y_b.to(self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                pred, _ = self.model(X_b)
                loss     = self.criterion(pred, y_b)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg_t.grad_clip)
                self.optimizer.step()
            total += loss.item() * len(X_b)
        return total / len(loader.dataset)

    def _save(self, epoch, val_loss):
        path = self.cfg_m.checkpoint_dir / f"{self.ticker}_best.pt"
        torch.save({"epoch": epoch, "model_state": self.model.state_dict(),
                    "val_loss": val_loss}, path)
        logger.success(f"[{self.ticker}] Checkpoint saved (val={val_loss:.6f})")

    def load_best(self):
        path = self.cfg_m.checkpoint_dir / f"{self.ticker}_best.pt"
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        logger.info(f"[{self.ticker}] Loaded best (epoch {ckpt['epoch']})")

    def train(self) -> Dict:
        tr_dl  = self._loader(self.data["X_train"], self.data["y_train"], shuffle=True)
        val_dl = self._loader(self.data["X_val"],   self.data["y_val"])
        te_dl  = self._loader(self.data["X_test"],  self.data["y_test"])
        stopper = EarlyStopping(self.cfg_t.patience, self.cfg_t.min_delta)

        pbar = tqdm(range(1, self.cfg_t.epochs + 1), desc=f"[{self.ticker}] Training")
        for epoch in pbar:
            tr  = self._run_epoch(tr_dl,  train=True)
            val = self._run_epoch(val_dl, train=False)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val)
            else:
                self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(tr)
            self.history["val_loss"].append(val)
            self.history["lr"].append(lr)
            self.writer.add_scalars("Loss", {"train": tr, "val": val}, epoch)
            self.writer.add_scalar("LR", lr, epoch)

            if val < self.best_val_loss:
                self.best_val_loss = val
                self._save(epoch, val)

            pbar.set_postfix(tr=f"{tr:.5f}", val=f"{val:.5f}", lr=f"{lr:.2e}")

            if self.cfg_t.early_stopping and stopper.step(val, epoch):
                break

        self.load_best()
        metrics = self.evaluate(te_dl)
        self.history["test_metrics"] = metrics
        logger.success(f"[{self.ticker}] Test metrics: {metrics}")

        for k, v in metrics.items():
            self.writer.add_scalar(f"Test/{k}", v, 0)
        self.writer.flush()
        self.writer.close()

        with open(self.cfg_m.checkpoint_dir / f"{self.ticker}_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    @torch.no_grad()
    def evaluate(self, loader) -> Dict:
        """
        Evaluate on real price scale.
        Model predicts scaled log-returns for each sample independently.
        Each prediction is converted: price = prev_close * exp(log_return)
        This avoids cumsum error compounding.
        """
        self.model.eval()
        preds, targets = [], []
        for X_b, y_b in loader:
            pred, _ = self.model(X_b.to(self.device))
            preds.append(pred.cpu().numpy())
            targets.append(y_b.numpy())

        preds   = np.concatenate(preds).reshape(-1, 1)   # scaled returns
        targets = np.concatenate(targets).reshape(-1, 1)

        # Inverse-transform: scaled return → log return
        pred_lr   = self.target_scaler.inverse_transform(preds).ravel()
        target_lr = self.target_scaler.inverse_transform(targets).ravel()

        # Convert each log-return independently using its own base price
        # base price for sample i = close_prices[train+val+seq_len + i - 1]
        n_test = len(pred_lr)
        if self.close_prices is not None:
            # The test set starts at index: total - n_test in close_prices
            start_idx = len(self.close_prices) - n_test - 1
            base_prices = self.close_prices[start_idx: start_idx + n_test]
            # Each prediction: tomorrow's price = today_close * exp(log_return)
            pred_prices   = base_prices * np.exp(pred_lr)
            target_prices = base_prices * np.exp(target_lr)
            return compute_metrics(target_prices, pred_prices)
        else:
            return compute_metrics(target_lr, pred_lr)

    @torch.no_grad()
    def predict(self, X: np.ndarray, base_price: float = None):
        self.model.eval()
        pred, attn = self.model(torch.tensor(X, dtype=torch.float32).to(self.device))
        scaled_ret = pred.cpu().numpy().reshape(-1, 1)
        log_ret    = self.target_scaler.inverse_transform(scaled_ret).ravel()
        if base_price:
            prices = base_price * np.exp(log_ret)
            return prices, (attn.cpu().numpy() if attn is not None else None)
        return log_ret, (attn.cpu().numpy() if attn is not None else None)