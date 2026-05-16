# ============================================================
# utils/metrics.py — Evaluation Metrics
# ============================================================

import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    if len(y_true) > 1:
        da = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
    else:
        da = float("nan")

    naive   = y_true[:-1]
    u_num   = np.sqrt(np.mean((y_pred[1:] - y_true[1:]) ** 2))
    u_den   = np.sqrt(np.mean((naive       - y_true[1:]) ** 2)) + 1e-12
    theil_u = u_num / u_den

    return {
        "MSE":     round(float(mse),     4),
        "RMSE":    round(float(rmse),    4),
        "MAE":     round(float(mae),     4),
        "MAPE":    round(float(mape),    4),
        "R2":      round(float(r2),      4),
        "DA":      round(float(da),      2),
        "Theil_U": round(float(theil_u), 4),
    }