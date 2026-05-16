# ============================================================
# utils/data_pipeline.py — Final version with DA-boosting features
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List, Dict
from sklearn.preprocessing import MinMaxScaler
import ta
from loguru import logger
import joblib

from config import config


class StockDataPipeline:
    def __init__(self, ticker: str):
        self.ticker         = ticker.upper()
        self.cfg            = config.data
        self.feature_scaler = None
        self.target_scaler  = None
        self.feature_names: List[str] = []
        self._close_prices  = None

    def download(self) -> pd.DataFrame:
        cache = self.cfg.raw_data_dir / f"{self.ticker}.parquet"
        if cache.exists():
            logger.info(f"[{self.ticker}] Loading cache")
            return pd.read_parquet(cache)
        logger.info(f"[{self.ticker}] Downloading...")
        df = yf.Ticker(self.ticker).history(
            start=self.cfg.start_date, end=self.cfg.end_date,
            interval=self.cfg.interval, auto_adjust=True,
        )
        df.index = pd.to_datetime(df.index)
        if df.empty:
            raise ValueError(f"No data for {self.ticker}")
        df.to_parquet(cache)
        logger.success(f"[{self.ticker}] {len(df)} rows")
        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df    = df.copy()
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        # ── Price returns ─────────────────────────────────────
        df["Returns"]      = close.pct_change()
        df["Log_Returns"]  = np.log(close / close.shift(1))
        df["HL_pct"]       = (high - low) / close
        df["OC_pct"]       = (close - df["Open"]) / df["Open"]

        # ── MA ratios ─────────────────────────────────────────
        df["SMA5_ratio"]   = ta.trend.sma_indicator(close, 5)  / close
        df["SMA20_ratio"]  = ta.trend.sma_indicator(close, 20) / close
        df["SMA50_ratio"]  = ta.trend.sma_indicator(close, 50) / close
        df["EMA12_ratio"]  = ta.trend.ema_indicator(close, 12) / close

        # ── Momentum ──────────────────────────────────────────
        df["RSI"]          = ta.momentum.rsi(close, 14) / 100.0
        df["MACD_diff"]    = ta.trend.macd_diff(close)
        df["ROC_5"]        = ta.momentum.roc(close, 5) / 100.0

        # ── Volatility ────────────────────────────────────────
        bb               = ta.volatility.BollingerBands(close, 20)
        df["BB_pct"]     = bb.bollinger_pband()
        df["ATR_ratio"]  = ta.volatility.average_true_range(high, low, close, 14) / close

        # ── Volume ────────────────────────────────────────────
        df["Vol_ratio"]  = vol / (vol.rolling(20).mean() + 1)

        # ── DA-boosting: lagged direction labels ──────────────
        # Give model explicit signal of recent directions
        df["Dir_1"]  = np.sign(df["Log_Returns"].shift(1))   # yesterday
        df["Dir_2"]  = np.sign(df["Log_Returns"].shift(2))   # 2 days ago
        df["Dir_3"]  = np.sign(df["Log_Returns"].shift(3))   # 3 days ago
        df["Dir_5"]  = np.sign(df["Log_Returns"].shift(5))   # 5 days ago (week)

        # Streak: how many consecutive same-direction days (normalised)
        ret_sign = np.sign(df["Log_Returns"])
        streak   = ret_sign.groupby(
            (ret_sign != ret_sign.shift()).cumsum()
        ).cumcount() + 1
        df["Streak"] = streak * ret_sign / 10.0   # sign tells direction, magnitude tells length

        # Close position in week's range
        df["Week_pos"] = (close - close.rolling(5).min()) / (
            close.rolling(5).max() - close.rolling(5).min() + 1e-9
        )

        keep = [
            "Close",
            "Returns", "Log_Returns", "HL_pct", "OC_pct",
            "SMA5_ratio", "SMA20_ratio", "SMA50_ratio", "EMA12_ratio",
            "RSI", "MACD_diff", "ROC_5",
            "BB_pct", "ATR_ratio", "Vol_ratio",
            "Dir_1", "Dir_2", "Dir_3", "Dir_5",
            "Streak", "Week_pos",
        ]
        df = df[keep].replace([np.inf, -np.inf], np.nan).dropna()
        logger.info(f"[{self.ticker}] Features: {len(df.columns)} | Rows: {len(df)}")
        return df

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = self.cfg.sequence_length
        horizon = self.cfg.prediction_horizon
        self.feature_names  = list(df.columns)
        self._close_prices  = df["Close"].values

        self.feature_scaler = MinMaxScaler((0, 1))
        scaled = self.feature_scaler.fit_transform(df.values)

        self.target_scaler  = MinMaxScaler((0, 1))
        log_rets = df[["Log_Returns"]].values
        scaled_returns = self.target_scaler.fit_transform(log_rets).ravel()

        X, y = [], []
        for i in range(seq_len, len(scaled) - horizon + 1):
            X.append(scaled[i - seq_len: i])
            y.append(scaled_returns[i: i + horizon])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    @staticmethod
    def split(X, y, train=0.80, val=0.10):
        n = len(X); t = int(n * train); v = t + int(n * val)
        return (X[:t], y[:t]), (X[t:v], y[t:v]), (X[v:], y[v:])

    def run(self) -> Dict:
        raw  = self.download()
        feat = self.build_features(raw)
        X, y = self.create_sequences(feat)
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = self.split(X, y)

        proc = self.cfg.processed_data_dir
        joblib.dump(self.feature_scaler, proc / f"{self.ticker}_feat_scaler.joblib")
        joblib.dump(self.target_scaler,  proc / f"{self.ticker}_tgt_scaler.joblib")
        np.save(proc / f"{self.ticker}_close.npy", self._close_prices)

        logger.info(
            f"[{self.ticker}] Train:{X_tr.shape} Val:{X_v.shape} "
            f"Test:{X_te.shape} | Features:{X_tr.shape[2]}"
        )
        return {
            "ticker":       self.ticker,
            "X_train":      X_tr, "y_train": y_tr,
            "X_val":        X_v,  "y_val":   y_v,
            "X_test":       X_te, "y_test":  y_te,
            "scaler":       self.target_scaler,
            "features":     self.feature_names,
            "close_prices": self._close_prices,
        }

    def inverse_price(self, arr: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(
            arr.reshape(-1, 1)
        ).ravel()

    def inverse_transform_price(self, log_returns: np.ndarray) -> np.ndarray:
        """Convert log returns to absolute prices using last known close."""
        base = float(self._close_prices[-1])
        return base * np.exp(np.cumsum(np.asarray(log_returns, dtype=np.float64)))