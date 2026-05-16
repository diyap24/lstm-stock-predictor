# ============================================================
# app.py — Flask backend for the dashboard
# ============================================================

import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from loguru import logger

from config import config


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = config.dashboard.secret_key
    CORS(app)

    # ── Pages ─────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("dashboard.html")

    # ── Quote ─────────────────────────────────────────────────
    @app.route("/api/quote/<ticker>")
    def get_quote(ticker: str):
        try:
            t   = yf.Ticker(ticker.upper())
            hi  = t.history(period="2d", interval="1d")
            if hi.empty:
                return jsonify({"error": "No data"}), 404
            last = hi.iloc[-1]
            prev = hi.iloc[-2] if len(hi) > 1 else last
            chg  = float(last["Close"] - prev["Close"])
            pct  = chg / float(prev["Close"]) * 100
            info = t.info
            return jsonify({
                "ticker":   ticker.upper(),
                "price":    round(float(last["Close"]), 2),
                "change":   round(chg, 2),
                "pct":      round(pct, 2),
                "volume":   int(last["Volume"]),
                "high":     round(float(last["High"]), 2),
                "low":      round(float(last["Low"]),  2),
                "open":     round(float(last["Open"]), 2),
                "name":     info.get("longName", ticker.upper()),
                "sector":   info.get("sector", "Equity"),
                "mktcap":   info.get("marketCap", 0),
                "pe":       info.get("trailingPE", None),
                "52wHigh":  info.get("fiftyTwoWeekHigh", None),
                "52wLow":   info.get("fiftyTwoWeekLow",  None),
            })
        except Exception as e:
            logger.error(f"Quote error [{ticker}]: {e}")
            return jsonify({"error": str(e)}), 500

    # ── Price History ─────────────────────────────────────────
    @app.route("/api/history/<ticker>")
    def get_history(ticker: str):
        period = request.args.get("period", "1y")
        try:
            df = yf.Ticker(ticker.upper()).history(period=period)
            if df.empty:
                return jsonify({"error": "No data"}), 404
            return jsonify({
                "dates":  [str(d.date()) for d in df.index],
                "open":   [round(v, 2)   for v in df["Open"]],
                "high":   [round(v, 2)   for v in df["High"]],
                "low":    [round(v, 2)   for v in df["Low"]],
                "close":  [round(v, 2)   for v in df["Close"]],
                "volume": [int(v)         for v in df["Volume"]],
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── LSTM Prediction ───────────────────────────────────────
    @app.route("/api/predict/<ticker>")
    def get_prediction(ticker: str):
        """
        Try to use the trained model. Falls back to simulation if no checkpoint.
        """
        ckpt_path = config.model.checkpoint_dir / f"{ticker.upper()}_best.pt"

        if ckpt_path.exists():
            try:
                from utils.data_pipeline import StockDataPipeline
                from models.trainer import Trainer

                pipe    = StockDataPipeline(ticker.upper())
                data    = pipe.run()
                trainer = Trainer(data)
                trainer.load_best()

                X_last     = data["X_test"][-1:]
                base_price = float(data["close_prices"][-1])
                preds      = []
                X_input    = X_last.copy()

                for _ in range(7):
                    log_ret, _ = trainer.predict(X_input, base_price=base_price)
                    next_p     = float(log_ret[0])
                    preds.append(round(next_p, 2))
                    base_price = next_p
                    X_input    = np.roll(X_input, -1, axis=1)

                dates = [
                    (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d")
                    for i in range(7)
                ]
                current = float(data["close_prices"][-1])
                conf    = min(0.95, 0.70 + (data["X_test"].shape[0] / 5000))

                return jsonify({
                    "dates":       dates,
                    "predictions": preds,
                    "current":     round(current, 2),
                    "confidence":  round(conf, 3),
                    "model":       "LSTM",
                })
            except Exception as e:
                logger.error(f"Model predict error [{ticker}]: {e}")
                # Fall through to simulation

        # Simulation fallback
        try:
            df   = yf.Ticker(ticker.upper()).history(period="3mo")
            last = float(df["Close"].iloc[-1]) if not df.empty else 150.0
        except:
            last = 150.0

        np.random.seed(abs(hash(ticker)) % 2**31)
        preds = []
        p     = last
        for _ in range(7):
            p = p * (1 + np.random.normal(0.0003, 0.008))
            preds.append(round(p, 2))

        dates = [
            (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d")
            for i in range(7)
        ]
        return jsonify({
            "dates":       dates,
            "predictions": preds,
            "current":     round(last, 2),
            "confidence":  0.78,
            "model":       "simulation",
        })

    # ── Model Metrics ─────────────────────────────────────────
    @app.route("/api/metrics/<ticker>")
    def get_metrics(ticker: str):
        """Return saved test metrics from training history JSON."""
        hist_path = config.model.checkpoint_dir / f"{ticker.upper()}_history.json"
        if not hist_path.exists():
            return jsonify({}), 404
        try:
            with open(hist_path) as f:
                hist = json.load(f)
            metrics = hist.get("test_metrics", {})
            return jsonify(metrics)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── Training History (loss curves) ────────────────────────
    @app.route("/api/training/<ticker>")
    def get_training(ticker: str):
        hist_path = config.model.checkpoint_dir / f"{ticker.upper()}_history.json"
        if not hist_path.exists():
            return jsonify({}), 404
        try:
            with open(hist_path) as f:
                hist = json.load(f)
            return jsonify({
                "train_loss": hist.get("train_loss", []),
                "val_loss":   hist.get("val_loss",   []),
                "lr":         hist.get("lr",          []),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── Watchlist ─────────────────────────────────────────────
    @app.route("/api/watchlist")
    def get_watchlist():
        return jsonify({"tickers": config.data.tickers[:6]})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=config.dashboard.port, debug=True)

# Render needs this
import os
if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)