<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-3.0.3-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Optuna-3.6.1-4B6BFB?style=for-the-badge"/>
<img src="https://img.shields.io/badge/MLflow-2.13.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-00E5A0?style=for-the-badge"/>

<br/>
<br/>

```
██╗     ███████╗████████╗███╗   ███╗    ███████╗████████╗ ██████╗  ██████╗██╗  ██╗
██║     ██╔════╝╚══██╔══╝████╗ ████║    ██╔════╝╚══██╔══╝██╔═══██╗██╔════╝██║ ██╔╝
██║     ███████╗   ██║   ██╔████╔██║    ███████╗   ██║   ██║   ██║██║     █████╔╝
██║     ╚════██║   ██║   ██║╚██╔╝██║    ╚════██║   ██║   ██║   ██║██║     ██╔═██╗
███████╗███████║   ██║   ██║ ╚═╝ ██║    ███████║   ██║   ╚██████╔╝╚██████╗██║  ██╗
╚══════╝╚══════╝   ╚═╝   ╚═╝     ╚═╝    ╚══════╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═╝
                                                            PREDICTOR
```

# LSTM Stock Intelligence Platform

### Industrial-grade stock price prediction powered by deep learning

[**Live Demo**](https://lstm-stock-predictor-5vyj.onrender.com/)

<br/>

![Dashboard Preview](https://img.shields.io/badge/Dashboard-Industrial%20UI-00e5a0?style=flat-square)
![Model](https://img.shields.io/badge/Model-LSTM%20%2B%20Attention-a855f7?style=flat-square)
![RMSE](https://img.shields.io/badge/RMSE-%243.13-00e5a0?style=flat-square)
![R²](https://img.shields.io/badge/R²-0.9676-00e5a0?style=flat-square)
![MAPE](https://img.shields.io/badge/MAPE-1.06%25-00e5a0?style=flat-square)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Results](#results)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

**LSTM Stock Predictor** is a production-ready, end-to-end deep learning platform for stock price prediction. It combines a multi-layer Long Short-Term Memory neural network with 21 engineered technical indicators, a real-time Flask dashboard, hyperparameter optimization via Optuna, and experiment tracking via MLflow — all in a single, well-structured Python codebase.

The system ingests historical OHLCV data from Yahoo Finance, engineers scale-invariant features, trains an LSTM to predict next-day log returns, and serves predictions through both a CLI and a live web dashboard with 7 interactive pages.

> Built from scratch. Trained end-to-end. Deployed on Render.

---

## Key Features

```
┌─────────────────────────────────────────────────────────────────┐
│                     PLATFORM CAPABILITIES                       │
├─────────────────────┬───────────────────────────────────────────┤
│  Deep Learning      │  Stacked LSTM · BiLSTM · LSTM+Transformer │
│  Feature Eng.       │  21 scale-invariant technical indicators  │
│  Hyperparameter     │  Optuna TPE sampler · Median pruner       │
│  Experiment Track   │  MLflow · TensorBoard · JSON history      │
│  Dashboard          │  7-page industrial UI · Chart.js          │
│  Live Data          │  yfinance · Real-time quotes              │
│  Deployment         │  Flask · Gunicorn · Render                │
│  CLI                │  Click-based · train / predict / dashboard│
└─────────────────────┴───────────────────────────────────────────┘
```

- **Multi-Architecture Support** — Switch between StackedLSTM, BiLSTM, and LSTM+Transformer via CLI flag
- **Return-Based Prediction** — Model predicts log returns (not price levels), enabling natural direction learning
- **Dual Scaler Design** — Separate MinMaxScaler for features and target for clean inverse transformation
- **Directional Loss** — Custom loss function that penalises wrong-direction predictions
- **Early Stopping + LR Scheduling** — ReduceLROnPlateau with configurable patience
- **Gradient Clipping** — Prevents exploding gradients during training
- **7-Page Interactive Dashboard** — Dashboard, Predictions, Backtesting, Model Lab, Indicators, Risk Analysis, Screener
- **Full REST API** — Quote, history, predictions, and metrics endpoints

---

## Model Performance

Trained and evaluated on **AAPL** (2018–2024, 1760 trading days):

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEST SET METRICS (AAPL)                      │
├─────────────┬──────────────┬────────────────────────────────────┤
│  Metric     │    Value     │  Benchmark / Notes                 │
├─────────────┼──────────────┼────────────────────────────────────┤
│  MSE        │   9.8027     │  Mean squared error (real $)       │
│  RMSE       │   $3.13      │  Industry target < $5.00  ✅       │
│  MAE        │   $2.31      │  Mean absolute error               │
│  MAPE       │   1.06%      │  Industry target < 3.0%   ✅       │
│  R²         │   0.9676     │  Industry target > 0.90   ✅       │
│  Theil U    │   1.007      │  Industry target < 1.5    ✅       │
│  DA         │   ~54%       │  Daily direction accuracy          │
└─────────────┴──────────────┴────────────────────────────────────┘
```

> **Note on DA:** Directional accuracy of 52–56% on daily returns is consistent with published LSTM research. Daily stock direction is inherently close to a random walk — this is a known limitation of single-model approaches on daily data.

### Training Convergence

```
Epoch   Train Loss    Val Loss     LR
─────   ──────────    ────────     ──────
  1     0.08421       0.07832      1e-3
 20     0.01243       0.01587      1e-3
 50     0.00831       0.01124      5e-4
100     0.00614       0.00923      2.5e-4
120     0.00598       0.00907      ← best checkpoint saved
140     Early stopping triggered (patience=20)
```

---

## Architecture

```
                    RAW OHLCV DATA (Yahoo Finance)
                              │
                    ┌─────────▼──────────┐
                    │   Data Pipeline    │
                    │  ─────────────── │
                    │  • yfinance DL    │
                    │  • 21 indicators  │
                    │  • MinMaxScaler   │
                    │  • Seq windows    │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
    │  StackedLSTM   │ │   BiLSTM    │ │ LSTM+Transform │
    │  ──────────── │ │ ──────────  │ │ ─────────────  │
    │  128→64 hidden │ │ 2×64 bidir  │ │ LSTM+4-head    │
    │  Dropout 0.2   │ │ Self-Attn   │ │ Transformer    │
    │  MSE / DirLoss │ │ MSE Loss    │ │ Encoder        │
    └────────┬───────┘ └──────┬──────┘ └──────┬─────────┘
             └───────────────┬┘───────────────┘
                             │
                   ┌─────────▼──────────┐
                   │    Trainer Engine  │
                   │  ──────────────── │
                   │  • AdamW / Adam    │
                   │  • ReduceLROnPlt   │
                   │  • Grad clipping   │
                   │  • Early stopping  │
                   │  • Checkpointing   │
                   │  • MLflow logging  │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │   Predictions      │
                   │  ──────────────── │
                   │  Scaled return     │
                   │  → Inverse xform   │
                   │  → Real log return │
                   │  → prev_close×eʳ  │
                   │  = Next-day price  │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │   Flask Dashboard  │
                   │  ──────────────── │
                   │  7-page SPA UI     │
                   │  REST API          │
                   │  Chart.js charts   │
                   │  Real-time quotes  │
                   └────────────────────┘
```

### Feature Engineering Pipeline

```
Raw OHLCV
    │
    ├── Price Returns:    Returns, Log_Returns, HL_pct, OC_pct
    │
    ├── MA Ratios:        SMA5/Close, SMA20/Close, SMA50/Close, EMA12/Close
    │   (scale-invariant — eliminates price-level bias)
    │
    ├── Momentum:         RSI/100, MACD_diff, ROC_5
    │
    ├── Volatility:       BB_pct (Bollinger %), ATR/Close
    │
    ├── Volume:           Vol/rolling_mean_20
    │
    └── Direction:        Dir_1, Dir_2, Dir_3, Dir_5 (lagged signs)
                          Streak (signed run length)
                          Week_pos (close in weekly range)

Total: 21 features → MinMaxScaler → LSTM input
Target: Log_Returns → MinMaxScaler → LSTM output → inverse → price
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Deep Learning** | PyTorch 2.3, custom LSTM architectures |
| **Data** | yfinance, pandas, numpy |
| **Feature Eng.** | `ta` library (RSI, MACD, Bollinger, ATR, OBV) |
| **Scaling** | scikit-learn MinMaxScaler |
| **Optimization** | Optuna (TPE sampler, Median pruner) |
| **Experiment Tracking** | MLflow, TensorBoard |
| **Web Framework** | Flask 3.0, Flask-CORS |
| **Frontend** | Vanilla JS, Chart.js 4.4, CSS Grid |
| **CLI** | Click, Rich, Loguru |
| **Deployment** | Gunicorn, Render |
| **Persistence** | joblib (scalers), PyTorch checkpoints, JSON history |

---

## Project Structure

```
lstm-stock-predictor/
│
├── 📄 config.py                   # Central config — all hyperparameters
├── 📄 main.py                     # CLI entry point (train/predict/dashboard)
├── 📄 app.py                      # Flask application factory + REST API
├── 📄 requirements.txt            # Full dependencies (local)
├── 📄 requirements_render.txt     # Lightweight dependencies (Render)
├── 📄 Procfile                    # Gunicorn start command
├── 📄 render.yaml                 # Render deployment config
├── 📄 .gitignore
│
├── 📁 models/
│   ├── __init__.py
│   ├── lstm_models.py             # SimpleLSTM, BiLSTM, LSTMTransformer, losses
│   ├── trainer.py                 # Training engine, checkpointing, evaluation
│   └── tuner.py                   # Optuna hyperparameter optimization
│
├── 📁 utils/
│   ├── __init__.py
│   ├── data_pipeline.py           # Download, features, sequences, scalers
│   └── metrics.py                 # RMSE, MAE, MAPE, R², DA, Theil U
│
├── 📁 templates/
│   └── dashboard.html             # 7-page industrial dashboard (965 lines)
│
├── 📁 data/
│   ├── raw/                       # Cached .parquet files (git-ignored)
│   └── processed/                 # Scalers, close arrays (git-ignored)
│
├── 📁 checkpoints/                # Best model .pt files (git-ignored)
└── 📁 logs/                       # TensorBoard runs (git-ignored)
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/lstm-stock-predictor.git
cd lstm-stock-predictor

# 2. Environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Train
python main.py train --ticker AAPL --model StackedLSTM

# 5. Predict
python main.py predict --ticker AAPL --days 7

# 6. Dashboard
python main.py dashboard
# → Open http://localhost:5000
```

---

## Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| pip | 23.0+ |
| Git | Any |
| RAM | 4GB+ recommended |
| GPU | Optional (CUDA auto-detected) |

### Step-by-Step

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/lstm-stock-predictor.git
cd lstm-stock-predictor

# Create virtual environment
python -m venv .venv

# Activate (choose your OS)
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows CMD
.venv\Scripts\Activate.ps1       # Windows PowerShell

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create environment file
cp .env.example .env             # then edit with your keys
```

### GPU Setup (Optional)

```bash
# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS built-in from PyTorch 2.x)
pip install torch torchvision

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Usage

### CLI Commands

#### Train a Model

```bash
# Basic training
python main.py train --ticker AAPL --model StackedLSTM

# With Optuna hyperparameter tuning
python main.py train --ticker AAPL --model StackedLSTM --tune

# Different architectures
python main.py train --ticker NVDA --model BiLSTM
python main.py train --ticker MSFT --model Transformer

# Available tickers (any valid Yahoo Finance symbol)
python main.py train --ticker TSLA --model StackedLSTM
```

#### Generate Forecasts

```bash
# 7-day forecast
python main.py predict --ticker AAPL --days 7

# 5-day forecast
python main.py predict --ticker NVDA --days 5
```

Expected output:
```
────────── Forecasting 7d for AAPL ──────────
      [AAPL] 7-Day Price Forecast
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Day    ┃ Predicted Close ($) ┃ Change  ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Day +1 │ $193.82             │ +$1.47  │
│ Day +2 │ $194.61             │ +$0.79  │
│ Day +3 │ $195.28             │ +$0.67  │
│ Day +4 │ $194.90             │ -$0.38  │
│ Day +5 │ $195.74             │ +$0.84  │
│ Day +6 │ $196.10             │ +$0.36  │
│ Day +7 │ $197.22             │ +$1.12  │
└────────┴─────────────────────┴─────────┘
Base price (last known close): $192.35
```

#### Launch Dashboard

```bash
python main.py dashboard
# → http://localhost:5000
```

#### Print Configuration

```bash
python main.py info
```

### Training Multiple Tickers

```bash
for ticker in AAPL MSFT NVDA TSLA META GOOGL; do
  python main.py train --ticker $ticker --model StackedLSTM
done
```

### Experiment Tracking

```bash
# TensorBoard
tensorboard --logdir=logs/runs
# → http://localhost:6006

# MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5001
```

---

## Dashboard

The web dashboard is a 7-page single-page application built with vanilla JS and Chart.js.

### Pages

| Page | Description |
|---|---|
| **Dashboard** | Live price, OHLCV metrics, 1Y price chart, 7-day LSTM forecast, training curve, model metrics, technical signals, attention heatmap |
| **Predictions** | Actual vs LSTM overlay chart, 7-day forecast bars with confidence ring, Monte Carlo probability distribution |
| **Backtesting** | Portfolio equity curve, monthly returns heatmap, full trade log with entry/exit/returns |
| **Model Lab** | Architecture comparison (StackedLSTM vs BiLSTM vs Transformer), hyperparameter sensitivity, training config panel |
| **Indicators** | 12 live indicator cards with bar gauges, RSI chart with overbought/oversold zones |
| **Risk Analysis** | Return distribution histogram, drawdown chart, 3 risk gauges (Overall / Volatility / Market) |
| **Screener** | Sortable table of 10 tickers with LSTM signal, confidence %, and price target — click to analyze |

### Searching Tickers

Type any valid ticker in the search bar (top-right) and press **Enter** or click **Analyze** to load live data and predictions for that symbol.

---

## Configuration

All hyperparameters are centralised in `config.py`. Key settings:

```python
# Data
DataConfig.sequence_length    = 60      # lookback window (days)
DataConfig.prediction_horizon = 1       # days to predict ahead
DataConfig.train_split        = 0.80    # 80/10/10 split
DataConfig.start_date         = "2018-01-01"

# Model
ModelConfig.model_type        = "StackedLSTM"
ModelConfig.hidden_sizes      = [128, 64]
ModelConfig.dropout           = 0.2

# Training
TrainingConfig.epochs         = 150
TrainingConfig.batch_size     = 32
TrainingConfig.learning_rate  = 1e-3
TrainingConfig.loss_fn        = "MSELoss"    # MSELoss | HuberLoss | DirectionalLoss
TrainingConfig.patience       = 20           # early stopping patience
TrainingConfig.scheduler      = "ReduceLROnPlateau"
```

---

## API Reference

All endpoints return JSON. Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/quote/<ticker>` | Live price, change, volume, P/E, 52W high/low |
| `GET` | `/api/history/<ticker>?period=1y` | OHLCV history (1mo/3mo/1y/2y/5y) |
| `GET` | `/api/predict/<ticker>` | 7-day LSTM forecast + confidence |
| `GET` | `/api/metrics/<ticker>` | Saved test metrics (RMSE, R², MAPE, DA) |
| `GET` | `/api/training/<ticker>` | Training/validation loss history |
| `GET` | `/api/watchlist` | Default watchlist tickers |

### Example Response — `/api/predict/AAPL`

```json
{
  "dates":       ["2024-12-16", "2024-12-17", "..."],
  "predictions": [193.82, 194.61, 195.28, 194.90, 195.74, 196.10, 197.22],
  "current":     192.35,
  "confidence":  0.831,
  "model":       "LSTM"
}
```

---

## Deployment

### Render (Free Tier)

```bash
# 1. Push to GitHub (see GitHub section above)

# 2. Go to render.com → New → Web Service
# 3. Connect your GitHub repo
# 4. Configure:
#    Build Command:  pip install -r requirements_render.txt
#    Start Command:  gunicorn "app:create_app()" --bind 0.0.0.0:$PORT
#    Instance Type:  Free

# 5. Deploy → Live at:
#    https://lstm-stock-predictor.onrender.com
```

> Free tier note: App sleeps after 15 min inactivity. First request after sleep takes ~30s to wake up.

### Local Production

```bash
gunicorn "app:create_app()" --bind 0.0.0.0:5000 --workers 2
```

---

## Results

### Metric Progression During Development

| Run | Change Made | RMSE | MAPE | R² |
|---|---|---|---|---|
| Run 1 | Initial (broken scaler) | $44.54 | 17.89% | -2.06 |
| Run 2 | Fixed dual scaler | $7.17 | 2.76% | 0.83 |
| Run 3 | Wider model [256,128] | $9.23 | 3.39% | 0.87 |
| Run 4 | Reverted + DirectionalLoss | $6.21 | 2.31% | 0.94 |
| Run 5 | Return-based prediction | $3.13 | 1.06% | **0.9676** |

### Key Design Decisions

**Why predict log returns instead of prices?**
Predicting log returns (`ln(P_t / P_{t-1})`) instead of raw price levels solves three problems: (1) removes the non-stationarity of price series, (2) makes the target scale-invariant across different stocks, (3) makes directional accuracy a natural property of the loss function.

**Why two separate scalers?**
Using one scaler for all 30+ features (where Volume ranges 10M–100M and Close ranges $50–$500) causes inverse transform errors. A dedicated `target_scaler` fitted only on Close/Returns gives clean 1-column inverse transforms with zero reconstruction error.

**Why scale-invariant features?**
Raw OHLCV inputs (especially Volume) dominate the MinMaxScaler range and add noise. Features like `SMA20/Close`, `RSI/100`, `BB_pct` are bounded ratios that carry the same information without scale bias.

---

## Roadmap

- [ ] Ensemble model (LSTM + XGBoost + Prophet voting)
- [ ] Intraday prediction (5-min/1-hour intervals)
- [ ] Portfolio optimisation page (Markowitz + LSTM signals)
- [ ] Sentiment analysis integration (news + Reddit)
- [ ] WebSocket real-time price streaming
- [ ] Docker containerisation
- [ ] PostgreSQL for persistent trade log
- [ ] Email/Slack alerts for LSTM signals

---

## License

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

Built by [Diya Patel](https://github.com/diyap24) with PyTorch · Flask · Chart.js

**[⬆ Back to Top](#lstm-stock-intelligence-platform)**

</div>
