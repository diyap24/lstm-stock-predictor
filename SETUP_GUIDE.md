# Stock Price Prediction with LSTM — VS Code Setup Guide

## Prerequisites
- Python 3.10+
- Git
- VS Code (latest)

---

## Step 1 — Clone / Open the Project

```bash
# If downloaded as a zip, extract it. Then:
cd stock_lstm_predictor
code .           # opens the project folder in VS Code
```

---

## Step 2 — Install Recommended VS Code Extensions

Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`) and install:

| Extension                      | ID                                   |
|-------------------------------|--------------------------------------|
| Python                         | ms-python.python                     |
| Pylance                        | ms-python.vscode-pylance             |
| Jupyter                        | ms-toolsai.jupyter                   |
| GitLens                        | eamodio.gitlens                      |
| Better Comments                | aaron-bond.better-comments           |
| Error Lens                     | usernamehw.errorlens                 |
| autoDocstring                  | njpwerner.autodocstring              |
| TODO Highlight                 | wayou.vscode-todo-highlight          |
| Thunder Client (API testing)   | rangav.vscode-thunder-client         |
| Docker                         | ms-azuretools.vscode-docker          |

---

## Step 3 — Create and Activate Virtual Environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

**VS Code:** Press `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.venv`.

---

## Step 4 — Install All Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> **GPU users (CUDA 12.x):**
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
>
> **Apple Silicon (MPS):**
> ```bash
> pip install torch torchvision   # MPS is built-in from PyTorch 2.x
> ```

---

## Step 5 — Configure Environment Variables

Create a `.env` file at the project root:

```env
ALPHA_VANTAGE_KEY=your_key_here
SECRET_KEY=change-this-in-production
WANDB_API_KEY=optional_wandb_key
```

---

## Step 6 — Project Structure Overview

```
stock_lstm_predictor/
├── config.py                   # Central configuration (all tunable params)
├── main.py                     # CLI entry point
├── app.py                      # Flask dashboard factory
├── requirements.txt
├── .env                        # Secret keys (not committed)
│
├── utils/
│   ├── data_pipeline.py        # Download, indicators, sequences, splits
│   └── metrics.py              # RMSE, MAE, MAPE, R², Directional Accuracy
│
├── models/
│   ├── lstm_models.py          # Stacked LSTM, BiLSTM, LSTM+Transformer
│   ├── trainer.py              # Training loop (AMP, grad clip, checkpoints)
│   └── tuner.py                # Optuna hyperparameter optimization
│
├── data/
│   ├── raw/                    # Downloaded OHLCV parquet files (auto-created)
│   └── processed/              # Scalers and engineered data (auto-created)
│
├── checkpoints/                # Best model weights (.pt files)
├── logs/                       # TensorBoard logs, run history
├── models/registry/            # MLflow model registry
│
└── templates/
    └── dashboard.html          # Full-stack industrial web dashboard
```

---

## Step 7 — Run Your First Training

```bash
# Train StackedLSTM on Apple Inc. (downloads data automatically)
python main.py train --ticker AAPL --model StackedLSTM

# Train with Optuna hyperparameter tuning first (slower but better)
python main.py train --ticker AAPL --model StackedLSTM --tune

# Train Bidirectional LSTM on NVDA
python main.py train --ticker NVDA --model BiLSTM

# Train Transformer-LSTM hybrid
python main.py train --ticker MSFT --model Transformer
```

**What happens:**
1. Downloads OHLCV data from Yahoo Finance (cached as `.parquet`)
2. Engineers 25+ technical indicators (RSI, MACD, Bollinger, OBV, etc.)
3. Creates sliding-window sequences (60-day lookback → 5-day forecast)
4. Trains with AMP, gradient clipping, CosineAnnealingWarmRestarts
5. Early stopping (patience=25) saves best checkpoint
6. Logs all metrics to MLflow and exports history JSON

---

## Step 8 — Generate Predictions

```bash
# 7-day price forecast for TSLA
python main.py predict --ticker TSLA --days 7
```

---

## Step 9 — Launch the Dashboard

```bash
python main.py dashboard
# → Open http://localhost:5000 in your browser
```

**Dashboard features:**
- Live price quotes and daily metrics (Open, High, Low, Volume, Market Cap)
- Interactive 1M / 3M / 1Y / 2Y / 5Y price chart
- 7-day LSTM forecast bars with confidence ring
- Training loss curve (Train vs Val)
- Volume histogram
- Technical signal panel (RSI, MACD, Bollinger, ADX, Stochastic, OBV)
- LSTM attention weight heatmap (explainability)
- Watchlist sidebar with live tickers
- Real-time ticker strip

---

## Step 10 — Monitor Training with TensorBoard

```bash
tensorboard --logdir=logs/runs
# → Open http://localhost:6006
```

---

## Step 11 — View MLflow Experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# → Open http://localhost:5001
```

---

## Step 12 — Run Hyperparameter Optimization Standalone

```python
# Inside VS Code Python terminal
from utils.data_pipeline import StockDataPipeline
from models.tuner import HyperparamTuner

pipe  = StockDataPipeline("AAPL")
data  = pipe.run()
tuner = HyperparamTuner(data)
best  = tuner.run()   # runs 50 Optuna trials
print(best)
```

---

## Step 13 — Adjust Key Hyperparameters

Edit `config.py` to change any parameter without touching model code:

```python
# Larger model
ModelConfig.hidden_sizes = [512, 256, 128, 64]
ModelConfig.num_layers   = 4
ModelConfig.use_attention = True

# Longer training
TrainingConfig.epochs        = 300
TrainingConfig.patience      = 40
TrainingConfig.learning_rate = 5e-4

# Different architecture
ModelConfig.model_type = "BiLSTM"   # StackedLSTM | BiLSTM | Transformer

# Longer lookback window
DataConfig.sequence_length    = 90
DataConfig.prediction_horizon = 10
```

---

## Step 14 — VS Code Debugging

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Train AAPL",
      "type": "debugpy",
      "request": "launch",
      "module": "main",
      "args": ["train", "--ticker", "AAPL", "--model", "StackedLSTM"],
      "console": "integratedTerminal"
    },
    {
      "name": "Dashboard",
      "type": "debugpy",
      "request": "launch",
      "module": "main",
      "args": ["dashboard"],
      "console": "integratedTerminal"
    }
  ]
}
```

Press `F5` to launch any configuration with breakpoints.

---

## Step 15 — Run Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: ta` | `pip install ta` |
| `yfinance rate limit` | Wait 1 min; data is cached after first run |
| CUDA out of memory | Reduce `batch_size` in `config.py` (try 32 or 16) |
| MPS errors on Mac | Set `TrainingConfig.device = "cpu"` temporarily |
| MLflow DB locked | Delete `mlflow.db` and restart |
| Flask port in use | Change `DashboardConfig.port = 5001` |

---

## Architecture Summary

```
Raw OHLCV Data
    ↓
Yahoo Finance (yfinance)
    ↓
Technical Indicators (ta library: RSI, MACD, Bollinger, ATR, OBV, etc.)
    ↓
MinMaxScaler / StandardScaler
    ↓
Sliding Window Sequences (seq_len=60, horizon=5)
    ↓
┌─────────────────────────────────┐
│  LSTM Architecture Options      │
│  • StackedLSTM + Attention      │
│  • Bidirectional LSTM + MHA     │
│  • LSTM + Transformer Encoder   │
└─────────────────────────────────┘
    ↓
HuberLoss / MSE / LogCosh
    ↓
AdamW + CosineAnnealingWarmRestarts
    ↓
AMP + Gradient Clipping + Early Stopping
    ↓
Checkpointing + MLflow + TensorBoard
    ↓
Predictions → Inverse Transform → Price Forecast
```
