# ============================================================
# main.py — CLI Entry Point
# ============================================================

import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import click
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from config import config
from utils.data_pipeline import StockDataPipeline
from models.trainer import Trainer
from models.tuner import HyperparamTuner

console = Console()


@click.group()
def cli():
    """📈 Stock Price Prediction with LSTM — Industrial-Grade CLI"""


@cli.command()
@click.option("--ticker", "-t", default="AAPL", help="Stock ticker symbol")
@click.option("--model",  "-m", default="StackedLSTM",
              type=click.Choice(["StackedLSTM", "BiLSTM", "Transformer"]),
              help="Model architecture")
@click.option("--tune/--no-tune", default=False, help="Run Optuna tuning first")
def train(ticker: str, model: str, tune: bool):
    """Train an LSTM model for a given ticker."""
    console.rule(f"[bold cyan]Training {model} for {ticker.upper()}")

    # 1. Data pipeline
    logger.info(f"Building data pipeline for {ticker}...")
    pipe = StockDataPipeline(ticker)
    data = pipe.run()

    # 2. Optional tuning
    if tune:
        console.rule("[bold yellow]Hyperparameter Tuning")
        tuner = HyperparamTuner(data)
        best  = tuner.run()
        rprint(f"\n[green]Best params:[/green] {best}")

    # 3. Train
    trainer = Trainer(data, model_type=model)
    history = trainer.train()

    # 4. Display results
    table = Table(title=f"[{ticker}] Test Metrics", style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k, v in history.get("test_metrics", {}).items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)


@cli.command()
@click.option("--ticker", "-t", default="AAPL")
@click.option("--days",   "-d", default=5, help="Days to forecast")
def predict(ticker: str, days: int):
    """Generate a multi-day forecast for a ticker."""
    import numpy as np
    console.rule(f"[bold green]Forecasting {days}d for {ticker.upper()}")

    pipe    = StockDataPipeline(ticker)
    data    = pipe.run()
    trainer = Trainer(data)
    trainer.load_best()

    features = data["features"]
    scaler   = pipe.feature_scaler
    lr_idx   = features.index("Log_Returns")
    dir_idxs = [features.index(f) for f in ("Dir_1", "Dir_2", "Dir_3") if f in features]

    seq         = data["X_test"][-1:].copy()  # (1, seq_len, n_feat)
    log_returns = []

    for _ in range(days):
        log_ret, _ = trainer.predict(seq)
        log_returns.append(log_ret[0])

        # Build next row: copy last timestep, update key features
        new_row = seq[0, -1, :].copy()
        # Cascade direction signals (Dir_1 → Dir_2 → Dir_3)
        for k in range(len(dir_idxs) - 1, 0, -1):
            new_row[dir_idxs[k]] = new_row[dir_idxs[k - 1]]
        if dir_idxs:
            pred_sign = 1.0 if log_ret[0] > 0 else (-1.0 if log_ret[0] < 0 else 0.0)
            new_row[dir_idxs[0]] = np.clip(
                pred_sign * scaler.scale_[dir_idxs[0]] + scaler.min_[dir_idxs[0]], 0.0, 1.0
            )
        new_row[lr_idx] = np.clip(
            log_ret[0] * scaler.scale_[lr_idx] + scaler.min_[lr_idx], 0.0, 1.0
        )
        seq = np.concatenate(
            [seq[:, 1:, :], new_row[np.newaxis, np.newaxis, :]], axis=1
        )

    prices = pipe.inverse_transform_price(np.array(log_returns))

    table = Table(title=f"[{ticker}] {days}-Day Forecast", style="green")
    table.add_column("Day", style="bold")
    table.add_column("Predicted Close ($)", justify="right")
    for i, p in enumerate(prices, 1):
        table.add_row(f"Day +{i}", f"${p:,.2f}")
    console.print(table)


@cli.command()
def dashboard():
    """Launch the web dashboard."""
    from app import create_app
    app = create_app()
    console.rule("[bold magenta]Launching Dashboard")
    console.print(f"[cyan]Open: http://localhost:{config.dashboard.port}")
    app.run(
        host  = config.dashboard.host,
        port  = config.dashboard.port,
        debug = config.dashboard.debug,
    )


@cli.command()
def info():
    """Print current configuration."""
    import json
    rprint(json.dumps(config.summary(), indent=2, default=str))


if __name__ == "__main__":
    cli()
