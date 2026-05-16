# ============================================================
# config.py — Stage 3: fix DA, keep RMSE gains
# ============================================================

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class DataConfig:
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL","GOOGL","MSFT","AMZN","NVDA","TSLA","META","JPM","V"
    ])
    start_date: str         = "2018-01-01"
    end_date: str           = "2024-12-31"
    interval: str           = "1d"
    sequence_length: int    = 60    # back to 60 — worked best
    prediction_horizon: int = 1
    train_split: float      = 0.80
    val_split: float        = 0.10
    test_split: float       = 0.10
    scaler_type: str        = "MinMaxScaler"
    indicators: List[str]   = field(default_factory=lambda: [])
    raw_data_dir: Path      = BASE_DIR / "data" / "raw"
    processed_data_dir: Path= BASE_DIR / "data" / "processed"

    def __post_init__(self):
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    model_type: str         = "StackedLSTM"
    input_size: int         = 13
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    num_layers: int         = 2
    dropout: float          = 0.2
    recurrent_dropout: float= 0.0
    bidirectional: bool     = False
    use_attention: bool     = False
    output_size: int        = 1
    l1_reg: float           = 0.0
    l2_reg: float           = 1e-4
    batch_norm: bool        = False
    layer_norm: bool        = False
    checkpoint_dir: Path    = BASE_DIR / "checkpoints"
    model_registry_dir: Path= BASE_DIR / "models" / "registry"

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_registry_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    epochs: int             = 150
    batch_size: int         = 32
    learning_rate: float    = 1e-3
    weight_decay: float     = 1e-5
    optimizer: str          = "Adam"
    momentum: float         = 0.9
    amsgrad: bool           = False
    scheduler: str          = "ReduceLROnPlateau"
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    T_0: int                = 50
    early_stopping: bool    = True
    patience: int           = 20
    min_delta: float        = 1e-6
    grad_clip: float        = 1.0
    use_amp: bool           = False
    loss_fn: str            = "DirectionalLoss"  # targets DA directly
    huber_delta: float      = 1.0
    device: str             = "auto"
    log_interval: int       = 10
    save_best_only: bool    = True
    tensorboard: bool       = False
    mlflow_tracking: bool   = False
    log_dir: Path           = BASE_DIR / "logs"
    runs_dir: Path          = BASE_DIR / "logs" / "runs"

    def __post_init__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TuningConfig:
    n_trials: int        = 30
    timeout: int         = 1800
    sampler: str         = "TPESampler"
    pruner: str          = "MedianPruner"
    lr_range: tuple      = (1e-4, 1e-2)
    hidden_range: tuple  = (64, 256)
    dropout_range: tuple = (0.1, 0.4)
    batch_range: tuple   = (16, 128)
    layers_range: tuple  = (1, 3)


@dataclass
class APIConfig:
    alpha_vantage_key: str = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY","demo"))
    wandb_project: str     = "stock-lstm-predictor"
    mlflow_uri: str        = "sqlite:///mlflow.db"


@dataclass
class DashboardConfig:
    host: str            = "0.0.0.0"
    port: int            = 5000
    debug: bool          = False
    secret_key: str      = field(default_factory=lambda: os.getenv("SECRET_KEY","dev-secret"))
    update_interval: int = 60


class Config:
    data      = DataConfig()
    model     = ModelConfig()
    training  = TrainingConfig()
    tuning    = TuningConfig()
    api       = APIConfig()
    dashboard = DashboardConfig()

    @classmethod
    def summary(cls) -> Dict:
        return {
            "data":     cls.data.__dict__,
            "model":    cls.model.__dict__,
            "training": cls.training.__dict__,
        }


config = Config()