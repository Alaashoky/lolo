"""
AI Ensemble configuration and defaults.
"""

from __future__ import annotations

# Default configuration for the AI ensemble system
DEFAULT_CONFIG: dict = {
    "ai": {
        "enabled": True,
        "models_dir": "models/ai",
        "lookback_window": 60,
        "prediction_horizon": 1,
        "confidence_threshold": 0.55,
        "ensemble_weight": 0.4,
        "retrain_interval_hours": 24,
        "models": {
            "lstm": {
                "enabled": True,
                "weight": 0.20,
                "units": [128, 64, 32],
                "dropout": 0.2,
                "epochs": 50,
                "batch_size": 32,
                "lookback": 60,
            },
            "transformer": {
                "enabled": True,
                "weight": 0.15,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "epochs": 50,
                "batch_size": 32,
            },
            "xgboost": {
                "enabled": True,
                "weight": 0.15,
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
            },
            "random_forest": {
                "enabled": True,
                "weight": 0.10,
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
            },
            "cnn": {
                "enabled": True,
                "weight": 0.15,
                "filters": [32, 64],
                "kernel_size": 3,
                "dropout": 0.25,
                "epochs": 50,
                "batch_size": 32,
            },
            "prophet": {
                "enabled": True,
                "weight": 0.10,
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10,
                "forecast_horizon": 5,
            },
            "sentiment": {
                "enabled": True,
                "weight": 0.075,
                "lookback_hours": 24,
                "news_weight": 0.6,
                "social_weight": 0.4,
            },
            "anomaly": {
                "enabled": True,
                "weight": 0.075,
                "contamination": 0.05,
                "n_estimators": 100,
            },
        },
        "meta_learner": {
            "algorithm": "logistic_regression",
            "lookback_performance": 100,
            "min_samples": 20,
            "adaptation_rate": 0.1,
        },
        "feature_engineering": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
            "atr_period": 14,
            "volume_ma_period": 20,
        },
        "training": {
            "train_ratio": 0.80,
            "val_ratio": 0.10,
            "test_ratio": 0.10,
            "min_samples": 500,
            "cross_val_folds": 5,
        },
    }
}

# Signal encoding
SIGNAL_LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}
SIGNAL_TO_INT = {"SELL": 0, "HOLD": 1, "BUY": 2, "NEUTRAL": 1}
