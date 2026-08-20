"""Model training pipeline for Gargoyle ML.

Trains a classical classifier (LogisticRegression or RandomForest) on feature-extracted
traffic data, evaluates performance, performs threshold sweeps, compares with rule baseline,
and serializes the trained model artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import (
    EvaluationMetrics,
    calculate_metrics,
    evaluate_rule_baseline,
    evaluate_threshold_sweep,
    format_metrics_report,
)
from features.extract import FeatureExtractor, load_dataset
from features.spec import FEATURE_NAMES, NUM_FEATURES, SPEC_VERSION
from training.model_config import ModelConfig

try:
    import numpy as np  # type: ignore
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def train_model(
    X_train: Any,
    y_train: Any,
    config: ModelConfig,
) -> Any:
    """Train a classifier based on the provided configuration."""
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required to train models. Install dependencies first.")

    if config.model_type == "logistic_regression":
        model = LogisticRegression(
            C=config.lr_c,
            max_iter=config.lr_max_iter,
            class_weight=config.lr_class_weight,
            solver=config.lr_solver,
            random_state=config.random_state,
        )
    elif config.model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            class_weight=config.rf_class_weight,
            min_samples_split=config.rf_min_samples_split,
            random_state=config.random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    model.fit(X_train, y_train)
    return model


def run_training_pipeline(
    data_path: Path | str,
    config: Optional[ModelConfig] = None,
    save_artifacts: bool = True,
) -> Tuple[Any, EvaluationMetrics, EvaluationMetrics, Dict[str, Any]]:
    """Execute the end-to-end training and evaluation pipeline.

    Args:
        data_path: Path to processed features CSV.
        config: Training configuration (uses defaults if None).
        save_artifacts: Whether to save model and metrics to disk.

    Returns:
        Tuple of (trained_model, test_metrics, baseline_metrics, training_summary_dict)
    """
    if config is None:
        config = ModelConfig()

    X, y, labels = load_dataset(data_path, as_numpy=True)

    if HAS_SKLEARN:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )
    else:
        # Fallback manual split
        split_idx = int(len(X) * (1.0 - config.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

    model = train_model(X_train, y_train, config)

    # Predictions and probabilities on test set
    if hasattr(model, "predict_proba"):
        y_test_probs = model.predict_proba(X_test)[:, 1]
        y_test_pred = [1 if p >= config.decision_threshold else 0 for p in y_test_probs]
    else:
        y_test_probs = None
        y_test_pred = list(model.predict(X_test))

    test_metrics = calculate_metrics(
        y_test,
        y_test_pred,
        y_probs=y_test_probs,
        threshold=config.decision_threshold,
    )
    baseline_metrics = evaluate_rule_baseline(X_test, y_test)

    # Threshold sweep
    threshold_results = []
    if y_test_probs is not None:
        threshold_results = [
            m.to_dict() for m in evaluate_threshold_sweep(y_test, y_test_probs)
        ]

    summary = {
        "spec_version": SPEC_VERSION,
        "model_type": config.model_type,
        "config": config.to_dict(),
        "dataset": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": NUM_FEATURES,
            "feature_names": FEATURE_NAMES,
        },
        "test_metrics": test_metrics.to_dict(),
        "baseline_metrics": baseline_metrics.to_dict(),
        "threshold_sweep": threshold_results,
    }

    if save_artifacts:
        # Save model pickle
        model_path = Path(config.model_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metrics JSON
        metrics_path = Path(config.metrics_output_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return model, test_metrics, baseline_metrics, summary


def main() -> None:
    """CLI entrypoint for training."""
    parser = argparse.ArgumentParser(description="Train Gargoyle ML abuse detection model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/synthetic_features.csv",
        help="Path to feature dataset CSV (default: data/processed/synthetic_features.csv)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
        help="Classifier architecture (default: logistic_regression)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for test set (default: 0.2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    config = ModelConfig(
        model_type=args.model_type,
        test_size=args.test_size,
        decision_threshold=args.threshold,
        random_state=args.seed,
    )

    print(f"[*] Starting Gargoyle ML training pipeline (Model: {config.model_type}, Spec: v{SPEC_VERSION})...")
    model, test_metrics, baseline_metrics, summary = run_training_pipeline(
        data_path=args.data,
        config=config,
        save_artifacts=True,
    )

    print(f"[✓] Model saved to {config.model_output_path}")
    print(f"[✓] Metrics saved to {config.metrics_output_path}\n")

    report = format_metrics_report(
        test_metrics,
        baseline_metrics,
        title=f"Gargoyle ML Evaluation ({config.model_type.upper()})",
    )
    print(report)


if __name__ == "__main__":
    main()
