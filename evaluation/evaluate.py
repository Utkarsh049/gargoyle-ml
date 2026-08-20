"""Evaluation metrics and baseline comparison for Gargoyle ML.

Calculates Precision, Recall, F1, Specificity, ROC-AUC, and Confusion Matrix.
Includes threshold sweep analysis and comparative benchmark against Go rule-based baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.extract import load_dataset
from features.spec import FEATURE_NAMES, NUM_FEATURES

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

HAS_SKLEARN: bool = True


@dataclass
class EvaluationMetrics:
    """Performance evaluation metrics for a binary classifier."""

    total_samples: int
    positive_samples: int
    negative_samples: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    roc_auc: Optional[float]
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


def calculate_metrics(
    y_true: Sequence[int] | np.ndarray | Any,
    y_pred: Sequence[int] | np.ndarray | Any,
    y_probs: Optional[Sequence[float] | np.ndarray | Any] = None,
    threshold: float = 0.5,
) -> EvaluationMetrics:
    """Calculate comprehensive evaluation metrics from ground truth and predictions."""
    total = len(y_true)
    if total == 0:
        raise ValueError("Cannot calculate metrics on empty dataset")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    positives = tp + fn
    negatives = tn + fp

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    roc_auc: Optional[float] = None
    if y_probs is not None and HAS_SKLEARN and len(set(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = None

    return EvaluationMetrics(
        total_samples=total,
        positive_samples=positives,
        negative_samples=negatives,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        specificity=round(specificity, 4),
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        threshold=threshold,
    )


def evaluate_threshold_sweep(
    y_true: Sequence[int] | np.ndarray | Any,
    y_probs: Sequence[float] | np.ndarray | Any,
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> List[EvaluationMetrics]:
    """Evaluate model performance across a range of decision thresholds."""
    results: List[EvaluationMetrics] = []
    for t in thresholds:
        y_pred = [1 if p >= t else 0 for p in y_probs]
        m = calculate_metrics(y_true, y_pred, y_probs=y_probs, threshold=t)
        results.append(m)
    return results


def evaluate_rule_baseline(
    X: Sequence[Sequence[float]] | np.ndarray | Any,
    y_true: Sequence[int] | np.ndarray | Any,
) -> EvaluationMetrics:
    """Evaluate a naive rule-based abuse detector baseline on the dataset.

    Rule heuristic flags as abuse if:
      - requests_last_60s > 100 OR
      - failed_auth_count_last_5m > 10 OR
      - header_anomaly_score > 0.5
    """
    y_pred: List[int] = []
    for row in X:
        req_60s = row[0]
        auth_fail_5m = row[4]
        anomaly = row[5]

        is_flagged = 1 if (req_60s > 100.0 or auth_fail_5m > 10.0 or anomaly > 0.5) else 0
        y_pred.append(is_flagged)

    return calculate_metrics(y_true, y_pred, threshold=0.5)


def format_metrics_report(
    model_metrics: EvaluationMetrics,
    baseline_metrics: Optional[EvaluationMetrics] = None,
    title: str = "Gargoyle Model Evaluation Report",
) -> str:
    """Generate a clean ASCII/Markdown report comparing model and baseline."""
    lines: List[str] = [
        f"============================================================",
        f" {title}",
        f"============================================================",
        f" Total Test Samples: {model_metrics.total_samples} (Abuse: {model_metrics.positive_samples}, Normal: {model_metrics.negative_samples})",
        f" Decision Threshold: {model_metrics.threshold}",
        f"------------------------------------------------------------",
        f" {'Metric':<20} | {'ML Model':<14} | {'Rule Baseline':<14}",
        f"------------------------------------------------------------",
        f" {'Precision (No FP)':<20} | {model_metrics.precision:<14.4f} | {baseline_metrics.precision if baseline_metrics else 'N/A':<14}",
        f" {'Recall (Detection)':<20} | {model_metrics.recall:<14.4f} | {baseline_metrics.recall if baseline_metrics else 'N/A':<14}",
        f" {'F1-Score':<20} | {model_metrics.f1_score:<14.4f} | {baseline_metrics.f1_score if baseline_metrics else 'N/A':<14}",
        f" {'Specificity (TNR)':<20} | {model_metrics.specificity:<14.4f} | {baseline_metrics.specificity if baseline_metrics else 'N/A':<14}",
        f" {'Accuracy':<20} | {model_metrics.accuracy:<14.4f} | {baseline_metrics.accuracy if baseline_metrics else 'N/A':<14}",
        f" {'ROC-AUC':<20} | {str(model_metrics.roc_auc) if model_metrics.roc_auc is not None else 'N/A':<14} | {'N/A':<14}",
        f"------------------------------------------------------------",
        f" Confusion Matrix (ML Model):",
        f"   True Positives  (Caught Abuse):    {model_metrics.tp}",
        f"   False Positives (Blocked Legitimate): {model_metrics.fp}  <-- Priority to minimize!",
        f"   True Negatives  (Allowed Legitimate): {model_metrics.tn}",
        f"   False Negatives (Missed Abuse):       {model_metrics.fn}",
        f"============================================================",
    ]
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for standalone model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Gargoyle ML abuse detection model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/synthetic_features.csv",
        help="Path to test features CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/abuse_model.pkl",
        help="Path to trained pickle model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for abuse classification (default: 0.5)",
    )

    args = parser.parse_args()

    import pickle

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[!] Error: Model file not found: {model_path}")
        sys.exit(1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X, y, labels = load_dataset(args.data, as_numpy=True)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        y_pred = [1 if p >= args.threshold else 0 for p in probs]
    else:
        probs = None
        y_pred = list(model.predict(X))

    model_metrics = calculate_metrics(y, y_pred, y_probs=probs, threshold=args.threshold)
    baseline_metrics = evaluate_rule_baseline(X, y)

    report = format_metrics_report(model_metrics, baseline_metrics)
    print(report)


if __name__ == "__main__":
    main()
