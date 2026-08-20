"""Feature extraction pipeline for Gargoyle ML.

Extracts the 6 canonical features defined in features/feature_spec.md from:
1. Raw HTTP request logs (using sliding time windows per client IP)
2. Pre-computed feature datasets (CSV / JSON)
"""

from __future__ import annotations

from collections import defaultdict, deque
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.spec import (
    FEATURE_NAMES,
    NUM_FEATURES,
    SPEC_VERSION,
    get_cold_start_vector,
    validate_feature_vector,
)

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class ClientRequestRecord:
    """Individual client request event in memory for sliding window computation."""

    timestamp: float
    path: str
    status_code: int
    header_anomaly_score: float


class FeatureExtractor:
    """Extracts features from streaming or batch request logs per client."""

    WINDOW_60S: float = 60.0
    WINDOW_5M: float = 300.0

    def __init__(self) -> None:
        # Client IP -> deque of ClientRequestRecord
        self._client_history: Dict[str, deque[ClientRequestRecord]] = defaultdict(deque)

    def _normalize_path(self, path: str) -> str:
        """Normalize endpoint path by stripping query params and trailing slashes."""
        clean_path = path.split("?")[0].strip()
        if len(clean_path) > 1 and clean_path.endswith("/"):
            clean_path = clean_path[:-1]
        return clean_path

    def compute_features_for_event(
        self,
        client_ip: str,
        timestamp: float,
        path: str,
        status_code: int,
        header_anomaly_score: float,
    ) -> List[float]:
        """Compute the 6-feature vector for an incoming request event.

        Updates client history and computes features over the trailing windows.
        """
        norm_path = self._normalize_path(path)
        current_record = ClientRequestRecord(
            timestamp=timestamp,
            path=norm_path,
            status_code=status_code,
            header_anomaly_score=header_anomaly_score,
        )

        history = self._client_history[client_ip]
        history.append(current_record)

        # Evict records older than 5 minutes (300s)
        cutoff_5m = timestamp - self.WINDOW_5M
        while history and history[0].timestamp < cutoff_5m:
            history.popleft()

        # Filter for 60s window
        cutoff_60s = timestamp - self.WINDOW_60S
        records_60s = [r for r in history if r.timestamp >= cutoff_60s]

        # Feature 0: requests_last_60s
        requests_last_60s = float(len(records_60s))

        # Feature 1 & 2: avg_interval_ms & interval_stddev_ms
        if len(records_60s) >= 2:
            timestamps = [r.timestamp for r in records_60s]
            deltas_ms = [
                (timestamps[i] - timestamps[i - 1]) * 1000.0
                for i in range(1, len(timestamps))
            ]
            avg_interval_ms = float(sum(deltas_ms) / len(deltas_ms))

            if len(deltas_ms) >= 2:
                variance = sum((d - avg_interval_ms) ** 2 for d in deltas_ms) / (len(deltas_ms) - 1)
                interval_stddev_ms = float(math.sqrt(max(0.0, variance)))
            else:
                interval_stddev_ms = 0.0
        else:
            avg_interval_ms = 0.0
            interval_stddev_ms = 0.0

        # Feature 3: distinct_endpoints_last_5m
        endpoints_5m = {r.path for r in history}
        distinct_endpoints_last_5m = float(len(endpoints_5m))

        # Feature 4: failed_auth_count_last_5m
        failed_auth_count_last_5m = float(
            sum(1 for r in history if r.status_code in (401, 403))
        )

        # Feature 5: header_anomaly_score
        anomaly_score = float(max(0.0, min(1.0, header_anomaly_score)))

        vector = [
            requests_last_60s,
            avg_interval_ms,
            interval_stddev_ms,
            distinct_endpoints_last_5m,
            failed_auth_count_last_5m,
            anomaly_score,
        ]

        validate_feature_vector(vector)
        return vector

    def extract_from_raw_logs(
        self,
        raw_logs_path: Path | str,
    ) -> Tuple[List[List[float]], List[int], List[str]]:
        """Extract features from a CSV file of raw request logs.

        Returns:
            Tuple of (feature_vectors, binary_labels, attack_type_names)
        """
        path = Path(raw_logs_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw logs file not found: {path}")

        X: List[List[float]] = []
        y: List[int] = []
        labels: List[str] = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row["timestamp_iso"])
                ip = row["client_ip"]
                req_path = row["path"]
                status = int(row["status_code"])
                anomaly = float(row["header_anomaly_score"])
                is_abuse = int(row.get("is_abusive", 0))
                label = row.get("label", "normal")

                vec = self.compute_features_for_event(
                    client_ip=ip,
                    timestamp=ts,
                    path=req_path,
                    status_code=status,
                    header_anomaly_score=anomaly,
                )
                X.append(vec)
                y.append(is_abuse)
                labels.append(label)

        return X, y, labels


def load_dataset(
    csv_path: Path | str,
    as_numpy: bool = True,
) -> Tuple[Any, Any, List[str]]:
    """Load a pre-extracted feature dataset CSV.

    Args:
        csv_path: Path to dataset CSV.
        as_numpy: If True and NumPy is available, return X and y as np.ndarray.

    Returns:
        Tuple of (X, y, labels)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    X_list: List[List[float]] = []
    y_list: List[int] = []
    labels: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vec = [float(row[col]) for col in FEATURE_NAMES]
            validate_feature_vector(vec)
            X_list.append(vec)
            y_list.append(int(row["is_abusive"]))
            labels.append(row.get("label", "normal"))

    if as_numpy and HAS_NUMPY:
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), labels

    return X_list, y_list, labels
