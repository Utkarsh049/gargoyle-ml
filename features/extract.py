"""Feature extraction pipeline for Gargoyle ML.

Extracts the 6 canonical features defined in features/feature_spec.md from:
1. Raw HTTP request logs (using sliding time windows per client IP / session)
2. Simulator logs (ground_truth_5k.csv with JSON headers, ISO timestamps)
3. Pre-computed feature datasets (CSV / JSON)
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure repo root is on sys.path when executed as a script
if __package__ in (None, ""):
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

import numpy as np


@dataclass
class ClientRequestRecord:
    """Individual client request event in memory for sliding window computation."""

    timestamp: float
    path: str
    status_code: int
    header_anomaly_score: float


def compute_header_anomaly_from_headers(headers: Dict[str, Any] | str) -> float:
    """Compute header anomaly score [0.0, 1.0] from request headers dictionary or JSON string.

    Evaluates:
      - Missing User-Agent (+0.4)
      - Known scanner/script User-Agent (+0.5)
      - Generic/truncated browser User-Agent (+0.3)
      - Missing Accept-Language header (+0.2)
      - Wildcard Accept (*/*) (+0.1)
    """
    if isinstance(headers, str):
        try:
            headers_dict = json.loads(headers)
        except Exception:
            headers_dict = {}
    elif isinstance(headers, dict):
        headers_dict = headers
    else:
        headers_dict = {}

    score = 0.0
    ua = headers_dict.get("User-Agent", "")
    if not ua:
        score += 0.4
    else:
        ua_lower = ua.lower()
        tool_keywords = [
            "python-requests",
            "sqlmap",
            "nikto",
            "curl",
            "go-http-client",
            "headlesschrome",
            "wpscan",
            "dirbuster",
            "postman",
        ]
        if any(k in ua_lower for k in tool_keywords):
            score += 0.5
        elif "mozilla" in ua_lower and not any(
            b in ua_lower for b in ["chrome/", "firefox/", "safari/", "edg/", "version/"]
        ):
            # Truncated or generic Mozilla header without actual browser engine
            score += 0.3

    if "Accept-Language" not in headers_dict:
        score += 0.2

    accept = headers_dict.get("Accept", "")
    if accept == "*/*":
        score += 0.1

    return min(1.0, max(0.0, round(score, 3)))


def parse_timestamp_to_seconds(ts_val: Union[str, float, int]) -> float:
    """Parse timestamp string (ISO-8601 or float) to epoch seconds."""
    if isinstance(ts_val, (int, float)):
        return float(ts_val)
    ts_str = ts_val.strip()
    try:
        return float(ts_str)
    except ValueError:
        pass

    # ISO-8601 parsing
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.timestamp()
    except Exception:
        pass

    # Fallback to datetime strptime
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt).timestamp()
        except ValueError:
            continue

    raise ValueError(f"Unable to parse timestamp: {ts_str}")


class FeatureExtractor:
    """Extracts features from streaming or batch request logs per client."""

    WINDOW_60S: float = 60.0
    WINDOW_5M: float = 300.0

    def __init__(self) -> None:
        # Client IP/Key -> deque of ClientRequestRecord
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
            avg_interval_ms = sum(deltas_ms) / len(deltas_ms)

            if len(deltas_ms) >= 2:
                variance = sum((d - avg_interval_ms) ** 2 for d in deltas_ms) / (len(deltas_ms) - 1)
                interval_stddev_ms = math.sqrt(max(0.0, variance))
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
        anomaly_score = max(0.0, min(1.0, header_anomaly_score))

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

    def extract_from_simulator_logs(
        self,
        raw_csv_path: Path | str,
        output_csv_path: Optional[Path | str] = None,
    ) -> Tuple[List[List[float]], List[int], List[str]]:
        """Extract features from simulator raw logs (ground_truth_5k.csv format).

        Columns: timestamp, batch_id, sequence_num, client_id, source_identifier,
                 method, endpoint, headers_json, status_code, latency_ms, true_label
        """
        path = Path(raw_csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Simulator logs file not found: {path}")

        X: List[List[float]] = []
        y: List[int] = []
        labels: List[str] = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = parse_timestamp_to_seconds(row["timestamp"])
                client_id = row.get("client_id", "client")
                batch_id = row.get("batch_id", "default")
                client_key = f"{client_id}:{batch_id}"

                req_path = row.get("endpoint", row.get("path", "/"))
                status = int(row.get("status_code", 200))

                if "headers_json" in row:
                    anomaly = compute_header_anomaly_from_headers(row["headers_json"])
                else:
                    anomaly = float(row.get("header_anomaly_score", 0.0))

                true_label = row.get("true_label", row.get("label", "normal"))
                is_abuse = 0 if true_label == "normal" else 1

                vec = self.compute_features_for_event(
                    client_ip=client_key,
                    timestamp=ts,
                    path=req_path,
                    status_code=status,
                    header_anomaly_score=anomaly,
                )
                X.append(vec)
                y.append(is_abuse)
                labels.append(true_label)

        if output_csv_path:
            out_path = Path(output_csv_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as out_f:
                writer = csv.writer(out_f)
                writer.writerow(FEATURE_NAMES + ["label", "is_abusive"])
                for vec, label, is_abuse in zip(X, labels, y):
                    writer.writerow(vec + [label, is_abuse])

        return X, y, labels

    def extract_from_raw_logs(
        self,
        raw_logs_path: Path | str,
    ) -> Tuple[List[List[float]], List[int], List[str]]:
        """Extract features from a CSV file of generic raw request logs."""
        path = Path(raw_logs_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw logs file not found: {path}")

        # Check if simulator format
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first_row = next(reader, {})
            if "headers_json" in first_row or "true_label" in first_row:
                return self.extract_from_simulator_logs(path)

        X: List[List[float]] = []
        y: List[int] = []
        labels: List[str] = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = parse_timestamp_to_seconds(row.get("timestamp_iso", row.get("timestamp", 0)))
                ip = row.get("client_ip", "127.0.0.1")
                req_path = row.get("path", row.get("endpoint", "/"))
                status = int(row.get("status_code", 200))
                anomaly = float(row.get("header_anomaly_score", 0.0))
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

    if as_numpy:
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), labels

    return X_list, y_list, labels


def main() -> None:
    """CLI for feature extraction."""
    parser = argparse.ArgumentParser(description="Extract features from raw HTTP traffic logs")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/ground_truth_5k.csv",
        help="Path to input raw traffic logs CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/real_features.csv",
        help="Path to output processed features CSV",
    )

    args = parser.parse_args()

    extractor = FeatureExtractor()
    print(f"[*] Extracting features from {args.input} (Spec: v{SPEC_VERSION})...")
    X, y, labels = extractor.extract_from_simulator_logs(args.input, output_csv_path=args.output)
    print(f"[✓] Successfully extracted {len(X)} feature vectors -> {args.output}")
    print(f"    - Abusive samples: {sum(y)}")
    print(f"    - Normal samples:  {len(y) - sum(y)}")


if __name__ == "__main__":
    main()
