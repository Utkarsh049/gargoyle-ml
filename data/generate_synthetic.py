"""Synthetic traffic and feature data generator for Gargoyle ML.

Generates structurally valid synthetic dataset representing normal traffic
and various abuse patterns (DoS rate bursts, brute force auth, endpoint scans,
header anomaly bots) to bootstrap the ML pipeline before real simulator data arrives.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.spec import (
    FEATURE_NAMES,
    NUM_FEATURES,
    SPEC_VERSION,
    validate_feature_vector,
)


@dataclass
class SyntheticTrafficSample:
    """A single synthetic feature sample with associated ground-truth labels."""

    requests_last_60s: float
    avg_interval_ms: float
    interval_stddev_ms: float
    distinct_endpoints_last_5m: float
    failed_auth_count_last_5m: float
    header_anomaly_score: float
    label: str
    is_abusive: int

    def to_feature_vector(self) -> List[float]:
        """Return the 6 features in exact canonical order."""
        return [
            self.requests_last_60s,
            self.avg_interval_ms,
            self.interval_stddev_ms,
            self.distinct_endpoints_last_5m,
            self.failed_auth_count_last_5m,
            self.header_anomaly_score,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return asdict(self)


@dataclass
class RawRequestLog:
    """A simulated raw HTTP request log entry."""

    timestamp_iso: str
    client_ip: str
    method: str
    path: str
    status_code: int
    user_agent: str
    header_anomaly_score: float
    label: str
    is_abusive: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return asdict(self)


class SyntheticDataGenerator:
    """Generator for synthetic traffic logs and feature vectors."""

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.rng = random.Random(seed)

    def _sample_normal(self) -> SyntheticTrafficSample:
        """Generate a feature vector representing legitimate user traffic."""
        requests = float(self.rng.randint(1, 25))
        if requests <= 1:
            avg_interval = 0.0
            std_interval = 0.0
        else:
            avg_interval = round(self.rng.uniform(1500.0, 15000.0), 2)
            std_interval = round(self.rng.uniform(400.0, 5000.0), 2)

        distinct_endpoints = float(self.rng.randint(1, min(int(requests), 6)))
        # Occasional human typo on login (3% probability)
        failed_auth = float(1 if self.rng.random() < 0.03 else 0)
        # Clean browser headers
        header_anomaly = round(self.rng.uniform(0.0, 0.15), 3)

        return SyntheticTrafficSample(
            requests_last_60s=requests,
            avg_interval_ms=avg_interval,
            interval_stddev_ms=std_interval,
            distinct_endpoints_last_5m=distinct_endpoints,
            failed_auth_count_last_5m=failed_auth,
            header_anomaly_score=header_anomaly,
            label="normal",
            is_abusive=0,
        )

    def _sample_rate_burst(self) -> SyntheticTrafficSample:
        """Generate a feature vector representing high-rate DoS / burst flooding."""
        requests = float(self.rng.randint(120, 1800))
        avg_interval = round(self.rng.uniform(10.0, 180.0), 2)
        # Tight timing with very low variance (scripted loop)
        std_interval = round(self.rng.uniform(0.1, 20.0), 2)
        distinct_endpoints = float(self.rng.randint(1, 3))
        failed_auth = float(self.rng.randint(0, 2))
        header_anomaly = round(self.rng.uniform(0.0, 0.4), 3)

        return SyntheticTrafficSample(
            requests_last_60s=requests,
            avg_interval_ms=avg_interval,
            interval_stddev_ms=std_interval,
            distinct_endpoints_last_5m=distinct_endpoints,
            failed_auth_count_last_5m=failed_auth,
            header_anomaly_score=header_anomaly,
            label="rate_burst",
            is_abusive=1,
        )

    def _sample_brute_force(self) -> SyntheticTrafficSample:
        """Generate a feature vector representing credential stuffing / auth brute force."""
        requests = float(self.rng.randint(35, 400))
        avg_interval = round(self.rng.uniform(80.0, 650.0), 2)
        std_interval = round(self.rng.uniform(2.0, 45.0), 2)
        distinct_endpoints = float(self.rng.randint(1, 2))  # Targeting /login or /token
        failed_auth = float(self.rng.randint(15, min(int(requests), 350)))
        header_anomaly = round(self.rng.uniform(0.1, 0.7), 3)

        return SyntheticTrafficSample(
            requests_last_60s=requests,
            avg_interval_ms=avg_interval,
            interval_stddev_ms=std_interval,
            distinct_endpoints_last_5m=distinct_endpoints,
            failed_auth_count_last_5m=failed_auth,
            header_anomaly_score=header_anomaly,
            label="brute_force",
            is_abusive=1,
        )

    def _sample_endpoint_scan(self) -> SyntheticTrafficSample:
        """Generate a feature vector representing endpoint scanning / dir-busting."""
        requests = float(self.rng.randint(40, 500))
        avg_interval = round(self.rng.uniform(60.0, 500.0), 2)
        std_interval = round(self.rng.uniform(5.0, 80.0), 2)
        # High distinct endpoint count
        distinct_endpoints = float(self.rng.randint(25, 180))
        failed_auth = float(self.rng.randint(2, 25))
        header_anomaly = round(self.rng.uniform(0.3, 0.9), 3)

        return SyntheticTrafficSample(
            requests_last_60s=requests,
            avg_interval_ms=avg_interval,
            interval_stddev_ms=std_interval,
            distinct_endpoints_last_5m=distinct_endpoints,
            failed_auth_count_last_5m=failed_auth,
            header_anomaly_score=header_anomaly,
            label="endpoint_scan",
            is_abusive=1,
        )

    def _sample_header_bot(self) -> SyntheticTrafficSample:
        """Generate a feature vector representing bots with abnormal/malicious headers."""
        requests = float(self.rng.randint(5, 60))
        avg_interval = round(self.rng.uniform(400.0, 4000.0), 2)
        std_interval = round(self.rng.uniform(20.0, 300.0), 2)
        distinct_endpoints = float(self.rng.randint(1, 8))
        failed_auth = float(self.rng.randint(0, 4))
        # High header anomaly score (scanner tools, missing User-Agent, etc.)
        header_anomaly = round(self.rng.uniform(0.65, 1.0), 3)

        return SyntheticTrafficSample(
            requests_last_60s=requests,
            avg_interval_ms=avg_interval,
            interval_stddev_ms=std_interval,
            distinct_endpoints_last_5m=distinct_endpoints,
            failed_auth_count_last_5m=failed_auth,
            header_anomaly_score=header_anomaly,
            label="header_bot",
            is_abusive=1,
        )

    def generate_features_dataset(
        self,
        num_samples: int = 1000,
        abuse_ratio: float = 0.25,
    ) -> List[SyntheticTrafficSample]:
        """Generate a balanced or realistic dataset of synthetic feature vectors.

        Args:
            num_samples: Total number of samples to generate.
            abuse_ratio: Fraction of abusive samples (default 0.25 for realistic imbalance).

        Returns:
            List of validated SyntheticTrafficSample instances.
        """
        num_abuse = int(num_samples * abuse_ratio)
        num_normal = num_samples - num_abuse

        samples: List[SyntheticTrafficSample] = []

        # Generate normal traffic
        for _ in range(num_normal):
            s = self._sample_normal()
            validate_feature_vector(s.to_feature_vector())
            samples.append(s)

        # Generate abuse traffic evenly across attack categories
        attack_generators = [
            self._sample_rate_burst,
            self._sample_brute_force,
            self._sample_endpoint_scan,
            self._sample_header_bot,
        ]

        for i in range(num_abuse):
            generator = attack_generators[i % len(attack_generators)]
            s = generator()
            validate_feature_vector(s.to_feature_vector())
            samples.append(s)

        self.rng.shuffle(samples)
        return samples

    def generate_raw_traffic_logs(
        self,
        num_records: int = 500,
    ) -> List[RawRequestLog]:
        """Generate a sequence of synthetic raw HTTP request logs."""
        endpoints = [
            "/api/v1/users",
            "/api/v1/auth/login",
            "/api/v1/products",
            "/api/v1/orders",
            "/api/v1/search",
            "/admin/dashboard",
            "/api/v1/checkout",
            "/.env",
            "/wp-admin/login.php",
        ]
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
            "curl/7.81.0",
            "python-requests/2.28.1",
            "sqlmap/1.6#stable",
            "Nikto/2.1.6",
        ]

        logs: List[RawRequestLog] = []
        base_time = 1700000000.0  # Unix timestamp baseline

        for i in range(num_records):
            is_abuse = self.rng.random() < 0.25
            if not is_abuse:
                ip = f"192.168.1.{self.rng.randint(10, 50)}"
                method = self.rng.choice(["GET", "POST"])
                path = self.rng.choice(endpoints[:5])
                status = 200 if self.rng.random() > 0.05 else 400
                ua = self.rng.choice(user_agents[:2])
                anomaly = round(self.rng.uniform(0.0, 0.1), 2)
                label = "normal"
                abusive_flag = 0
            else:
                ip = f"10.0.0.{self.rng.randint(100, 110)}"
                attack_type = self.rng.choice(["rate_burst", "brute_force", "endpoint_scan", "header_bot"])
                label = attack_type
                abusive_flag = 1

                if attack_type == "brute_force":
                    path = "/api/v1/auth/login"
                    method = "POST"
                    status = 401
                    ua = self.rng.choice(user_agents)
                    anomaly = round(self.rng.uniform(0.2, 0.6), 2)
                elif attack_type == "endpoint_scan":
                    path = self.rng.choice(endpoints)
                    method = "GET"
                    status = self.rng.choice([404, 403, 200])
                    ua = self.rng.choice(user_agents[2:])
                    anomaly = round(self.rng.uniform(0.5, 0.9), 2)
                else:
                    path = "/api/v1/search"
                    method = "GET"
                    status = 200
                    ua = self.rng.choice(user_agents)
                    anomaly = round(self.rng.uniform(0.3, 0.8), 2)

            log_entry = RawRequestLog(
                timestamp_iso=f"{base_time + i * 0.5:.3f}",
                client_ip=ip,
                method=method,
                path=path,
                status_code=status,
                user_agent=ua,
                header_anomaly_score=anomaly,
                label=label,
                is_abusive=abusive_flag,
            )
            logs.append(log_entry)

        return logs


def save_features_to_csv(samples: List[SyntheticTrafficSample], filepath: Path | str) -> None:
    """Save synthetic feature samples to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = FEATURE_NAMES + ["label", "is_abusive"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_dict())


def save_raw_logs_to_csv(logs: List[RawRequestLog], filepath: Path | str) -> None:
    """Save synthetic raw request logs to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "timestamp_iso",
        "client_ip",
        "method",
        "path",
        "status_code",
        "user_agent",
        "header_anomaly_score",
        "label",
        "is_abusive",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for log in logs:
            writer.writerow(log.to_dict())


def main() -> None:
    """CLI entrypoint for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic traffic logs and feature vectors for Gargoyle ML"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1200,
        help="Total number of feature vector samples to generate (default: 1200)",
    )
    parser.add_argument(
        "--raw-records",
        type=int,
        default=800,
        help="Total number of raw request logs to generate (default: 800)",
    )
    parser.add_argument(
        "--abuse-ratio",
        type=float,
        default=0.25,
        help="Proportion of abusive traffic (default: 0.25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-features",
        type=str,
        default="data/processed/synthetic_features.csv",
        help="Output path for processed feature vectors CSV",
    )
    parser.add_argument(
        "--output-raw",
        type=str,
        default="data/raw/synthetic_traffic.csv",
        help="Output path for raw traffic logs CSV",
    )

    args = parser.parse_args()

    generator = SyntheticDataGenerator(seed=args.seed)

    print(f"[+] Generating {args.samples} synthetic feature samples (Spec v{SPEC_VERSION})...")
    features = generator.generate_features_dataset(
        num_samples=args.samples,
        abuse_ratio=args.abuse_ratio,
    )
    save_features_to_csv(features, args.output_features)
    print(f"[✓] Saved feature dataset to {args.output_features}")

    print(f"[+] Generating {args.raw_records} raw request logs...")
    raw_logs = generator.generate_raw_traffic_logs(num_records=args.raw_records)
    save_raw_logs_to_csv(raw_logs, args.output_raw)
    print(f"[✓] Saved raw traffic logs to {args.output_raw}")


if __name__ == "__main__":
    main()
