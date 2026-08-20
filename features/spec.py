"""Feature specification contract and schema definitions for Gargoyle ML.

This module provides the single source of programmatic truth for feature
ordering, data types, validation, and metadata matching features/feature_spec.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, List, Sequence, Tuple

import numpy as np

SPEC_VERSION: str = "1.0.0"
ONNX_INPUT_NAME: str = "float_input"
ONNX_INPUT_SHAPE: Tuple[int | None, int] = (None, 6)
NUM_FEATURES: int = 6


@dataclass(frozen=True)
class Feature:
    """Metadata describing a single feature in the Gargoyle abuse detection vector."""

    index: int
    name: str
    dtype: str
    min_value: float
    max_value: float
    cold_start_default: float
    description: str

    def validate_value(self, value: float) -> bool:
        """Check if a scalar feature value is within valid bounds and finite."""
        if not math.isfinite(value):
            return False
        return self.min_value <= value <= self.max_value


FEATURE_SPEC: Tuple[Feature, ...] = (
    Feature(
        index=0,
        name="requests_last_60s",
        dtype="float32",
        min_value=1.0,
        max_value=float("inf"),
        cold_start_default=1.0,
        description="Total request count from client in trailing 60s window.",
    ),
    Feature(
        index=1,
        name="avg_interval_ms",
        dtype="float32",
        min_value=0.0,
        max_value=60000.0,
        cold_start_default=0.0,
        description="Mean inter-request interval (ms) in trailing 60s window.",
    ),
    Feature(
        index=2,
        name="interval_stddev_ms",
        dtype="float32",
        min_value=0.0,
        max_value=60000.0,
        cold_start_default=0.0,
        description="Standard deviation of inter-request intervals (ms) in 60s window.",
    ),
    Feature(
        index=3,
        name="distinct_endpoints_last_5m",
        dtype="float32",
        min_value=1.0,
        max_value=float("inf"),
        cold_start_default=1.0,
        description="Distinct normalized endpoints accessed in trailing 5m window.",
    ),
    Feature(
        index=4,
        name="failed_auth_count_last_5m",
        dtype="float32",
        min_value=0.0,
        max_value=float("inf"),
        cold_start_default=0.0,
        description="HTTP 401/403 response count in trailing 5m window.",
    ),
    Feature(
        index=5,
        name="header_anomaly_score",
        dtype="float32",
        min_value=0.0,
        max_value=1.0,
        cold_start_default=0.0,
        description="Heuristic header anomaly penalty score in [0.0, 1.0].",
    ),
)

FEATURE_NAMES: List[str] = [f.name for f in FEATURE_SPEC]


def get_feature_by_index(index: int) -> Feature:
    """Retrieve feature definition by zero-based index."""
    if 0 <= index < len(FEATURE_SPEC):
        return FEATURE_SPEC[index]
    raise IndexError(f"Feature index {index} out of bounds (0..{len(FEATURE_SPEC)-1})")


def get_feature_by_name(name: str) -> Feature:
    """Retrieve feature definition by name."""
    for f in FEATURE_SPEC:
        if f.name == name:
            return f
    raise KeyError(f"Unknown feature name: {name}")


def get_cold_start_vector(as_numpy: bool = False) -> Any:
    """Return default cold-start feature vector.

    Args:
        as_numpy: If True and numpy is available, return as np.ndarray(dtype=float32).
                  Otherwise, return as list of float.
    """
    defaults = [f.cold_start_default for f in FEATURE_SPEC]
    if as_numpy:
        return np.array(defaults, dtype=np.float32)
    return defaults


def validate_feature_vector(vector: Sequence[float] | Any) -> bool:
    """Validate that a feature vector conforms to spec size, dtype, and finite ranges.

    Args:
        vector: 1D array or sequence of length 6.

    Returns:
        True if valid.

    Raises:
        ValueError: If length or values violate the specification.
    """
    if isinstance(vector, np.ndarray):
        if vector.ndim != 1 or vector.shape[0] != NUM_FEATURES:
            raise ValueError(
                f"Expected 1D feature vector of length {NUM_FEATURES}, got shape {vector.shape}"
            )
        values = [float(x) for x in vector]
    else:
        try:
            values = [float(x) for x in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid vector elements: {exc}") from exc

        if len(values) != NUM_FEATURES:
            raise ValueError(
                f"Expected 1D feature vector of length {NUM_FEATURES}, got length {len(values)}"
            )

    for i, (val, feature) in enumerate(zip(values, FEATURE_SPEC)):
        if not math.isfinite(val):
            raise ValueError(f"Feature '{feature.name}' (index {i}) contains NaN or infinite value")
        if val < feature.min_value or val > feature.max_value:
            raise ValueError(
                f"Feature '{feature.name}' (index {i}) value {val} out of bounds "
                f"[{feature.min_value}, {feature.max_value}]"
            )

    return True


def to_dict() -> dict[str, Any]:
    """Export the feature specification as a structured dictionary."""
    return {
        "spec_version": SPEC_VERSION,
        "onnx_input_name": ONNX_INPUT_NAME,
        "onnx_input_shape": list(ONNX_INPUT_SHAPE),
        "num_features": NUM_FEATURES,
        "features": [asdict(f) for f in FEATURE_SPEC],
    }
