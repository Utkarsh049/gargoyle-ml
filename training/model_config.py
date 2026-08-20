"""Model configuration and hyperparameters for Gargoyle ML training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional


@dataclass
class ModelConfig:
    """Configuration and hyperparameters for model training."""

    model_type: Literal["logistic_regression", "random_forest"] = "logistic_regression"

    # Common parameters
    random_state: int = 42
    test_size: float = 0.2
    decision_threshold: float = 0.5

    # Logistic Regression hyperparameters
    lr_c: float = 1.0
    lr_max_iter: int = 1000
    lr_class_weight: Optional[str] = "balanced"
    lr_solver: str = "lbfgs"

    # Random Forest hyperparameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 6
    rf_class_weight: Optional[str] = "balanced"
    rf_min_samples_split: int = 2

    # Artifact paths
    model_output_path: str = "models/abuse_model.pkl"
    onnx_output_path: str = "models/abuse_model.onnx"
    metrics_output_path: str = "models/train_metrics.json"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        """Save configuration to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> ModelConfig:
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


DEFAULT_CONFIG = ModelConfig()
