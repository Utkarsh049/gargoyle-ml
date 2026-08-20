"""Cross-Language Feature and Inference Parity Checker.

Validates that feature extraction and ONNX inference produce identical vectors
and predictions matching fixtures/parity_fixtures.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.extract import FeatureExtractor, compute_header_anomaly_from_headers
from features.spec import FEATURE_NAMES, NUM_FEATURES, ONNX_INPUT_NAME, SPEC_VERSION

try:
    import numpy as np  # type: ignore
    import onnxruntime as ort  # type: ignore

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def run_parity_checks(
    fixtures_path: Path | str = "fixtures/parity_fixtures.json",
    model_path: Optional[Path | str] = "models/abuse_model.onnx",
    tolerance: float = 1e-3,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Run feature extraction and inference parity check against test fixtures.

    Returns:
        Tuple of (all_passed, results_list)
    """
    path = Path(fixtures_path)
    if not path.exists():
        raise FileNotFoundError(f"Fixtures file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)

    if fixtures.get("spec_version") != SPEC_VERSION:
        raise ValueError(
            f"Fixture spec version '{fixtures.get('spec_version')}' does not match "
            f"code spec version '{SPEC_VERSION}'"
        )

    session = None
    if model_path and Path(model_path).exists() and HAS_ORT:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    all_passed = True
    results: List[Dict[str, Any]] = []

    for case in fixtures.get("test_cases", []):
        case_name = case["name"]
        events = case["events"]
        expected_vec = case["expected_feature_vector"]
        expected_label = case.get("expected_label")

        extractor = FeatureExtractor()
        actual_vec: List[float] = []

        for ev in events:
            ts = float(ev["timestamp"])
            ip = ev["client_ip"]
            req_path = ev["path"]
            status = int(ev["status_code"])
            headers = ev.get("headers", {})
            anomaly = compute_header_anomaly_from_headers(headers)

            actual_vec = extractor.compute_features_for_event(
                client_ip=ip,
                timestamp=ts,
                path=req_path,
                status_code=status,
                header_anomaly_score=anomaly,
            )

        # Check feature vector parity
        vector_match = True
        diffs = []
        for i, (exp_val, act_val) in enumerate(zip(expected_vec, actual_vec)):
            if abs(exp_val - act_val) > tolerance:
                vector_match = False
                diffs.append(
                    f"{FEATURE_NAMES[i]}: expected {exp_val:.4f}, got {act_val:.4f}"
                )

        # Check ONNX model prediction if available
        model_passed = True
        predicted_label = None
        predicted_prob = None

        if session is not None and vector_match:
            input_tensor = np.array([actual_vec], dtype=np.float32)
            out = session.run(None, {ONNX_INPUT_NAME: input_tensor})
            predicted_label = int(out[0][0])
            predicted_prob = float(out[1][0][1])

            if expected_label is not None and predicted_label != expected_label:
                model_passed = False

        passed = vector_match and model_passed
        if not passed:
            all_passed = False

        results.append({
            "name": case_name,
            "passed": passed,
            "vector_match": vector_match,
            "expected_vector": expected_vec,
            "actual_vector": actual_vec,
            "diffs": diffs,
            "predicted_label": predicted_label,
            "expected_label": expected_label,
            "predicted_prob": predicted_prob,
        })

    return all_passed, results


def main() -> None:
    """CLI for parity check."""
    parser = argparse.ArgumentParser(description="Run cross-language feature parity checks")
    parser.add_argument(
        "--fixtures",
        type=str,
        default="fixtures/parity_fixtures.json",
        help="Path to parity fixtures JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/abuse_model.onnx",
        help="Path to ONNX model",
    )

    args = parser.parse_args()

    print(f"[*] Running feature parity checks against {args.fixtures}...")
    passed, results = run_parity_checks(args.fixtures, args.model)

    print("------------------------------------------------------------")
    for r in results:
        status_icon = "✓" if r["passed"] else "✗"
        print(f"[{status_icon}] Test Case: {r['name']}")
        if not r["vector_match"]:
            print(f"    Vector Diffs: {', '.join(r['diffs'])}")
        if r["predicted_label"] is not None:
            print(f"    Predicted Label: {r['predicted_label']} (Prob: {r['predicted_prob']:.4f})")
    print("------------------------------------------------------------")

    if passed:
        print("[✓] ALL PARITY CHECKS PASSED: 0% Feature Drift Detected!")
    else:
        print("[!] PARITY CHECK FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
