"""ONNX Model Export Pipeline for Gargoyle ML.

Converts trained scikit-learn models (LogisticRegression or RandomForestClassifier)
into portable ONNX format (abuse_model.onnx) for in-process inference in the Go core.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.spec import (
    NUM_FEATURES,
    ONNX_INPUT_NAME,
    SPEC_VERSION,
    get_cold_start_vector,
)

try:
    import numpy as np  # type: ignore
    import onnx  # type: ignore
    import onnxruntime as ort  # type: ignore
    from skl2onnx import convert_sklearn  # type: ignore
    from skl2onnx.common.data_types import FloatTensorType  # type: ignore

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


def export_model_to_onnx(
    model: Any,
    output_path: Path | str,
    input_name: str = ONNX_INPUT_NAME,
    num_features: int = NUM_FEATURES,
    target_opset: int = 15,
    doc_string: Optional[str] = None,
) -> Path:
    """Convert a trained scikit-learn model to ONNX format and save to disk.

    Args:
        model: Trained scikit-learn estimator.
        output_path: Destination path for the .onnx file.
        input_name: Name of the input tensor (default: "float_input").
        num_features: Number of continuous float features (default: 6).
        target_opset: ONNX opset version (default: 15).
        doc_string: Optional metadata documentation embedded in ONNX model.

    Returns:
        Path to the exported ONNX model.
    """
    if not HAS_ONNX:
        raise RuntimeError("skl2onnx and onnx are required for ONNX export.")

    initial_type = [(input_name, FloatTensorType([None, num_features]))]

    options: Dict[Any, Dict[str, Any]] = {
        type(model): {"zipmap": False}
    }

    if doc_string is None:
        doc_string = f"Gargoyle abuse detection model (Spec v{SPEC_VERSION})"

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options=options,
        target_opset=target_opset,
        doc_string=doc_string,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return out_file


def verify_onnx_model(
    onnx_path: Path | str,
    input_name: str = ONNX_INPUT_NAME,
    sample_input: Optional[np.ndarray] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that an ONNX model loads and runs inference properly.

    Args:
        onnx_path: Path to .onnx file.
        input_name: Expected input tensor name.
        sample_input: Optional test tensor (default: cold-start vector batch of 1).

    Returns:
        Tuple of (is_valid, metadata_dict)
    """
    if not HAS_ONNX:
        raise RuntimeError("onnxruntime is required for ONNX verification.")

    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    # Check onnx graph validity
    onnx_proto = onnx.load(str(path))
    onnx.checker.check_model(onnx_proto)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    input_names = [inp.name for inp in inputs]
    output_names = [out.name for out in outputs]

    if input_name not in input_names:
        raise ValueError(f"Expected input tensor '{input_name}', but found: {input_names}")

    if sample_input is None:
        sample_input = np.array([get_cold_start_vector()], dtype=np.float32)

    results = session.run(None, {input_name: sample_input})

    meta = {
        "inputs": [{"name": inp.name, "shape": inp.shape, "type": inp.type} for inp in inputs],
        "outputs": [{"name": out.name, "shape": out.shape, "type": out.type} for out in outputs],
        "sample_output_labels": results[0].tolist() if len(results) > 0 else [],
        "sample_output_probabilities": results[1].tolist() if len(results) > 1 else [],
    }

    return True, meta


def main() -> None:
    """CLI entrypoint for ONNX model export."""
    parser = argparse.ArgumentParser(description="Export trained model to ONNX format")
    parser.add_argument(
        "--model",
        type=str,
        default="models/abuse_model.pkl",
        help="Path to trained pickle model (default: models/abuse_model.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/abuse_model.onnx",
        help="Path to output ONNX model (default: models/abuse_model.onnx)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify exported ONNX model with onnxruntime (default: True)",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[!] Error: Model file not found: {model_path}")
        sys.exit(1)

    print(f"[*] Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"[*] Exporting to ONNX -> {args.output} (Spec v{SPEC_VERSION})...")
    out_path = export_model_to_onnx(model, args.output)
    print(f"[✓] Successfully exported ONNX model to {out_path}")

    if args.verify:
        print("[*] Verifying ONNX model with onnxruntime...")
        is_valid, meta = verify_onnx_model(out_path)
        print(f"[✓] Model verification passed!")
        print(f"    - Input tensor:  {meta['inputs'][0]['name']} (Shape: {meta['inputs'][0]['shape']})")
        print(f"    - Output labels: {meta['outputs'][0]['name']} (Shape: {meta['outputs'][0]['shape']})")
        print(f"    - Output probs:  {meta['outputs'][1]['name']} (Shape: {meta['outputs'][1]['shape']})")


if __name__ == "__main__":
    main()
