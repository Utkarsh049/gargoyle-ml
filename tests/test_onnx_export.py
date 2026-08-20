"""Unit tests for ONNX model export and inference parity."""

import os
import pickle
import tempfile
import unittest

import numpy as np
import onnxruntime as ort
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from export.to_onnx import export_model_to_onnx, verify_onnx_model
from features.spec import (
    NUM_FEATURES,
    ONNX_INPUT_NAME,
    get_cold_start_vector,
)


class TestONNXExport(unittest.TestCase):
    """Test suite for ONNX export and runtime inference validation."""

    def setUp(self):
        # Create a small synthetic dataset for training test models
        np.random.seed(42)
        self.X_dummy = np.random.uniform(0.0, 100.0, size=(50, NUM_FEATURES)).astype(np.float32)
        self.y_dummy = np.random.choice([0, 1], size=50).astype(np.int64)

    def test_export_logistic_regression_parity(self):
        lr = LogisticRegression(random_state=42)
        lr.fit(self.X_dummy, self.y_dummy)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_lr.onnx")
            export_model_to_onnx(lr, onnx_path)
            self.assertTrue(os.path.exists(onnx_path))

            # Verify with onnxruntime
            is_valid, meta = verify_onnx_model(onnx_path)
            self.assertTrue(is_valid)

            # Compare scikit-learn vs ONNX runtime probabilities
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            test_batch = self.X_dummy[:10]

            sklearn_probs = lr.predict_proba(test_batch)
            onnx_results = session.run(None, {ONNX_INPUT_NAME: test_batch})
            onnx_probs = onnx_results[1]

            np.testing.assert_allclose(sklearn_probs, onnx_probs, rtol=1e-4, atol=1e-4)

    def test_export_random_forest_parity(self):
        rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(self.X_dummy, self.y_dummy)

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_rf.onnx")
            export_model_to_onnx(rf, onnx_path)
            self.assertTrue(os.path.exists(onnx_path))

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            test_batch = self.X_dummy[:10]

            sklearn_probs = rf.predict_proba(test_batch)
            onnx_results = session.run(None, {ONNX_INPUT_NAME: test_batch})
            onnx_probs = onnx_results[1]

            np.testing.assert_allclose(sklearn_probs, onnx_probs, rtol=1e-4, atol=1e-4)

    def test_cold_start_inference(self):
        model_path = "models/abuse_model.onnx"
        if not os.path.exists(model_path):
            self.skipTest("models/abuse_model.onnx not yet generated")

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        cold_start = np.array([get_cold_start_vector(as_numpy=True)], dtype=np.float32)

        results = session.run(None, {ONNX_INPUT_NAME: cold_start})
        labels = results[0]
        probs = results[1]

        self.assertEqual(len(labels), 1)
        self.assertEqual(probs.shape, (1, 2))
        # Cold start normal traffic should predict class 0 (Normal)
        self.assertEqual(labels[0], 0)
        self.assertGreater(probs[0, 0], 0.5)

    def test_invalid_input_shape_rejected(self):
        model_path = "models/abuse_model.onnx"
        if not os.path.exists(model_path):
            self.skipTest("models/abuse_model.onnx not yet generated")

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        # Only 5 features instead of 6
        invalid_input = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)

        with self.assertRaises(Exception):
            session.run(None, {ONNX_INPUT_NAME: invalid_input})


if __name__ == "__main__":
    unittest.main()
