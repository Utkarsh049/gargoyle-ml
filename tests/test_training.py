"""Unit tests for model training and configuration."""

import os
import tempfile
import unittest

from training.model_config import ModelConfig
from training.train import run_training_pipeline, train_model


class TestTrainingPipeline(unittest.TestCase):
    """Test suite for model training pipeline."""

    def test_model_config_serialization(self):
        config = ModelConfig(model_type="random_forest", rf_n_estimators=50, random_state=99)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            config.save_json(cfg_path)
            self.assertTrue(os.path.exists(cfg_path))

            loaded = ModelConfig.load_json(cfg_path)
            self.assertEqual(loaded.model_type, "random_forest")
            self.assertEqual(loaded.rf_n_estimators, 50)
            self.assertEqual(loaded.random_state, 99)

    def test_run_training_pipeline_logistic_regression(self):
        dataset_path = "data/processed/synthetic_features.csv"
        if not os.path.exists(dataset_path):
            self.skipTest(f"{dataset_path} not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")
            metrics_path = os.path.join(tmpdir, "metrics.json")

            config = ModelConfig(
                model_type="logistic_regression",
                model_output_path=model_path,
                metrics_output_path=metrics_path,
                test_size=0.2,
                random_state=42,
            )

            model, test_metrics, baseline_metrics, summary = run_training_pipeline(
                data_path=dataset_path,
                config=config,
                save_artifacts=True,
            )

            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(metrics_path))
            self.assertGreater(test_metrics.accuracy, 0.7)
            self.assertGreater(test_metrics.f1_score, 0.7)
            self.assertIn("threshold_sweep", summary)
            self.assertEqual(len(summary["threshold_sweep"]), 9)

    def test_run_training_pipeline_random_forest(self):
        dataset_path = "data/processed/synthetic_features.csv"
        if not os.path.exists(dataset_path):
            self.skipTest(f"{dataset_path} not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "rf_model.pkl")
            metrics_path = os.path.join(tmpdir, "rf_metrics.json")

            config = ModelConfig(
                model_type="random_forest",
                rf_n_estimators=20,
                rf_max_depth=4,
                model_output_path=model_path,
                metrics_output_path=metrics_path,
                test_size=0.2,
                random_state=42,
            )

            model, test_metrics, baseline_metrics, summary = run_training_pipeline(
                data_path=dataset_path,
                config=config,
                save_artifacts=True,
            )

            self.assertTrue(os.path.exists(model_path))
            self.assertTrue(os.path.exists(metrics_path))
            self.assertGreater(test_metrics.accuracy, 0.8)
            self.assertGreater(test_metrics.f1_score, 0.8)


if __name__ == "__main__":
    unittest.main()
