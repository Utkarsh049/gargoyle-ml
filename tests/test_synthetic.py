"""Unit tests for synthetic data generation and serialization."""

import csv
import os
import tempfile
import unittest

from data.generate_synthetic import (
    SyntheticDataGenerator,
    save_features_to_csv,
    save_raw_logs_to_csv,
)
from features.spec import FEATURE_NAMES, NUM_FEATURES, validate_feature_vector


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test suite for synthetic traffic and feature generation."""

    def setUp(self):
        self.generator = SyntheticDataGenerator(seed=123)

    def test_feature_dataset_shape_and_validity(self):
        samples = self.generator.generate_features_dataset(num_samples=100, abuse_ratio=0.3)
        self.assertEqual(len(samples), 100)

        abuse_count = sum(1 for s in samples if s.is_abusive == 1)
        self.assertEqual(abuse_count, 30)

        normal_count = sum(1 for s in samples if s.is_abusive == 0)
        self.assertEqual(normal_count, 70)

        for s in samples:
            vec = s.to_feature_vector()
            self.assertEqual(len(vec), NUM_FEATURES)
            self.assertTrue(validate_feature_vector(vec))

    def test_attack_label_diversity(self):
        samples = self.generator.generate_features_dataset(num_samples=200, abuse_ratio=0.5)
        labels = {s.label for s in samples}
        expected_labels = {"normal", "rate_burst", "brute_force", "endpoint_scan", "header_bot"}
        self.assertEqual(labels, expected_labels)

    def test_raw_logs_generation(self):
        logs = self.generator.generate_raw_traffic_logs(num_records=50)
        self.assertEqual(len(logs), 50)

        for entry in logs:
            self.assertTrue(entry.client_ip)
            self.assertTrue(entry.path.startswith("/"))
            self.assertIn(entry.status_code, [200, 400, 401, 403, 404])
            self.assertIn(entry.is_abusive, [0, 1])
            self.assertGreaterEqual(entry.header_anomaly_score, 0.0)
            self.assertLessEqual(entry.header_anomaly_score, 1.0)

    def test_save_and_load_features_csv(self):
        samples = self.generator.generate_features_dataset(num_samples=20, abuse_ratio=0.5)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_features.csv")
            save_features_to_csv(samples, csv_path)

            self.assertTrue(os.path.exists(csv_path))

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                expected_headers = FEATURE_NAMES + ["label", "is_abusive"]
                self.assertEqual(reader.fieldnames, expected_headers)
                rows = list(reader)
                self.assertEqual(len(rows), 20)

    def test_save_and_load_raw_logs_csv(self):
        logs = self.generator.generate_raw_traffic_logs(num_records=20)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_raw.csv")
            save_raw_logs_to_csv(logs, csv_path)

            self.assertTrue(os.path.exists(csv_path))

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertIn("timestamp_iso", reader.fieldnames)
                self.assertIn("client_ip", reader.fieldnames)
                self.assertIn("header_anomaly_score", reader.fieldnames)
                rows = list(reader)
                self.assertEqual(len(rows), 20)


if __name__ == "__main__":
    unittest.main()
