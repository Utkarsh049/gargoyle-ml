"""Unit tests for sliding-window feature extraction and dataset loader."""

import os
import tempfile
import unittest

from features.extract import ClientRequestRecord, FeatureExtractor, load_dataset
from features.spec import NUM_FEATURES, validate_feature_vector


class TestFeatureExtractor(unittest.TestCase):
    """Test suite for sliding-window feature extractor."""

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_cold_start_single_request(self):
        vec = self.extractor.compute_features_for_event(
            client_ip="192.168.1.100",
            timestamp=1000.0,
            path="/api/v1/users",
            status_code=200,
            header_anomaly_score=0.05,
        )
        self.assertEqual(len(vec), NUM_FEATURES)
        self.assertTrue(validate_feature_vector(vec))

        # requests_last_60s: 1.0
        self.assertEqual(vec[0], 1.0)
        # avg_interval_ms: 0.0 (no previous request)
        self.assertEqual(vec[1], 0.0)
        # interval_stddev_ms: 0.0
        self.assertEqual(vec[2], 0.0)
        # distinct_endpoints_last_5m: 1.0
        self.assertEqual(vec[3], 1.0)
        # failed_auth_count_last_5m: 0.0
        self.assertEqual(vec[4], 0.0)
        # header_anomaly_score: 0.05
        self.assertAlmostEqual(vec[5], 0.05, places=3)

    def test_inter_request_intervals_and_variance(self):
        ip = "10.0.0.1"
        # 3 requests spaced exactly 200ms apart (0.2s)
        self.extractor.compute_features_for_event(ip, 1000.0, "/api/v1/resource", 200, 0.0)
        self.extractor.compute_features_for_event(ip, 1000.2, "/api/v1/resource", 200, 0.0)
        vec = self.extractor.compute_features_for_event(ip, 1000.4, "/api/v1/resource", 200, 0.0)

        # 3 requests
        self.assertEqual(vec[0], 3.0)
        # avg_interval_ms should be 200.0 ms
        self.assertAlmostEqual(vec[1], 200.0, places=1)
        # standard deviation should be 0.0 ms (perfectly uniform)
        self.assertAlmostEqual(vec[2], 0.0, places=1)

    def test_distinct_endpoints_and_failed_auth(self):
        ip = "10.0.0.2"
        # Hits 3 distinct paths, 2 failed auths (401, 403)
        self.extractor.compute_features_for_event(ip, 1000.0, "/api/login", 401, 0.2)
        self.extractor.compute_features_for_event(ip, 1001.0, "/api/login", 401, 0.2)
        self.extractor.compute_features_for_event(ip, 1002.0, "/api/users", 200, 0.0)
        vec = self.extractor.compute_features_for_event(ip, 1003.0, "/api/admin", 403, 0.5)

        # 4 requests in 60s
        self.assertEqual(vec[0], 4.0)
        # 3 distinct endpoints: /api/login, /api/users, /api/admin
        self.assertEqual(vec[3], 3.0)
        # 3 failed auths (401, 401, 403)
        self.assertEqual(vec[4], 3.0)
        # Anomaly score from current request
        self.assertAlmostEqual(vec[5], 0.5, places=2)

    def test_window_eviction(self):
        ip = "10.0.0.3"
        # Old request at t=100.0 (350 seconds ago, outside 5m window)
        self.extractor.compute_features_for_event(ip, 100.0, "/api/old", 401, 0.0)

        # Current request at t=460.0
        vec = self.extractor.compute_features_for_event(ip, 460.0, "/api/new", 200, 0.0)

        # Old request should have been evicted from 5m window
        self.assertEqual(vec[0], 1.0)  # only current in 60s
        self.assertEqual(vec[3], 1.0)  # only /api/new in 5m
        self.assertEqual(vec[4], 0.0)  # old 401 is evicted


class TestLoadDataset(unittest.TestCase):
    """Test suite for load_dataset helper."""

    def test_load_existing_dataset(self):
        dataset_path = "data/processed/synthetic_features.csv"
        if not os.path.exists(dataset_path):
            self.skipTest(f"{dataset_path} not found")

        X, y, labels = load_dataset(dataset_path, as_numpy=False)
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(labels))
        self.assertEqual(len(X[0]), NUM_FEATURES)


if __name__ == "__main__":
    unittest.main()
