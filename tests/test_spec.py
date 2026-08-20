"""Unit tests for feature specification definition and validation."""

import math
import unittest

from features.spec import (
    FEATURE_NAMES,
    FEATURE_SPEC,
    NUM_FEATURES,
    SPEC_VERSION,
    get_cold_start_vector,
    get_feature_by_index,
    get_feature_by_name,
    to_dict,
    validate_feature_vector,
)


class TestFeatureSpec(unittest.TestCase):
    """Test suite for feature specification and validation rules."""

    def test_spec_metadata(self):
        self.assertEqual(SPEC_VERSION, "1.0.0")
        self.assertEqual(NUM_FEATURES, 6)
        self.assertEqual(len(FEATURE_SPEC), 6)
        self.assertEqual(len(FEATURE_NAMES), 6)
        self.assertEqual(
            FEATURE_NAMES,
            [
                "requests_last_60s",
                "avg_interval_ms",
                "interval_stddev_ms",
                "distinct_endpoints_last_5m",
                "failed_auth_count_last_5m",
                "header_anomaly_score",
            ],
        )

    def test_feature_indices_ordered(self):
        for expected_idx, feature in enumerate(FEATURE_SPEC):
            self.assertEqual(feature.index, expected_idx)

    def test_get_feature_by_index(self):
        f0 = get_feature_by_index(0)
        self.assertEqual(f0.name, "requests_last_60s")
        self.assertEqual(f0.dtype, "float32")

        with self.assertRaises(IndexError):
            get_feature_by_index(6)

        with self.assertRaises(IndexError):
            get_feature_by_index(-1)

    def test_get_feature_by_name(self):
        f = get_feature_by_name("header_anomaly_score")
        self.assertEqual(f.index, 5)
        self.assertEqual(f.min_value, 0.0)
        self.assertEqual(f.max_value, 1.0)

        with self.assertRaises(KeyError):
            get_feature_by_name("non_existent_feature")

    def test_cold_start_vector(self):
        v = get_cold_start_vector()
        self.assertEqual(len(v), 6)
        self.assertEqual(v, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertTrue(validate_feature_vector(v))

    def test_validate_feature_vector_valid(self):
        valid_vec = [10.0, 500.0, 50.0, 3.0, 1.0, 0.2]
        self.assertTrue(validate_feature_vector(valid_vec))

    def test_validate_feature_vector_invalid_length(self):
        with self.assertRaises(ValueError):
            validate_feature_vector([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            validate_feature_vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_validate_feature_vector_nan_inf(self):
        with self.assertRaises(ValueError):
            validate_feature_vector([1.0, 2.0, float("nan"), 1.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            validate_feature_vector([1.0, 2.0, float("inf"), 1.0, 0.0, 0.0])

    def test_validate_feature_vector_out_of_bounds(self):
        # requests_last_60s < 1.0
        with self.assertRaises(ValueError):
            validate_feature_vector([0.5, 100.0, 10.0, 1.0, 0.0, 0.0])

        # header_anomaly_score > 1.0
        with self.assertRaises(ValueError):
            validate_feature_vector([1.0, 100.0, 10.0, 1.0, 0.0, 1.5])

        # header_anomaly_score < 0.0
        with self.assertRaises(ValueError):
            validate_feature_vector([1.0, 100.0, 10.0, 1.0, 0.0, -0.1])

    def test_to_dict_export(self):
        d = to_dict()
        self.assertEqual(d["spec_version"], "1.0.0")
        self.assertEqual(d["num_features"], 6)
        self.assertEqual(len(d["features"]), 6)
        self.assertEqual(d["features"][0]["name"], "requests_last_60s")


if __name__ == "__main__":
    unittest.main()
