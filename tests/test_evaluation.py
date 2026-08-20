"""Unit tests for evaluation metrics and baseline calculations."""

import unittest

from evaluation.evaluate import (
    EvaluationMetrics,
    calculate_metrics,
    evaluate_rule_baseline,
    evaluate_threshold_sweep,
    format_metrics_report,
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for metric calculation and report formatting."""

    def test_perfect_predictions(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        m = calculate_metrics(y_true, y_pred, threshold=0.5)

        self.assertEqual(m.accuracy, 1.0)
        self.assertEqual(m.precision, 1.0)
        self.assertEqual(m.recall, 1.0)
        self.assertEqual(m.f1_score, 1.0)
        self.assertEqual(m.specificity, 1.0)
        self.assertEqual(m.tp, 2)
        self.assertEqual(m.fp, 0)
        self.assertEqual(m.tn, 2)
        self.assertEqual(m.fn, 0)

    def test_imperfect_predictions(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]  # 1 TP, 1 FN, 1 FP, 1 TN
        m = calculate_metrics(y_true, y_pred, threshold=0.5)

        self.assertEqual(m.tp, 1)
        self.assertEqual(m.fp, 1)
        self.assertEqual(m.tn, 1)
        self.assertEqual(m.fn, 1)
        self.assertAlmostEqual(m.precision, 0.5, places=3)
        self.assertAlmostEqual(m.recall, 0.5, places=3)
        self.assertAlmostEqual(m.f1_score, 0.5, places=3)

    def test_threshold_sweep(self):
        y_true = [1, 0, 1, 0]
        y_probs = [0.8, 0.2, 0.6, 0.4]
        sweep = evaluate_threshold_sweep(y_true, y_probs, thresholds=[0.3, 0.5, 0.7])

        self.assertEqual(len(sweep), 3)
        self.assertEqual(sweep[0].threshold, 0.3)
        self.assertEqual(sweep[1].threshold, 0.5)
        self.assertEqual(sweep[2].threshold, 0.7)

    def test_rule_baseline(self):
        # Sample with high request rate (>100) -> should be flagged by rule
        X = [
            [150.0, 50.0, 1.0, 1.0, 0.0, 0.0],  # Abuse by rate
            [5.0, 2000.0, 500.0, 2.0, 0.0, 0.0],  # Normal
            [20.0, 500.0, 20.0, 1.0, 15.0, 0.1],  # Abuse by auth failures
        ]
        y_true = [1, 0, 1]
        baseline = evaluate_rule_baseline(X, y_true)

        self.assertEqual(baseline.tp, 2)
        self.assertEqual(baseline.fp, 0)
        self.assertEqual(baseline.tn, 1)
        self.assertEqual(baseline.fn, 0)
        self.assertEqual(baseline.precision, 1.0)
        self.assertEqual(baseline.recall, 1.0)

    def test_format_metrics_report(self):
        m = calculate_metrics([1, 0], [1, 0])
        report = format_metrics_report(m, title="Test Report")
        self.assertIn("Test Report", report)
        self.assertIn("Precision", report)
        self.assertIn("Confusion Matrix", report)


if __name__ == "__main__":
    unittest.main()
