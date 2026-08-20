"""Unit tests for cross-language feature parity validation."""

import os
import unittest

from export.parity_check import run_parity_checks


class TestFeatureParity(unittest.TestCase):
    """Test suite for feature parity checks against canonical fixtures."""

    def test_fixtures_parity_verification(self):
        fixtures_path = "fixtures/parity_fixtures.json"
        model_path = "models/abuse_model.onnx"

        if not os.path.exists(fixtures_path):
            self.skipTest(f"{fixtures_path} not found")

        all_passed, results = run_parity_checks(
            fixtures_path=fixtures_path,
            model_path=model_path if os.path.exists(model_path) else None,
            tolerance=1e-3,
        )

        for r in results:
            self.assertTrue(
                r["vector_match"],
                f"Feature drift in test case '{r['name']}': {r['diffs']}",
            )
            self.assertTrue(
                r["passed"],
                f"Parity check failed for test case '{r['name']}'",
            )

        self.assertTrue(all_passed)


if __name__ == "__main__":
    unittest.main()
