"""Static checks for the long-generation comparison cases."""
from __future__ import annotations

import unittest

from models.flashnext.tests.api import TestSpec
from models.flashnext.tests.catalog import build_catalog
from models.flashnext.tests.runner import SuiteConfig


class ComparisonCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog()
        cls.config = SuiteConfig()

    def test_long_cases_report_32_token_windows(self):
        for test_id in (
            "product-long-answer",
            "product-long-slabpacks",
            "product-long-physical-miss",
            "physical-miss-calibration",
        ):
            spec = self.catalog[test_id]
            command = spec.script(self.config, None)
            self.assertIn("--tokens", command)
            self.assertIn("256", command)
            self.assertIn("--window", command)
            self.assertIn("32", command)
            self.assertIn("--rounds", command)
            self.assertEqual(command[command.index("--rounds") + 1], "1")
            self.assertIn("--phase", command)
            self.assertIn("answer", command)

    def test_product_long_runs_one_current_answer(self):
        spec = self.catalog["product-long-answer"]
        command = spec.script(self.config, None)
        self.assertIn("bench_product_long.py", " ".join(command))
        self.assertIn("current", command)
        self.assertEqual(spec.controls["rounds"], "one; long runs never provide promotion statistics")

    def test_fusion_stack_sets_both_fusion_variants(self):
        spec = self.catalog["fusion-stack"]
        command = spec.script(self.config, None)
        self.assertIn("slabpack60_skew_8a_up,slabpack60_skew_8b_up", command)
        self.assertIn("Up-QMV/SwiGLU on", spec.controls["control"])
        self.assertIn("Up-QMV/SwiGLU on", spec.controls["candidate"])

    def test_chunk_case_has_worker_premise(self):
        spec = self.catalog["chunk-after-workers"]
        self.assertIsInstance(spec, TestSpec)
        why = spec.why.lower()
        self.assertIn("worker", why)
        self.assertIn("premise", why)
        self.assertIn("selected", spec.controls["worker count"].lower())

    def test_cache_aware_long_case_is_absent(self):
        self.assertNotIn("product-long-cache-aware", self.catalog)

    def test_physical_miss_gate_is_a_valid_block(self):
        spec = self.catalog["product-long-physical-miss"]
        output = (
            '{"type":"premise","offline_ceiling":'
            '{"minimum_mb_per_token":20.0,"predicted_mb_per_token":1.66,'
            '"passes":false}}'
        )
        interpretation = spec.interpret(0, output, [])
        self.assertIn("Premise blocked", interpretation)
        self.assertIn("No model comparison ran", interpretation)


if __name__ == "__main__":
    unittest.main()
