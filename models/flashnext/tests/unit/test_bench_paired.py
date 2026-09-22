"""Pure-Python tests for paired production-benchmark statistics."""
from __future__ import annotations

import contextlib
import io
import unittest

from models.flashnext.tests.bench.bench_production import _two_sided_sign_p, report_paired


def _results(base_rates, candidate_rates):
    def arms(rates):
        return [
            {"gen_rate": rate, "mb_per_token": 100.0}
            for rate in rates
        ]

    return [
        {"condition": "baseline", "arms": arms(base_rates)},
        {"condition": "candidate", "arms": arms(candidate_rates)},
    ]


class PairedSignTest(unittest.TestCase):
    def test_all_improvements_are_reported_as_improvements(self):
        p_value, wins, losses, ties = _two_sided_sign_p([1.0, 2.0, 3.0])
        self.assertEqual((wins, losses, ties), (3, 0, 0))
        self.assertAlmostEqual(p_value, 0.25)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report_paired(_results([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]))
        text = output.getvalue()
        self.assertIn("improved in 3 of 3 non-tied pairs", text)
        self.assertIn("regressed in 0", text)
        self.assertIn("two-sided sign test p = 0.250", text)
        self.assertIn("improvement direction", text)

    def test_all_regressions_are_reported_as_regressions(self):
        p_value, wins, losses, ties = _two_sided_sign_p([-1.0, -2.0, -3.0])
        self.assertEqual((wins, losses, ties), (0, 3, 0))
        self.assertAlmostEqual(p_value, 0.25)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report_paired(_results([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]))
        text = output.getvalue()
        self.assertIn("improved in 0 of 3 non-tied pairs", text)
        self.assertIn("regressed in 3", text)
        self.assertIn("two-sided sign test p = 0.250", text)
        self.assertIn("regression direction", text)
        self.assertNotIn("The pairs do not separate", text)

    def test_ties_are_excluded_from_sign_test_but_reported(self):
        p_value, wins, losses, ties = _two_sided_sign_p([0.0, 0.0, 0.0])
        self.assertEqual((wins, losses, ties), (0, 0, 3))
        self.assertEqual(p_value, 1.0)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report_paired(_results([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]))
        text = output.getvalue()
        self.assertIn("improved in 0 of 0 non-tied pairs", text)
        self.assertIn("ties: 3", text)
        self.assertIn("All matched pairs tied", text)

    def test_mixed_direction_is_unresolved_with_two_sided_test(self):
        p_value, wins, losses, ties = _two_sided_sign_p([1.0, 1.0, -1.0, -1.0])
        self.assertEqual((wins, losses, ties), (2, 2, 0))
        self.assertEqual(p_value, 1.0)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report_paired(_results([1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 1.0, 1.0]))
        text = output.getvalue()
        self.assertIn("improved in 2 of 4 non-tied pairs", text)
        self.assertIn("regressed in 2", text)
        self.assertIn("improvement versus regression direction", text)
        self.assertIn("collect more matched pairs", text)


if __name__ == "__main__":
    unittest.main()
