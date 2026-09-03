from __future__ import annotations

import unittest

from models.flashnext.gpustat import GPUMeter


class GPUMeterTests(unittest.TestCase):
    def test_summary_exposes_relative_signal_only(self):
        meter = GPUMeter.__new__(GPUMeter)
        meter.samples = [10.0, 30.0]

        summary = meter.summary()

        self.assertEqual(summary["samples"], 2)
        self.assertAlmostEqual(summary["relative_busy_fraction"], 0.2)
        self.assertTrue(summary["relative_only"])
        self.assertNotIn("busy_fraction", summary)

    def test_empty_summary_has_no_absolute_gpu_value(self):
        meter = GPUMeter.__new__(GPUMeter)
        meter.samples = []

        self.assertEqual(meter.summary(), {"samples": 0})


if __name__ == "__main__":
    unittest.main()
