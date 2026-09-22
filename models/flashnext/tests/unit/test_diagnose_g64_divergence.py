import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from models.flashnext.tests.bench.diagnose_g64_divergence import G64DivergenceProbe
from macqwen.testsuite.catalog import build_catalog


class G64DivergenceProbeTests(unittest.TestCase):
    def test_materialize_handles_mlx_bfloat16(self):
        import mlx.core as mx

        from models.flashnext.tests.bench.diagnose_g64_divergence import _materialize

        value = _materialize(mx.array([1.0, 2.0], dtype=mx.bfloat16))
        self.assertEqual(value.dtype, __import__("numpy").float32)
        self.assertEqual(value.tolist(), [1.0, 2.0])

    def test_probe_bounds_records_and_writes_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = G64DivergenceProbe(directory, max_records=1)
            self.assertEqual(probe.max_records, 1)
            report = probe.write_report("/checkpoint", "prompt", 16)
            payload = json.loads(report.read_text())
            self.assertEqual(payload["diagnostic"], "g64-divergence")
            self.assertTrue(payload["source_sha256"])
            self.assertEqual(payload["records"], [])

    def test_catalog_exposes_bounded_diagnostic(self):
        case = build_catalog("flashnext")["g64-divergence"]
        self.assertEqual(case.category, "diagnostic")
        context = mock.Mock(python="python", checkpoint="model")
        context.output.side_effect = lambda name: Path("/results/run") / name
        command = case.script(context, None)
        self.assertIn("models.flashnext.tests.bench.diagnose_g64_divergence", command)
        self.assertIn("--tokens", command)
        self.assertEqual(command[command.index("--tokens") + 1], "16")


if __name__ == "__main__":
    unittest.main()
