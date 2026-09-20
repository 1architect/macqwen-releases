from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from macqwen.measurement import (
    MeasurementRun, append_record, canonical_measurement_path, read_records,
    validate_path,
)


class MeasurementTests(unittest.TestCase):
    def test_canonical_path_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = canonical_measurement_path(root, "flashnext", "G64 kernel")
            self.assertEqual(path.suffix, ".jsonl")
            self.assertEqual(validate_path(path, root, "flashnext"), path.resolve())
            with self.assertRaises(ValueError):
                validate_path(root / "results.json", root, "flashnext")

    def test_lifecycle_is_append_only_and_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            run = MeasurementRun(path, runtime="k2_horizon", experiment="baseline")
            run.start()
            run.arm(arm_id="a", condition="control", round_index=0, command=["python"],
                    metrics={"common": {"rate_tps": 1.0}}, tokens=[1, 2], token_digest="x")
            run.validation("a", "passed", digest=True)
            run.finish()
            records = read_records(path)
            self.assertEqual([record["type"] for record in records],
                             ["run", "arm", "validation", "summary"])
            self.assertTrue(all(record["schema"] == 1 for record in records))

    def test_invalid_record_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                append_record(Path(directory) / "x.jsonl", {"schema": 1, "type": "nope"})


if __name__ == "__main__":
    unittest.main()
