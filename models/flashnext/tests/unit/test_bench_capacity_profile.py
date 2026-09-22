"""Checkpoint-free regression for the prepared capacity sweep profile."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from models.flashnext.tests.bench import bench_slab_production as bench


class CapacityProfileTests(unittest.TestCase):
    def test_prepared_sweep_resets_each_arm_from_verified_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            profile.write_text('{"ranked_counts": {}}')

            def allocation(budget, min_slots=4):
                return {0: list(range(budget))}

            packs = {}
            for name, budget in zip(bench.CAPACITY_ARMS, (56, 60, 64)):
                path = root / f"pack-{budget}.bin"
                path.write_bytes(b"pack")
                packs[name] = {
                    "path": str(path),
                    "pack_bytes": path.stat().st_size,
                    "allocation_digest": bench.allocation_directory_digest(
                        allocation(budget, 4)
                    ),
                }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "version": 2,
                "routing_profile": {
                    "path": str(profile),
                    "digest": hashlib.sha256(profile.read_bytes()).hexdigest()[:16],
                },
                "packs": packs,
            }))
            arm_profile = root / "arm.json"
            with (
                patch.object(bench, "CAPACITY_MANIFEST", manifest),
                patch.object(bench, "FROZEN_ARM_PINS", arm_profile),
                patch(
                    "models.flashnext.expert_cache.get_skew_slab_allocation",
                    side_effect=allocation,
                ),
                patch.dict(os.environ, {"FLASHNEXT_PIN_CACHE": "missing"}),
            ):
                verified = bench.require_prepared_capacity_packs()
                self.assertEqual(verified, profile)
                self.assertEqual(os.environ["FLASHNEXT_PIN_CACHE"], str(profile.resolve()))

                bench.install_frozen_pins(verified)
                self.assertEqual(arm_profile.read_bytes(), profile.read_bytes())
                arm_profile.write_text("arm changed its private copy")
                bench.install_frozen_pins(verified)
                self.assertEqual(arm_profile.read_bytes(), profile.read_bytes())
                self.assertEqual(os.environ["FLASHNEXT_PIN_CACHE"], str(arm_profile))

    def test_capacity_run_installs_verified_profile_before_first_arm(self):
        class StopBeforeModel(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.json"
            profile.write_text("frozen profile")
            arm_profile = root / "arm.json"
            with (
                patch.object(bench, "require_prepared_capacity_packs", return_value=profile),
                patch.object(bench, "FROZEN_ARM_PINS", arm_profile),
                patch.object(bench, "output_path", return_value=root / "result.json"),
                patch.object(bench, "wait_for_quiescence"),
                patch.object(bench, "run_arm", side_effect=StopBeforeModel) as run_arm,
                patch.object(sys, "argv", ["bench", "--capacity-sweep", "--pairs", "1"]),
                patch.dict(os.environ, {"FLASHNEXT_PIN_CACHE": "missing"}),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(StopBeforeModel):
                    bench.main()
                run_arm.assert_called_once()
                self.assertEqual(arm_profile.read_bytes(), profile.read_bytes())
                self.assertEqual(os.environ["FLASHNEXT_PIN_CACHE"], str(arm_profile))


if __name__ == "__main__":
    unittest.main()
