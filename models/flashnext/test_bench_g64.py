from __future__ import annotations

import unittest
from unittest.mock import patch

from models.flashnext.bench_production import (
    COMPARISONS,
    LOAD_TIME_SETTINGS,
    check_g64_kernel_checkpoint,
    g64_kernel_status,
    inspect_g64_runtime,
)


class G64BenchmarkTests(unittest.TestCase):
    @staticmethod
    def _backend(path="custom-metal", missing=None, slab=False, with_executors=True):
        layers = []
        for index in range(48):
            projection = type("Projection", (), {"slab": object() if slab else None, "group_size": 64})()
            executors = {}
            if with_executors and index != 0 and index != missing:
                executors = {
                    "one": type("Executor", (), {"last_path": path})(),
                    "two": type("Executor", (), {"last_path": path})(),
                }
            switch = type("Switch", (), {
                "metal_runtime_capable": True,
                "gate_proj": projection,
                "up_proj": projection,
                "down_proj": projection,
                "slab_pack": None,
                "_metal_executors": executors,
            })()
            layers.append(type("Layer", (), {
                "mlp": type("MLP", (), {"switch_mlp": switch})(),
            })())
        return type("Backend", (), {
            "language": type("Language", (), {
                "model": type("Model", (), {"layers": layers})(),
            })(),
        })()

    def test_kernel_comparison_is_opt_in_and_pack_free(self):
        comparison = COMPARISONS["g64-kernel"]
        self.assertEqual(set(comparison), {"g64-reference", "g64-metal"})
        for env in comparison.values():
            self.assertEqual(env["FLASHNEXT_METAL_RUNTIME"], "1")
            for key in ("FLASHNEXT_SLAB", "FLASHNEXT_SLAB_GLOBAL", "FLASHNEXT_SLAB_PACK",
                        "FLASHNEXT_SLAB_G64", "FLASHNEXT_STREAM_PACK"):
                self.assertEqual(env[key], "0")
                self.assertIn(key, LOAD_TIME_SETTINGS)
        self.assertEqual(comparison["g64-reference"]["FLASHNEXT_METAL_G64"], "0")
        self.assertEqual(comparison["g64-metal"]["FLASHNEXT_METAL_G64"], "1")
        self.assertIn("FLASHNEXT_METAL_G64", LOAD_TIME_SETTINGS)

    def test_status_reports_verification_until_executor_gate(self):
        with patch("models.flashnext.metal_runtime.G64_RUNTIME_READY", False, create=True):
            self.assertEqual(g64_kernel_status(), "verification")

    def test_non_g64_checkpoint_rejected_before_model_load(self):
        with patch(
            "models.flashnext.bench_chat_parity.checkpoint_runtime_capability",
            return_value={"group_size": 32},
        ):
            with self.assertRaises(SystemExit):
                check_g64_kernel_checkpoint("/checkpoint")

    def test_runtime_state_rejects_wrong_path_and_slab(self):
        with self.assertRaises(SystemExit):
            inspect_g64_runtime(self._backend(path="reference"), True)

    def test_runtime_state_accepts_multiple_executors_per_layer(self):
        backend = self._backend()
        self.assertEqual(
            inspect_g64_runtime(
                self._backend(with_executors=False), True, phase="before"
            )["executor_count"], 0
        )
        self.assertEqual(inspect_g64_runtime(backend, True)["executor_layers"], list(range(1, 48)))

    def test_runtime_state_rejects_missing_layer(self):
        with self.assertRaises(SystemExit):
            inspect_g64_runtime(self._backend(missing=19), True)

    def test_runtime_state_rejects_slab_pack(self):
        with self.assertRaises(SystemExit):
            inspect_g64_runtime(self._backend(slab=True), True)


if __name__ == "__main__":
    unittest.main()
