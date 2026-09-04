"""Pure-Python registry checks. These tests do not load MLX or a model."""
from __future__ import annotations

import os
import ast
from pathlib import Path
import re
import sys
import unittest
from types import SimpleNamespace

from macqwen.backend_settings import Setting, SettingRegistry
from .registry import get_registry


class SettingsRegistryTests(unittest.TestCase):
    def test_every_production_flashnext_read_is_registered_or_allowlisted(self):
        registered = {
            setting.env_key
            for setting in get_registry().for_backend("flashnext")
            if setting.env_key
        }
        # Internal storage, kernel, prefill, and speculative probes stay
        # explicit until a user-facing setting provider owns them.
        allowlisted_internal = {
            # Storage and allocator diagnostics.
            "FLASHNEXT_DLPACK", "FLASHNEXT_F_NOCACHE", "FLASHNEXT_RDAHEAD",
            "FLASHNEXT_MMAP_ADVICE", "FLASHNEXT_NGRAM_DONTNEED",
            "FLASHNEXT_HYBRID_CUTOFF", "FLASHNEXT_RESIDENT_ROWS",
            "FLASHNEXT_BULK", "FLASHNEXT_CHUNK", "FLASHNEXT_PIN_CACHE",
            "FLASHNEXT_HOST_WINDOW",
            # Routing and profile internals.
            "FLASHNEXT_RENORM", "FLASHNEXT_RENORM_BLEND", "FLASHNEXT_TOPK_THRESHOLD",
            "FLASHNEXT_FAST_THRESHOLD", "FLASHNEXT_FAST_SENSITIVE",
            "FLASHNEXT_FAST_PROTECTED", "FLASHNEXT_FAST_PROTECTED_THRESHOLD",
            "FLASHNEXT_FAST_TOP_COUNT", "FLASHNEXT_FAST_TOP_THRESHOLD",
            "FLASHNEXT_FAST_TOP_MIN_KEEP", "FLASHNEXT_FAST_MID_COUNT",
            "FLASHNEXT_FAST_MID_THRESHOLD", "FLASHNEXT_FAST_MIN_KEEP",
            "FLASHNEXT_FAST_SENSITIVE_MIN_KEEP", "FLASHNEXT_FAST_MID_MIN_KEEP",
            "FLASHNEXT_FAST_RENORM", "FLASHNEXT_OVERLAP",
            "FLASHNEXT_MTP_THRESHOLD",
            # Prefill and model-shape controls.
            "FLASHNEXT_COMPILE_NORM", "FLASHNEXT_NGRAM_DIRECT",
            "FLASHNEXT_QSA_CHUNK_THRESHOLD", "FLASHNEXT_QSA_QUERY_CHUNK",
            "FLASHNEXT_PREFILL_FULL_LOGITS_MAX_TOKENS", "FLASHNEXT_PREFILL_RELEASE_BYTES",
            "FLASHNEXT_PREFILL_CLEAR_CACHE", "FLASHNEXT_METAL_VERIFY",
            # Speculative and tracing diagnostics.
            "FLASHNEXT_DRAFT_MIN_MARGIN", "FLASHNEXT_DRAFT_FUSED_ARGMAX",
            "FLASHNEXT_SPEC_TRACE", "FLASHNEXT_PROFILE_SCORE_SYNC",
        }
        found = set()
        root = Path(__file__).resolve().parents[1]

        class EnvironmentReads(ast.NodeVisitor):
            def visit_Call(self, node):
                function = node.func
                is_get = (
                    isinstance(function, ast.Attribute)
                    and function.attr == "get"
                    and isinstance(function.value, ast.Attribute)
                    and isinstance(function.value.value, ast.Name)
                    and function.value.value.id == "os"
                    and function.value.attr == "environ"
                )
                is_getenv = (
                    isinstance(function, ast.Attribute)
                    and function.attr == "getenv"
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "os"
                )
                if (is_get or is_getenv) and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        found.update(re.findall(r"FLASHNEXT_[A-Z0-9_]+", argument.value))
                self.generic_visit(node)

            def visit_Subscript(self, node):
                value = node.value
                is_environ = (
                    isinstance(value, ast.Attribute)
                    and value.attr == "environ"
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "os"
                )
                key = node.slice
                if is_environ and isinstance(key, ast.Constant) and isinstance(key.value, str):
                    found.update(re.findall(r"FLASHNEXT_[A-Z0-9_]+", key.value))
                self.generic_visit(node)

        for path in root.rglob("*.py"):
            relative = path.relative_to(root)
            if any(part in {"settings", "tests"} for part in relative.parts):
                continue
            if path.name.startswith(("test_", "bench_")) or path.name in {
                "capture_dispatches.py", "metal_trace.py",
            }:
                continue
            EnvironmentReads().visit(ast.parse(path.read_text(), filename=str(path)))
        missing = found - registered - allowlisted_internal
        self.assertFalse(missing, sorted(missing))

    def test_discovery_does_not_import_mlx(self):
        before = "mlx.core" in sys.modules
        registry = get_registry()
        self.assertTrue(registry.settings)
        self.assertEqual(before, "mlx.core" in sys.modules)

    def test_aliases_and_defaults(self):
        registry = get_registry()
        self.assertIs(registry.get("flashnext", "mode"), registry.get("flashnext", "routing"))
        defaults = registry.defaults("flashnext")
        self.assertEqual(defaults["routing"], "exact-quality")
        self.assertEqual(defaults["swap-epsilon"], 0.02)

    def test_production_environment_controls_are_registered(self):
        registry = get_registry()
        keys = {setting.env_key for setting in registry.for_backend("flashnext")}
        required = {
            "FLASHNEXT_METAL_RUNTIME", "FLASHNEXT_SLAB_GLOBAL", "FLASHNEXT_SLAB_PACK",
            "FLASHNEXT_SLAB_POLICY", "FLASHNEXT_FUSED_SHARED", "FLASHNEXT_FUSED_SHARED_PARTS",
            "FLASHNEXT_FUSED_UP_SWIGLU", "FLASHNEXT_STREAM_PACK", "FLASHNEXT_PREAD_CHUNK",
            "FLASHNEXT_IO_WORKERS", "FLASHNEXT_READ",
        }
        self.assertTrue(required <= keys)

    def test_render_reports_environment_source_and_active_state(self):
        registry = get_registry()
        os.environ["FLASHNEXT_FUSED_UP_SWIGLU"] = "1"
        backend = SimpleNamespace(
            routing_profile="exact-quality", threshold=0.85,
            swap_epsilon=0.02, resident_experts=32, pin_budget_gb=6.0,
            tail_experts=6, tail_warmup=8, fusion_block=23,
            fusion_min_margin=1.0, fusion_min_block=20,
            fusion_margin_tokens=8, fusion_max_prompt=512,
            fusion_model="", routing=SimpleNamespace(pinned={}), model_path="",
        )
        try:
            text = registry.render(backend, "flashnext")
        finally:
            os.environ.pop("FLASHNEXT_FUSED_UP_SWIGLU", None)
        self.assertIn("fused-up-swiglu", text)
        self.assertIn("environment", text)

    def test_duplicate_aliases_fail(self):
        setting = Setting("one", ("alias",), 1, int, "live", "test", "public", "test", lambda _: 1)
        duplicate = Setting("two", ("alias",), 2, int, "live", "test", "public", "test", lambda _: 2)
        with self.assertRaises(ValueError):
            SettingRegistry((setting, duplicate))


if __name__ == "__main__":
    unittest.main()
