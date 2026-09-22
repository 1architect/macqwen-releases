"""Synthetic tests for the bounded-memory parity diagnostic.

No checkpoint, no live model, no heavy allocation. Every test stays far
below the 8 GiB child limit the diagnostic enforces.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "models" / "flashnext" / "diagnose_eigenlabs_parity.py"


class LoadSubsetExclusionTests(unittest.TestCase):
    def test_excludes_switch_mlp_and_ngram(self):
        import mlx.core as mx

        sys.path.insert(0, str(REPO))
        from models.flashnext.diagnose_eigenlabs_parity import _load_subset

        requested = []

        class StubStore:
            refs = {
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": SimpleNamespace(shape=(4,)),
                "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shards.0.weight": SimpleNamespace(shape=(4,)),
                "language_model.model.layers.0.self_attn.q_proj.weight": SimpleNamespace(shape=(4,)),
                "language_model.model.layers.0.mlp.gate.weight": SimpleNamespace(shape=(4,)),
            }

            def rows(self, key, indices):
                requested.append(key)
                return mx.zeros((1, 4))

        _load_subset(StubStore(), ("language_model.model.layers.0",))
        self.assertEqual(sorted(requested), [
            "language_model.model.layers.0.mlp.gate.weight",
            "language_model.model.layers.0.self_attn.q_proj.weight",
        ])


class SparseSwitchTests(unittest.TestCase):
    def _fake_bank(self):
        import numpy as np

        bank = {}
        specs = {"gate_proj": 3, "up_proj": 5, "down_proj": 7,
                 "gate_scales": 11, "up_scales": 13, "down_scales": 17,
                 "gate_biases": 19, "up_biases": 23, "down_biases": 29}
        for name, base in specs.items():
            if "scales" in name or "biases" in name:
                shape = (4, 64, 2)
            else:
                shape = (4, 64, 8)
            count = 1
            for dim in shape:
                count *= dim
            bank[name] = (np.arange(count).reshape(shape) + base) % 200
        return bank

    def _as_mx_bank(self, raw):
        import mlx.core as mx
        import numpy as np

        bank = {}
        for name, values in raw.items():
            if "scales" in name or "biases" in name:
                bank[name] = mx.array(
                    (values % 64).astype(np.float32)).astype(mx.bfloat16)
            else:
                bank[name] = mx.array(values.astype(np.uint32))
        return bank

    def test_reads_only_requested_experts_and_matches_naive(self):
        import mlx.core as mx
        import mlx.nn as nn

        sys.path.insert(0, str(REPO))
        from models.flashnext.diagnose_eigenlabs_parity import SparseStockSwitchGLU
        from mlx_vlm.models.activations import swiglu

        bank = self._as_mx_bank(self._fake_bank())
        fetched = []

        class StubStore:
            def rows(self, key, indices):
                fetched.append((key.split(".")[-2], list(indices)))
                part = key.split(".")[-1]
                table = {"weight": 0, "scales": 1, "biases": 2}[part]
                proj = key.split(".")[-2]
                mapping = {"gate_proj": ("gate_proj", "gate_scales", "gate_biases"),
                           "up_proj": ("up_proj", "up_scales", "up_biases"),
                           "down_proj": ("down_proj", "down_scales", "down_biases")}[proj]
                full = bank[mapping[table]]
                return mx.stack([full[e] for e in indices])

        activation = lambda u, g: swiglu(g, u)
        switch = SparseStockSwitchGLU(
            StubStore(), "prefix", 32, 4, "affine", 4, 64, 64, activation)
        x = (mx.random.uniform(shape=(1, 2, 64)).astype(mx.bfloat16) - 0.5) * 0.2
        mx.eval(x)
        indices = mx.array([[[0, 2, 1], [1, 0, 2]]], dtype=mx.uint32)
        got = switch(x, indices)
        mx.eval(got)
        seen = {proj for proj, _ in fetched}
        self.assertEqual(seen, {"gate_proj", "up_proj", "down_proj"})
        experts = sorted({e for _, rows in fetched for e in rows})
        self.assertEqual(experts, [0, 1, 2])
        self.assertEqual(tuple(got.shape), (1, 2, 3, 64))

        def expert_linears(expert):
            mods = []
            for proj, out_dims in (("gate_proj", 64), ("up_proj", 64), ("down_proj", 64)):
                module = nn.QuantizedLinear(64, out_dims, bias=False,
                                            group_size=32, bits=4, mode="affine")
                module.weight = bank[proj][expert]
                module.scales = bank[proj.replace("proj", "scales")][expert]
                module.biases = bank[proj.replace("proj", "biases")][expert]
                mods.append(module)
            return mods

        flat = x.reshape(-1, 64)
        host = [0, 2, 1, 1, 0, 2]
        rows = []
        slots_per_token = 3
        for slot, expert in enumerate(host):
            gate, up, down = expert_linears(expert)
            token = flat[[slot // slots_per_token]]
            rows.append(down(activation(up(token), gate(token)))[0])
        expect = mx.stack(rows).reshape(1, 2, 3, 64)
        mx.eval(expect)
        self.assertTrue(bool(mx.array_equal(got, expect).item()))


class VerifyRowsNoFullLoadTests(unittest.TestCase):
    def _write_shard(self, root: Path):
        header_entries = {}

        def add(name, dtype, shape, payload: bytes):
            start = sum(len(v) for v in blobs) if (blobs := add.buffers) else 0
            add.buffers.append(payload)
            header_entries[name] = {"dtype": dtype, "shape": list(shape),
                                    "data_offsets": [start, start + len(payload)]}

        add.buffers = []
        import numpy as np

        for part, code in (("weight", "<u4"), ("scales", "<u2"), ("biases", "<u2")):
            arr = (np.arange(2 * 8 * 4).reshape(2, 8, 4) + len(part)).astype(code)
            add(f"language_model.model.layers.0.mlp.switch_mlp.gate_proj.{part}",
                "U32" if part == "weight" else "BF16", (2, 8, 4), arr.tobytes())
        header = json.dumps(header_entries).encode()
        with open(root / "model-00001-of-00001.safetensors", "wb") as handle:
            handle.write(struct.pack("<Q", len(header)))
            handle.write(header)
            for buffer in add.buffers:
                handle.write(buffer)
        index = {"weight_map": {
            key: "model-00001-of-00001.safetensors" for key in header_entries}}
        (root / "model.safetensors.index.json").write_text(json.dumps(index))

    def test_verify_rows_passes_without_mx_load(self):
        sys.path.insert(0, str(REPO))
        import mlx.core as mx
        from models.flashnext.diagnose_eigenlabs_parity import cmd_verify_rows

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_shard(root)

            def forbidden(_path):
                raise AssertionError("mx.load must not run in verify-rows")

            with mock.patch.object(mx, "load", forbidden):
                result = cmd_verify_rows(SimpleNamespace(
                    model=str(root), out=str(root), layer="0"))
        self.assertTrue(result)


class WatchdogTests(unittest.TestCase):
    def test_over_limit_child_is_terminated(self):
        sys.path.insert(0, str(REPO))
        from models.flashnext import diagnose_eigenlabs_parity as diag

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            result = diag._enforce_child_limit(child, "probe", 1e-9, 30.0)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
        self.assertTrue(result.killed_for_memory)
        self.assertEqual(result.returncode, 128 + 15)
        self.assertIsNotNone(child.poll())


class ModeDispatchTests(unittest.TestCase):
    def test_split_end_modes_exist_and_ends_removed(self):
        sys.path.insert(0, str(REPO))
        from models.flashnext import diagnose_eigenlabs_parity as diag

        for mode in ("dump-stock-embedding", "dump-stock-final", "dump-stock-head"):
            self.assertIn(mode, diag.MODES)
        self.assertNotIn("dump-stock-ends", diag.MODES)


class IsolationTests(unittest.TestCase):
    def test_stock_paths_import_no_macqwen_runtime(self):
        script = (
            "import sys; sys.path.insert(0, %r); "
            "from models.flashnext.diagnose_eigenlabs_parity import "
            "_stock_predicate, _module_overrides, SparseStockSwitchGLU; "
            "banned = [m for m in sys.modules if m in ("
            "'models.flashnext.loader', 'models.flashnext.adaptive_topk', "
            "'models.flashnext.patch_rmsnorm')]; "
            "assert not banned, banned; print('isolated OK')"
        ) % str(REPO)
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("isolated OK", proc.stdout)

    def test_production_sources_ignore_diagnostic(self):
        for relative in ("macqwen/backends/flashnext.py",
                         "models/flashnext/loader.py",
                         "models/flashnext/expert_cache.py"):
            text = (REPO / relative).read_text()
            self.assertNotIn("diagnose_eigenlabs", text)


if __name__ == "__main__":
    unittest.main()
