import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import numpy as np

from models.flashnext.slab_pack import (
    HEADER_MAGIC,
    HEADER_SIZE,
    HEADER_VERSION_G64,
    Q4G32_LAYOUT,
    Q4G64_LAYOUT,
    SlabPack,
    build_slab_pack,
    get_slab_pack_cache_path,
    validate_slab_allocation,
)


class _G64Store:
    dir = "/mock/g64"

    def __init__(self, dtype_ok=True):
        self.refs = {}
        self.dtype_ok = dtype_ok

    def shape(self, name):
        projection, part = name.rsplit(".", 1)
        projection = projection.rsplit(".", 1)[-1]
        return ((288, 640, 320) if projection in ("gate_proj", "up_proj") else (288, 2560, 80)) if part == "weight" else ((288, 640, 40) if projection in ("gate_proj", "up_proj") else (288, 2560, 10))

    def rows_np(self, name, indices):
        shape = self.shape(name)[1:]
        dtype = np.uint32 if name.endswith(".weight") else np.uint16
        marker = sum((index + 1) * ord(char) for index, char in enumerate(name)) % 251
        values = np.arange(
            len(indices) * int(np.prod(shape)), dtype=dtype
        ).reshape((len(indices),) + shape)
        return values + np.asarray(marker, dtype=dtype)


class _G32Store(_G64Store):
    def shape(self, name):
        projection, part = name.rsplit(".", 1)
        projection = projection.rsplit(".", 1)[-1]
        if part == "weight":
            tail = (640, 320) if projection in ("gate_proj", "up_proj") else (2560, 80)
        else:
            tail = (640, 80) if projection in ("gate_proj", "up_proj") else (2560, 20)
        return (288,) + tail


class SlabPackG64Tests(unittest.TestCase):
    def test_descriptor_has_exact_offsets_and_shapes(self):
        self.assertEqual(Q4G32_LAYOUT.record_stride, 3072000)
        self.assertEqual(Q4G64_LAYOUT.record_stride, 2764800)
        self.assertEqual(Q4G64_LAYOUT.record_stride, 675 * 4096)
        self.assertEqual(Q4G64_LAYOUT.shape("gate_proj", "scales"), (640, 40))
        self.assertEqual(Q4G64_LAYOUT.shape("down_proj", "biases"), (2560, 10))
        self.assertEqual(Q4G64_LAYOUT.offset("gate_proj", "biases"), 870400)
        self.assertEqual(Q4G64_LAYOUT.offset("up_proj", "weight"), 921600)
        self.assertEqual(Q4G64_LAYOUT.offset("down_proj", "biases"), 2713600)

    def test_unknown_group_size_is_rejected(self):
        from models.flashnext.slab_pack import get_slab_layout
        with self.assertRaises(ValueError):
            get_slab_layout(128)

    def test_validation_rejects_g32_metadata_for_g64(self):
        store = _G64Store()
        allocation = {2: [7]}
        self.assertTrue(validate_slab_allocation(store, allocation, layout=Q4G64_LAYOUT))
        self.assertFalse(validate_slab_allocation(store, allocation, layout=Q4G32_LAYOUT))

    def test_validation_rejects_wrong_dtype(self):
        store = _G64Store()
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for part in ("weight", "scales", "biases"):
                name = f"language_model.model.layers.2.mlp.switch_mlp.{projection}.{part}"
                store.refs[name] = SimpleNamespace(dtype="F32")
        self.assertFalse(validate_slab_allocation(store, {2: [7]}, layout=Q4G64_LAYOUT))

    def test_build_writes_g64_header_and_component_offsets(self):
        store = _G64Store()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g64.bin"
            self.assertEqual(
                build_slab_pack(store, {2: [7]}, path, layout=Q4G64_LAYOUT),
                HEADER_SIZE + Q4G64_LAYOUT.record_stride,
            )
            raw = path.read_bytes()
            magic, version, count, stride, header_size, _model, layout_id = __import__("struct").unpack_from("<IIIII8sI", raw)
            self.assertEqual((magic, version, count, stride, header_size, layout_id), (HEADER_MAGIC, HEADER_VERSION_G64, 1, Q4G64_LAYOUT.record_stride, HEADER_SIZE, 64))
            record = raw[HEADER_SIZE:]
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for part in ("weight", "scales", "biases"):
                    expected = store.rows_np(
                        f"language_model.model.layers.2.mlp.switch_mlp.{projection}.{part}",
                        [7],
                    ).tobytes()
                    offset = Q4G64_LAYOUT.offset(projection, part)
                    self.assertEqual(record[offset:offset + len(expected)], expected)

    def test_g32_default_header_and_cache_key_stay_compatible(self):
        store = _G32Store()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g32.bin"
            build_slab_pack(store, {2: [7]}, path)
            import struct
            header = struct.unpack_from("<IIIII8sI", path.read_bytes())
            self.assertEqual(header[1], 1)
            self.assertEqual(header[3], Q4G32_LAYOUT.record_stride)
            implicit = get_slab_pack_cache_path("/mock/model", {2: [7]}, tmp, model_identity="a")
            explicit = get_slab_pack_cache_path("/mock/model", {2: [7]}, tmp, model_identity="a", layout=Q4G32_LAYOUT)
            self.assertEqual(implicit, explicit)

    def test_open_detects_g64_and_rejects_g32_expectation(self):
        store = _G64Store()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g64.bin"
            build_slab_pack(store, {2: [7]}, path, layout=Q4G64_LAYOUT)
            fake_core = types.SimpleNamespace(from_dlpack=lambda value: value)
            fake_mlx = types.ModuleType("mlx")
            fake_mlx.core = fake_core
            with mock.patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}):
                pack = SlabPack(path, lock_memory=False)
                self.assertIs(pack.layout, Q4G64_LAYOUT)
                pack.close()
                with self.assertRaises(ValueError):
                    SlabPack(path, lock_memory=False, expected_layout=Q4G32_LAYOUT)

    def test_cache_paths_do_not_cross_layouts(self):
        allocation = {2: [7]}
        g32 = get_slab_pack_cache_path("/mock/model", allocation, "/tmp/cache", model_identity="a", layout=Q4G32_LAYOUT)
        g64 = get_slab_pack_cache_path("/mock/model", allocation, "/tmp/cache", model_identity="a", layout=Q4G64_LAYOUT)
        self.assertNotEqual(g32, g64)


if __name__ == "__main__":
    unittest.main()
