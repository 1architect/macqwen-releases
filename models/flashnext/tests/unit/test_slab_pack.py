import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from models.flashnext.slab_pack import (
    HEADER_MAGIC,
    HEADER_VERSION,
    HEADER_SIZE,
    RECORD_STRIDE,
    GATE_WEIGHT_OFFSET,
    GATE_SCALES_OFFSET,
    GATE_BIASES_OFFSET,
    UP_WEIGHT_OFFSET,
    UP_SCALES_OFFSET,
    UP_BIASES_OFFSET,
    DOWN_WEIGHT_OFFSET,
    DOWN_SCALES_OFFSET,
    DOWN_BIASES_OFFSET,
    build_slab_pack,
    SlabPack,
    get_or_create_slab_pack,
    libc_mlock,
    libc_munlock,
)


class MockStore:
    def __init__(self):
        self.dir = "/mock/model"
        self._data = {}

    def rows_np(self, name: str, indices: list[int]) -> np.ndarray:
        # Determine expected size from name
        if "scales" in name or "biases" in name:
            shape = (len(indices), 102400 // 2)
            dtype = np.uint16  # bfloat16 raw bits
        else:
            shape = (len(indices), 819200 // 4)
            dtype = np.uint32

        if name not in self._data:
            rng = np.random.default_rng(42)
            self._data[name] = rng.integers(0, 65535, size=shape, dtype=dtype)
        return self._data[name]


class TestSlabPack(unittest.TestCase):
    def test_offsets_and_stride(self):
        self.assertEqual(RECORD_STRIDE, 3072000)
        self.assertEqual(RECORD_STRIDE % 4096, 0)
        self.assertEqual(HEADER_SIZE, 4096)
        self.assertEqual(GATE_WEIGHT_OFFSET, 0)
        self.assertEqual(GATE_SCALES_OFFSET, 819200)
        self.assertEqual(GATE_BIASES_OFFSET, 921600)
        self.assertEqual(UP_WEIGHT_OFFSET, 1024000)
        self.assertEqual(UP_SCALES_OFFSET, 1843200)
        self.assertEqual(UP_BIASES_OFFSET, 1945600)
        self.assertEqual(DOWN_WEIGHT_OFFSET, 2048000)
        self.assertEqual(DOWN_SCALES_OFFSET, 2867200)
        self.assertEqual(DOWN_BIASES_OFFSET, 2969600)
        self.assertEqual(DOWN_BIASES_OFFSET + 102400, RECORD_STRIDE)

    def test_build_and_open_pack(self):
        store = MockStore()
        allocation = {
            5: [10, 20],
            11: [30, 40],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_slab.bin"
            total_size = build_slab_pack(store, allocation, out_file)
            expected_size = HEADER_SIZE + 4 * RECORD_STRIDE
            self.assertEqual(total_size, expected_size)
            self.assertEqual(out_file.stat().st_size, expected_size)

            # Open with SlabPack
            pack = SlabPack(out_file, lock_memory=False)
            self.assertEqual(pack.expert_count, 4)
            self.assertEqual(pack.layer_to_base_slot, {5: 0, 11: 2})
            self.assertEqual(pack.layer_expert_to_slot[(5, 10)], 0)
            self.assertEqual(pack.layer_expert_to_slot[(5, 20)], 1)
            self.assertEqual(pack.layer_expert_to_slot[(11, 30)], 2)
            self.assertEqual(pack.layer_expert_to_slot[(11, 40)], 3)

            # Verify byte values for (5, 10)
            expected_gate_w = store.rows_np("language_model.model.layers.5.mlp.switch_mlp.gate_proj.weight", [10])
            read_bytes = pack._mm[HEADER_SIZE + GATE_WEIGHT_OFFSET : HEADER_SIZE + GATE_SCALES_OFFSET]
            self.assertEqual(read_bytes, expected_gate_w.tobytes())

            pack.close()


if __name__ == "__main__":
    unittest.main()
