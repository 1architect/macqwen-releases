import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.flashnext.expert_cache import (
    _GLOBAL_SLAB_CACHE,
    get_global_slab_allocation,
    get_skew_slab_allocation,
)
from models.flashnext.slab_pack import build_slab_pack, SlabPack, HEADER_SIZE, RECORD_STRIDE
from models.flashnext.test_slab_pack import MockStore


class TestSkewAllocation(unittest.TestCase):
    def setUp(self):
        _GLOBAL_SLAB_CACHE.clear()

    def test_skew_allocation_48_slots(self):
        alloc = get_skew_slab_allocation(48, min_slots=4, max_slots=6, num_layers=12)
        if not alloc:
            self.skipTest("pins.json not present")
        self.assertEqual(len(alloc), 12)
        self.assertEqual(sum(len(v) for v in alloc.values()), 48)
        for l, exps in alloc.items():
            self.assertEqual(len(exps), 4)

    def test_skew_allocation_56_slots(self):
        alloc = get_skew_slab_allocation(56, min_slots=4, max_slots=6, num_layers=12)
        if not alloc:
            self.skipTest("pins.json not present")
        self.assertEqual(len(alloc), 12)
        self.assertEqual(sum(len(v) for v in alloc.values()), 56)
        for l, exps in alloc.items():
            self.assertGreaterEqual(len(exps), 4)
            self.assertLessEqual(len(exps), 6)

    def test_skew_allocation_fallback_uniform(self):
        # Empty pin file should fallback cleanly to empty dict
        with patch.dict(os.environ, {"FLASHNEXT_PIN_CACHE": "/nonexistent/path/pins.json"}):
            alloc = get_skew_slab_allocation(56)
            self.assertEqual(alloc, {})

    def test_slab_pack_56_slots_build(self):
        store = MockStore()
        # 8 layers * 4 + 4 layers * 6 = 56 slots
        alloc = {layer: list(range(4)) for layer in range(8)}
        for l in (20, 31, 35, 47):
            alloc[l] = list(range(6))
        total_slots = sum(len(v) for v in alloc.values())
        self.assertEqual(total_slots, 56)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_slab_56.bin"
            total_size = build_slab_pack(store, alloc, out_file)
            expected_size = HEADER_SIZE + 56 * RECORD_STRIDE
            self.assertEqual(total_size, expected_size)
            self.assertEqual(expected_size, 172036096)
            self.assertEqual(out_file.stat().st_size, expected_size)

            pack = SlabPack(out_file, lock_memory=False)
            self.assertEqual(pack.expert_count, 56)
            self.assertEqual(len(pack.layer_expert_to_slot), 56)
            pack.close()


if __name__ == "__main__":
    unittest.main()
