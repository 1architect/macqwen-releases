"""The locked expert working set stays within budget and never alters bytes."""
from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from models.flashnext.store import SafeTensorStore

NAME = "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"
ROWS = 8
# 64 KiB per row: four 16 KiB pages, so rows never share a page.
COLUMNS = 16384


def write_checkpoint(directory: str) -> np.ndarray:
    data = (np.arange(ROWS * COLUMNS, dtype=np.uint32) * 2654435761).reshape(ROWS, COLUMNS)
    payload = data.tobytes()
    header = {NAME: {"dtype": "U32", "shape": [ROWS, COLUMNS], "data_offsets": [0, len(payload)]}}
    blob = json.dumps(header).encode()
    # Pad the header so the tensor starts on a page boundary.
    pad = (-(8 + len(blob))) % 16384
    blob += b" " * pad
    with open(os.path.join(directory, "model-00001-of-00001.safetensors"), "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        out.write(payload)
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
        json.dump({"weight_map": {NAME: "model-00001-of-00001.safetensors"}}, out)
    return data


class ExpertLockTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.expected = write_checkpoint(self._dir.name)
        self.row_bytes = COLUMNS * 4

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def build(self, rows_in_budget: float):
        budget_gb = rows_in_budget * self.row_bytes / 1e9
        with mock.patch.dict(os.environ, {"FLASHNEXT_EXPERT_LOCK_GB": str(budget_gb)}):
            self.store = SafeTensorStore(self._dir.name)
        return self.store

    def read(self, rows):
        out = self.store.rows_np(NAME, list(rows), "pread")
        self.store.drain_expert_locks()
        return out

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLASHNEXT_EXPERT_LOCK_GB", None)
            self.store = SafeTensorStore(self._dir.name)
        self.read([0, 1])
        self.assertEqual(self.store.expert_lock_stats()["locked_rows"], 0)

    def test_budget_is_respected_and_oldest_rows_leave_first(self):
        store = self.build(3)
        self.read([0, 1, 2])
        self.read([0])        # refresh row 0
        self.read([3])        # evicts row 1, the least recently read
        self.assertEqual(
            [row for _name, row in store._locked], [2, 0, 3]
        )
        stats = store.expert_lock_stats()
        self.assertLessEqual(stats["locked_bytes"], stats["budget_bytes"])

    def test_locking_never_changes_the_bytes(self):
        self.build(2)
        for _ in range(3):
            np.testing.assert_array_equal(self.read([5, 1, 7]), self.expected[[5, 1, 7]])

    def test_pinned_rows_are_left_to_the_pin(self):
        store = self.build(4)
        store.pin_rows(NAME, [2])
        self.read([2, 3])
        self.assertNotIn((NAME, 2), store._locked)
        self.assertIn((NAME, 3), store._locked)

    def test_unpin_forgets_rows_that_lost_their_lock(self):
        store = self.build(4)
        self.read([4])
        store.pin_rows(NAME, [4])
        store.unpin_all()
        self.assertNotIn((NAME, 4), store._locked)
        self.read([4])
        self.assertIn((NAME, 4), store._locked)

    def test_close_releases_every_lock(self):
        store = self.build(4)
        self.read([0, 1])
        store.close()
        self.assertEqual(store.expert_lock_stats()["locked_bytes"], 0)
        self.store = SafeTensorStore(self._dir.name)


if __name__ == "__main__":
    unittest.main()
