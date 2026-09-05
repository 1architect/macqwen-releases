"""Every read mode must return the same bytes.

`resident` maps rows held by `mlock` and reads the rest. A wrong gate would
return the right shape with the wrong contents, which no timing test catches.
"""
from __future__ import annotations

import json
import os
import subprocess
import struct
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
import unittest.mock

import numpy as np

from models.flashnext.store import (
    SafeTensorStore,
    begin_read_profile,
    finish_read_profile,
)

ROWS = 12
COLUMNS = 64


def write_checkpoint(directory: str) -> np.ndarray:
    data = (np.arange(ROWS * COLUMNS, dtype=np.uint32) * 7919).reshape(ROWS, COLUMNS)
    payload = data.tobytes()
    header = {
        "block.experts.weight": {
            "dtype": "U32",
            "shape": [ROWS, COLUMNS],
            "data_offsets": [0, len(payload)],
        }
    }
    blob = json.dumps(header).encode()
    with open(os.path.join(directory, "model-00001-of-00001.safetensors"), "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        out.write(payload)
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
        json.dump(
            {"weight_map": {"block.experts.weight": "model-00001-of-00001.safetensors"}},
            out,
        )
    return data


class ReadModeTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.expected = write_checkpoint(self._dir.name)
        self.store = SafeTensorStore(self._dir.name)
        self.name = "block.experts.weight"

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def rows(self, indices, mode):
        return self.store.rows_np(self.name, list(indices), mode)

    def test_resident_matches_pread_with_nothing_pinned(self):
        wanted = [0, 3, 7, 11]
        np.testing.assert_array_equal(
            self.rows(wanted, "resident"), self.expected[wanted]
        )

    def test_resident_matches_pread_when_every_row_is_pinned(self):
        wanted = [1, 4, 9]
        self.store.pin_rows(self.name, wanted)
        np.testing.assert_array_equal(
            self.rows(wanted, "resident"), self.expected[wanted]
        )

    def test_resident_mixes_pinned_and_unpinned_rows(self):
        self.store.pin_rows(self.name, [2, 5])
        wanted = [5, 0, 2, 8]
        np.testing.assert_array_equal(
            self.rows(wanted, "resident"), self.expected[wanted]
        )

    def test_unpinning_sends_every_row_back_to_the_read_path(self):
        self.store.pin_rows(self.name, [2, 5])
        self.store.unpin_all()
        self.assertEqual(self.store._pinned_rows, set())
        np.testing.assert_array_equal(
            self.rows([2, 5], "resident"), self.expected[[2, 5]]
        )

    def test_pin_size_matches_the_locked_byte_count(self):
        wanted = [1, 4, 9]
        expected = self.store.pin_size(self.name, wanted)
        self.assertEqual(self.store.pin_rows(self.name, wanted), expected)

    def test_pin_size_does_not_open_a_shared_mapping(self):
        with unittest.mock.patch.object(
            self.store, "_shared_view", side_effect=AssertionError("mapped")
        ):
            size = self.store.pin_size(self.name, [1, 4, 9])
        self.assertGreater(size, 0)

    def test_every_mode_agrees(self):
        wanted = [11, 0, 6]
        self.store.pin_rows(self.name, [6])
        reference = self.expected[wanted]
        for mode in ("pread", "preadv", "shared_mmap", "resident"):
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(self.rows(wanted, mode), reference)

    def test_rows_into_agrees_with_rows_np(self):
        wanted = [3, 6, 1]
        self.store.pin_rows(self.name, [6, 1])
        out = self.store.empty_rows(self.name, len(wanted))
        self.store.rows_into(self.name, wanted, out, "resident")
        np.testing.assert_array_equal(out, self.expected[wanted])

    def test_read_profile_counts_pread_service(self):
        wanted = [3, 6, 1]
        begin_read_profile()
        actual = self.rows(wanted, "pread")
        profile = finish_read_profile()

        np.testing.assert_array_equal(actual, self.expected[wanted])
        self.assertEqual(profile["pread_calls"], len(wanted))
        self.assertEqual(profile["pread_bytes"], actual.nbytes)
        self.assertEqual(len(profile["pread_intervals"]), len(wanted))
        self.assertTrue(all(end >= start for start, end in profile["pread_intervals"]))

    def test_close_can_run_twice_after_positioned_reads(self):
        self.rows([1], "pread")
        self.store.close()
        self.store.close()

    def test_shared_map_creation_is_singleton_under_concurrency(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            maps = list(pool.map(lambda _item: self.store._map(
                "model-00001-of-00001.safetensors"
            ), range(32)))
        self.assertEqual(len({id(handle) for handle in maps}), 1)

    def test_expert_record_view_writes_strided_rows(self):
        wanted = [3, 6, 1]
        record_stride = self.expected[0].nbytes + 128
        offset = 32
        buffer = np.zeros(len(wanted) * record_stride, dtype=np.uint8)
        view = self.store.expert_record_view(
            self.name, buffer, len(wanted), offset, record_stride
        )

        self.store.rows_into(self.name, wanted, view, "pread")

        np.testing.assert_array_equal(view, self.expected[wanted])
        self.assertTrue(np.all(buffer[:offset] == 0))

    def test_many_shards_fit_a_low_descriptor_limit(self):
        script = r'''
import json, os, resource, struct, sys
from models.flashnext.store import SafeTensorStore

directory = sys.argv[1]
shards = []
weight_map = {}
payload = b"x"
for index in range(131):
    shard = f"model-{index + 1:05d}-of-00131.safetensors"
    name = f"block.{index}.experts.weight"
    header = json.dumps({name: {"dtype": "U8", "shape": [1],
                                "data_offsets": [0, 1]}}).encode()
    with open(os.path.join(directory, shard), "wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.write(payload)
    shards.append(shard)
    weight_map[name] = shard
with open(os.path.join(directory, "model.safetensors.index.json"), "w") as stream:
    json.dump({"weight_map": weight_map}, stream)

# Leave room for the descriptors held by a live model process.
resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
reserved = [open(os.devnull, "rb") for _ in range(16)]
store = SafeTensorStore(directory)
names = list(weight_map)
for name in names:
    assert store.pin_size(name, [0]) == __import__("mmap").PAGESIZE
for name in names[:96]:
    assert int(store.rows_np(name, [0], "shared_mmap")[0]) == 120, name
for name in names:
    assert int(store.rows_np(name, [0], "pread")[0]) == 120, name
store.close()
for stream in reserved:
    stream.close()
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", script, directory],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()


class ResidencyTrackerTests(unittest.TestCase):
    """The gate must be cheap and honest about what it does not know."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.expected = write_checkpoint(self._dir.name)
        self.name = "block.experts.weight"

    def tearDown(self):
        self.store.close()
        self._dir.cleanup()

    def build(self, tracking: str, cap: str = "4"):
        with unittest.mock.patch.dict(
            "os.environ",
            {"FLASHNEXT_TRACK_RESIDENT": tracking, "FLASHNEXT_RESIDENT_ROWS": cap},
        ):
            self.store = SafeTensorStore(self._dir.name)
        return self.store

    def test_untracked_claims_only_pinned_rows(self):
        store = self.build("0")
        store.pin_rows(self.name, [2])
        self.assertTrue(store.believed_resident(self.name, 2))
        store.rows_np(self.name, [5], "resident")
        self.assertFalse(store.believed_resident(self.name, 5))

    def test_tracking_claims_a_row_it_has_read(self):
        store = self.build("1")
        self.assertFalse(store.believed_resident(self.name, 5))
        store.rows_np(self.name, [5], "resident")
        self.assertTrue(store.believed_resident(self.name, 5))

    def test_tracking_can_be_enabled_after_the_store_opens(self):
        store = self.build("0")
        store.set_residency_tracking(True)
        store.rows_np(self.name, [5], "resident")
        self.assertTrue(store.believed_resident(self.name, 5))

    def test_the_lru_forgets_the_least_recently_used(self):
        store = self.build("1", cap="3")
        for row in (0, 1, 2):
            store.rows_np(self.name, [row], "resident")
        store.believed_resident(self.name, 0)      # refresh row 0
        store.rows_np(self.name, [3], "resident")  # evicts row 1, not row 0
        self.assertTrue(store.believed_resident(self.name, 0))
        self.assertFalse(store.believed_resident(self.name, 1))

    def test_tracking_never_changes_the_bytes(self):
        store = self.build("1")
        wanted = [3, 0, 2]
        for _ in range(3):
            np.testing.assert_array_equal(
                store.rows_np(self.name, wanted, "resident"), self.expected[wanted]
            )

    def test_a_pinned_row_stays_claimed_even_when_untracked(self):
        store = self.build("0")
        store.pin_rows(self.name, [1])
        for _ in range(10):
            store.rows_np(self.name, [7], "resident")
        self.assertTrue(store.believed_resident(self.name, 1))


class GateIsReachableTests(unittest.TestCase):
    """The residency gate is only consulted on the `resident` read path.

    A benchmark that forgets to set the read mode measures nothing and reports
    it as though the gate never claimed a row. That happened once.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        write_checkpoint(self._dir.name)
        self.name = "block.experts.weight"

    def tearDown(self):
        store = getattr(self, "store", None)
        if store is not None:
            store.close()
        self._dir.cleanup()

    def store_with(self, read_mode: str):
        with unittest.mock.patch.dict(
            "os.environ",
            {"FLASHNEXT_READ": read_mode, "FLASHNEXT_TRACK_RESIDENT": "1"},
        ):
            self.store = SafeTensorStore(self._dir.name)
        return self.store

    def consulted(self, store, rows):
        seen = []
        original = type(store).believed_resident
        type(store).believed_resident = (
            lambda self, name, row: (
                seen.append(row), original(self, name, row)
            )[1]
        )
        try:
            store.rows_np(self.name, rows)
        finally:
            type(store).believed_resident = original
        return seen

    def test_the_resident_mode_consults_the_gate(self):
        store = self.store_with("resident")
        self.assertEqual(store._read_mode, "resident")
        self.assertEqual(self.consulted(store, [1, 2]), [1, 2])

    def test_the_default_mode_never_consults_the_gate(self):
        store = self.store_with("pread")
        self.assertEqual(self.consulted(store, [1, 2]), [])

    def test_the_benchmark_selects_the_mode_the_gate_needs(self):
        import inspect

        from models.flashnext import bench_residency

        source = inspect.getsource(bench_residency.main)
        self.assertIn('os.environ["FLASHNEXT_READ"] = "resident"', source)
