"""Small-row sidecar: built copies land byte-exact in the stream record."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import mlx.core as mx
import numpy as np

from models.flashnext import small_sidecar
from models.flashnext.slab_pack import RECORD_STRIDE
from models.flashnext.store import SafeTensorStore

PREFIX = "language_model.model.layers.3.mlp.switch_mlp"
EXPERTS = 5


class SmallSidecarTests(unittest.TestCase):
    def test_build_and_scatter_match_checkpoint_rows(self):
        mx.random.seed(3)
        tensors = {}
        for projection, part in small_sidecar.PARTS:
            # 51,200 uint16 values per row is one 102,400-byte row.
            tensors[f"{PREFIX}.{projection}.{part}"] = mx.random.randint(
                0, 65535, (EXPERTS, 51_200)
            ).astype(mx.uint16)
        with tempfile.TemporaryDirectory() as directory:
            shard = "model-00001-of-00001.safetensors"
            mx.save_safetensors(os.path.join(directory, shard), tensors)
            with open(os.path.join(directory, "model.safetensors.index.json"), "w") as out:
                json.dump({"weight_map": {key: shard for key in tensors}}, out)
            store = SafeTensorStore(directory)
            sidecar_dir = os.path.join(directory, "sidecar")
            try:
                written = small_sidecar.build(store, [3], sidecar_dir, lambda _layer: PREFIX)
                self.assertEqual(written, {3: EXPERTS})
                with open(os.path.join(sidecar_dir, "manifest.json"), "w") as out:
                    json.dump({
                        "format": small_sidecar.FORMAT,
                        "checkpoint_identity": "synthetic",
                        "layers": [3],
                    }, out)
                with mock.patch.dict(os.environ, {small_sidecar.FLAG: sidecar_dir}), \
                        mock.patch(
                            "models.flashnext.routing.checkpoint_identity_for_store",
                            return_value="synthetic",
                        ):
                    sidecar = small_sidecar.for_store(store)
                self.assertEqual(sidecar.layers, {3})
                wanted = [4, 1, 3]
                buffer = np.zeros(len(wanted) * RECORD_STRIDE, dtype=np.uint8)
                sidecar.read_into(3, wanted[1:], buffer, 1)
                sidecar.read_into(3, wanted[:1], buffer, 0)
                from models.flashnext import slab_pack

                offsets = {
                    ("gate_proj", "scales"): slab_pack.GATE_SCALES_OFFSET,
                    ("gate_proj", "biases"): slab_pack.GATE_BIASES_OFFSET,
                    ("up_proj", "scales"): slab_pack.UP_SCALES_OFFSET,
                    ("up_proj", "biases"): slab_pack.UP_BIASES_OFFSET,
                    ("down_proj", "scales"): slab_pack.DOWN_SCALES_OFFSET,
                    ("down_proj", "biases"): slab_pack.DOWN_BIASES_OFFSET,
                }
                for slot, expert in enumerate(wanted):
                    for key, offset in offsets.items():
                        name = f"{PREFIX}.{key[0]}.{key[1]}"
                        expected = store.rows_np(name, [expert]).tobytes()
                        start = slot * RECORD_STRIDE + offset
                        self.assertEqual(
                            bytes(buffer[start : start + small_sidecar.ROW_BYTES]),
                            expected, f"{key} expert {expert}",
                        )
                sidecar.close()
            finally:
                store.close()

    def test_wrong_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "manifest.json"), "w") as out:
                json.dump({
                    "format": small_sidecar.FORMAT,
                    "checkpoint_identity": "other", "layers": [1],
                }, out)

            class Store:
                pass

            with mock.patch.dict(os.environ, {small_sidecar.FLAG: directory}), \
                    mock.patch(
                        "models.flashnext.routing.checkpoint_identity_for_store",
                        return_value="this",
                    ):
                with self.assertRaises(ValueError):
                    small_sidecar.for_store(Store())


if __name__ == "__main__":
    unittest.main()
