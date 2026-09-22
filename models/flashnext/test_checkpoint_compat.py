"""Synthetic regression tests for FlashNext checkpoint families.

No model load. No MLX import. Reads only config/index JSON and the
pure-Python capability detector.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from models.flashnext.checkpoint_compat import (
    MTP_INDEXED,
    MTP_NONE,
    MTP_SIDECAR,
    REQUIRED_MTP_SUBSTRINGS,
    describe_checkpoint,
    detect_mtp_source,
    is_mtp_key,
    mtp_keys_from_weight_map,
    validate_mtp_structure,
)


def _write_checkpoint(
    root: Path,
    *,
    dirname: str = "ckpt",
    model_type: str = "qwen4_exp",
    text_type: str = "qwen4_exp_text",
    experts: int = 512,
    top_k: int = 10,
    group: int = 32,
    extra_index_keys: dict | None = None,
    sidecar: bool = False,
) -> Path:
    path = root / dirname
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": model_type,
        "quantization": {"group_size": group, "bits": 4, "mode": "affine"},
        "text_config": {
            "model_type": text_type,
            "num_hidden_layers": 48,
            "hidden_size": 2560,
            "num_experts": experts,
            "num_experts_per_tok": top_k,
            "moe_intermediate_size": 640,
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    weight_map = {"language_model.model.embed_tokens.weight": "model-00001-of-00002.safetensors"}
    for key in (extra_index_keys or {}):
        weight_map[key] = extra_index_keys[key]
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for shard in sorted(set(weight_map.values())):
        (path / shard).write_bytes(b"")
    if sidecar:
        (path / "model-mtp.safetensors").write_bytes(b"")
    return path


def _indexed_mtp_map(shard: str = "model-00022-of-00022.safetensors") -> dict:
    return {name: shard for name in REQUIRED_MTP_SUBSTRINGS}


class CheckpointFamilyTests(unittest.TestCase):
    def test_old_g32_no_mtp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_checkpoint(Path(tmp))
            source, keys, _shards = detect_mtp_source(path, {"a.weight": "s.safetensors"})
            self.assertEqual(source, MTP_NONE)
            self.assertEqual(keys, [])
            caps = describe_checkpoint(path)
            self.assertTrue(caps.compatible)
            self.assertEqual(caps.mtp_source, MTP_NONE)
            self.assertEqual(caps.experts, 512)
            self.assertEqual(caps.backbone_group, 32)

    def test_sidecar_mtp_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_checkpoint(Path(tmp), sidecar=True)
            source, _keys, shards = detect_mtp_source(path, {"a.weight": "s.safetensors"})
            self.assertEqual(source, MTP_SIDECAR)
            self.assertEqual(shards, ["model-mtp.safetensors"])

    def test_indexed_mtp_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_extra = _indexed_mtp_map()
            path = _write_checkpoint(Path(tmp), extra_index_keys=index_extra)
            caps = describe_checkpoint(path)
            self.assertEqual(caps.mtp_source, MTP_INDEXED)
            self.assertEqual(len(caps.mtp_keys), len(REQUIRED_MTP_SUBSTRINGS))
            self.assertEqual(caps.mtp_shards, ["model-00022-of-00022.safetensors"])
            self.assertEqual(validate_mtp_structure(caps.mtp_keys), [])

    def test_reap_g64_shape_reports_no_mtp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_checkpoint(Path(tmp), experts=288, top_k=8, group=64)
            caps = describe_checkpoint(path)
            self.assertTrue(caps.compatible)
            self.assertEqual(caps.experts, 288)
            self.assertEqual(caps.backbone_group, 64)
            self.assertEqual(caps.mtp_source, MTP_NONE)

    def test_malformed_mtp_reports_missing(self):
        partial = {"language_model.mtp.fc_embedding.weight": "s.safetensors"}
        missing = validate_mtp_structure(partial)
        self.assertIn("language_model.mtp.pre_fc_norm_embedding.weight", missing)
        self.assertIn("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight", missing)

    def test_both_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_checkpoint(
                Path(tmp), extra_index_keys=_indexed_mtp_map(), sidecar=True
            )
            with self.assertRaises(ValueError):
                detect_mtp_source(path)

    def test_no_basename_heuristic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_extra = _indexed_mtp_map()
            first = _write_checkpoint(root, dirname="vontra-looking-name", extra_index_keys=index_extra)
            second = _write_checkpoint(root, dirname="plain-name", extra_index_keys=dict(index_extra))
            self.assertEqual(describe_checkpoint(first).mtp_source, MTP_INDEXED)
            self.assertEqual(describe_checkpoint(second).mtp_source, MTP_INDEXED)

    def test_mtp_key_predicate_covers_non_expert_tensors(self):
        self.assertTrue(is_mtp_key("language_model.mtp.pre_fc_norm_embedding.weight"))
        self.assertTrue(is_mtp_key("language_model.mtp.fc_hidden.scales"))
        self.assertTrue(is_mtp_key("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight"))
        self.assertFalse(is_mtp_key("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"))
        self.assertFalse(is_mtp_key("language_model.model.embed_tokens.weight"))

    def test_mtp_keys_sorted_from_weight_map(self):
        weight_map = {
            "language_model.mtp.fc_hidden.weight": "s2",
            "language_model.model.embed_tokens.weight": "s1",
            "language_model.mtp.fc_embedding.weight": "s2",
        }
        self.assertEqual(
            mtp_keys_from_weight_map(weight_map),
            [
                "language_model.mtp.fc_embedding.weight",
                "language_model.mtp.fc_hidden.weight",
            ],
        )

    def test_vontra_static_checkpoint(self):
        for candidate in (
            os.path.expanduser("~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"),
            "/Users/gioma/models/Qwen3.8-Flash-Next-MLX-4bit-MTP",
        ):
            if os.path.isdir(candidate):
                path = candidate
                break
        else:
            self.skipTest("Vontra checkpoint not present")
        caps = describe_checkpoint(path)
        self.assertTrue(caps.compatible)
        self.assertEqual(caps.layers, 48)
        self.assertEqual(caps.hidden_size, 2560)
        self.assertEqual(caps.experts, 512)
        self.assertEqual(caps.top_k, 10)
        self.assertEqual(caps.moe_intermediate, 640)
        self.assertEqual(caps.backbone_group, 32)
        self.assertEqual(caps.backbone_bits, 4)
        self.assertEqual(caps.mtp_source, MTP_INDEXED)
        self.assertEqual(len(caps.mtp_keys), 76)
        self.assertEqual(caps.mtp_shards, ["model-00022-of-00022.safetensors"])
        self.assertEqual(validate_mtp_structure(caps.mtp_keys), [])

    def test_vontra_norm_convention_is_one_centered(self):
        """Lock VONTRA_NORM=one-centered from stored tensor values.

        Reads kilobytes via ranged I/O only. No model construction.
        The Vontra artifact stores direct gains (median near 1.0), unlike
        the official zero-centered offsets, so the loader must keep the
        legacy one-centered fallback for it.
        """
        for candidate in (
            os.path.expanduser("~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"),
            "/Users/gioma/models/Qwen3.8-Flash-Next-MLX-4bit-MTP",
        ):
            if os.path.isdir(candidate):
                path = candidate
                break
        else:
            self.skipTest("Vontra checkpoint not present")
        self.assertEqual(describe_checkpoint(path).norm_convention, "")
        probes = {
            "language_model.model.hyper_connection_mixer.hc_norm.weight": (2.0, 6.0),
            "language_model.model.layers.0.linear_attn.norm.weight": (0.8, 1.2),
        }
        for key, (low, high) in probes.items():
            values = _read_bf16_vector(path, key)
            ordered = sorted(values)
            median = ordered[len(ordered) // 2]
            neg = sum(1 for value in values if value < 0) / len(values)
            # One-centered gains sit near 1.0 or above; zero-centered
            # offsets would sit near 0.0 with a wide negative fraction.
            self.assertGreater(median, low, key)
            self.assertLess(median, high, key)
            self.assertLess(neg, 0.10, key)
        # The small MTP embedding norm is a positive gain too: no negative
        # values at all, which a zero-centered offset vector would not show.
        small = _read_bf16_vector(
            path, "language_model.mtp.pre_fc_norm_embedding.weight"
        )
        self.assertTrue(all(value > 0.05 for value in small))
        self.assertLess(max(small), 1.0)


def _read_bf16_vector(model_dir: str, key: str) -> list[float]:
    """Read one BF16 vector with stdlib only. No MLX, no numpy."""
    import struct

    path = Path(os.path.expanduser(model_dir))
    weight_map = json.loads(
        (path / "model.safetensors.index.json").read_text()
    )["weight_map"]
    shard = weight_map[key]
    with open(path / shard, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        meta = header[key]
        count = 1
        for dim in meta["shape"]:
            count *= dim
        handle.seek(8 + header_len + meta["data_offsets"][0])
        raw = handle.read(count * 2)
    out = []
    for (bits,) in struct.iter_unpack("<H", raw):
        out.append(struct.unpack(">f", struct.pack(">I", bits << 16))[0])
    return out


if __name__ == "__main__":
    unittest.main()
