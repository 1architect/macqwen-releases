"""Focused compatibility checks for checkpoint naming and norm selection."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen4_exp.config import TextConfig
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpGatedDeltaNet, Qwen4ExpPLELayer
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpRMSNorm

from models.flashnext import loader
from models.flashnext.loader import _ngram_shard_prefix, _norm_one_centered
from models.flashnext.patch_rmsnorm import apply, configure


class LoaderCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        apply()

    def setUp(self):
        self._clean_norm_env = patch.dict(os.environ, {"FLASHNEXT_NORM_CONVENTION": ""})
        self._clean_norm_env.start()

    def tearDown(self):
        self._clean_norm_env.stop()

    def test_norm_conventions_are_numerically_distinct(self):
        for dtype in (mx.float32, mx.bfloat16):
            for group_size in (None, 2):
                x = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=dtype)
                w = mx.array([0.1, 0.2, 0.3, 0.4], dtype=dtype)
                norm = Qwen4ExpRMSNorm(4, group_size=group_size, eps=1e-6)
                norm.weight = w
                y = x.astype(mx.float32)
                if group_size is not None:
                    y = y.reshape(1, 2, group_size)
                    ew = w.astype(mx.float32).reshape(2, group_size)
                else:
                    ew = w.astype(mx.float32)
                base = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + 1e-6)
                for one_centered, gain in ((True, ew), (False, 1.0 + ew)):
                    norm._flashnext_one_centered = one_centered
                    expected = (base * gain).reshape(x.shape).astype(dtype)
                    self.assertTrue(mx.array_equal(norm(x), expected).item())

    def test_configure_is_per_model_and_supports_grouped_norms(self):
        first = SimpleNamespace(norm=Qwen4ExpRMSNorm(4), grouped=Qwen4ExpRMSNorm(4, 2))
        second = SimpleNamespace(norm=Qwen4ExpRMSNorm(4), grouped=Qwen4ExpRMSNorm(4, 2))
        first.named_modules = lambda: [("norm", first.norm), ("grouped", first.grouped)]
        second.named_modules = lambda: [("norm", second.norm), ("grouped", second.grouped)]
        self.assertEqual(configure(first, one_centered=False), 2)
        self.assertEqual(configure(second, one_centered=True), 2)
        self.assertFalse(first.norm._flashnext_one_centered)
        self.assertTrue(second.norm._flashnext_one_centered)
        self.assertFalse(first.grouped._flashnext_one_centered)

    def test_ngram_accepts_reap_and_legacy_names(self):
        reap = SimpleNamespace(refs={"base.shards.3.weight": object()})
        old = SimpleNamespace(refs={"base.shard_3.weight": object()})
        self.assertEqual(_ngram_shard_prefix(reap, "base", 3), "base.shards.3")
        self.assertEqual(_ngram_shard_prefix(old, "base", 3), "base.shard_3")
        self.assertIsNone(_ngram_shard_prefix(old, "base", 4))

    def test_norm_override_is_explicit_and_invalid_values_fail(self):
        store = SimpleNamespace(refs={}, dir=".")
        with patch.dict(os.environ, {"FLASHNEXT_NORM_CONVENTION": "zero"}):
            self.assertFalse(_norm_one_centered({}, store))
        with patch.dict(os.environ, {"FLASHNEXT_NORM_CONVENTION": "one"}):
            self.assertTrue(_norm_one_centered({}, store))
        with patch.dict(os.environ, {"FLASHNEXT_NORM_CONVENTION": "typo"}):
            with self.assertRaises(ValueError):
                _norm_one_centered({}, store)

    def test_unknown_checkpoint_keeps_legacy_norm_mode(self):
        store = SimpleNamespace(refs={}, dir=".")
        self.assertTrue(_norm_one_centered({}, store))

    def test_known_norm_fingerprint_selects_zero_mode(self):
        payload = b"x" * 32
        digest = __import__("hashlib").sha256(payload).hexdigest()
        ref = SimpleNamespace(shard="norm.bin", start=0, shape=(16,), dtype="uint8")
        store = SimpleNamespace(refs={loader._REAP_NORM_KEY: ref}, dir=".")
        with patch.object(loader, "_REAP_NORM_FINGERPRINT", digest), patch.object(
            loader, "_nbytes", return_value=len(payload)
        ), patch("builtins.open", unittest.mock.mock_open(read_data=payload)):
            self.assertFalse(_norm_one_centered({}, store))

    def test_nonmatching_norm_fingerprint_keeps_legacy_mode(self):
        ref = SimpleNamespace(shard="norm.bin", start=0, shape=(16,), dtype="uint8")
        store = SimpleNamespace(refs={loader._REAP_NORM_KEY: ref}, dir=".")
        with patch.object(loader, "_nbytes", return_value=3), patch(
            "builtins.open", unittest.mock.mock_open(read_data=b"bad")
        ):
            self.assertTrue(_norm_one_centered({}, store))

    def test_ngram_swap_replaces_complete_old_and_new_sets(self):
        def model():
            table = SimpleNamespace(shards=[object(), object()], shard_sizes=(3, 4), dims=2)
            layer = SimpleNamespace(
                ple=SimpleNamespace(ple_embedding=SimpleNamespace(ngram_embedding=table))
            )
            return SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[None, layer])))

        for spelling in ("shards.", "shard_"):
            current = model()
            refs = {
                f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.{spelling}{i}.weight": object()
                for i in range(2)
            }
            store = SimpleNamespace(refs=refs)
            with patch.object(loader, "StreamingQuantizedEmbedding", side_effect=lambda *a: a[1]):
                self.assertEqual(loader._swap_ngram(current, store, 1, "affine"), 2)
            self.assertEqual(len(current.language_model.model.layers[1].ple.ple_embedding.ngram_embedding.shards), 2)

    def test_ngram_swap_rejects_incomplete_set_before_replacement(self):
        table = SimpleNamespace(shards=[object(), object()], shard_sizes=(3, 4), dims=2)
        original = SimpleNamespace(ple=SimpleNamespace(ple_embedding=SimpleNamespace(ngram_embedding=table)))
        current = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[None, original])))
        refs = {"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight": object()}
        with patch.object(loader, "StreamingQuantizedEmbedding"):
            with self.assertRaisesRegex(ValueError, "incomplete n-gram"):
                loader._swap_ngram(current, SimpleNamespace(refs=refs), 1, "affine")
        self.assertIs(current.language_model.model.layers[1].ple.ple_embedding.ngram_embedding, table)

    def test_conv1d_sanitization_uses_each_target_module_layout(self):
        config = TextConfig(
            model_type="qwen4_exp", hidden_size=4, num_hidden_layers=1,
            num_attention_heads=1, linear_num_value_heads=1,
            linear_num_key_heads=1, linear_key_head_dim=4,
            linear_value_head_dim=4, linear_conv_kernel_dim=4,
            num_experts=1, num_experts_per_tok=1,
            shared_expert_intermediate_size=4, moe_intermediate_size=4,
            rms_norm_eps=1e-6, vocab_size=100, num_key_value_heads=1,
            max_position_embeddings=10, ple_layer_ids=[1], ple_embed_dim=16,
            ngram_vocab_size_base=16, split_ngram_parts=2, eos_token_id=99,
        )
        root = nn.Module()
        root.ple = Qwen4ExpPLELayer(config, 0, 0)
        root.linear = Qwen4ExpGatedDeltaNet(config)
        ple_key = "ple.conv1d.weight"
        linear_key = "linear.conv1d.weight"
        ple_source = mx.arange(16 * 4, dtype=mx.float32).reshape(16, 1, 4)
        linear_source = mx.arange(12 * 4, dtype=mx.float32).reshape(12, 4, 1)
        weights = {ple_key: ple_source, linear_key: linear_source}

        loader._sanitize_conv1d_weights(root, weights)

        self.assertEqual(weights[ple_key].shape, root.ple.conv1d.weight.shape)
        self.assertTrue(mx.array_equal(weights[ple_key], ple_source.moveaxis(2, 1)).item())
        self.assertEqual(weights[linear_key].shape, root.linear.conv1d.weight.shape)
        self.assertTrue(mx.array_equal(weights[linear_key], linear_source).item())
        loader._sanitize_conv1d_weights(root, weights)
        self.assertTrue(mx.array_equal(weights[ple_key], ple_source.moveaxis(2, 1)).item())

        root.load_weights(list(weights.items()), strict=False)
        x = mx.arange(2 * 12 * 16, dtype=mx.float32).reshape(2, 12, 16)
        expected = mx.conv1d(
            x, ple_source.moveaxis(2, 1), dilation=root.ple.conv1d.dilation, groups=16
        )
        actual = root.ple.conv1d(x)
        self.assertTrue(mx.array_equal(actual, expected).item())


if __name__ == "__main__":
    unittest.main()
