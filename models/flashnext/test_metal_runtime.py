"""Small, checkpoint-free tests for the experimental Metal MoE runtime."""
from __future__ import annotations

from types import SimpleNamespace
import sys
import unittest

import numpy as np

from models.flashnext.metal_runtime import (
    MetalMoEExecutor,
    probe_capabilities,
    weighted_combine,
)


class MetalRuntimeArgumentTests(unittest.TestCase):
    def test_constructor_rejects_non_positive_dimensions(self):
        for kwargs in (
            {"expert_count": 0, "hidden_size": 2, "top_k": 1},
            {"expert_count": 2, "hidden_size": 0, "top_k": 1},
            {"expert_count": 2, "hidden_size": 2, "top_k": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    MetalMoEExecutor(**kwargs)

    def test_constructor_rejects_top_k_larger_than_expert_count(self):
        with self.assertRaises(ValueError):
            MetalMoEExecutor(expert_count=2, hidden_size=2, top_k=3)

    def test_combine_rejects_incompatible_route_shapes(self):
        expert_outputs = np.zeros((2, 3, 2), dtype=np.float32)
        routes = np.zeros((2, 2), dtype=np.int32)
        scores = np.ones((2, 1), dtype=np.float32)

        with self.assertRaises(ValueError):
            weighted_combine(expert_outputs, routes, scores)

    def test_combine_rejects_an_expert_index_out_of_range(self):
        expert_outputs = np.zeros((2, 3, 2), dtype=np.float32)
        routes = np.array([[0, 3]], dtype=np.int32)
        scores = np.ones((1, 2), dtype=np.float32)

        with self.assertRaises(ValueError):
            weighted_combine(expert_outputs, routes, scores)

    def test_execute_rejects_a_malformed_q4_projection_pack(self):
        executor = MetalMoEExecutor(
            expert_count=2,
            hidden_size=32,
            top_k=1,
            backend="reference",
        )
        # Q4 packs store eight four-bit values per uint32 word.  This weight
        # row has three words instead of four for a 32-wide input.
        bad_pack = (
            np.zeros((2, 4, 3), dtype=np.uint32),
            np.zeros((2, 4, 1), dtype=np.float32),
            np.zeros((2, 4, 1), dtype=np.float32),
        )
        valid_pack = (
            np.zeros((2, 4, 4), dtype=np.uint32),
            np.zeros((2, 4, 1), dtype=np.float32),
            np.zeros((2, 4, 1), dtype=np.float32),
        )
        projections = {
            "gate_proj": bad_pack,
            "up_proj": valid_pack,
            "down_proj": (
                np.zeros((2, 32, 4), dtype=np.uint32),
                np.zeros((2, 32, 1), dtype=np.float32),
                np.zeros((2, 32, 1), dtype=np.float32),
            ),
        }

        with self.assertRaisesRegex(ValueError, "weight shape"):
            executor.execute(
                np.zeros((1, 32), dtype=np.float32),
                np.zeros((1, 1), dtype=np.int32),
                projections,
            )

    def test_execute_rejects_a_route_with_another_token_count(self):
        executor = MetalMoEExecutor(
            expert_count=2, hidden_size=32, top_k=1, backend="reference"
        )
        with self.assertRaisesRegex(ValueError, "same token count"):
            executor.execute(
                np.zeros((2, 32), dtype=np.float32),
                np.zeros((1, 1), dtype=np.int32),
                {},
            )


class MetalRuntimeCapabilityTests(unittest.TestCase):
    def test_capabilities_report_unavailable_backend_without_constructing_kernel(self):
        backend = SimpleNamespace(available=False, supports_custom_moe=False)

        capabilities = probe_capabilities(backend)

        self.assertFalse(capabilities.available)
        self.assertFalse(capabilities.supports_custom_moe)

    def test_capabilities_enable_executor_only_when_custom_moe_is_supported(self):
        backend = SimpleNamespace(available=True, supports_custom_moe=True)

        capabilities = probe_capabilities(backend)
        executor = MetalMoEExecutor(
            expert_count=2,
            hidden_size=2,
            top_k=1,
            backend=backend,
        )

        self.assertTrue(capabilities.available)
        self.assertTrue(capabilities.supports_custom_moe)
        self.assertTrue(executor.capabilities.supports_custom_moe)

    def test_unsupported_backend_fails_before_execution(self):
        backend = SimpleNamespace(available=True, supports_custom_moe=False)
        executor = MetalMoEExecutor(
            expert_count=2,
            hidden_size=2,
            top_k=1,
            backend=backend,
        )

        with self.assertRaisesRegex(RuntimeError, "custom MoE"):
            executor.execute(
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 1), dtype=np.int32),
                np.ones((1, 1), dtype=np.float32),
            )


class MetalRuntimeMathTests(unittest.TestCase):
    def test_weighted_combine_is_deterministic_for_tiny_routes(self):
        expert_outputs = np.array(
            [
                [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]],
                [[3.0, 4.0], [30.0, 40.0], [300.0, 400.0]],
            ],
            dtype=np.float32,
        )
        routes = np.array([[2, 0], [1, 2]], dtype=np.int32)
        scores = np.array([[0.25, 0.75], [0.4, 0.6]], dtype=np.float32)
        expected = np.array([[25.75, 51.5], [192.0, 256.0]], dtype=np.float32)

        first = weighted_combine(expert_outputs, routes, scores)
        second = weighted_combine(expert_outputs, routes, scores)

        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, first)

    def test_executor_uses_the_same_weighted_route_math(self):
        backend = SimpleNamespace(available=True, supports_custom_moe=True)
        executor = MetalMoEExecutor(
            expert_count=3,
            hidden_size=2,
            top_k=2,
            backend=backend,
        )
        expert_outputs = np.array(
            [
                [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]],
                [[3.0, 4.0], [30.0, 40.0], [300.0, 400.0]],
            ],
            dtype=np.float32,
        )
        routes = np.array([[2, 0], [1, 2]], dtype=np.int32)
        scores = np.array([[0.25, 0.75], [0.4, 0.6]], dtype=np.float32)

        actual = executor.execute(expert_outputs, routes, scores)

        np.testing.assert_array_equal(
            actual,
            np.array([[25.75, 51.5], [192.0, 256.0]], dtype=np.float32),
        )


class NativeMetalSchedulerTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "darwin", "native Metal requires Darwin")
    def test_all_native_strategies_keep_the_dependency_chain_exact(self):
        from models.flashnext.metal_native import probe_native, run_dependency_chain

        status = probe_native()
        if not status.available:
            self.skipTest(status.reason)
        values = np.arange(128, dtype=np.float32)
        expected = values + np.float32(4)
        for strategy in ("serial", "barrier", "fence"):
            with self.subTest(strategy=strategy):
                actual = run_dependency_chain(values, steps=4, strategy=strategy)
                np.testing.assert_array_equal(actual, expected)


class MLXQ4G32SmokeTests(unittest.TestCase):
    """Exercise the custom path with real tiny MLX packs, not the checkpoint."""

    @classmethod
    def setUpClass(cls):
        if sys.platform != "darwin":
            raise unittest.SkipTest("custom Metal requires Darwin")
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise unittest.SkipTest(f"MLX unavailable: {exc}") from exc
        if not hasattr(mx, "gather_qmm") or not hasattr(mx, "quantize"):
            raise unittest.SkipTest("MLX lacks quantize or gather_qmm")
        if mx.default_device().type != mx.DeviceType.gpu:
            raise unittest.SkipTest("MLX is not using a GPU device")
        cls.mx = mx

    def pack(self, seed, output_width=32, input_width=32, expert_count=2):
        mx = self.mx
        values = (
            (
                mx.arange(expert_count * output_width * input_width, dtype=mx.float32)
                + seed * 3
            )
            % 17
            - 8
        ) / 64
        return mx.quantize(
            values.reshape(expert_count, output_width, input_width).astype(
                mx.bfloat16
            ),
            group_size=32,
            bits=4,
        )

    def assert_mlx_close(self, actual, expected, message):
        mx = self.mx
        actual_np = np.asarray(actual.astype(mx.float32))
        expected_np = np.asarray(expected.astype(mx.float32))
        delta = np.abs(actual_np - expected_np)
        denominator = np.maximum(np.abs(expected_np), 1e-6)
        max_abs = float(delta.max()) if delta.size else 0.0
        max_rel = float((delta / denominator).max()) if delta.size else 0.0
        np.testing.assert_allclose(
            actual_np,
            expected_np,
            rtol=2e-2,
            atol=3e-3,
            err_msg=(
                f"{message}; max_abs={max_abs:.6g}; "
                f"max_rel={max_rel:.6g}"
            ),
        )

    def assert_mlx_equal(self, actual, expected, message):
        mx = self.mx
        actual_np = np.asarray(actual.astype(mx.float32))
        expected_np = np.asarray(expected.astype(mx.float32))
        np.testing.assert_array_equal(actual_np, expected_np, err_msg=message)

    def test_q4g32_metal_path_stays_within_gather_qmm_tolerance(self):
        mx = self.mx
        packs = {
            name: self.pack(seed)
            for seed, name in enumerate(("gate_proj", "up_proj", "down_proj"))
        }
        x = (
            ((mx.arange(32, dtype=mx.float32) % 13) - 6) / 8
        ).reshape(1, 32).astype(mx.bfloat16)
        routes = mx.array([[0, 1]], dtype=mx.uint32)
        executor = MetalMoEExecutor(2, 32, 2, backend="metal")

        actual = executor.execute(x, routes, packs)
        mx.eval(actual)

        qmm_args = dict(
            rhs_indices=routes,
            transpose=True,
            group_size=32,
            bits=4,
            mode="affine",
            sorted_indices=False,
        )
        gate = mx.gather_qmm(
            x, *packs["gate_proj"], **qmm_args
        ).squeeze(-2)
        up = mx.gather_qmm(x, *packs["up_proj"], **qmm_args).squeeze(-2)
        activation = up * (gate / (1 + mx.exp(-gate)))
        expected = mx.gather_qmm(
            mx.expand_dims(activation, -2),
            *packs["down_proj"],
            **qmm_args,
        ).squeeze(-2)
        scores = mx.array([[0.25, 0.75]], dtype=mx.float32)
        actual_weighted = weighted_combine(actual, routes, scores)
        fused_weighted = executor.execute(
            x, routes, packs, scores=scores
        )
        expected_weighted = weighted_combine(expected, routes, scores)
        mx.eval(expected)
        mx.eval(actual_weighted, fused_weighted, expected_weighted)

        self.assertEqual(executor.last_path, "custom-metal")
        # Compare the final routed output. The custom kernel writes BF16, and
        # MLX gather_qmm accumulates in float32, so allow one BF16 rounding
        # step plus small affine Q4 error.
        self.assert_mlx_close(
            actual_weighted,
            expected_weighted,
            "Q4/G32 weighted output diverged from gather_qmm",
        )
        self.assert_mlx_close(
            fused_weighted,
            expected_weighted,
            "fused down and route output diverged from gather_qmm",
        )

    def test_q4g32_metal_path_handles_wide_hidden_and_intermediate_shapes(self):
        mx = self.mx
        hidden_size = 256
        intermediate_size = 128
        expert_count = 4
        top_k = 3
        packs = {
            "gate_proj": self.pack(
                0, intermediate_size, hidden_size, expert_count
            ),
            "up_proj": self.pack(
                1, intermediate_size, hidden_size, expert_count
            ),
            "down_proj": self.pack(
                2, hidden_size, intermediate_size, expert_count
            ),
        }
        # Two tokens and three routes exercise flattened token/slot indexing.
        x = (
            ((mx.arange(2 * hidden_size, dtype=mx.float32) % 29) - 14) / 16
        ).reshape(2, hidden_size).astype(mx.bfloat16)
        routes = mx.array([[0, 2, 3], [3, 1, 0]], dtype=mx.uint32)
        scores = mx.array(
            [[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]],
            dtype=mx.float32,
        )
        executor = MetalMoEExecutor(
            expert_count, hidden_size, top_k, backend="metal", max_width=256
        )

        actual = executor.execute(x, routes, packs)
        mx.eval(actual)

        qmm_args = dict(
            transpose=True,
            group_size=32,
            bits=4,
            mode="affine",
            sorted_indices=False,
        )
        # gather_qmm's leading axes are model-layout axes. Run one token per
        # call so this reference retains the executor's flattened contract.
        expected_tokens = []
        for token in range(2):
            token_x = x[token : token + 1]
            token_routes = routes[token : token + 1]
            args = {"rhs_indices": token_routes, **qmm_args}
            gate = mx.gather_qmm(
                token_x, *packs["gate_proj"], **args
            ).squeeze(-2)
            up = mx.gather_qmm(token_x, *packs["up_proj"], **args).squeeze(-2)
            activation = up * (gate / (1 + mx.exp(-gate)))
            down = mx.gather_qmm(
                mx.expand_dims(activation, -2),
                *packs["down_proj"],
                **args,
            ).squeeze(-2)
            expected_tokens.append(down)
        expected = mx.concatenate(expected_tokens, axis=0)
        actual_weighted = weighted_combine(actual, routes, scores)
        expected_weighted = weighted_combine(expected, routes, scores)
        mx.eval(actual_weighted, expected_weighted)

        self.assertEqual(executor.last_path, "custom-metal")
        self.assert_mlx_close(
            actual_weighted,
            expected_weighted,
            "wide Q4/G32 weighted output diverged from gather_qmm",
        )

    def test_production_shape_is_bit_identical(self):
        mx = self.mx
        hidden_size, intermediate_size = 2560, 640
        packs = {
            "gate_proj": self.pack(0, intermediate_size, hidden_size, 2),
            "up_proj": self.pack(1, intermediate_size, hidden_size, 2),
            "down_proj": self.pack(2, hidden_size, intermediate_size, 2),
        }
        x = (((mx.arange(hidden_size, dtype=mx.float32) % 31) - 15) / 64)
        x = x.reshape(1, hidden_size).astype(mx.bfloat16)
        routes = mx.array([[0, 1]], dtype=mx.uint32)
        scores = mx.array([[0.375, 0.625]], dtype=mx.float32)
        executor = MetalMoEExecutor(2, hidden_size, 2, backend="metal")
        actual = executor.execute(x, routes, packs, scores=scores)
        qmm = dict(
            rhs_indices=routes, transpose=True, group_size=32, bits=4,
            mode="affine", sorted_indices=False,
        )
        gate = mx.gather_qmm(x, *packs["gate_proj"], **qmm).squeeze(-2)
        up = mx.gather_qmm(x, *packs["up_proj"], **qmm).squeeze(-2)
        from mlx_vlm.models.activations import swiglu

        activation = swiglu(gate, up)
        down = mx.gather_qmm(
            mx.expand_dims(activation, -2), *packs["down_proj"], **qmm
        ).squeeze(-2)
        expected = weighted_combine(down, routes, scores)
        mx.eval(actual, expected)
        self.assert_mlx_equal(
            actual, expected, "production Q4/G32 output is not bit-identical"
        )


if __name__ == "__main__":
    unittest.main()
