"""Focused Q4/G64 executor checks.

These tests keep the G64 path checkpoint-free.  Metal integration tests remain
optional because CI does not provide MLX or an Apple GPU.
"""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from models.flashnext.metal_runtime import MetalMoEExecutor, Q4G32Projection


def _pack(values: np.ndarray, group_size: int) -> tuple[np.ndarray, ...]:
    experts, outputs, inputs = values.shape
    packed = np.zeros((experts, outputs, inputs // 8), dtype=np.uint32)
    for k in range(inputs):
        packed[:, :, k // 8] |= (
            (values[:, :, k].astype(np.uint32) & 15) << ((k & 7) * 4)
        )
    groups = inputs // group_size
    scales = np.ones((experts, outputs, groups), dtype=np.float32)
    biases = np.zeros_like(scales)
    return packed, scales, biases


class _CaptureBackend(SimpleNamespace):
    def __init__(self):
        super().__init__(available=True, supports_custom_moe=True)
        self.calls = []

    def metal_kernel(self, **kwargs):
        self.calls.append(kwargs)

        def run(*, inputs, **_):
            # Kernel compilation is the target of these checks.  Execution is
            # covered by the MLX smoke suite when Metal is available.
            x = inputs[0]
            routes = inputs[4]
            return np.zeros((x.shape[0], routes.shape[1], 32), dtype=x.dtype)

        return run


class Q4G64ReferenceTests(unittest.TestCase):
    def test_numpy_projection_uses_64_value_metadata_groups(self):
        hidden, inter, experts = 64, 64, 2
        rng = np.random.default_rng(7)
        values = {
            "gate_proj": _pack(rng.integers(0, 16, (experts, inter, hidden), dtype=np.uint8), 64),
            "up_proj": _pack(rng.integers(0, 16, (experts, inter, hidden), dtype=np.uint8), 64),
            "down_proj": _pack(rng.integers(0, 16, (experts, hidden, inter), dtype=np.uint8), 64),
        }
        x = rng.normal(size=(1, hidden)).astype(np.float32)
        routes = np.array([[0, 1]], dtype=np.uint32)
        executor = MetalMoEExecutor(
            experts, hidden, 2, backend="reference", group_size=64
        )
        actual = executor.execute(x, routes, values)
        self.assertEqual(actual.shape, (1, 2, hidden))
        self.assertTrue(np.isfinite(actual).all())


class Q4G64KernelSourceTests(unittest.TestCase):
    def test_kernel_source_selects_group_64_and_keeps_simd_width_32(self):
        backend = _CaptureBackend()
        executor = MetalMoEExecutor(1, 64, 1, backend=backend, group_size=64)
        x = np.zeros((1, 64), dtype=np.float32)
        routes = np.zeros((1, 1), dtype=np.uint32)
        projection = _pack(np.zeros((1, 32, 64), dtype=np.uint8), 64)
        executor._metal_projection(
            x, routes, Q4G32Projection(*projection), 32, False,
            proj_name="gate_proj"
        )
        source = backend.calls[0]["source"]
        header = backend.calls[0]["header"]
        self.assertIn("qmv_mixed_impl<T, 64, 4>", source)
        self.assertIn("simdgroup_index_in_threadgroup", source)
        self.assertIn("const int in_vec_size", header)
        self.assertNotIn("const int& in_vec_size", header)


class Q4G64MetalExactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        if mx.default_device().type != mx.DeviceType.gpu:
            raise unittest.SkipTest("MLX is not using a GPU device")
        cls.mx = mx

    def test_fused_and_unfused_scores_match_gather_qmm_bfloat16(self):
        mx = self.mx
        hidden, inter, experts = 256, 128, 2
        rng = np.random.default_rng(17)

        def pack(output_width, input_width):
            dense = mx.array(
                rng.normal(size=(experts, output_width, input_width)).astype(np.float32)
            ).astype(mx.bfloat16)
            return mx.quantize(dense, group_size=64, bits=4)

        projections = {
            "gate_proj": pack(inter, hidden),
            "up_proj": pack(inter, hidden),
            "down_proj": pack(hidden, inter),
        }
        x = mx.array(rng.normal(size=(1, hidden)).astype(np.float32)).astype(mx.bfloat16)
        routes = mx.array([[0, 1]], dtype=mx.uint32)
        scores = mx.array([[0.375, 0.625]], dtype=mx.float32)
        qmm = dict(
            rhs_indices=routes, transpose=True, group_size=64, bits=4,
            mode="affine", sorted_indices=False,
        )
        gate = mx.gather_qmm(x, *projections["gate_proj"], **qmm).squeeze(-2)
        up = mx.gather_qmm(x, *projections["up_proj"], **qmm).squeeze(-2)
        from mlx_vlm.models.activations import swiglu

        activation = swiglu(gate, up)
        down = mx.gather_qmm(
            mx.expand_dims(activation, -2), *projections["down_proj"], **qmm
        )
        expected = (down[:, :, 0, :] * scores[..., None]).sum(axis=1)
        for fused in (False, True):
            actual = MetalMoEExecutor(
                experts, hidden, 2, backend="metal", group_size=64,
                fused_up_swiglu=fused,
            ).execute(x, routes, projections, scores=scores)
            mx.eval(actual, expected)
            np.testing.assert_array_equal(
                np.asarray(actual.view(mx.uint16)),
                np.asarray(expected.view(mx.uint16)),
            )

    def test_batch_two_ten_slot_route_order_matches_gather_qmm(self):
        mx = self.mx
        hidden, inter, experts, top_k = 256, 128, 10, 10
        rng = np.random.default_rng(19)

        def pack(output_width, input_width):
            dense = mx.array(
                rng.normal(size=(experts, output_width, input_width)).astype(np.float32)
            ).astype(mx.bfloat16)
            return mx.quantize(dense, group_size=64, bits=4)

        projections = {
            "gate_proj": pack(inter, hidden),
            "up_proj": pack(inter, hidden),
            "down_proj": pack(hidden, inter),
        }
        x = mx.array(rng.normal(size=(2, hidden)).astype(np.float32)).astype(mx.bfloat16)
        routes = mx.array([list(reversed(range(top_k))), list(range(top_k))], dtype=mx.uint32)
        scores = mx.array([[1.0] + [0.0] * (top_k - 1)] * 2, dtype=mx.float32)
        qmm = dict(
            transpose=True, group_size=64, bits=4, mode="affine", sorted_indices=False,
        )
        expected_rows = []
        from mlx_vlm.models.activations import swiglu
        for row in range(2):
            args = {"rhs_indices": routes[row : row + 1], **qmm}
            gate = mx.gather_qmm(x[row : row + 1], *projections["gate_proj"], **args).squeeze(-2)
            up = mx.gather_qmm(x[row : row + 1], *projections["up_proj"], **args).squeeze(-2)
            act = swiglu(gate, up)
            down = mx.gather_qmm(
                mx.expand_dims(act, -2), *projections["down_proj"], **args
            )[:, :, 0, :]
            expected_rows.append((down * scores[row : row + 1, :, None]).sum(axis=1))
        expected = mx.concatenate(expected_rows, axis=0)
        actual = MetalMoEExecutor(
            experts, hidden, top_k, backend="metal", group_size=64,
        ).execute(x, routes, projections, scores=scores)
        mx.eval(actual, expected)
        np.testing.assert_array_equal(
            np.asarray(actual.view(mx.uint16)), np.asarray(expected.view(mx.uint16))
        )


if __name__ == "__main__":
    unittest.main()
