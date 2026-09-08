import os
import sys
import unittest

import numpy as np


class G64PackedExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "darwin" or os.environ.get("GITHUB_ACTIONS") == "true":
            raise unittest.SkipTest("G64 packed Metal requires a local macOS GPU")
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        if mx.default_device().type != mx.DeviceType.gpu:
            raise unittest.SkipTest("MLX is not using a GPU device")
        cls.mx = mx
        from mlx_vlm.models.activations import swiglu
        from models.flashnext.metal_runtime import MetalMoEExecutor
        from models.flashnext.slab_pack import HEADER_SIZE, Q4G64_LAYOUT
        cls.swiglu = swiglu
        cls.MetalMoEExecutor = MetalMoEExecutor
        cls.HEADER_SIZE = HEADER_SIZE
        cls.layout = Q4G64_LAYOUT

    def make_packs(self):
        mx = self.mx
        hidden, inter, experts = 2560, 640, 2

        def pack(seed, output_width, input_width):
            values = (
                ((mx.arange(experts * output_width * input_width, dtype=mx.float32)
                  + seed * 3) % 17) - 8
            ) / 64
            return mx.quantize(
                values.reshape(experts, output_width, input_width).astype(mx.bfloat16),
                group_size=64,
                bits=4,
            )

        packs = {
            "gate_proj": pack(0, inter, hidden),
            "up_proj": pack(1, inter, hidden),
            "down_proj": pack(2, hidden, inter),
        }
        mx.eval(*(part for projection in packs.values() for part in projection))
        return packs

    def make_slab_pack(self, packs):
        raw = bytearray(self.HEADER_SIZE + 2 * self.layout.record_stride)
        for projection, parts in packs.items():
            for part_name, values in zip(("weight", "scales", "biases"), parts):
                if str(values.dtype) == "mlx.core.bfloat16":
                    values = self.mx.view(values, self.mx.uint16)
                    self.mx.eval(values)
                rows = np.asarray(values)
                component_offset = self.layout.offset(projection, part_name)
                for expert in range(2):
                    start = self.HEADER_SIZE + expert * self.layout.record_stride + component_offset
                    row = np.ascontiguousarray(rows[expert]).tobytes()
                    raw[start:start + len(row)] = row
        return self.mx.array(np.frombuffer(raw, dtype=np.uint8))

    def make_input(self, dtype):
        mx = self.mx
        x = (((mx.arange(2560, dtype=mx.float32) % 31) - 15) / 64).reshape(1, 2560)
        return x.astype(dtype)

    def expected(self, x, packs, routes, scores, shared_y=None):
        mx = self.mx
        logical_routes = mx.array([[0, 1]], dtype=mx.uint32)
        args = dict(
            rhs_indices=logical_routes,
            transpose=True,
            group_size=64,
            bits=4,
            mode="affine",
            sorted_indices=False,
        )
        gate = mx.gather_qmm(x, *packs["gate_proj"], **args).squeeze(-2)
        up = mx.gather_qmm(x, *packs["up_proj"], **args).squeeze(-2)
        activation = self.swiglu(gate, up)
        down = mx.gather_qmm(
            mx.expand_dims(activation, -2), *packs["down_proj"], **args
        ).squeeze(-2)
        # G64 follows the generic MLX score product and reduction order.
        expected = (down * scores[..., None]).sum(axis=-2)
        if shared_y is not None:
            expected = expected + shared_y
        return expected

    def assert_exact(self, actual, expected, label):
        mx = self.mx
        mx.eval(actual, expected)
        actual_np = np.asarray(actual.astype(mx.float32))
        expected_np = np.asarray(expected.astype(mx.float32))
        np.testing.assert_array_equal(actual_np, expected_np, err_msg=label)

    def test_g64_packed_mixed_and_allresident_match_reference(self):
        mx = self.mx
        packs = self.make_packs()
        slab_pack = self.make_slab_pack(packs)
        scores = mx.array([[0.375, 0.625]], dtype=mx.float32)
        x = self.make_input(mx.bfloat16)
        expected = self.expected(x, packs, None, scores)

        for label, routes in (
            ("mixed resident and cold", mx.array([[0x80000000, 1]], dtype=mx.uint32)),
            ("all resident", mx.array([[0x80000000, 0x80000001]], dtype=mx.uint32)),
        ):
            executor = self.MetalMoEExecutor(2, 2560, 2, backend="metal", group_size=64)
            actual = executor.execute(x, routes, packs, scores=scores, slab_pack=slab_pack)
            self.assertEqual(executor.last_path, "custom-metal")
            self.assert_exact(actual, expected, label)

    def test_g64_packed_fused_up_bfloat16_and_float32_shared_output(self):
        mx = self.mx
        packs = self.make_packs()
        slab_pack = self.make_slab_pack(packs)
        scores = mx.array([[0.375, 0.625]], dtype=mx.float32)
        shared_y = (((mx.arange(2560, dtype=mx.float32) % 29) - 14) / 32).astype(mx.bfloat16).reshape(1, 2560)
        routes = mx.array([[0x80000000, 1]], dtype=mx.uint32)
        for dtype in (mx.bfloat16, mx.float32):
            x = self.make_input(dtype)
            expected = self.expected(x, packs, None, scores, shared_y=shared_y)
            for fused in (False, True):
                executor = self.MetalMoEExecutor(
                    2, 2560, 2, backend="metal", group_size=64,
                    fused_up_swiglu=fused,
                )
                actual = executor.execute(
                    x, routes, packs, scores=scores, slab_pack=slab_pack,
                    shared_y=shared_y,
                )
                self.assert_exact(actual, expected, f"dtype={dtype}, fused={fused}")

    def test_g64_unpacked_production_shape_matches_reference(self):
        mx = self.mx
        packs = self.make_packs()
        x = self.make_input(mx.bfloat16)
        routes = mx.array([[0, 1]], dtype=mx.uint32)
        scores = mx.array([[0.375, 0.625]], dtype=mx.float32)
        expected = self.expected(x, packs, routes, scores)
        executor = self.MetalMoEExecutor(2, 2560, 2, backend="metal", group_size=64)
        actual = executor.execute(x, routes, packs, scores=scores)
        self.assertEqual(executor.last_path, "custom-metal")
        self.assert_exact(actual, expected, "unpacked G64 production shape")

    def test_g64_bfloat16_scores_match_generic_reduction(self):
        mx = self.mx
        packs = self.make_packs()
        x = self.make_input(mx.bfloat16)
        routes = mx.array([[0, 1]], dtype=mx.uint32)
        scores = mx.array([[0.375, 0.625]], dtype=mx.bfloat16)
        expected = self.expected(x, packs, routes, scores)
        executor = self.MetalMoEExecutor(2, 2560, 2, backend="metal", group_size=64)
        actual = executor.execute(x, routes, packs, scores=scores)
        self.assert_exact(actual, expected, "BF16-score G64 reduction")


if __name__ == "__main__":
    unittest.main()
