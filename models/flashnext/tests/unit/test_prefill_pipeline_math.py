"""Segmented sorted gather_qmm must equal one sorted call, bit for bit.

The prefill pipeline computes each expert chunk's contiguous segment of the
sorted rows separately. This checks that premise on real MLX kernels with
production-like shapes (Q4/G32, hidden 2560, intermediate 640, top 8).
"""
from __future__ import annotations

import unittest

import mlx.core as mx
from mlx_vlm.models.switch_layers import _gather_sort, _scatter_unsort


def quantized(experts, out_dims, in_dims, key):
    weight = mx.random.normal((experts, out_dims, in_dims), key=key).astype(mx.bfloat16) * 0.05
    return mx.quantize(weight, group_size=32, bits=4)


class SegmentedGatherTests(unittest.TestCase):
    def test_segments_match_single_call(self):
        mx.random.seed(0)
        tokens, slots, hidden, inter, experts = 96, 8, 2560, 640, 40
        x = (mx.random.normal((1, tokens, hidden)) * 0.5).astype(mx.bfloat16)
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.random.randint(0, experts, (1, tokens, slots)).astype(mx.uint32)
        gate = quantized(experts, inter, hidden, mx.random.key(1))
        down = quantized(experts, hidden, inter, mx.random.key(2))

        def run(xs, local, gw, dw):
            g = mx.gather_qmm(xs, *gw, rhs_indices=local, transpose=True,
                              group_size=32, bits=4, sorted_indices=True)
            return mx.gather_qmm(g, *dw, rhs_indices=local, transpose=True,
                                 group_size=32, bits=4, sorted_indices=True)

        xs, local, inverse = _gather_sort(x, indices)
        reference = _scatter_unsort(run(xs, local, gate, down), inverse, indices.shape)

        counts = [0] * experts
        for value in indices.reshape(-1).tolist():
            counts[value] += 1
        outputs, row = [], 0
        for start in range(0, experts, 11):
            end = min(experts, start + 11)
            rows = sum(counts[start:end])
            if not rows:
                continue
            gw = tuple(part[start:end] for part in gate)
            dw = tuple(part[start:end] for part in down)
            seg_local = local[row:row + rows] - start
            outputs.append(run(xs[row:row + rows], seg_local, gw, dw))
            row += rows
        pipelined = _scatter_unsort(mx.concatenate(outputs, axis=0), inverse, indices.shape)
        self.assertTrue(mx.array_equal(reference, pipelined).item())


if __name__ == "__main__":
    unittest.main()
