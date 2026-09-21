// Bonsai-2 ternary gate/up QMM tail (D4 prototype, qmv-structured).
//
// Entry block for mx.fast.metal_kernel. Same launch geometry as the
// wheel qmv fast path (64 threads per 8 rows) and same traversal
// shape, except K advances in 640-weight steps (128 ternary bytes,
// five intact G128 groups). Each lane owns 20 weights (4 bytes,
// unrolled divmod, branch-free); lanes straddling a group boundary
// accumulate both adjacent groups with arithmetic selects. FP32
// activations, F16 scales/biases promoted exactly in registers,
// stock per-group order scale*accum + sum*bias. Gate/up shape only.
{
  uint3 group = threadgroup_position_in_grid;
  uint simd_lid = thread_index_in_simdgroup;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint3 tid = uint3(group.x, group.y, 0);

  thread float result[4] = {0, 0, 0, 0};

  // 5120 / 640 = 8 steps. Strides per step: weights 128 B,
  // scales/biases 5 groups, activations 640.
  const device uint8_t* ws = (const device uint8_t*)tpack;
  const int out_row = tid.y * 8 + simd_gid * 4;

  for (int step = 0; step < 8; step++) {
    // Lane weights: 20 per lane inside the 640-block. A lane touches
    // at most two adjacent groups; scales follow the lane's groups.
    int wlo = step * 640 + (int)simd_lid * 20;
    int ga = wlo / 128;
    int gb = (wlo + 19) / 128;
    for (int row = 0; row < 4; row++) {
      const device uint8_t* prow =
          ws + (out_row + row) * 1040;
      const device float* xrow =
          x + tid.x * in_vec_size[0] + step * 640;
      float acc0 = 0.0f;
      float sum0 = 0.0f;
      float acc1 = 0.0f;
      float sum1 = 0.0f;
      for (int i = 0; i < 20; i++) {
        int wpos = wlo + i;
        int g = wpos / 128;
        int m = wpos - g * 128;
        uint v = (uint)prow[g * 26 + m / 5];
        // digit = (v / 3^(m%5)) % 3 via descending chain
        uint t = v;
        uint d0 = t % 3;
        t /= 3;
        uint d1 = t % 3;
        t /= 3;
        uint d2 = t % 3;
        t /= 3;
        uint d3 = t % 3;
        uint d4 = t / 3;
        int dm = m % 5;
        uint dig = dm == 0 ? d0 : (dm == 1 ? d1 : (dm == 2 ? d2 : (dm == 3 ? d3 : d4)));
        float xv = xrow[wpos - step * 640];
        // Membership: weight belongs to ga-side iff wpos < (ga+1)*128.
        float in0 = (float)(wpos < (ga + 1) * 128);
        float in1 = 1.0f - in0;
        acc0 += in0 * xv * dig;
        sum0 += in0 * xv;
        acc1 += in1 * xv * dig;
        sum1 += in1 * xv;
      }
      float s0 = float(scales[(out_row + row) * 40 + ga]);
      float b0 = float(biases[(out_row + row) * 40 + ga]);
      float s1 = float(scales[(out_row + row) * 40 + gb]);
      float b1 = float(biases[(out_row + row) * 40 + gb]);
      result[row] += s0 * acc0 + sum0 * b0 + s1 * acc1 + sum1 * b1;
    }
  }

  for (int row = 0; row < 4; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[tid.x * out_vec_size[0] + out_row + row] = result[row];
    }
  }
}
