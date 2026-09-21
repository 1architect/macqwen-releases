// Bonsai-2 narrow-F16 gate/up QMM tail (D3 prototype).
//
// Entry block for mx.fast.metal_kernel (the harness wraps this source
// inside the generated kernel function, so no template definitions may
// appear here; only calls into the wheel header helpers are allowed).
// Gate/up shape only: 17408x5120, Q2/G128, FP32 activations, F16
// scales/biases promoted exactly to F32 in registers, stock per-group
// order scale*accum + sum*bias. All other projections keep stock path.
{
  uint3 group = threadgroup_position_in_grid;
  uint simd_lid = thread_index_in_simdgroup;
  uint simd_gid = simdgroup_index_in_threadgroup;
  uint3 tid = uint3(group.x, group.y, 0);

  const device uint8_t* ws = (const device uint8_t*)w;

  thread float x_thread[16];
  thread float result[4] = {0, 0, 0, 0};

  // Q2/G128 constants: pack_factor 16, bytes_per_pack 4,
  // values_per_thread 16, block_size 512, scale_step 8.
  const int in_vec_size_w = in_vec_size[0] >> 2;
  const int in_vec_size_g = in_vec_size[0] >> 7;
  const int out_row = tid.y * 8 + simd_gid * 4;

  ws += out_row * in_vec_size_w + simd_lid * 4;
  scales += out_row * in_vec_size_g + simd_lid / 8;
  biases += out_row * in_vec_size_g + simd_lid / 8;
  x += tid.x * in_vec_size[0] + simd_lid * 16;
  y += tid.x * out_vec_size[0] + out_row;

  for (int k = 0; k < in_vec_size[0]; k += 512) {
    float sum = load_vector<float, float, 16, 2>(x, x_thread);

    for (int row = 0; row < 4; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device half* sl = scales + row * in_vec_size_g;
      const device half* bl = biases + row * in_vec_size_g;

      float s = float(sl[0]);
      float b = float(bl[0]);
      result[row] += qdot<float, 16, 2>(wl, x_thread, s, b, sum);
    }

    ws += 512 >> 2;
    scales += 512 >> 7;
    biases += 512 >> 7;
    x += 512;
  }

  for (int row = 0; row < 4; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = result[row];
    }
  }
}
