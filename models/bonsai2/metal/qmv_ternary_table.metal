// Bonsai-2 ternary gate/up QMM tail (D4 prototype, table variant).
//
// Entry block for mx.fast.metal_kernel. One thread per output row:
// 17408 threads, each dots its 5120-wide row (40 groups of 128).
// Ternary bytes: 26 per group (130 trits, first 128 used) via the
// 256x5 base-243 table. FP32 activations, F16 scales/biases promoted
// exactly in registers, stock per-group order scale*accum + sum*bias.
// Gate/up shape only; all other projections keep the stock path.
{
  uint tid = thread_position_in_grid.x;
  if (tid >= (uint)out_vec_size[0]) {
    return;
  }
  const device uint8_t* prow = tpack + tid * 1040;
  const device half* srow = scales + tid * 40;
  const device half* brow = biases + tid * 40;
  const device float* xx = x;  // single-row decode
  float yacc = 0.0f;
  for (int gg = 0; gg < 40; gg++) {
    float s = float(srow[gg]);
    float b = float(brow[gg]);
    const device uint8_t* pb = prow + gg * 26;
    const device float* xg = xx + gg * 128;
    float acc = 0.0f;
    float sum = 0.0f;
    for (int bb = 0; bb < 25; bb++) {
      uint8_t v = pb[bb];
      uint d0 = stable[((uint)v) * 5 + 0];
      uint d1 = stable[((uint)v) * 5 + 1];
      uint d2 = stable[((uint)v) * 5 + 2];
      uint d3 = stable[((uint)v) * 5 + 3];
      uint d4 = stable[((uint)v) * 5 + 4];
      int o = 5 * bb;
      acc += xg[o + 0] * d0 + xg[o + 1] * d1 + xg[o + 2] * d2 +
          xg[o + 3] * d3 + xg[o + 4] * d4;
      sum += xg[o + 0] + xg[o + 1] + xg[o + 2] + xg[o + 3] + xg[o + 4];
    }
    // tail weights 125,126,127 = byte 25 digits 0,1,2
    {
      uint8_t v = pb[25];
      acc += xg[125] * stable[((uint)v) * 5 + 0] +
          xg[126] * stable[((uint)v) * 5 + 1] +
          xg[127] * stable[((uint)v) * 5 + 2];
      sum += xg[125] + xg[126] + xg[127];
    }
    yacc += s * acc + sum * b;
  }
  y[tid] = yacc;
}
