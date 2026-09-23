Xcode Performance > Shaders for decode.gputrace (one token, chat defaults at
fd5e398, keep-warm off). Replay GPU total 77.52 ms. Cost column 0.00% for every
pipeline and the Counters tab is unavailable, so SIMD groups are the only
per-kernel measure. Top rows, transcribed from the user's screenshots:

affine_qmv_fast_bfloat16_t_gs_32_b_4_batch_0        7619
affine_qmv_bfloat16_t_gs_32_b_4_batch_0             4839
v_copyfloat32bfloat16                               3033
custom_kernel_gated_delta_step__bfloat16_t_float_128 3024
v_Squarefloat32float32                              2011
v_Sigmoidbfloat16bfloat16                           1971
Ff4IBroadcastBCGf4IBroadcastCBHf4IMultiply... (compiled) 1884
v_copybfloat16float32                               1839
vv_Multiplybfloat16                                 1753
vv_Addbfloat16                                      1638
col_reduce_small_1_reduce_sumfloat16                1578
CV2IBroadcastADV2IBroadcastBEV2OMultiply... (compiled) 1572
gg1_copybfloat16bfloat16                            1557
row_reduce_looped_1_reduce_sumfloat32               1002
Cf4IAsTypeADf4IBroadcastCBEf4IBroadcastBCFf4OMultiply... 943
BV2ISigmoidACV2IBroadcastABDV2IBroadcastBAEV2OMultiply... 801
gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0    780
