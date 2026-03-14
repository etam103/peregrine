# Peregrine vs ML Frameworks — Wall-Clock Benchmark

All benchmarks run on CPU with `nice -n 10`. Times in microseconds (lower is better).

**Versions:**
- PyTorch: 2.10.0
- MLX: 0.30.6
- TensorFlow: 2.20.0
- tinygrad: ?
- JAX: 0.9.0.1

| Operation | Peregrine | PyTorch | MLX | TensorFlow | tinygrad | JAX | Best |
|-----------|----------: | --------: | --------: | ----------: | --------: | --------: | ----:|
| matmul_128x128         |       6.0 |      7.4 |     22.5 |       53.3 |    430.6 |     59.1 | Peregrine |
| matmul_256x256         |      69.5 |     35.9 |     51.7 |      128.0 |    446.8 |    153.8 | PyTorch |
| matmul_512x512         |     216.3 |    154.5 |    146.3 |      689.5 |    474.7 |    552.0 | MLX |
| matmul_1024x1024       |    1072.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9059.1 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      13.0 |     35.4 |     28.7 |       53.1 |    187.3 |     44.5 | Peregrine |
| add_500k               |     111.4 |     62.9 |     80.1 |       78.9 |    194.8 |     63.4 | PyTorch |
| add_1M                 |     121.3 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     510.0 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     888.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     43.0 |     29.0 |       41.1 |    195.8 |     31.1 | Peregrine |
| mul_500k               |     116.0 |     62.2 |     81.9 |       72.7 |    195.4 |     60.1 | JAX |
| mul_1M                 |     171.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     531.7 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     888.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      94.2 |     44.0 |     60.9 |       69.3 |    229.8 |     48.6 | PyTorch |
| exp_500k               |     177.5 |    149.8 |    237.8 |      116.0 |    238.6 |    124.5 | TensorFlow |
| exp_1M                 |     297.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1102.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2167.4 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     35.1 |     26.5 |       37.4 |    340.2 |     99.0 | Peregrine |
| relu_1M                |     114.8 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     30.6 |     15.3 |       11.2 |    640.7 |     30.9 | Peregrine |
| softmax_8x512          |       4.2 |     33.4 |     17.8 |       13.8 |    628.8 |     33.7 | Peregrine |
| mlp_fwd_64x784         |      33.1 |     29.2 |     48.3 |      221.1 |   1930.1 |    180.9 | PyTorch |
| mlp_fwd_256x784_wide   |     423.7 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     818.9 |   1268.2 |    857.0 |     8654.6 |  28348.8 |   5194.2 | Peregrine |
| train_step_256_wide    |    3318.7 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     32.3 |     24.3 |       49.9 |    187.8 |     27.2 | Peregrine |
| square_100k            |       8.8 |     33.6 |     23.0 |       15.0 |    198.4 |     28.6 | Peregrine |
| rsqrt_100k             |      72.8 |     32.1 |     31.2 |       56.2 |        — |     81.0 | MLX |
| floor_100k             |      47.5 |     32.9 |     23.6 |       16.6 |    479.6 |     28.0 | TensorFlow |
| ceil_100k              |      47.5 |     33.7 |     23.5 |       16.5 |    392.0 |     28.2 | TensorFlow |
| round_100k             |      47.5 |     34.5 |     20.7 |       47.1 |        — |     32.8 | MLX |
| sign_100k              |      55.4 |     33.9 |     26.9 |       43.3 |    919.6 |     35.6 | MLX |
| expm1_100k             |     145.3 |     94.5 |    107.1 |      152.6 |        — |     99.3 | PyTorch |
| log2_100k              |     112.8 |     64.2 |    100.5 |      141.4 |    207.8 |     56.6 | JAX |
| log10_100k             |     112.5 |     65.0 |    116.4 |      151.2 |        — |     58.3 | JAX |
| log1p_100k             |     113.6 |     63.1 |    140.7 |       96.5 |        — |    105.5 | PyTorch |
| erf_100k               |      96.0 |     37.4 |    112.3 |       55.9 |        — |     55.2 | PyTorch |
| sinh_100k              |      52.0 |    102.9 |    108.0 |      135.4 |    671.1 |    144.6 | Peregrine |
| cosh_100k              |      47.2 |    100.7 |     93.6 |      128.4 |    600.8 |     78.6 | Peregrine |
| arcsin_100k            |      53.1 |     58.4 |    103.2 |       53.9 |   3639.7 |    123.4 | Peregrine |
| arccos_100k            |     109.5 |     68.7 |    116.0 |       51.0 |        — |    207.9 | TensorFlow |
| arctan_100k            |      54.1 |     74.7 |     96.9 |       58.7 |   3810.2 |    222.9 | Peregrine |
| arcsinh_100k           |     133.5 |    131.6 |    349.9 |      140.3 |        — |    140.3 | PyTorch |
| maximum_100k           |      12.7 |     29.7 |     26.8 |       38.6 |    194.7 |     30.0 | Peregrine |
| minimum_100k           |      12.7 |     31.1 |     30.2 |       43.6 |    392.5 |     31.3 | Peregrine |
| power_100k             |     392.2 |    214.0 |    230.0 |      253.2 |        — |    144.1 | JAX |
| arctan2_100k           |    1118.2 |    119.4 |    151.2 |       50.1 |        — |    317.5 | TensorFlow |
| logaddexp_100k         |     415.0 |    125.2 |    272.0 |      320.2 |        — |    146.0 | PyTorch |
| clip_100k              |       8.8 |     31.4 |     38.4 |       41.4 |    558.4 |     35.1 | Peregrine |
| where_100k             |      94.9 |     35.3 |     27.1 |       65.0 |    289.3 |     33.1 | MLX |
| greater_100k           |      71.4 |     34.2 |     22.9 |       45.6 |    195.8 |     28.3 | MLX |
| equal_100k             |      71.3 |     29.6 |     21.3 |       59.6 |    296.2 |     31.9 | MLX |
| sum_axis_256x512       |     114.7 |     34.4 |     22.6 |       50.7 |    212.7 |     50.4 | MLX |
| mean_axis_256x512      |     114.7 |     36.6 |     23.3 |       48.4 |    308.2 |     45.6 | MLX |
| max_axis_256x512       |     154.3 |     40.0 |     39.8 |       47.0 |    217.2 |     45.5 | MLX |
| min_axis_256x512       |     154.2 |     37.4 |     40.1 |       46.2 |    346.6 |     49.1 | PyTorch |
| var_256x512            |     235.7 |    355.1 |     56.2 |      188.2 |        — |     82.7 | MLX |
| prod_axis_256x512      |     148.9 |     32.8 |     24.0 |       49.0 |        — |     55.9 | MLX |
| logsumexp_256x512      |     385.5 |    144.9 |    122.2 |      289.9 |        — |    284.8 | MLX |
| cumsum_256x512         |     122.6 |     61.1 |    133.4 |      161.8 |    656.1 |    215.5 | PyTorch |
| argmax_axis_256x512    |     157.5 |     66.1 |    179.8 |       61.6 |   1358.4 |    175.1 | TensorFlow |
| sum_axis_1024x1024     |     940.4 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    1922.6 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      34.7 |     28.3 |     54.9 |       45.0 |   1880.5 |     36.3 | PyTorch |
| triu_256x256           |      41.1 |     33.4 |     52.0 |       44.9 |   1908.4 |     36.7 | PyTorch |
| repeat_64x128_2x3      |     124.7 |     39.0 |     25.4 |       72.7 |        — |     28.0 | MLX |
| pad_64x128             |      17.2 |      4.5 |     15.2 |       84.7 |     93.1 |     18.6 | PyTorch |
| stack_8x64x128         |      25.6 |      9.4 |     44.2 |       49.4 |    958.2 |    160.1 | PyTorch |
| diagonal_512x512       |       0.8 |      0.7 |     33.8 |       11.1 |        — |      9.2 | PyTorch |
| silu_100k              |      65.2 |     59.2 |     87.3 |      209.5 |    342.2 |     52.8 | JAX |
| softplus_100k          |     298.7 |    125.8 |    275.7 |      103.5 |    818.7 |    154.8 | TensorFlow |
| mish_100k              |     537.0 |    308.2 |    390.4 |      242.1 |   1197.4 |    232.9 | JAX |
| leaky_relu_100k        |       8.1 |     44.8 |     79.5 |       18.4 |        — |     29.9 | Peregrine |
| elu_100k               |     145.1 |    105.8 |    128.3 |      134.3 |    905.2 |     77.7 | JAX |
| hard_tanh_100k         |      51.5 |     33.0 |     35.9 |       40.2 |        — |     39.5 | PyTorch |
| relu6_100k             |      51.5 |     36.2 |     44.5 |       49.0 |    755.9 |    111.4 | PyTorch |
| hardswish_100k         |      85.1 |     32.1 |     71.8 |      207.6 |        — |     28.5 | JAX |
| gelu_100k              |      78.3 |     48.4 |    140.8 |      251.2 |    888.2 |    210.8 | PyTorch |
| selu_100k              |     184.9 |    130.8 |     85.3 |      150.4 |    829.0 |     90.3 | MLX |
| softsign_100k          |      35.5 |    134.4 |     42.4 |       47.9 |        — |     58.5 | Peregrine |
| cross_entropy_64x10    |       2.7 |     44.2 |     23.0 |      601.5 |   3528.0 |     53.2 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.6 |     19.3 |       40.2 |   1170.6 |     12.1 | Peregrine |
| mse_loss_64x10         |       4.1 |      5.2 |     19.2 |       35.7 |    464.1 |     23.8 | Peregrine |
| huber_loss_64x10       |       5.2 |      5.0 |     35.7 |      224.5 |        — |     47.1 | PyTorch |
| smooth_l1_loss_64x10   |       5.0 |      5.5 |     31.5 |      231.7 |        — |     47.1 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.8 |     18.2 |      374.1 |        — |     57.7 | Peregrine |
| cosine_sim_loss_64x64  |      13.8 |     11.3 |    108.2 |      233.7 |        — |     55.0 | PyTorch |
| rmsnorm_64x512         |      58.6 |     55.8 |     34.2 |      463.9 |        — |     69.0 | MLX |
| conv1d_1x32x128_k3     |      20.3 |     50.8 |     27.4 |      513.6 |        — |     74.4 | Peregrine |
| avgpool2d_1x16x32x32   |      25.5 |     34.0 |    276.8 |       61.2 |        — |     45.1 | Peregrine |
| groupnorm_4x64x16x16   |      74.0 |     41.6 |    225.0 |      783.4 |        — |    268.0 | PyTorch |
| rnn_seq32_128_256      |     184.5 |    272.3 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1129.8 |    814.9 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     854.4 |    791.7 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     808.9 |   1303.8 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     934.3 |   1149.0 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     931.0 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1306.4 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     108.4 |    265.4 |    492.1 |      130.5 |   2664.4 |    546.7 | Peregrine |
| rand_normal_100k       |     774.8 |   1003.5 |    723.2 |      356.9 |   3442.1 |    616.2 | TensorFlow |
| rand_bernoulli_100k    |     309.1 |    257.7 |    534.9 |      207.4 |        — |    538.4 | TensorFlow |
| rand_uniform_1M        |    1083.7 |   2647.3 |   4661.4 |      430.8 |   2497.5 |   2310.1 | TensorFlow |
| rand_normal_1M         |    7824.1 |  10124.9 |   6707.2 |     2082.2 |   3372.7 |   2887.5 | TensorFlow |
| rfft_1k                |       2.1 |      4.7 |     22.7 |       39.6 |        — |     22.8 | Peregrine |
| rfft_4k                |       6.8 |     15.9 |     30.0 |       51.6 |        — |     70.6 | Peregrine |
| rfft_16k               |      29.8 |     69.3 |     77.6 |      105.9 |        — |    125.0 | Peregrine |
| fft_1k                 |       3.2 |      7.2 |     21.8 |        8.0 |        — |     18.2 | Peregrine |
| fft_4k                 |      12.0 |     28.0 |     40.9 |       16.2 |        — |     70.9 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     17.5 |       64.2 |        — |      3.8 | Peregrine |
| solve_64x64            |      11.8 |     18.2 |     94.2 |       23.6 |        — |     32.6 | Peregrine |
| inv_64x64              |      36.6 |     26.6 |     48.4 |       31.6 |        — |     44.9 | PyTorch |
| cholesky_64x64         |       9.1 |     44.1 |     22.9 |       18.8 |        — |     22.1 | Peregrine |
| svd_64x64              |     278.7 |    287.0 |    297.2 |      511.4 |        — |    297.4 | Peregrine |
| qr_64x64               |      40.5 |     78.4 |     58.6 |       82.1 |        — |     65.0 | Peregrine |
| eigh_64x64             |     389.7 |    224.3 |    231.3 |      148.5 |        — |    237.3 | TensorFlow |
| det_64x64              |      22.6 |     19.9 |        — |       22.0 |        — |     28.4 | PyTorch |
| solve_128x128          |      49.4 |     44.9 |    191.3 |       75.0 |        — |     85.2 | PyTorch |
| inv_128x128            |      96.2 |     60.3 |     90.3 |      139.8 |        — |     83.8 | PyTorch |
| cholesky_128x128       |      50.4 |     54.4 |     26.3 |       57.1 |        — |     35.8 | MLX |
| svd_128x128            |     997.4 |   1002.4 |    993.0 |     1879.1 |        — |   1016.5 | MLX |
| qr_128x128             |     189.8 |    225.5 |    195.3 |      335.3 |        — |    189.9 | Peregrine |
| eigh_128x128           |    1896.7 |    702.3 |    715.3 |      703.8 |        — |    742.0 | PyTorch |
| det_128x128            |      51.3 |     49.8 |        — |       82.7 |        — |     75.8 | PyTorch |
| solve_256x256          |     191.6 |    181.2 |    762.1 |      376.0 |        — |    265.5 | PyTorch |
| inv_256x256            |     490.6 |    302.4 |    251.1 |      871.6 |        — |    333.4 | MLX |
| cholesky_256x256       |     232.2 |     90.3 |     56.6 |      283.0 |        — |    116.4 | MLX |
| svd_256x256            |    6111.2 |   5652.9 |   5885.9 |     8264.6 |        — |   6050.1 | PyTorch |
| qr_256x256             |    1053.4 |   1009.1 |   1041.3 |     1736.1 |        — |    997.5 | JAX |
| eigh_256x256           |    6198.4 |   3500.3 |   3517.1 |     4679.9 |        — |   3525.8 | PyTorch |
| det_256x256            |     219.3 |    202.8 |        — |      434.4 |        — |    209.8 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1214.5 |    919.9 |        — |     2420.7 |   1281.0 |   2140.6 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2395.2 |   1944.3 |        — |     3756.2 |   1278.0 |   3416.1 | tinygrad |
| add_layernorm_196x768  |     104.2 |    101.8 |        — |     1221.2 |   1164.3 |    219.5 | PyTorch |
| add_layernorm_196x1024 |     131.3 |    109.3 |        — |     1267.3 |   1175.4 |    283.1 | PyTorch |
| matmul_f32_196x768x3072 |     609.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14755.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1461.0 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26546.5 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 1.02x** (Peregrine is slower)
- **Peregrine vs MLX: 0.73x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.54x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.09x** (Peregrine is faster)
- **Peregrine vs JAX: 0.67x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 62/141 ops
- PyTorch: 37/141 ops
- MLX: 20/141 ops
- TensorFlow: 12/141 ops
- JAX: 9/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
