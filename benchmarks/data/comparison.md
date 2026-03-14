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
| matmul_128x128         |      11.3 |      5.8 |     24.0 |       94.8 |    419.8 |     77.8 | PyTorch |
| matmul_256x256         |      68.9 |     31.8 |     47.1 |      198.1 |    420.7 |    146.8 | PyTorch |
| matmul_512x512         |     218.8 |    142.2 |    164.3 |      691.1 |    469.4 |    498.7 | PyTorch |
| matmul_1024x1024       |    1038.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9231.5 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.8 |     39.9 |     28.3 |       47.2 |    187.0 |     37.0 | Peregrine |
| add_500k               |      63.7 |     62.8 |     79.8 |       83.0 |    185.2 |     59.5 | JAX |
| add_1M                 |     129.7 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     507.0 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     890.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.9 |     39.5 |     28.8 |       44.7 |    185.0 |     29.0 | Peregrine |
| mul_500k               |      63.6 |     65.6 |     82.6 |       74.9 |    187.0 |     58.9 | JAX |
| mul_1M                 |     129.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     516.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     882.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.2 |     61.3 |     60.9 |       67.1 |    222.5 |     46.3 | JAX |
| exp_500k               |     253.7 |    140.7 |    223.0 |       95.1 |    221.6 |    119.3 | TensorFlow |
| exp_1M                 |     507.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1272.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2119.9 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.7 |     36.6 |     24.9 |       37.9 |    339.0 |     97.7 | Peregrine |
| relu_1M                |      83.9 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     28.2 |     16.1 |       11.3 |    626.6 |     31.4 | Peregrine |
| softmax_8x512          |       4.3 |     31.5 |     18.6 |       14.2 |    614.4 |     33.7 | Peregrine |
| mlp_fwd_64x784         |      32.7 |     27.9 |     51.8 |      250.7 |   1770.1 |    185.8 | PyTorch |
| mlp_fwd_256x784_wide   |     423.0 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     820.8 |   1228.9 |    771.5 |     8445.4 |  23395.5 |   5048.7 | MLX |
| train_step_256_wide    |    3299.7 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.6 |     35.8 |     24.5 |       50.4 |    162.4 |     28.8 | Peregrine |
| square_100k            |       8.6 |     34.7 |     23.5 |       16.2 |    174.7 |     29.4 | Peregrine |
| rsqrt_100k             |      21.5 |     40.2 |     32.5 |       50.6 |        — |     92.5 | Peregrine |
| floor_100k             |       8.6 |     37.1 |     23.7 |       17.5 |    411.6 |     28.6 | Peregrine |
| ceil_100k              |       8.6 |     40.2 |     23.6 |       17.6 |    349.4 |     32.0 | Peregrine |
| round_100k             |       8.7 |     40.6 |     23.5 |       44.4 |        — |     28.5 | Peregrine |
| sign_100k              |       8.7 |     35.6 |     27.7 |       47.7 |    790.0 |     36.6 | Peregrine |
| expm1_100k             |      63.2 |    114.9 |    107.7 |      149.7 |        — |     98.8 | Peregrine |
| log2_100k              |      56.7 |     84.8 |    101.9 |      155.9 |    167.0 |     56.2 | JAX |
| log10_100k             |      59.0 |     91.2 |    107.0 |      146.0 |        — |     56.1 | JAX |
| log1p_100k             |      76.9 |     81.4 |    127.7 |       92.9 |        — |    106.4 | Peregrine |
| erf_100k               |     102.6 |     51.7 |    100.6 |       57.5 |        — |     54.0 | PyTorch |
| sinh_100k              |      52.0 |    131.2 |     93.5 |      127.9 |    528.6 |    114.4 | Peregrine |
| cosh_100k              |      47.2 |    132.6 |     89.5 |      130.2 |    462.1 |     69.4 | Peregrine |
| arcsin_100k            |      53.1 |     70.4 |     93.8 |       57.2 |   2903.8 |    111.7 | Peregrine |
| arccos_100k            |      61.8 |     89.0 |    114.3 |       51.9 |        — |    200.7 | TensorFlow |
| arctan_100k            |      54.1 |     98.5 |     93.1 |       59.9 |   3086.5 |    223.4 | Peregrine |
| arcsinh_100k           |     208.7 |    163.3 |    342.3 |      139.1 |        — |    137.6 | JAX |
| maximum_100k           |      15.9 |     36.3 |     29.7 |       43.0 |    192.0 |     31.3 | Peregrine |
| minimum_100k           |      12.7 |     35.0 |     26.5 |       41.5 |    371.0 |     25.9 | Peregrine |
| power_100k             |     156.7 |    242.4 |    216.5 |      283.1 |        — |    145.3 | JAX |
| arctan2_100k           |      96.9 |    142.7 |    148.8 |       74.8 |        — |    312.7 | TensorFlow |
| logaddexp_100k         |     277.6 |    148.7 |    258.5 |      363.8 |        — |    160.5 | PyTorch |
| clip_100k              |       8.8 |     37.4 |     34.8 |       42.5 |    530.9 |     43.2 | Peregrine |
| where_100k             |      16.7 |     47.7 |     27.4 |       66.1 |    276.5 |     30.1 | Peregrine |
| greater_100k           |      12.7 |     46.1 |     23.9 |       53.9 |    189.5 |     26.5 | Peregrine |
| equal_100k             |      12.7 |     23.8 |     24.3 |       57.1 |    288.1 |     27.1 | Peregrine |
| sum_axis_256x512       |      19.2 |     37.0 |     22.6 |       51.6 |    205.5 |     51.9 | Peregrine |
| mean_axis_256x512      |      19.2 |     40.8 |     24.7 |       52.3 |    286.0 |     53.1 | Peregrine |
| max_axis_256x512       |      13.9 |     50.9 |     40.8 |       52.8 |    204.0 |     45.1 | Peregrine |
| min_axis_256x512       |      13.7 |     48.0 |     41.3 |       48.9 |    324.3 |     45.2 | Peregrine |
| var_256x512            |      45.7 |    274.2 |     61.5 |      219.3 |        — |     84.0 | Peregrine |
| prod_axis_256x512      |      24.2 |     36.5 |     25.5 |       51.4 |        — |     58.2 | Peregrine |
| logsumexp_256x512      |      95.5 |    192.8 |    106.6 |      340.8 |        — |    279.2 | Peregrine |
| cumsum_256x512         |     118.9 |     79.0 |    128.2 |      190.6 |    608.8 |    212.8 | PyTorch |
| argmax_axis_256x512    |      51.7 |     93.7 |    170.3 |       72.3 |   1283.6 |    168.4 | Peregrine |
| sum_axis_1024x1024     |     174.1 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.9 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      42.9 |     35.9 |     55.5 |       52.6 |   1818.2 |     35.4 | JAX |
| triu_256x256           |      34.6 |     34.5 |     56.0 |       55.6 |   1792.0 |     36.3 | PyTorch |
| repeat_64x128_2x3      |       7.4 |     41.5 |     30.1 |       75.2 |        — |     28.0 | Peregrine |
| pad_64x128             |       2.6 |      4.1 |     17.9 |       84.1 |     90.7 |     18.0 | Peregrine |
| stack_8x64x128         |       3.8 |      8.7 |     44.6 |       59.1 |    909.9 |    157.8 | Peregrine |
| diagonal_512x512       |       0.8 |      0.7 |     28.8 |       12.5 |        — |      8.9 | PyTorch |
| silu_100k              |      64.0 |     73.1 |     82.8 |      225.8 |    323.7 |     53.2 | JAX |
| softplus_100k          |     180.6 |    156.3 |    260.0 |      126.6 |    775.3 |    155.1 | TensorFlow |
| mish_100k              |     286.0 |    308.6 |    368.8 |      244.3 |   1185.5 |    231.0 | JAX |
| leaky_relu_100k        |       8.7 |     37.6 |     76.1 |       19.5 |        — |     29.1 | Peregrine |
| elu_100k               |      60.0 |    116.1 |    116.6 |      130.5 |    875.6 |     77.4 | Peregrine |
| hard_tanh_100k         |       8.7 |     37.7 |     34.6 |       41.9 |        — |     39.3 | Peregrine |
| relu6_100k             |       8.7 |     36.7 |     43.5 |       51.5 |    730.0 |    112.2 | Peregrine |
| hardswish_100k         |      10.0 |     36.5 |     69.3 |      206.0 |        — |     28.6 | Peregrine |
| gelu_100k              |      95.3 |     64.3 |    135.4 |      250.0 |    843.4 |    206.4 | PyTorch |
| selu_100k              |      63.7 |    133.3 |     85.1 |      142.0 |    740.9 |     81.7 | Peregrine |
| softsign_100k          |      38.1 |    108.4 |     43.6 |       48.1 |        — |     57.7 | Peregrine |
| cross_entropy_64x10    |       2.6 |     35.9 |     22.5 |      643.9 |   3329.7 |     54.4 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.3 |     19.9 |       44.6 |   1115.0 |     12.5 | Peregrine |
| mse_loss_64x10         |       3.6 |      4.8 |     21.6 |       40.5 |    443.0 |     23.9 | Peregrine |
| huber_loss_64x10       |       5.0 |      4.8 |     33.5 |      243.2 |        — |     47.9 | PyTorch |
| smooth_l1_loss_64x10   |       4.8 |      5.0 |     32.6 |      236.3 |        — |     48.3 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.5 |     18.6 |      412.8 |        — |     60.9 | Peregrine |
| cosine_sim_loss_64x64  |      13.5 |     10.2 |    108.7 |      258.4 |        — |     73.1 | PyTorch |
| rmsnorm_64x512         |      57.6 |     63.5 |     32.9 |      465.6 |        — |     71.5 | MLX |
| conv1d_1x32x128_k3     |      21.3 |     46.8 |     27.6 |      503.6 |        — |     74.7 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     42.4 |    260.7 |       62.1 |        — |     43.2 | Peregrine |
| groupnorm_4x64x16x16   |      72.5 |     50.2 |    221.6 |      760.5 |        — |    262.9 | PyTorch |
| rnn_seq32_128_256      |     195.7 |    270.0 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1149.3 |    805.0 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     824.9 |    783.1 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     810.2 |   1188.3 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     925.0 |   1058.6 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     920.8 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1275.9 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     106.3 |    257.2 |    479.8 |      124.6 |   2390.2 |    556.2 | Peregrine |
| rand_normal_100k       |     236.4 |    972.8 |    685.7 |      332.9 |   3258.2 |    642.2 | Peregrine |
| rand_bernoulli_100k    |     303.4 |    250.0 |    447.9 |      218.9 |        — |    534.1 | TensorFlow |
| rand_uniform_1M        |    1064.3 |   2566.4 |   4540.3 |      418.9 |   2358.0 |   2286.9 | TensorFlow |
| rand_normal_1M         |    2368.7 |   9702.3 |   6575.8 |     2066.6 |   3274.5 |   2881.3 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     26.1 |       43.4 |        — |     60.4 | Peregrine |
| rfft_4k                |       6.5 |     14.8 |     32.5 |       54.2 |        — |     66.2 | Peregrine |
| rfft_16k               |      30.3 |     65.4 |     77.6 |      104.7 |        — |    116.4 | Peregrine |
| fft_1k                 |       3.3 |      6.8 |     24.1 |        9.0 |        — |     39.3 | Peregrine |
| fft_4k                 |      12.1 |     26.2 |     42.9 |       17.6 |        — |     54.9 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     19.8 |       69.6 |        — |      4.0 | Peregrine |
| solve_64x64            |      12.1 |     18.5 |    100.3 |       24.5 |        — |     32.2 | Peregrine |
| inv_64x64              |      37.3 |     26.0 |     51.5 |       32.5 |        — |     37.2 | PyTorch |
| cholesky_64x64         |       9.7 |     42.6 |     21.7 |       19.4 |        — |     20.1 | Peregrine |
| svd_64x64              |     275.8 |    281.2 |    284.9 |      495.9 |        — |    303.4 | Peregrine |
| qr_64x64               |      41.2 |     82.9 |     58.4 |       85.1 |        — |     62.9 | Peregrine |
| eigh_64x64             |     379.7 |    215.6 |    231.2 |      148.5 |        — |    236.3 | TensorFlow |
| det_64x64              |      23.2 |     19.9 |        — |       23.4 |        — |     28.6 | PyTorch |
| solve_128x128          |      50.1 |     45.1 |    187.8 |       76.7 |        — |     85.1 | PyTorch |
| inv_128x128            |      92.2 |     62.0 |     90.5 |      138.9 |        — |     82.8 | PyTorch |
| cholesky_128x128       |      50.6 |     54.6 |     26.2 |       58.5 |        — |     38.0 | MLX |
| svd_128x128            |     985.0 |    992.5 |    997.9 |     1823.2 |        — |   1018.0 | Peregrine |
| qr_128x128             |     188.2 |    222.1 |    198.1 |      326.7 |        — |    190.2 | Peregrine |
| eigh_128x128           |    1840.8 |    704.5 |    726.7 |      710.6 |        — |    746.2 | PyTorch |
| det_128x128            |      52.3 |     49.8 |        — |       82.1 |        — |     76.3 | PyTorch |
| solve_256x256          |     188.3 |    187.8 |    749.3 |      377.2 |        — |    265.0 | PyTorch |
| inv_256x256            |     468.3 |    301.8 |    248.8 |      848.2 |        — |    333.5 | MLX |
| cholesky_256x256       |     226.2 |     76.7 |     55.6 |      281.1 |        — |    116.8 | MLX |
| svd_256x256            |    6035.5 |   5722.5 |   5559.6 |     7963.8 |        — |   5770.7 | MLX |
| qr_256x256             |    1038.9 |    996.6 |   1016.0 |     1697.1 |        — |    968.5 | JAX |
| eigh_256x256           |    6091.8 |   3448.9 |   3478.1 |     4532.8 |        — |   3553.5 | PyTorch |
| det_256x256            |     212.3 |    203.6 |        — |      433.5 |        — |    204.8 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1430.0 |    946.2 |        — |     2378.5 |   1218.0 |   2115.3 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2643.4 |   1905.1 |        — |     3683.8 |   1234.8 |   3392.5 | tinygrad |
| add_layernorm_196x768  |     105.9 |     98.0 |        — |     1223.8 |   1123.8 |    232.8 | PyTorch |
| add_layernorm_196x1024 |     135.9 |    104.2 |        — |     1270.2 |   1125.8 |    272.0 | PyTorch |
| matmul_f32_196x768x3072 |     614.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14480.4 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1476.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26314.0 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.62x** (Peregrine is faster)
- **Peregrine vs MLX: 0.45x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.33x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.06x** (Peregrine is faster)
- **Peregrine vs JAX: 0.42x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 88/141 ops
- PyTorch: 27/141 ops
- JAX: 11/141 ops
- TensorFlow: 8/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
