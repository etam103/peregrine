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
| matmul_128x128         |      13.4 |      5.5 |     23.6 |       95.5 |    427.0 |     78.3 | PyTorch |
| matmul_256x256         |      59.1 |     30.0 |     46.4 |      193.9 |    425.9 |    147.1 | PyTorch |
| matmul_512x512         |     209.5 |    139.6 |    168.1 |      684.9 |    436.5 |    488.6 | PyTorch |
| matmul_1024x1024       |    1053.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9003.1 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     40.5 |     28.8 |       49.5 |    190.2 |     34.2 | Peregrine |
| add_500k               |      49.6 |     57.3 |     79.9 |       78.4 |    190.2 |     61.3 | Peregrine |
| add_1M                 |     125.5 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     551.1 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     967.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.9 |     39.0 |     28.2 |       42.2 |    194.1 |     33.6 | Peregrine |
| mul_500k               |      63.0 |     58.5 |     81.2 |       77.5 |    197.3 |     59.8 | PyTorch |
| mul_1M                 |     125.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     542.0 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     908.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.6 |     61.4 |     59.4 |       67.6 |    248.7 |     46.4 | JAX |
| exp_500k               |     248.3 |    136.7 |    227.3 |       99.8 |    228.3 |    122.2 | TensorFlow |
| exp_1M                 |     496.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    2486.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    4984.4 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     39.8 |     25.3 |       37.9 |    343.4 |     98.6 | Peregrine |
| relu_1M                |      83.8 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     26.3 |     16.1 |       11.5 |    631.6 |     30.9 | Peregrine |
| softmax_8x512          |       4.2 |     34.8 |     21.3 |       14.3 |    628.3 |     35.9 | Peregrine |
| mlp_fwd_64x784         |      33.5 |     27.5 |     52.1 |      249.4 |   1824.2 |    180.7 | PyTorch |
| mlp_fwd_256x784_wide   |     429.7 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     813.5 |   1277.8 |    776.7 |     8576.3 |  23361.5 |   4991.8 | MLX |
| train_step_256_wide    |    3326.9 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     39.9 |     24.7 |       48.0 |    168.0 |     30.2 | Peregrine |
| square_100k            |       8.8 |     38.8 |     23.5 |       16.3 |    177.9 |     31.5 | Peregrine |
| rsqrt_100k             |      21.9 |     40.1 |     36.5 |       51.9 |        — |     80.6 | Peregrine |
| floor_100k             |       8.8 |     40.8 |     23.5 |       18.0 |    420.6 |     26.8 | Peregrine |
| ceil_100k              |       8.8 |     40.0 |     23.5 |       17.8 |    353.1 |     36.1 | Peregrine |
| round_100k             |       8.8 |     42.7 |     23.5 |       44.2 |        — |     29.4 | Peregrine |
| sign_100k              |       8.8 |     40.2 |     27.7 |       50.8 |    794.9 |     33.9 | Peregrine |
| expm1_100k             |      64.3 |    111.5 |    107.5 |      141.9 |        — |     94.7 | Peregrine |
| log2_100k              |      56.6 |     89.8 |    103.6 |      154.6 |    165.5 |     56.2 | JAX |
| log10_100k             |      59.0 |     86.1 |    108.6 |      155.8 |        — |     56.3 | JAX |
| log1p_100k             |      76.9 |     83.7 |    128.1 |       89.4 |        — |    116.7 | Peregrine |
| erf_100k               |     102.7 |     57.8 |    100.8 |       55.6 |        — |     54.8 | JAX |
| sinh_100k              |      52.0 |    129.0 |     93.5 |      131.3 |    536.5 |    110.8 | Peregrine |
| cosh_100k              |      47.2 |    128.5 |     89.4 |      123.1 |    467.7 |     69.5 | Peregrine |
| arcsin_100k            |      53.1 |     81.7 |     95.2 |       57.7 |   2928.5 |    111.2 | Peregrine |
| arccos_100k            |      61.9 |     90.2 |    110.2 |       56.5 |        — |    203.6 | TensorFlow |
| arctan_100k            |      54.1 |     94.9 |     93.1 |       58.1 |   3056.8 |    223.3 | Peregrine |
| arcsinh_100k           |     209.0 |    151.5 |    333.1 |      136.3 |        — |    112.0 | JAX |
| maximum_100k           |      12.7 |     38.0 |     28.5 |       44.2 |    191.8 |     30.8 | Peregrine |
| minimum_100k           |      12.7 |     40.0 |     28.2 |       40.0 |    384.4 |     28.5 | Peregrine |
| power_100k             |     156.7 |    238.0 |    231.8 |      271.7 |        — |    140.8 | JAX |
| arctan2_100k           |      96.9 |    130.3 |    148.8 |       70.4 |        — |    317.5 | TensorFlow |
| logaddexp_100k         |     277.7 |    146.3 |    262.2 |      354.5 |        — |    142.2 | JAX |
| clip_100k              |       8.8 |     40.9 |     35.2 |       42.9 |    546.6 |     36.1 | Peregrine |
| where_100k             |      16.7 |     52.0 |     28.5 |       65.9 |    275.5 |     33.8 | Peregrine |
| greater_100k           |      12.7 |     48.9 |     25.4 |       47.9 |    190.5 |     27.8 | Peregrine |
| equal_100k             |      12.7 |     31.2 |     25.2 |       55.6 |    292.6 |     27.2 | Peregrine |
| sum_axis_256x512       |      19.2 |     40.6 |     23.0 |       50.6 |    206.1 |     46.8 | Peregrine |
| mean_axis_256x512      |      19.2 |     43.2 |     24.8 |       50.7 |    299.2 |     46.0 | Peregrine |
| max_axis_256x512       |      13.9 |     56.1 |     40.0 |       48.8 |    205.0 |     44.6 | Peregrine |
| min_axis_256x512       |      13.9 |     59.8 |     41.4 |       50.0 |    330.4 |     47.4 | Peregrine |
| var_256x512            |      46.5 |    274.3 |     59.6 |      216.3 |        — |     78.4 | Peregrine |
| prod_axis_256x512      |      24.6 |     38.4 |     25.3 |       51.1 |        — |     56.0 | Peregrine |
| logsumexp_256x512      |      97.2 |    198.4 |    106.8 |      335.1 |        — |    275.6 | Peregrine |
| cumsum_256x512         |     121.0 |     83.7 |    128.3 |      188.7 |    630.6 |    208.6 | PyTorch |
| argmax_axis_256x512    |      52.7 |     91.9 |    170.7 |       73.5 |   1316.6 |    173.3 | Peregrine |
| sum_axis_1024x1024     |     174.1 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     428.1 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      34.7 |     40.5 |     56.5 |       51.4 |   1808.6 |     38.5 | Peregrine |
| triu_256x256           |      34.4 |     39.4 |     55.8 |       54.7 |   1841.1 |     36.4 | Peregrine |
| repeat_64x128_2x3      |       6.0 |     49.0 |     31.3 |       77.8 |        — |     28.5 | Peregrine |
| pad_64x128             |       2.5 |      4.1 |     19.2 |       83.8 |     89.7 |     18.3 | Peregrine |
| stack_8x64x128         |       3.8 |      8.6 |     44.9 |       52.9 |    918.0 |    163.5 | Peregrine |
| diagonal_512x512       |       0.8 |      0.6 |     29.2 |       12.5 |        — |      8.5 | PyTorch |
| silu_100k              |      65.2 |     73.2 |     84.6 |      220.9 |    330.6 |     53.4 | JAX |
| softplus_100k          |     184.1 |    155.2 |    263.6 |      124.2 |    793.6 |    155.3 | TensorFlow |
| mish_100k              |     291.6 |    311.2 |    377.6 |      238.8 |   1153.5 |    230.7 | JAX |
| leaky_relu_100k        |       8.8 |     40.8 |     78.2 |       20.1 |        — |     30.8 | Peregrine |
| elu_100k               |      61.2 |    125.0 |    118.8 |      133.0 |    877.7 |     77.9 | Peregrine |
| hard_tanh_100k         |       8.8 |     41.0 |     34.4 |       42.8 |        — |     35.5 | Peregrine |
| relu6_100k             |       8.8 |     40.0 |     45.8 |       54.4 |    739.5 |    113.5 | Peregrine |
| hardswish_100k         |      10.2 |     40.7 |     68.9 |      209.2 |        — |     29.9 | Peregrine |
| gelu_100k              |      97.2 |     71.7 |    137.4 |      240.9 |    864.0 |    205.3 | PyTorch |
| selu_100k              |      64.9 |    122.2 |     85.9 |      134.8 |    751.5 |     82.0 | Peregrine |
| softsign_100k          |      38.8 |    125.7 |     46.6 |       47.9 |        — |     57.5 | Peregrine |
| cross_entropy_64x10    |       2.6 |     38.3 |     23.3 |      634.3 |   3443.7 |     56.4 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.6 |     19.1 |       44.0 |   1142.9 |     12.3 | Peregrine |
| mse_loss_64x10         |       3.9 |      4.8 |     21.5 |       39.9 |    457.0 |     24.5 | Peregrine |
| huber_loss_64x10       |       5.1 |      4.8 |     32.2 |      242.2 |        — |     47.9 | PyTorch |
| smooth_l1_loss_64x10   |       5.0 |      5.0 |     32.3 |      242.9 |        — |     47.7 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     17.8 |      382.8 |        — |     62.3 | Peregrine |
| cosine_sim_loss_64x64  |      13.8 |     10.3 |    109.8 |      242.5 |        — |     69.2 | PyTorch |
| rmsnorm_64x512         |      58.6 |     67.9 |     32.6 |      441.1 |        — |     68.7 | MLX |
| conv1d_1x32x128_k3     |      20.0 |     55.5 |     28.4 |      505.1 |        — |     73.0 | Peregrine |
| avgpool2d_1x16x32x32   |      25.5 |     46.2 |    260.9 |       63.4 |        — |     41.6 | Peregrine |
| groupnorm_4x64x16x16   |      74.0 |     52.7 |    222.2 |      746.8 |        — |    264.5 | PyTorch |
| rnn_seq32_128_256      |     187.2 |    266.0 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1136.3 |    790.1 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     796.8 |    776.8 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     809.2 |   1315.6 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     935.9 |   1159.7 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     919.8 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1288.9 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     106.4 |    262.6 |    481.2 |      137.4 |   2410.5 |    545.7 | Peregrine |
| rand_normal_100k       |     241.0 |    994.7 |    686.4 |      355.6 |   3309.8 |    614.9 | Peregrine |
| rand_bernoulli_100k    |     309.2 |    250.1 |    451.4 |      229.5 |        — |    521.3 | TensorFlow |
| rand_uniform_1M        |    1081.9 |   2643.6 |   4624.1 |      455.1 |   2430.8 |   2264.3 | TensorFlow |
| rand_normal_1M         |    2410.6 |   9962.2 |   6709.4 |     2189.8 |   3293.8 |   2808.2 | TensorFlow |
| rfft_1k                |       2.1 |      4.5 |     22.3 |       43.4 |        — |     59.7 | Peregrine |
| rfft_4k                |       6.7 |     15.0 |     31.2 |       55.4 |        — |     64.4 | Peregrine |
| rfft_16k               |      29.5 |     66.4 |     78.5 |      106.7 |        — |    116.1 | Peregrine |
| fft_1k                 |       3.4 |      6.7 |     21.6 |        8.9 |        — |     18.2 | Peregrine |
| fft_4k                 |      12.2 |     26.9 |     43.0 |       17.5 |        — |     55.1 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     18.2 |       70.0 |        — |      3.8 | Peregrine |
| solve_64x64            |      11.8 |     23.3 |    104.6 |       25.4 |        — |     32.5 | Peregrine |
| inv_64x64              |      36.9 |     25.1 |     49.2 |       33.5 |        — |     43.9 | PyTorch |
| cholesky_64x64         |       9.0 |     42.5 |     21.7 |       20.1 |        — |     19.8 | Peregrine |
| svd_64x64              |     274.6 |    280.7 |    290.1 |      504.7 |        — |    298.9 | Peregrine |
| qr_64x64               |      39.8 |     85.0 |     64.0 |       86.4 |        — |     66.0 | Peregrine |
| eigh_64x64             |     385.1 |    213.9 |    232.3 |      147.2 |        — |    234.8 | TensorFlow |
| det_64x64              |      22.5 |     19.1 |        — |       23.5 |        — |     28.1 | PyTorch |
| solve_128x128          |      48.9 |     43.3 |    190.4 |       79.0 |        — |     86.7 | PyTorch |
| inv_128x128            |      92.5 |     58.1 |     90.6 |      143.1 |        — |     81.0 | PyTorch |
| cholesky_128x128       |      50.6 |     52.8 |     26.5 |       60.1 |        — |     35.8 | MLX |
| svd_128x128            |     983.1 |    979.4 |    969.4 |     1835.7 |        — |   1016.5 | MLX |
| qr_128x128             |     187.7 |    223.7 |    191.3 |      336.5 |        — |    190.2 | Peregrine |
| eigh_128x128           |    1876.7 |    699.6 |    720.1 |      729.5 |        — |    733.7 | PyTorch |
| det_128x128            |      51.1 |     48.2 |        — |       84.7 |        — |     76.2 | PyTorch |
| solve_256x256          |     190.1 |    178.2 |    735.5 |      390.9 |        — |    258.9 | PyTorch |
| inv_256x256            |     478.6 |    297.2 |    241.4 |      874.0 |        — |    330.7 | MLX |
| cholesky_256x256       |     230.7 |     80.5 |     57.0 |      291.2 |        — |    116.2 | MLX |
| svd_256x256            |    6029.4 |   5701.0 |   5725.6 |     8334.8 |        — |   5885.4 | PyTorch |
| qr_256x256             |    1039.2 |   1002.8 |   1009.2 |     1747.8 |        — |    976.2 | JAX |
| eigh_256x256           |    6400.8 |   3448.1 |   3475.3 |     4683.2 |        — |   3569.4 | PyTorch |
| det_256x256            |     213.7 |    201.7 |        — |      444.6 |        — |    204.9 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1867.9 |    911.1 |        — |     2552.0 |   1252.4 |   2141.8 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3320.3 |   1958.8 |        — |     3971.7 |   1259.6 |   3367.6 | tinygrad |
| add_layernorm_196x768  |     109.6 |    101.7 |        — |     1284.4 |   1129.9 |    227.8 | PyTorch |
| add_layernorm_196x1024 |     138.2 |    104.3 |        — |     1311.5 |   1124.6 |    277.2 | PyTorch |
| matmul_f32_196x768x3072 |     660.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14856.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1490.6 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26901.5 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.61x** (Peregrine is faster)
- **Peregrine vs MLX: 0.45x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.33x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.06x** (Peregrine is faster)
- **Peregrine vs JAX: 0.42x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 90/141 ops
- PyTorch: 26/141 ops
- JAX: 10/141 ops
- TensorFlow: 8/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
