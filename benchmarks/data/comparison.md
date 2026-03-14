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
| matmul_128x128         |      31.7 |      6.7 |     21.0 |       94.9 |    445.8 |     80.3 | PyTorch |
| matmul_256x256         |      79.6 |     32.8 |     45.5 |      189.9 |    422.2 |    150.1 | PyTorch |
| matmul_512x512         |     217.8 |    133.2 |    165.7 |      667.9 |    421.4 |    502.4 | PyTorch |
| matmul_1024x1024       |     967.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9108.9 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.8 |     40.7 |     28.1 |       51.1 |    184.7 |     27.9 | Peregrine |
| add_500k               |      62.9 |     64.4 |     78.7 |       79.1 |    186.6 |     59.7 | JAX |
| add_1M                 |      96.0 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     615.9 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     870.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     52.5 |     28.9 |       42.2 |    188.2 |     30.6 | Peregrine |
| mul_500k               |      63.7 |     59.4 |     78.6 |       71.4 |    186.3 |     57.9 | JAX |
| mul_1M                 |     125.4 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     574.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |    1031.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.3 |     83.2 |     57.8 |       66.9 |    217.2 |     46.9 | JAX |
| exp_500k               |     106.2 |    149.8 |    221.6 |      101.8 |    220.3 |    119.3 | TensorFlow |
| exp_1M                 |     136.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     452.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     813.8 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.5 |     46.4 |     25.4 |       39.5 |    330.5 |     97.2 | Peregrine |
| relu_1M                |      82.3 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     45.1 |     20.4 |       11.4 |    617.6 |     30.8 | Peregrine |
| softmax_8x512          |       4.2 |     48.6 |     21.2 |       14.3 |    617.7 |     32.8 | Peregrine |
| mlp_fwd_64x784         |      31.8 |     27.2 |     51.0 |      245.7 |   1769.3 |    176.9 | PyTorch |
| mlp_fwd_256x784_wide   |     391.4 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     803.1 |   1270.2 |    768.6 |     8301.6 |  23893.6 |   5003.0 | MLX |
| train_step_256_wide    |    3310.9 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.6 |     29.5 |     26.2 |       47.0 |    167.1 |     32.6 | Peregrine |
| square_100k            |       8.6 |     28.1 |     25.1 |       14.7 |    178.6 |     27.7 | Peregrine |
| rsqrt_100k             |      21.5 |     33.3 |     33.3 |       50.6 |        — |     92.1 | Peregrine |
| floor_100k             |       9.2 |     34.3 |     24.8 |       16.2 |    425.7 |     28.5 | Peregrine |
| ceil_100k              |       9.0 |     34.1 |     25.2 |       16.2 |    363.4 |     28.6 | Peregrine |
| round_100k             |       8.8 |     31.2 |     23.7 |       47.0 |        — |     31.2 | Peregrine |
| sign_100k              |       8.5 |     30.7 |     28.8 |       46.0 |    803.4 |     39.8 | Peregrine |
| expm1_100k             |      63.2 |     73.9 |    104.4 |      147.7 |        — |     99.1 | Peregrine |
| log2_100k              |      55.6 |     88.0 |     97.6 |      145.0 |    165.9 |     55.9 | Peregrine |
| log10_100k             |      58.0 |     85.6 |    106.7 |      135.9 |        — |     55.8 | JAX |
| log1p_100k             |      76.8 |     82.2 |    133.0 |       95.9 |        — |    104.7 | Peregrine |
| erf_100k               |     102.8 |     59.9 |    100.8 |       53.9 |        — |     42.6 | JAX |
| sinh_100k              |      51.2 |    121.0 |     93.4 |      124.6 |    545.7 |    105.5 | Peregrine |
| cosh_100k              |      46.4 |    119.5 |     91.8 |      122.1 |    470.8 |     68.8 | Peregrine |
| arcsin_100k            |      52.1 |     80.0 |     94.3 |       53.8 |   2925.0 |    112.0 | Peregrine |
| arccos_100k            |      60.7 |     87.5 |    110.3 |       54.0 |        — |    192.2 | TensorFlow |
| arctan_100k            |      53.1 |     92.7 |     93.2 |       57.5 |   3150.7 |    214.8 | Peregrine |
| arcsinh_100k           |     207.2 |    155.7 |    332.7 |      131.2 |        — |    115.1 | JAX |
| maximum_100k           |      12.5 |     43.7 |     29.1 |       40.3 |    194.4 |     30.0 | Peregrine |
| minimum_100k           |      12.5 |     43.8 |     27.5 |       42.7 |    382.7 |     29.1 | Peregrine |
| power_100k             |     153.9 |    236.2 |    215.7 |      265.0 |        — |    140.5 | JAX |
| arctan2_100k           |      95.8 |    135.1 |    145.0 |       75.8 |        — |    315.1 | TensorFlow |
| logaddexp_100k         |     277.7 |    155.8 |    255.6 |      371.2 |        — |    147.8 | JAX |
| clip_100k              |       8.7 |     42.9 |     35.1 |       42.1 |    595.6 |     41.8 | Peregrine |
| where_100k             |      16.4 |     54.0 |     29.4 |       66.0 |    287.0 |     34.4 | Peregrine |
| greater_100k           |      12.8 |     44.7 |     24.4 |       49.6 |    195.1 |     29.1 | Peregrine |
| equal_100k             |      12.9 |     28.4 |     24.0 |       49.6 |    300.9 |     27.4 | Peregrine |
| sum_axis_256x512       |      18.9 |     39.0 |     22.6 |       48.0 |    208.8 |     46.1 | Peregrine |
| mean_axis_256x512      |      18.9 |     40.7 |     25.1 |       51.5 |    291.7 |     56.1 | Peregrine |
| max_axis_256x512       |      13.7 |     55.9 |     40.1 |       50.9 |    216.9 |     46.6 | Peregrine |
| min_axis_256x512       |      13.7 |     53.7 |     42.0 |       51.9 |    341.8 |     46.6 | Peregrine |
| var_256x512            |      45.7 |    272.0 |     64.8 |      239.6 |        — |     77.1 | Peregrine |
| prod_axis_256x512      |      24.2 |     39.8 |     27.3 |       53.3 |        — |     54.3 | Peregrine |
| logsumexp_256x512      |      95.5 |    194.4 |    106.8 |      358.8 |        — |    268.4 | Peregrine |
| cumsum_256x512         |     112.8 |     81.4 |    129.2 |      187.4 |    650.6 |    201.0 | PyTorch |
| argmax_axis_256x512    |      51.8 |     95.0 |    172.2 |       71.7 |   1368.1 |    164.9 | Peregrine |
| sum_axis_1024x1024     |     174.1 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     431.3 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.8 |     37.9 |     54.7 |       53.8 |   1842.7 |     35.1 | Peregrine |
| triu_256x256           |       7.6 |     38.6 |     57.9 |       53.8 |   1819.0 |     39.8 | Peregrine |
| repeat_64x128_2x3      |       7.4 |     49.0 |     32.1 |       75.4 |        — |     27.9 | Peregrine |
| pad_64x128             |       2.5 |      4.8 |     18.8 |       84.6 |     89.2 |     17.9 | Peregrine |
| stack_8x64x128         |       3.8 |      8.2 |     46.4 |       58.3 |    907.1 |    158.3 | Peregrine |
| diagonal_512x512       |       0.3 |      0.7 |     31.1 |       12.4 |        — |      8.2 | Peregrine |
| silu_100k              |      64.0 |     68.2 |     84.7 |      216.8 |    324.2 |     51.5 | JAX |
| softplus_100k          |     180.8 |    149.0 |    262.4 |      120.7 |    785.1 |    154.8 | TensorFlow |
| mish_100k              |     286.2 |    307.6 |    381.4 |      254.8 |   1131.9 |    236.8 | JAX |
| leaky_relu_100k        |       8.5 |     39.7 |     73.8 |       19.7 |        — |     28.7 | Peregrine |
| elu_100k               |      60.0 |    128.4 |    117.7 |      142.8 |    858.8 |     77.4 | Peregrine |
| hard_tanh_100k         |       8.7 |     43.8 |     35.4 |       42.0 |        — |     40.9 | Peregrine |
| relu6_100k             |       8.7 |     46.6 |     44.9 |       55.0 |    726.5 |    111.8 | Peregrine |
| hardswish_100k         |      10.0 |     39.3 |     67.4 |      225.7 |        — |     28.8 | Peregrine |
| gelu_100k              |      95.4 |     73.0 |    135.5 |      249.8 |    859.8 |    204.0 | PyTorch |
| selu_100k              |      63.7 |    133.7 |     86.8 |      132.9 |    749.9 |     81.7 | Peregrine |
| softsign_100k          |      38.1 |    123.3 |     43.9 |       47.4 |        — |     56.9 | Peregrine |
| cross_entropy_64x10    |       2.6 |     39.9 |     24.1 |      634.3 |   3598.1 |     52.3 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.5 |     19.6 |       46.6 |   1170.4 |     12.0 | Peregrine |
| mse_loss_64x10         |       3.8 |      5.0 |     23.6 |       40.6 |    463.6 |     24.7 | Peregrine |
| huber_loss_64x10       |       0.3 |      5.2 |     33.2 |      236.8 |        — |     47.2 | Peregrine |
| smooth_l1_loss_64x10   |       0.8 |      5.3 |     36.7 |      233.8 |        — |     47.0 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      7.2 |     17.8 |      375.5 |        — |     56.5 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     11.5 |    108.0 |      237.8 |        — |     61.9 | Peregrine |
| rmsnorm_64x512         |      19.2 |     67.9 |     33.6 |      438.4 |        — |     70.3 | Peregrine |
| conv1d_1x32x128_k3     |      20.4 |     57.9 |     27.5 |      509.1 |        — |     73.8 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     46.3 |    265.5 |       63.2 |        — |     43.0 | Peregrine |
| groupnorm_4x64x16x16   |      21.2 |     57.8 |    233.3 |      764.0 |        — |    263.4 | Peregrine |
| rnn_seq32_128_256      |     198.6 |    286.0 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1030.2 |    855.6 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     806.6 |    829.6 |        — |          — |        — |        — | Peregrine |
| optim_adam_64          |     803.9 |   1273.9 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     922.8 |   1176.1 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |    1250.9 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1400.6 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      60.5 |    257.8 |    480.8 |      121.3 |   2543.1 |    535.4 | Peregrine |
| rand_normal_100k       |     237.3 |    971.5 |    687.0 |      327.2 |   3500.1 |    600.1 | Peregrine |
| rand_bernoulli_100k    |     119.7 |    250.3 |    448.6 |      194.1 |        — |    523.5 | Peregrine |
| rand_uniform_1M        |     604.3 |   2581.5 |   4560.6 |      409.6 |   2587.8 |   2289.5 | TensorFlow |
| rand_normal_1M         |    2398.9 |   9939.1 |   6599.6 |     2080.2 |   3321.9 |   2890.7 | TensorFlow |
| rfft_1k                |       2.2 |      4.5 |     26.4 |       43.5 |        — |     45.5 | Peregrine |
| rfft_4k                |       6.5 |     14.9 |     35.0 |       54.6 |        — |     66.3 | Peregrine |
| rfft_16k               |      30.2 |     65.8 |     74.9 |      105.8 |        — |    116.1 | Peregrine |
| fft_1k                 |       3.5 |      6.8 |     22.3 |        9.0 |        — |     44.4 | Peregrine |
| fft_4k                 |      12.2 |     26.5 |     39.0 |       17.8 |        — |     58.1 | Peregrine |
| norm_l2_1k             |       1.1 |      1.4 |     19.1 |       70.7 |        — |      3.7 | Peregrine |
| solve_64x64            |      15.4 |     19.2 |     95.9 |       24.6 |        — |     32.1 | Peregrine |
| inv_64x64              |      37.3 |     26.7 |     47.2 |       32.6 |        — |     37.5 | PyTorch |
| cholesky_64x64         |       6.4 |     44.9 |     21.2 |       19.4 |        — |     20.2 | Peregrine |
| svd_64x64              |     275.9 |    283.8 |    289.1 |      502.9 |        — |    302.2 | Peregrine |
| qr_64x64               |      41.2 |     80.1 |     55.1 |       83.7 |        — |     63.2 | Peregrine |
| eigh_64x64             |     376.1 |    215.7 |    232.8 |      149.1 |        — |    236.3 | TensorFlow |
| det_64x64              |      19.2 |     20.2 |        — |       22.8 |        — |     30.0 | Peregrine |
| solve_128x128          |      50.2 |     48.4 |    185.7 |       76.4 |        — |     84.8 | PyTorch |
| inv_128x128            |      96.2 |     62.2 |     88.4 |      141.3 |        — |     82.8 | PyTorch |
| cholesky_128x128       |      35.3 |     51.3 |     27.6 |       59.1 |        — |     35.7 | MLX |
| svd_128x128            |     983.2 |    994.2 |    966.4 |     1829.8 |        — |   1005.1 | MLX |
| qr_128x128             |     188.2 |    214.2 |    192.3 |      326.0 |        — |    190.2 | Peregrine |
| eigh_128x128           |    1824.2 |    696.8 |    717.5 |      720.9 |        — |    735.3 | PyTorch |
| det_128x128            |      41.2 |     50.1 |        — |       84.3 |        — |     76.7 | Peregrine |
| solve_256x256          |     188.5 |    173.3 |    729.5 |      387.6 |        — |    264.6 | PyTorch |
| inv_256x256            |     451.9 |    287.9 |    249.3 |      855.7 |        — |    334.3 | MLX |
| cholesky_256x256       |     145.2 |     76.2 |     56.3 |      287.2 |        — |    124.6 | MLX |
| svd_256x256            |    5875.7 |   5571.2 |   5818.9 |     8009.5 |        — |   5638.1 | PyTorch |
| qr_256x256             |     982.2 |    954.7 |    984.5 |     1693.6 |        — |    961.6 | PyTorch |
| eigh_256x256           |    5916.2 |   3429.4 |   3481.7 |     4631.2 |        — |   3496.1 | PyTorch |
| det_256x256            |     140.8 |    200.7 |        — |      452.9 |        — |    206.2 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1772.8 |    823.2 |        — |     2349.1 |   1289.0 |   2072.1 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3172.2 |   1910.6 |        — |     3652.8 |   1282.7 |   3348.2 | tinygrad |
| add_layernorm_196x768  |     105.9 |     98.4 |        — |     1158.9 |   1116.1 |    215.5 | PyTorch |
| add_layernorm_196x1024 |     140.6 |    107.8 |        — |     1237.3 |   1111.1 |    279.3 | PyTorch |
| matmul_f32_196x768x3072 |     501.0 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14576.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1421.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26249.1 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.52x** (Peregrine is faster)
- **Peregrine vs MLX: 0.38x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.29x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.37x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 100/141 ops
- PyTorch: 18/141 ops
- JAX: 10/141 ops
- TensorFlow: 7/141 ops
- MLX: 5/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
