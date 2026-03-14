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
| matmul_128x128         |      28.2 |      6.1 |     19.8 |       96.3 |    430.4 |     79.2 | PyTorch |
| matmul_256x256         |      46.0 |     30.2 |     48.2 |      200.7 |    424.7 |    171.6 | PyTorch |
| matmul_512x512         |     260.2 |    140.6 |    171.8 |      696.7 |    464.5 |    510.9 | PyTorch |
| matmul_1024x1024       |    1128.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |   10609.4 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.9 |     42.0 |     29.3 |       48.2 |    193.2 |     33.2 | Peregrine |
| add_500k               |     166.3 |     57.3 |     81.5 |       91.3 |    189.2 |     68.8 | PyTorch |
| add_1M                 |     120.6 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     538.1 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     961.4 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.8 |     41.1 |     26.5 |       46.9 |    190.1 |     27.8 | Peregrine |
| mul_500k               |     109.3 |     56.7 |     81.4 |       79.8 |    190.0 |     59.8 | PyTorch |
| mul_1M                 |     130.1 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     608.7 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     861.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      93.2 |     64.3 |     60.6 |       66.3 |    226.7 |     46.2 | JAX |
| exp_500k               |     194.9 |    137.7 |    226.0 |      111.1 |    222.0 |    122.5 | TensorFlow |
| exp_1M                 |     319.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1162.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2124.0 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     38.2 |     27.2 |       41.7 |    341.7 |     98.4 | Peregrine |
| relu_1M                |     109.7 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     32.5 |     15.8 |       11.8 |    627.2 |     30.7 | Peregrine |
| softmax_8x512          |       4.3 |     37.1 |     17.8 |       14.8 |    629.5 |     34.9 | Peregrine |
| mlp_fwd_64x784         |      32.6 |     27.7 |     51.1 |      272.3 |   1835.5 |    190.7 | PyTorch |
| mlp_fwd_256x784_wide   |     426.7 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     827.1 |   1243.9 |    763.3 |     9241.6 |  24976.0 |   5170.9 | MLX |
| train_step_256_wide    |    3316.0 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     40.2 |     24.5 |       50.4 |    166.7 |     28.9 | Peregrine |
| square_100k            |       8.8 |     39.8 |     23.0 |       17.2 |    180.5 |     35.9 | Peregrine |
| rsqrt_100k             |      97.5 |     44.6 |     36.3 |       53.8 |        — |     92.4 | MLX |
| floor_100k             |       8.8 |     38.1 |     23.3 |       18.6 |    424.0 |     33.2 | Peregrine |
| ceil_100k              |       8.8 |     38.9 |     23.4 |       18.6 |    362.6 |     29.7 | Peregrine |
| round_100k             |       8.8 |     42.1 |     23.4 |       45.1 |        — |     30.2 | Peregrine |
| sign_100k              |       8.8 |     38.1 |     27.4 |       52.3 |    819.8 |     36.8 | Peregrine |
| expm1_100k             |     159.5 |    109.2 |    108.8 |      157.5 |        — |     99.9 | JAX |
| log2_100k              |      99.2 |     89.5 |    102.7 |      156.3 |    169.6 |     56.2 | JAX |
| log10_100k             |     108.2 |     85.2 |    111.2 |      157.3 |        — |     56.8 | JAX |
| log1p_100k             |     116.1 |     83.0 |    129.5 |      106.1 |        — |    113.9 | PyTorch |
| erf_100k               |     117.2 |     57.9 |    101.1 |       55.2 |        — |     57.4 | TensorFlow |
| sinh_100k              |      52.0 |    136.5 |     93.9 |      133.5 |    536.8 |    133.4 | Peregrine |
| cosh_100k              |      47.2 |    129.4 |     89.6 |      124.7 |    478.3 |     79.2 | Peregrine |
| arcsin_100k            |      53.1 |     80.2 |     95.8 |       58.2 |   2962.2 |    120.8 | Peregrine |
| arccos_100k            |     104.2 |     88.5 |    111.2 |       53.2 |        — |    203.6 | TensorFlow |
| arctan_100k            |      54.2 |     92.8 |     99.4 |       56.5 |   3129.7 |    221.9 | Peregrine |
| arcsinh_100k           |     146.5 |    162.5 |    334.1 |      142.5 |        — |    130.4 | JAX |
| maximum_100k           |      12.8 |     39.7 |     27.4 |       46.1 |    194.4 |     34.1 | Peregrine |
| minimum_100k           |      12.8 |     38.5 |     28.2 |       47.7 |    390.3 |     32.7 | Peregrine |
| power_100k             |     156.8 |    230.8 |    212.2 |      299.9 |        — |    152.9 | JAX |
| arctan2_100k           |      96.9 |    127.8 |    161.2 |       74.2 |        — |    319.9 | TensorFlow |
| logaddexp_100k         |     416.5 |    152.8 |    284.1 |      387.2 |        — |    150.3 | JAX |
| clip_100k              |       8.8 |     39.8 |     35.4 |       43.9 |    544.0 |     42.6 | Peregrine |
| where_100k             |      16.8 |     49.0 |     28.1 |       67.6 |    278.9 |     35.0 | Peregrine |
| greater_100k           |      12.8 |     47.8 |     24.4 |       51.9 |    200.4 |     30.9 | Peregrine |
| equal_100k             |      12.8 |     29.0 |     24.1 |       62.2 |    291.0 |     35.1 | Peregrine |
| sum_axis_256x512       |      19.2 |     39.7 |     23.6 |       48.7 |    212.6 |     46.2 | Peregrine |
| mean_axis_256x512      |      19.2 |     43.2 |     25.1 |       52.4 |    300.0 |     57.4 | Peregrine |
| max_axis_256x512       |      14.0 |     55.8 |     48.0 |       50.2 |    205.1 |     50.9 | Peregrine |
| min_axis_256x512       |      14.0 |     55.7 |     42.3 |       49.6 |    332.7 |     50.2 | Peregrine |
| var_256x512            |     240.2 |    273.2 |     61.8 |      222.5 |        — |     74.0 | MLX |
| prod_axis_256x512      |      24.2 |     38.8 |     26.1 |       49.7 |        — |     54.3 | Peregrine |
| logsumexp_256x512      |     389.7 |    193.1 |    108.1 |      339.3 |        — |    281.6 | MLX |
| cumsum_256x512         |     124.8 |     73.6 |    128.4 |      187.9 |    641.3 |    210.9 | PyTorch |
| argmax_axis_256x512    |     157.7 |     91.3 |    178.9 |       79.2 |   1317.8 |    168.3 | TensorFlow |
| sum_axis_1024x1024     |     177.4 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    1958.6 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      35.4 |     38.4 |     58.7 |       64.8 |   1844.1 |     38.2 | Peregrine |
| triu_256x256           |      34.8 |     38.1 |     58.2 |       55.2 |   1870.7 |     37.3 | Peregrine |
| repeat_64x128_2x3      |     127.0 |     47.7 |     32.2 |       76.4 |        — |     28.0 | JAX |
| pad_64x128             |      17.1 |      4.4 |     20.1 |       88.4 |     93.1 |     18.4 | PyTorch |
| stack_8x64x128         |      15.2 |      8.8 |     46.5 |       61.8 |    940.1 |    161.0 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     29.0 |       12.6 |        — |      9.6 | PyTorch |
| silu_100k              |      66.1 |     73.4 |     86.2 |      245.1 |    346.7 |     62.7 | JAX |
| softplus_100k          |     347.3 |    147.3 |    267.1 |      145.6 |    788.3 |    155.3 | TensorFlow |
| mish_100k              |     507.1 |    308.7 |    374.7 |      261.8 |   1180.0 |    237.6 | JAX |
| leaky_relu_100k        |       8.7 |     39.2 |     80.1 |       19.8 |        — |     32.8 | Peregrine |
| elu_100k               |     152.4 |    126.2 |    123.3 |      134.9 |    891.6 |     80.2 | JAX |
| hard_tanh_100k         |       8.1 |     42.5 |     36.3 |       43.3 |        — |     36.4 | Peregrine |
| relu6_100k             |       8.1 |     38.4 |     46.6 |       56.2 |    736.2 |    113.4 | Peregrine |
| hardswish_100k         |      10.2 |     39.0 |     67.9 |      223.3 |        — |     28.8 | Peregrine |
| gelu_100k              |      72.3 |     73.3 |    136.0 |      252.1 |    875.5 |    216.2 | Peregrine |
| selu_100k              |      65.7 |    132.8 |     85.5 |      134.6 |    755.2 |     82.2 | Peregrine |
| softsign_100k          |      35.6 |    123.7 |     47.4 |       45.1 |        — |     59.2 | Peregrine |
| cross_entropy_64x10    |       2.7 |     37.9 |     26.4 |      636.6 |   3465.3 |     55.3 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.4 |     18.5 |       46.2 |   1145.5 |     11.9 | Peregrine |
| mse_loss_64x10         |       3.7 |      4.8 |     22.4 |       39.7 |    454.7 |     23.8 | Peregrine |
| huber_loss_64x10       |       5.2 |      4.7 |     33.7 |      237.9 |        — |     47.1 | PyTorch |
| smooth_l1_loss_64x10   |       5.0 |      5.0 |     33.3 |      239.7 |        — |     48.4 | Peregrine |
| kl_div_loss_64x10      |       2.6 |      6.3 |     18.3 |      398.1 |        — |     62.8 | Peregrine |
| cosine_sim_loss_64x64  |      13.8 |     10.1 |    112.6 |      245.8 |        — |     73.1 | PyTorch |
| rmsnorm_64x512         |      58.8 |     65.5 |     34.4 |      445.7 |        — |     75.0 | MLX |
| conv1d_1x32x128_k3     |      20.9 |     51.8 |     31.1 |      573.0 |        — |     72.8 | Peregrine |
| avgpool2d_1x16x32x32   |      25.5 |     42.6 |    268.5 |       64.3 |        — |     44.8 | Peregrine |
| groupnorm_4x64x16x16   |      74.0 |     50.9 |    223.5 |      795.4 |        — |    274.5 | PyTorch |
| rnn_seq32_128_256      |     189.9 |    269.6 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1105.2 |    807.6 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     816.8 |    779.4 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     809.8 |   1285.0 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     929.8 |   1136.6 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     916.3 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1314.6 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     108.4 |    257.9 |    483.1 |      125.1 |   2440.1 |    545.4 | Peregrine |
| rand_normal_100k       |     772.4 |    990.2 |    688.2 |      356.0 |   3329.1 |    611.9 | TensorFlow |
| rand_bernoulli_100k    |     303.4 |    255.1 |    453.7 |      212.8 |        — |    527.5 | TensorFlow |
| rand_uniform_1M        |    1087.8 |   2618.2 |   4602.5 |      441.5 |   2505.7 |   2287.6 | TensorFlow |
| rand_normal_1M         |    7789.0 |   9933.2 |   6645.0 |     2197.2 |   3365.5 |   2932.8 | TensorFlow |
| rfft_1k                |       2.2 |      4.5 |     26.6 |       44.2 |        — |     49.0 | Peregrine |
| rfft_4k                |       7.5 |     15.0 |     35.1 |       58.6 |        — |     69.3 | Peregrine |
| rfft_16k               |      30.0 |     66.4 |     77.6 |      107.3 |        — |    125.1 | Peregrine |
| fft_1k                 |       3.3 |      6.7 |     24.2 |        9.3 |        — |     42.5 | Peregrine |
| fft_4k                 |      12.2 |     26.8 |     41.3 |       17.9 |        — |     59.1 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     18.9 |       70.5 |        — |      3.8 | Peregrine |
| solve_64x64            |      12.0 |     23.4 |     97.0 |       25.1 |        — |     32.7 | Peregrine |
| inv_64x64              |      37.3 |     25.2 |     48.4 |       33.6 |        — |     43.0 | PyTorch |
| cholesky_64x64         |       9.6 |     46.1 |     21.6 |       20.1 |        — |     20.1 | Peregrine |
| svd_64x64              |     277.3 |    279.1 |    289.5 |      513.0 |        — |    306.1 | Peregrine |
| qr_64x64               |      40.1 |     80.9 |     57.7 |       85.0 |        — |     64.1 | Peregrine |
| eigh_64x64             |     385.8 |    215.0 |    229.1 |      149.4 |        — |    237.7 | TensorFlow |
| det_64x64              |      22.5 |     19.1 |        — |       23.2 |        — |     28.4 | PyTorch |
| solve_128x128          |      49.0 |     43.6 |    188.7 |       78.1 |        — |     84.4 | PyTorch |
| inv_128x128            |      94.3 |     57.9 |     90.6 |      141.8 |        — |     81.9 | PyTorch |
| cholesky_128x128       |      49.4 |     58.2 |     26.0 |       59.6 |        — |     35.9 | MLX |
| svd_128x128            |     995.3 |   1000.4 |    996.3 |     1860.2 |        — |   1016.1 | Peregrine |
| qr_128x128             |     188.3 |    219.2 |    197.2 |      332.4 |        — |    192.8 | Peregrine |
| eigh_128x128           |    1868.4 |    705.0 |    718.9 |      750.8 |        — |    745.2 | PyTorch |
| det_128x128            |      50.9 |     48.5 |        — |       84.1 |        — |     76.2 | PyTorch |
| solve_256x256          |     189.1 |    172.6 |    731.4 |      382.9 |        — |    312.5 | PyTorch |
| inv_256x256            |     463.7 |    294.2 |    245.0 |      874.6 |        — |    335.4 | MLX |
| cholesky_256x256       |     232.8 |     78.5 |     54.8 |      286.7 |        — |    131.5 | MLX |
| svd_256x256            |    6007.6 |   5829.1 |   5656.1 |     8504.8 |        — |   5986.7 | MLX |
| qr_256x256             |    1025.5 |   1015.7 |    999.6 |     1770.8 |        — |   1000.6 | MLX |
| eigh_256x256           |    6093.6 |   3455.0 |   3497.5 |     4845.4 |        — |   3634.8 | PyTorch |
| det_256x256            |     212.7 |    212.6 |        — |      440.5 |        — |    218.1 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1299.2 |    939.9 |        — |     2400.3 |   1310.5 |   2175.2 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2122.8 |   1999.1 |        — |     3833.9 |   1262.3 |   3585.4 | tinygrad |
| add_layernorm_196x768  |     106.8 |    103.9 |        — |     1246.8 |   1149.7 |    229.5 | PyTorch |
| add_layernorm_196x1024 |     319.6 |    109.2 |        — |     1292.4 |   1161.3 |    274.2 | PyTorch |
| matmul_f32_196x768x3072 |     664.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14860.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1632.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26808.2 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.73x** (Peregrine is faster)
- **Peregrine vs MLX: 0.55x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.39x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.07x** (Peregrine is faster)
- **Peregrine vs JAX: 0.50x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 80/141 ops
- PyTorch: 28/141 ops
- TensorFlow: 11/141 ops
- JAX: 11/141 ops
- MLX: 10/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
