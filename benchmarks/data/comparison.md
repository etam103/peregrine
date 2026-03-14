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
| matmul_128x128         |      24.0 |      6.2 |     21.3 |       98.2 |    427.2 |     79.2 | PyTorch |
| matmul_256x256         |      51.3 |     31.7 |     48.6 |      197.3 |    425.9 |    152.6 | PyTorch |
| matmul_512x512         |     268.7 |    147.3 |    172.0 |      662.9 |    438.1 |    528.5 | PyTorch |
| matmul_1024x1024       |    1041.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |   10835.2 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |       9.8 |     40.4 |     31.9 |       55.6 |    187.8 |     35.1 | Peregrine |
| add_500k               |     123.2 |     60.4 |     80.6 |       80.5 |    187.0 |     61.0 | PyTorch |
| add_1M                 |     129.8 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     651.8 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |    1077.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      13.0 |     39.7 |     32.0 |       47.5 |    191.3 |     32.6 | Peregrine |
| mul_500k               |     171.0 |     58.2 |     84.8 |       77.2 |    189.2 |     59.1 | PyTorch |
| mul_1M                 |     169.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     677.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |    1024.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |     150.8 |     67.3 |     61.0 |       66.7 |    225.4 |     47.4 | JAX |
| exp_500k               |     232.8 |    139.8 |    223.2 |      106.0 |    225.1 |    121.8 | TensorFlow |
| exp_1M                 |     354.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1245.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2391.7 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.9 |     38.9 |     26.6 |       43.9 |    335.1 |     96.4 | Peregrine |
| relu_1M                |     118.0 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.5 |     37.2 |     16.7 |       11.3 |    635.4 |     31.0 | Peregrine |
| softmax_8x512          |       4.3 |     31.5 |     20.1 |       14.4 |    624.9 |     33.6 | Peregrine |
| mlp_fwd_64x784         |      32.7 |     27.8 |     55.8 |      247.4 |   1878.8 |    189.8 | PyTorch |
| mlp_fwd_256x784_wide   |     437.9 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     865.1 |   1360.7 |    772.8 |     8676.6 |  23803.2 |   5176.8 | MLX |
| train_step_256_wide    |    3493.2 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.7 |     37.9 |     25.1 |       47.4 |    165.2 |     28.5 | Peregrine |
| square_100k            |       8.6 |     41.9 |     24.2 |       16.2 |    197.7 |     31.0 | Peregrine |
| rsqrt_100k             |      85.0 |     43.6 |     32.6 |       53.5 |        — |     88.0 | MLX |
| floor_100k             |       8.2 |     39.7 |     24.5 |       16.1 |    436.5 |     36.4 | Peregrine |
| ceil_100k              |       7.9 |     39.4 |     23.1 |       16.0 |    359.9 |     30.8 | Peregrine |
| round_100k             |       7.9 |     41.5 |     24.1 |       46.7 |        — |     31.6 | Peregrine |
| sign_100k              |       8.2 |     40.1 |     27.7 |       48.0 |    869.8 |     36.2 | Peregrine |
| expm1_100k             |     157.1 |    110.2 |    103.7 |      153.6 |        — |     99.2 | JAX |
| log2_100k              |     120.0 |     87.0 |     97.8 |      157.5 |    169.4 |     56.4 | JAX |
| log10_100k             |     122.2 |     83.3 |    106.5 |      148.1 |        — |     56.8 | JAX |
| log1p_100k             |     126.4 |     90.5 |    127.7 |       92.4 |        — |    104.3 | PyTorch |
| erf_100k               |     124.3 |     58.4 |    104.3 |       59.8 |        — |     42.5 | JAX |
| sinh_100k              |      52.0 |    130.2 |     96.0 |      128.8 |    554.4 |    113.0 | Peregrine |
| cosh_100k              |      47.2 |    125.9 |     91.6 |      131.4 |    492.8 |     70.5 | Peregrine |
| arcsin_100k            |      53.1 |     74.7 |     94.1 |       58.3 |   3026.3 |    113.1 | Peregrine |
| arccos_100k            |     130.6 |     88.4 |    110.3 |       55.1 |        — |    206.9 | TensorFlow |
| arctan_100k            |      53.1 |     92.8 |     95.0 |       59.7 |   3164.9 |    215.5 | Peregrine |
| arcsinh_100k           |     150.3 |    154.7 |    338.4 |      142.8 |        — |    126.3 | JAX |
| maximum_100k           |      12.8 |     40.5 |     27.6 |       44.2 |    193.9 |     33.5 | Peregrine |
| minimum_100k           |      12.8 |     38.3 |     27.8 |       41.5 |    394.4 |     30.0 | Peregrine |
| power_100k             |     161.0 |    245.1 |    226.4 |      286.4 |        — |    158.3 | JAX |
| arctan2_100k           |      99.6 |    135.5 |    147.3 |       72.0 |        — |    314.8 | TensorFlow |
| logaddexp_100k         |     417.1 |    154.2 |    260.8 |      354.6 |        — |    154.2 | PyTorch |
| clip_100k              |       8.0 |     39.9 |     34.4 |       42.5 |    545.1 |     42.8 | Peregrine |
| where_100k             |      14.8 |     50.8 |     27.8 |       66.2 |    285.4 |     31.5 | Peregrine |
| greater_100k           |      10.2 |     50.0 |     25.9 |       56.6 |    203.8 |     34.9 | Peregrine |
| equal_100k             |       9.7 |     28.7 |     25.1 |       60.9 |    289.9 |     30.6 | Peregrine |
| sum_axis_256x512       |      18.9 |     39.1 |     24.0 |       51.9 |    216.8 |     54.0 | Peregrine |
| mean_axis_256x512      |      18.9 |     42.1 |     25.5 |       49.1 |    295.0 |     49.2 | Peregrine |
| max_axis_256x512       |      13.8 |     53.6 |     41.0 |       50.5 |    207.2 |     45.7 | Peregrine |
| min_axis_256x512       |      13.8 |     53.6 |     40.9 |       50.9 |    336.4 |     49.8 | Peregrine |
| var_256x512            |      45.8 |    277.9 |     61.9 |      233.5 |        — |     87.1 | Peregrine |
| prod_axis_256x512      |      24.2 |     39.7 |     25.6 |       50.5 |        — |     58.5 | Peregrine |
| logsumexp_256x512      |      95.6 |    194.5 |    106.9 |      341.0 |        — |    290.3 | Peregrine |
| cumsum_256x512         |     124.4 |     78.3 |    128.6 |      203.7 |    630.0 |    215.0 | PyTorch |
| argmax_axis_256x512    |     154.7 |     95.6 |    173.0 |       71.5 |   1368.3 |    176.2 | TensorFlow |
| sum_axis_1024x1024     |     174.8 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     428.3 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      34.7 |     40.5 |     75.8 |       50.8 |   2060.4 |     38.4 | Peregrine |
| triu_256x256           |      34.4 |     38.8 |    161.8 |       52.8 |   1972.1 |     36.0 | Peregrine |
| repeat_64x128_2x3      |     125.8 |     48.1 |     41.9 |       77.8 |        — |     28.3 | JAX |
| pad_64x128             |      17.4 |      4.3 |     21.3 |       84.7 |     90.1 |     19.3 | PyTorch |
| stack_8x64x128         |      15.9 |      8.9 |     57.1 |       60.3 |    981.9 |    160.8 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     30.2 |       12.5 |        — |      8.8 | PyTorch |
| silu_100k              |      65.0 |     69.7 |    101.3 |      236.8 |    346.3 |     52.5 | JAX |
| softplus_100k          |     345.1 |    155.2 |    266.5 |      124.8 |    854.8 |    161.6 | TensorFlow |
| mish_100k              |     498.1 |    312.0 |    378.7 |      243.6 |   1194.8 |    237.5 | JAX |
| leaky_relu_100k        |       8.0 |     41.9 |     78.2 |       19.6 |        — |     33.1 | Peregrine |
| elu_100k               |     149.5 |    128.7 |    120.9 |      139.7 |    908.6 |     94.8 | JAX |
| hard_tanh_100k         |       8.0 |     40.8 |     36.5 |       42.5 |        — |     45.1 | Peregrine |
| relu6_100k             |       8.0 |     39.2 |     47.4 |       49.8 |    736.2 |    113.5 | Peregrine |
| hardswish_100k         |      10.0 |     38.9 |     68.6 |      212.5 |        — |     33.3 | Peregrine |
| gelu_100k              |      81.1 |     67.1 |    136.3 |      249.7 |    859.7 |    225.8 | PyTorch |
| selu_100k              |      63.8 |    131.9 |     85.7 |      134.6 |    752.1 |     83.5 | Peregrine |
| softsign_100k          |      35.0 |    117.5 |     42.3 |       46.1 |        — |     68.9 | Peregrine |
| cross_entropy_64x10    |       2.6 |     38.7 |     23.4 |      623.1 |   3457.7 |     55.5 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.5 |     19.5 |       43.0 |   1131.0 |     11.9 | Peregrine |
| mse_loss_64x10         |       3.7 |      4.9 |     23.4 |       38.7 |    448.9 |     24.0 | Peregrine |
| huber_loss_64x10       |       5.2 |      4.8 |     34.7 |      239.6 |        — |     48.5 | PyTorch |
| smooth_l1_loss_64x10   |       5.0 |      5.1 |     34.6 |      236.2 |        — |     47.7 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.3 |     18.0 |      378.8 |        — |     59.0 | Peregrine |
| cosine_sim_loss_64x64  |      13.5 |     10.2 |    112.3 |      247.7 |        — |     73.1 | PyTorch |
| rmsnorm_64x512         |      57.8 |     66.1 |     33.2 |      439.9 |        — |     71.6 | MLX |
| conv1d_1x32x128_k3     |      20.9 |     54.0 |     28.1 |      525.4 |        — |     74.2 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     45.3 |    274.8 |       67.6 |        — |     41.8 | Peregrine |
| groupnorm_4x64x16x16   |      72.6 |     56.3 |    222.6 |      771.3 |        — |    271.1 | PyTorch |
| rnn_seq32_128_256      |     191.7 |    272.6 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1133.4 |    804.2 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     881.4 |    780.0 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     802.8 |   1332.2 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     938.4 |   1138.3 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     909.1 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1277.5 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     107.5 |    258.0 |    490.2 |      124.6 |   2466.0 |    521.0 | Peregrine |
| rand_normal_100k       |     771.5 |    984.6 |    690.9 |      352.0 |   3333.1 |    605.2 | TensorFlow |
| rand_bernoulli_100k    |     311.6 |    255.2 |    457.9 |      218.0 |        — |    541.9 | TensorFlow |
| rand_uniform_1M        |    1072.1 |   2620.7 |   4582.5 |      426.7 |   2483.6 |   2300.1 | TensorFlow |
| rand_normal_1M         |    7729.8 |  10049.3 |   6802.4 |     2081.3 |   3387.2 |   2960.5 | TensorFlow |
| rfft_1k                |       2.2 |      4.7 |     25.1 |       43.7 |        — |     46.3 | Peregrine |
| rfft_4k                |       6.5 |     15.3 |     29.9 |       54.8 |        — |     61.9 | Peregrine |
| rfft_16k               |      31.0 |     67.5 |     81.0 |      106.9 |        — |    122.3 | Peregrine |
| fft_1k                 |       3.3 |      6.8 |     32.1 |        8.9 |        — |     17.8 | Peregrine |
| fft_4k                 |      12.4 |     27.4 |     50.0 |       17.5 |        — |     54.7 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     18.7 |       69.4 |        — |      4.1 | Peregrine |
| solve_64x64            |      12.1 |     18.0 |    103.5 |       24.8 |        — |     33.2 | Peregrine |
| inv_64x64              |      36.7 |     25.4 |     52.0 |       32.4 |        — |     37.5 | PyTorch |
| cholesky_64x64         |       9.6 |     45.5 |     21.9 |       19.7 |        — |     20.8 | Peregrine |
| svd_64x64              |     276.8 |    283.2 |    299.5 |      486.1 |        — |    301.9 | Peregrine |
| qr_64x64               |      41.4 |     94.0 |     56.8 |       85.5 |        — |     64.8 | Peregrine |
| eigh_64x64             |     379.9 |    217.6 |    231.7 |      141.9 |        — |    236.5 | TensorFlow |
| det_64x64              |      23.5 |     19.9 |        — |       23.3 |        — |     28.7 | PyTorch |
| solve_128x128          |      50.1 |     43.9 |    191.1 |       77.5 |        — |     84.9 | PyTorch |
| inv_128x128            |      94.8 |     59.1 |     88.4 |      141.5 |        — |     82.8 | PyTorch |
| cholesky_128x128       |      50.6 |     54.1 |     26.7 |       58.4 |        — |     37.0 | MLX |
| svd_128x128            |     993.8 |    997.6 |    961.1 |     1845.5 |        — |   1020.0 | MLX |
| qr_128x128             |     188.5 |    229.0 |    197.5 |      332.6 |        — |    191.1 | Peregrine |
| eigh_128x128           |    1842.8 |    702.9 |    725.7 |      723.2 |        — |    749.7 | PyTorch |
| det_128x128            |      52.4 |     48.3 |        — |       82.0 |        — |     76.4 | PyTorch |
| solve_256x256          |     188.8 |    190.7 |    743.2 |      379.5 |        — |    264.5 | Peregrine |
| inv_256x256            |     474.1 |    284.8 |    252.9 |      869.3 |        — |    339.5 | MLX |
| cholesky_256x256       |     226.2 |     91.1 |     55.1 |      282.9 |        — |    117.6 | MLX |
| svd_256x256            |    6076.8 |   5784.2 |   5868.9 |     8301.4 |        — |   5945.1 | PyTorch |
| qr_256x256             |    1053.5 |   1033.8 |    994.0 |     1726.2 |        — |   1008.7 | MLX |
| eigh_256x256           |    6247.0 |   3371.9 |   3544.2 |     4605.4 |        — |   3541.1 | PyTorch |
| det_256x256            |     212.7 |    205.0 |        — |      437.7 |        — |    205.9 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1463.9 |   1073.6 |        — |     2448.6 |   1246.4 |   2193.2 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2408.4 |   2173.5 |        — |     3733.9 |   1319.5 |   3504.6 | tinygrad |
| add_layernorm_196x768  |     102.1 |    111.8 |        — |     1247.8 |   1145.8 |    245.5 | Peregrine |
| add_layernorm_196x1024 |     310.5 |    119.4 |        — |     1308.0 |   1134.7 |    298.3 | PyTorch |
| matmul_f32_196x768x3072 |     646.0 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14851.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1738.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26469.0 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.71x** (Peregrine is faster)
- **Peregrine vs MLX: 0.52x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.39x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.07x** (Peregrine is faster)
- **Peregrine vs JAX: 0.49x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 82/141 ops
- PyTorch: 29/141 ops
- JAX: 11/141 ops
- TensorFlow: 10/141 ops
- MLX: 8/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
