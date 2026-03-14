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
| matmul_128x128         |      25.2 |      6.1 |     20.1 |       95.7 |    427.0 |     79.4 | PyTorch |
| matmul_256x256         |      28.8 |     30.2 |     46.2 |      193.9 |    423.4 |    159.8 | Peregrine |
| matmul_512x512         |     270.3 |    141.6 |    167.4 |      613.4 |    534.3 |    526.0 | PyTorch |
| matmul_1024x1024       |     975.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9187.1 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.9 |     43.7 |     34.6 |       49.0 |    188.5 |     36.3 | Peregrine |
| add_500k               |      63.8 |     56.5 |     87.3 |       88.2 |    185.2 |     59.1 | PyTorch |
| add_1M                 |     110.1 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     604.5 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     920.0 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      13.4 |     43.5 |     34.8 |       46.6 |    188.0 |     33.2 | Peregrine |
| mul_500k               |      62.2 |     59.8 |     78.0 |       77.5 |    188.7 |     56.3 | JAX |
| mul_1M                 |     125.7 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     526.3 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     932.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      50.1 |     64.5 |     57.9 |       67.4 |    225.9 |     46.4 | JAX |
| exp_500k               |     109.5 |    140.6 |    219.6 |      101.2 |    220.4 |    118.2 | TensorFlow |
| exp_1M                 |     147.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     421.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     836.2 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       9.1 |     41.5 |     27.0 |       39.4 |    339.2 |     98.6 | Peregrine |
| relu_1M                |      88.6 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     30.3 |     18.8 |       12.0 |    622.8 |     31.5 | Peregrine |
| softmax_8x512          |       4.3 |     36.7 |     18.9 |       14.9 |    627.5 |     33.5 | Peregrine |
| mlp_fwd_64x784         |      32.0 |     27.8 |     52.5 |      247.7 |   1791.0 |    174.2 | PyTorch |
| mlp_fwd_256x784_wide   |     404.8 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     833.4 |   1319.8 |    787.2 |     8590.1 |  23878.1 |   5041.9 | MLX |
| train_step_256_wide    |    3505.6 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     42.8 |     26.4 |       48.5 |    162.5 |     30.1 | Peregrine |
| square_100k            |       8.6 |     41.7 |     26.7 |       16.7 |    177.1 |     29.2 | Peregrine |
| rsqrt_100k             |      22.0 |     42.9 |     32.1 |       51.8 |        — |     93.4 | Peregrine |
| floor_100k             |       9.1 |     38.4 |     19.5 |       18.4 |    415.2 |     31.6 | Peregrine |
| ceil_100k              |       8.8 |     44.4 |     21.4 |       18.3 |    352.9 |     27.1 | Peregrine |
| round_100k             |       8.9 |     42.3 |     25.9 |       43.0 |        — |     28.5 | Peregrine |
| sign_100k              |       8.7 |     40.6 |     28.7 |       49.8 |    807.5 |     49.0 | Peregrine |
| expm1_100k             |      64.8 |    110.9 |    107.1 |      149.7 |        — |    100.0 | Peregrine |
| log2_100k              |      57.3 |     86.4 |     99.6 |      151.1 |    164.0 |     57.0 | JAX |
| log10_100k             |      59.2 |     84.7 |    111.0 |      140.4 |        — |     58.1 | JAX |
| log1p_100k             |      76.9 |     86.1 |    128.9 |       97.0 |        — |    117.5 | Peregrine |
| erf_100k               |     102.9 |     58.0 |    101.0 |       57.9 |        — |     55.4 | JAX |
| sinh_100k              |      52.6 |    125.7 |     96.8 |      130.8 |    527.5 |    110.7 | Peregrine |
| cosh_100k              |      47.3 |    126.8 |     89.9 |      130.7 |    466.5 |     69.8 | Peregrine |
| arcsin_100k            |      53.1 |     75.5 |    106.7 |       62.0 |   2979.7 |    112.7 | Peregrine |
| arccos_100k            |      62.0 |     89.3 |    117.0 |       53.0 |        — |    207.2 | TensorFlow |
| arctan_100k            |      55.0 |     92.0 |     94.7 |       60.5 |   3120.7 |    221.3 | Peregrine |
| arcsinh_100k           |     209.6 |    154.5 |    337.6 |      144.6 |        — |    113.2 | JAX |
| maximum_100k           |      12.9 |     44.5 |     28.4 |       44.6 |    191.5 |     29.5 | Peregrine |
| minimum_100k           |      12.9 |     39.5 |     27.5 |       44.1 |    379.8 |     28.0 | Peregrine |
| power_100k             |     158.7 |    232.0 |    212.8 |      279.6 |        — |    144.7 | JAX |
| arctan2_100k           |      97.9 |    132.4 |    145.3 |       73.9 |        — |    323.0 | TensorFlow |
| logaddexp_100k         |     279.9 |    146.0 |    259.3 |      369.3 |        — |    151.8 | PyTorch |
| clip_100k              |       8.9 |     41.6 |     34.4 |       43.5 |    544.5 |     35.4 | Peregrine |
| where_100k             |      16.9 |     51.2 |     28.2 |       66.5 |    274.6 |     33.3 | Peregrine |
| greater_100k           |      12.9 |     49.1 |     25.2 |       57.4 |    191.5 |     28.0 | Peregrine |
| equal_100k             |      12.9 |     31.3 |     23.7 |       50.9 |    292.9 |     29.6 | Peregrine |
| sum_axis_256x512       |      19.5 |     38.8 |     22.9 |       53.0 |    209.9 |     45.5 | Peregrine |
| mean_axis_256x512      |      20.0 |     45.8 |     24.2 |       53.3 |    290.8 |     45.1 | Peregrine |
| max_axis_256x512       |      14.2 |     58.5 |     41.2 |       49.4 |    205.1 |     46.5 | Peregrine |
| min_axis_256x512       |      13.9 |     54.6 |     40.1 |       49.1 |    334.2 |     49.3 | Peregrine |
| var_256x512            |      46.6 |    276.7 |     58.1 |      245.5 |        — |     51.9 | Peregrine |
| prod_axis_256x512      |      24.9 |     39.5 |     25.9 |       49.4 |        — |     52.7 | Peregrine |
| logsumexp_256x512      |      97.5 |    197.2 |    107.2 |      354.6 |        — |    265.8 | Peregrine |
| cumsum_256x512         |     116.2 |     83.3 |    128.2 |      199.4 |    621.0 |    211.8 | PyTorch |
| argmax_axis_256x512    |      52.9 |     96.3 |    170.3 |       72.9 |   1311.0 |    173.3 | Peregrine |
| sum_axis_1024x1024     |     180.0 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     442.8 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       8.5 |     39.9 |     55.5 |       61.6 |   1850.8 |     35.9 | Peregrine |
| triu_256x256           |       8.4 |     40.3 |     54.7 |       57.8 |   1804.8 |     36.4 | Peregrine |
| repeat_64x128_2x3      |       7.4 |     45.1 |     30.5 |       75.7 |        — |     28.6 | Peregrine |
| pad_64x128             |       2.6 |      4.1 |     18.6 |       84.5 |     91.6 |     18.5 | Peregrine |
| stack_8x64x128         |       4.3 |      8.8 |     48.0 |       58.6 |    910.1 |    163.0 | Peregrine |
| diagonal_512x512       |       0.3 |      0.6 |     36.7 |       12.9 |        — |      7.5 | Peregrine |
| silu_100k              |      66.0 |     72.9 |     85.1 |      243.9 |    333.2 |     52.8 | JAX |
| softplus_100k          |     192.2 |    153.0 |    261.8 |      138.7 |    772.1 |    155.5 | TensorFlow |
| mish_100k              |     147.1 |    306.4 |    375.0 |      241.7 |   1146.1 |    230.2 | Peregrine |
| leaky_relu_100k        |       9.2 |     38.6 |     77.0 |       20.4 |        — |     30.4 | Peregrine |
| elu_100k               |      61.9 |    133.9 |    120.7 |      128.6 |    876.5 |     78.8 | Peregrine |
| hard_tanh_100k         |       8.9 |     41.8 |     34.8 |       43.7 |        — |     37.3 | Peregrine |
| relu6_100k             |       8.8 |     39.9 |     45.4 |       52.2 |    732.2 |    112.4 | Peregrine |
| hardswish_100k         |      10.3 |     40.8 |     68.8 |      199.5 |        — |     28.8 | Peregrine |
| gelu_100k              |     104.5 |     70.5 |    142.2 |      259.1 |    859.9 |    205.5 | PyTorch |
| selu_100k              |      66.0 |    130.1 |     89.5 |      146.4 |    744.6 |     95.8 | Peregrine |
| softsign_100k          |      39.3 |    137.5 |     43.5 |       48.4 |        — |     55.2 | Peregrine |
| cross_entropy_64x10    |       2.7 |     45.0 |     31.8 |      630.8 |   3444.9 |     52.3 | Peregrine |
| l1_loss_64x10          |       1.0 |      6.6 |     19.0 |       47.1 |   1150.7 |     12.5 | Peregrine |
| mse_loss_64x10         |       4.0 |      5.2 |     22.2 |       41.7 |    466.9 |     24.4 | Peregrine |
| huber_loss_64x10       |       0.3 |      4.9 |     33.9 |      258.0 |        — |     50.7 | Peregrine |
| smooth_l1_loss_64x10   |       0.8 |      5.4 |     36.7 |      253.0 |        — |     48.0 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     19.6 |      415.4 |        — |     55.5 | Peregrine |
| cosine_sim_loss_64x64  |       1.9 |     10.4 |    110.5 |      253.2 |        — |     47.8 | Peregrine |
| rmsnorm_64x512         |      20.0 |     64.6 |     33.7 |      442.7 |        — |     66.7 | Peregrine |
| conv1d_1x32x128_k3     |      21.1 |     57.9 |     27.2 |      597.6 |        — |     74.5 | Peregrine |
| avgpool2d_1x16x32x32   |      28.0 |     44.5 |    268.3 |       63.2 |        — |     45.1 | Peregrine |
| groupnorm_4x64x16x16   |      22.0 |     53.9 |    226.0 |      766.2 |        — |    283.9 | Peregrine |
| rnn_seq32_128_256      |     185.0 |    262.5 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |     995.1 |    821.5 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     821.0 |    782.1 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     806.0 |   1312.1 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     936.7 |   1168.3 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     920.3 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1306.3 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      61.4 |    262.0 |    494.9 |      129.7 |   2512.6 |    536.4 | Peregrine |
| rand_normal_100k       |     241.0 |   1000.7 |    690.8 |      333.3 |   3435.5 |    599.5 | Peregrine |
| rand_bernoulli_100k    |     120.7 |    254.7 |    451.3 |      206.7 |        — |    529.8 | Peregrine |
| rand_uniform_1M        |     612.7 |   2614.6 |   4591.9 |      439.2 |   2483.0 |   2238.8 | TensorFlow |
| rand_normal_1M         |    2420.7 |   9958.6 |   6722.8 |     2333.0 |   3350.3 |   2873.7 | TensorFlow |
| rfft_1k                |       2.0 |      4.4 |     21.4 |       44.3 |        — |     59.9 | Peregrine |
| rfft_4k                |       6.6 |     14.8 |     29.5 |       60.4 |        — |     70.4 | Peregrine |
| rfft_16k               |      29.6 |     65.2 |     75.4 |      113.0 |        — |    119.3 | Peregrine |
| fft_1k                 |       3.0 |      6.6 |     22.1 |        9.3 |        — |     29.9 | Peregrine |
| fft_4k                 |      11.9 |     26.2 |     39.5 |       18.0 |        — |     60.6 | Peregrine |
| norm_l2_1k             |       1.1 |      1.2 |     19.5 |       75.9 |        — |      4.0 | Peregrine |
| solve_64x64            |      11.5 |     18.2 |    100.2 |       26.5 |        — |     32.2 | Peregrine |
| inv_64x64              |      35.7 |     25.7 |     47.0 |       33.3 |        — |     42.2 | PyTorch |
| cholesky_64x64         |       5.7 |     48.0 |     21.9 |       20.2 |        — |     22.0 | Peregrine |
| svd_64x64              |     275.6 |    276.1 |    298.7 |      494.2 |        — |    298.1 | Peregrine |
| qr_64x64               |      40.1 |     80.1 |     56.7 |       85.4 |        — |     63.2 | Peregrine |
| eigh_64x64             |     381.8 |    215.7 |    234.4 |      147.7 |        — |    241.1 | TensorFlow |
| det_64x64              |      18.4 |     19.5 |        — |       23.7 |        — |     29.1 | Peregrine |
| solve_128x128          |      48.7 |     44.9 |    191.7 |       78.5 |        — |     91.2 | PyTorch |
| inv_128x128            |      95.8 |     59.8 |     87.1 |      142.2 |        — |     81.2 | PyTorch |
| cholesky_128x128       |      34.1 |     51.0 |     29.7 |       58.9 |        — |     36.5 | MLX |
| svd_128x128            |     984.9 |   1006.9 |   1000.4 |     1902.2 |        — |   1024.0 | Peregrine |
| qr_128x128             |     187.9 |    220.3 |    191.9 |      333.5 |        — |    201.4 | Peregrine |
| eigh_128x128           |    1852.4 |    706.7 |    726.0 |      721.8 |        — |    740.1 | PyTorch |
| det_128x128            |      40.9 |     49.4 |        — |       84.7 |        — |     77.1 | Peregrine |
| solve_256x256          |     189.2 |    185.9 |    746.1 |      372.7 |        — |    271.5 | PyTorch |
| inv_256x256            |     444.8 |    288.6 |    262.6 |      867.5 |        — |    330.2 | MLX |
| cholesky_256x256       |     145.8 |     87.5 |     71.0 |      287.2 |        — |    117.3 | MLX |
| svd_256x256            |    5833.8 |   5837.3 |   5884.9 |     8316.2 |        — |   5686.5 | JAX |
| qr_256x256             |     983.1 |    995.4 |    993.0 |     1726.4 |        — |    965.7 | JAX |
| eigh_256x256           |    5991.6 |   3460.3 |   3488.7 |     4686.8 |        — |   3618.8 | PyTorch |
| det_256x256            |     139.5 |    203.3 |        — |      438.6 |        — |    206.5 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1777.1 |   1043.4 |        — |     2357.2 |   1278.0 |   2133.2 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3443.8 |   2209.1 |        — |     3688.3 |   1276.5 |   3539.8 | tinygrad |
| add_layernorm_196x768  |     107.9 |    107.4 |        — |     1181.6 |   1132.1 |    236.4 | PyTorch |
| add_layernorm_196x1024 |     143.2 |    114.3 |        — |     1265.6 |   1138.3 |    306.5 | PyTorch |
| matmul_f32_196x768x3072 |     451.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14874.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1677.4 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26841.3 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.52x** (Peregrine is faster)
- **Peregrine vs MLX: 0.38x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.28x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.36x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 101/141 ops
- PyTorch: 18/141 ops
- JAX: 10/141 ops
- TensorFlow: 7/141 ops
- MLX: 4/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
