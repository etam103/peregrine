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
| matmul_128x128         |      28.3 |      5.7 |     22.2 |       96.3 |    427.9 |     79.8 | PyTorch |
| matmul_256x256         |      58.8 |     30.5 |     48.3 |      195.5 |    426.4 |    159.5 | PyTorch |
| matmul_512x512         |     244.7 |    140.7 |    173.3 |      688.8 |    421.4 |    505.6 | PyTorch |
| matmul_1024x1024       |    1020.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9968.7 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      13.2 |     40.4 |     33.0 |       48.6 |    187.7 |     30.7 | Peregrine |
| add_500k               |     118.9 |     58.6 |     81.2 |       84.1 |    187.1 |     65.4 | PyTorch |
| add_1M                 |     133.3 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     531.2 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |    1001.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |       9.6 |     37.0 |     29.7 |       43.7 |    191.0 |     35.1 | Peregrine |
| mul_500k               |     143.4 |     60.4 |     83.0 |       76.7 |    190.0 |     62.3 | PyTorch |
| mul_1M                 |     164.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     600.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     895.9 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |     120.5 |     69.1 |     62.8 |       70.9 |    221.6 |     48.2 | JAX |
| exp_500k               |     210.2 |    141.6 |    228.7 |      105.8 |    231.4 |    126.5 | TensorFlow |
| exp_1M                 |     316.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1120.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2221.2 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     42.2 |     26.6 |       41.6 |    338.0 |    101.3 | Peregrine |
| relu_1M                |     111.5 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     29.9 |     15.9 |       11.7 |    616.0 |     32.6 | Peregrine |
| softmax_8x512          |       4.4 |     36.6 |     18.2 |       14.7 |    628.0 |     34.4 | Peregrine |
| mlp_fwd_64x784         |      33.5 |     27.1 |     55.0 |      262.9 |   1818.8 |    181.7 | PyTorch |
| mlp_fwd_256x784_wide   |     419.1 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     815.9 |   1313.2 |    799.5 |     8930.0 |  25247.2 |   5168.4 | MLX |
| train_step_256_wide    |    3336.6 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.6 |     40.8 |     24.8 |       49.6 |    166.8 |     28.6 | Peregrine |
| square_100k            |       9.0 |     37.7 |     23.8 |       15.6 |    185.7 |     32.0 | Peregrine |
| rsqrt_100k             |      87.2 |     42.1 |     37.5 |       52.6 |        — |     88.7 | MLX |
| floor_100k             |       8.8 |     38.6 |     24.0 |       16.9 |    414.9 |     22.0 | Peregrine |
| ceil_100k              |       8.8 |     40.8 |     23.9 |       16.9 |    356.8 |     29.9 | Peregrine |
| round_100k             |       8.8 |     42.1 |     23.9 |       45.0 |        — |     31.0 | Peregrine |
| sign_100k              |       8.8 |     39.4 |     27.9 |       46.2 |    812.1 |     36.6 | Peregrine |
| expm1_100k             |     152.5 |    109.1 |    111.2 |      140.2 |        — |    101.0 | JAX |
| log2_100k              |     104.6 |     85.7 |    102.3 |      151.6 |    166.2 |     56.6 | JAX |
| log10_100k             |     113.2 |     82.3 |    110.1 |      155.6 |        — |     66.6 | JAX |
| log1p_100k             |     111.4 |     82.9 |    127.5 |       97.0 |        — |    110.8 | PyTorch |
| erf_100k               |     111.3 |     57.6 |    107.7 |       58.5 |        — |     56.6 | JAX |
| sinh_100k              |      52.1 |    126.8 |     98.7 |      140.7 |    553.9 |    112.0 | Peregrine |
| cosh_100k              |      47.2 |    125.0 |     92.8 |      126.3 |    470.5 |     77.2 | Peregrine |
| arcsin_100k            |      53.1 |     82.3 |     96.1 |       58.0 |   3035.2 |    113.8 | Peregrine |
| arccos_100k            |     109.8 |     88.5 |    110.8 |       52.9 |        — |    211.5 | TensorFlow |
| arctan_100k            |      54.9 |     96.6 |     93.2 |       57.9 |   3133.8 |    223.2 | Peregrine |
| arcsinh_100k           |     133.9 |    161.7 |    335.1 |      133.0 |        — |    123.2 | JAX |
| maximum_100k           |      12.8 |     38.7 |     28.2 |       41.6 |    191.2 |     32.8 | Peregrine |
| minimum_100k           |      12.7 |     40.6 |     27.3 |       40.5 |    385.1 |     32.7 | Peregrine |
| power_100k             |     394.0 |    240.1 |    215.1 |      273.0 |        — |    144.7 | JAX |
| arctan2_100k           |      97.0 |    134.5 |    149.2 |       72.6 |        — |    321.6 | TensorFlow |
| logaddexp_100k         |     415.8 |    150.2 |    258.1 |      365.0 |        — |    151.2 | PyTorch |
| clip_100k              |       8.2 |     40.1 |     35.0 |       42.3 |    541.5 |     47.2 | Peregrine |
| where_100k             |      15.0 |     50.0 |     27.0 |       67.0 |    284.1 |     32.5 | Peregrine |
| greater_100k           |      10.4 |     48.0 |     25.1 |       57.4 |    190.3 |     30.7 | Peregrine |
| equal_100k             |       9.9 |     31.6 |     23.7 |       56.6 |    294.9 |     27.6 | Peregrine |
| sum_axis_256x512       |     115.8 |     42.4 |     23.2 |       53.3 |    209.0 |     49.9 | MLX |
| mean_axis_256x512      |     114.6 |     41.9 |     24.1 |       51.5 |    293.6 |     51.9 | MLX |
| max_axis_256x512       |     157.4 |     55.9 |     41.2 |       48.9 |    202.6 |     45.2 | MLX |
| min_axis_256x512       |     157.5 |     55.4 |     41.8 |       47.3 |    329.9 |     46.3 | MLX |
| var_256x512            |     240.7 |    272.9 |     61.8 |      222.2 |        — |     80.0 | MLX |
| prod_axis_256x512      |     149.7 |     39.3 |     25.5 |       50.7 |        — |     56.8 | MLX |
| logsumexp_256x512      |     391.2 |    194.9 |    108.5 |      327.9 |        — |    285.6 | MLX |
| cumsum_256x512         |     124.5 |     76.9 |    130.3 |      189.8 |    614.9 |    212.2 | PyTorch |
| argmax_axis_256x512    |     157.5 |     92.8 |    178.3 |       74.9 |   1350.1 |    181.1 | TensorFlow |
| sum_axis_1024x1024     |     939.9 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    1934.2 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      34.6 |     37.9 |     59.3 |       53.6 |   1926.2 |     36.4 | Peregrine |
| triu_256x256           |      34.8 |     37.6 |     58.7 |       54.8 |   1841.6 |     36.5 | Peregrine |
| repeat_64x128_2x3      |     127.0 |     49.2 |     31.6 |       76.1 |        — |     28.8 | JAX |
| pad_64x128             |      17.3 |      4.4 |     19.9 |       84.7 |     89.9 |     19.8 | PyTorch |
| stack_8x64x128         |      15.7 |      8.7 |     46.7 |       57.5 |    934.7 |    177.8 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     29.0 |       12.7 |        — |      9.0 | PyTorch |
| silu_100k              |      65.7 |     72.3 |     86.7 |      222.8 |    336.3 |     71.8 | Peregrine |
| softplus_100k          |     382.3 |    157.2 |    262.2 |      128.5 |    785.3 |    163.6 | TensorFlow |
| mish_100k              |    2169.3 |    308.6 |    375.9 |      246.3 |   1174.2 |    240.1 | JAX |
| leaky_relu_100k        |       8.1 |     40.1 |     77.7 |       19.8 |        — |     30.8 | Peregrine |
| elu_100k               |     144.7 |    135.8 |    126.7 |      134.5 |    901.9 |     82.2 | JAX |
| hard_tanh_100k         |       8.1 |     42.0 |     36.8 |       43.1 |        — |     40.3 | Peregrine |
| relu6_100k             |       8.1 |     41.9 |     51.6 |       54.6 |    739.0 |    108.2 | Peregrine |
| hardswish_100k         |      10.2 |     40.7 |     68.1 |      212.7 |        — |     26.8 | Peregrine |
| gelu_100k              |      77.7 |     64.5 |    137.7 |      242.6 |    855.9 |    216.0 | PyTorch |
| selu_100k              |      64.9 |    121.5 |     85.5 |      129.5 |    746.3 |     83.2 | Peregrine |
| softsign_100k          |      35.9 |    119.8 |     43.1 |       48.4 |        — |     58.9 | Peregrine |
| cross_entropy_64x10    |       2.7 |     37.8 |     25.1 |      630.2 |   3493.1 |     55.9 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.6 |     19.1 |       44.3 |   1165.3 |     12.0 | Peregrine |
| mse_loss_64x10         |       4.0 |      5.1 |     22.6 |       39.8 |    453.3 |     25.2 | Peregrine |
| huber_loss_64x10       |       5.2 |      5.2 |     32.9 |      243.2 |        — |     48.4 | PyTorch |
| smooth_l1_loss_64x10   |       5.1 |      5.4 |     33.4 |      244.4 |        — |     48.2 | Peregrine |
| kl_div_loss_64x10      |       2.6 |      6.5 |     18.0 |      378.7 |        — |     58.8 | Peregrine |
| cosine_sim_loss_64x64  |      13.6 |     10.5 |    109.9 |      240.9 |        — |     64.3 | PyTorch |
| rmsnorm_64x512         |      57.6 |     69.9 |     33.6 |      439.7 |        — |     81.7 | MLX |
| conv1d_1x32x128_k3     |      19.6 |     53.9 |     29.8 |      507.6 |        — |     72.7 | Peregrine |
| avgpool2d_1x16x32x32   |      25.5 |     41.8 |    277.0 |       62.7 |        — |     43.7 | Peregrine |
| groupnorm_4x64x16x16   |      74.0 |     54.1 |    224.1 |      774.1 |        — |    278.3 | PyTorch |
| rnn_seq32_128_256      |     186.5 |    265.7 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1149.2 |    800.8 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     874.5 |    777.4 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     813.2 |   1309.3 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     941.4 |   1147.1 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     913.5 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1322.5 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     108.4 |    262.1 |    485.5 |      121.3 |   2484.7 |    536.9 | Peregrine |
| rand_normal_100k       |     783.0 |    987.3 |    697.3 |      343.7 |   3361.9 |    613.5 | TensorFlow |
| rand_bernoulli_100k    |     309.2 |    255.8 |    455.0 |      214.7 |        — |    548.5 | TensorFlow |
| rand_uniform_1M        |    1092.0 |   2619.8 |   4621.4 |      423.2 |   2454.1 |   2277.7 | TensorFlow |
| rand_normal_1M         |    7740.5 |   9961.3 |   6711.1 |     2094.5 |   3354.0 |   2918.6 | TensorFlow |
| rfft_1k                |       2.0 |      4.4 |     21.8 |       45.1 |        — |     59.3 | Peregrine |
| rfft_4k                |       7.2 |     14.8 |     31.9 |       54.9 |        — |     68.2 | Peregrine |
| rfft_16k               |      29.3 |     65.7 |     77.4 |      107.2 |        — |    116.2 | Peregrine |
| fft_1k                 |       3.1 |      6.8 |     25.6 |        9.0 |        — |     39.8 | Peregrine |
| fft_4k                 |      11.9 |     26.7 |     43.9 |       17.8 |        — |     66.2 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     20.0 |       70.3 |        — |      4.0 | Peregrine |
| solve_64x64            |      11.8 |     23.7 |     97.6 |       25.2 |        — |     32.5 | Peregrine |
| inv_64x64              |      36.2 |     25.2 |     49.3 |       33.2 |        — |     37.3 | PyTorch |
| cholesky_64x64         |       9.1 |     42.1 |     22.4 |       19.8 |        — |     22.4 | Peregrine |
| svd_64x64              |     275.9 |    278.6 |    291.2 |      497.6 |        — |    314.4 | Peregrine |
| qr_64x64               |      40.0 |     82.9 |     55.2 |       85.5 |        — |     62.8 | Peregrine |
| eigh_64x64             |     386.4 |    217.4 |    234.5 |      146.0 |        — |    243.6 | TensorFlow |
| det_64x64              |      22.5 |     19.6 |        — |       22.9 |        — |     28.9 | PyTorch |
| solve_128x128          |      49.0 |     45.5 |    191.0 |       78.5 |        — |     85.4 | PyTorch |
| inv_128x128            |      93.9 |     61.2 |     90.9 |      142.8 |        — |     88.5 | PyTorch |
| cholesky_128x128       |      49.8 |     54.5 |     26.5 |       59.8 |        — |     39.2 | MLX |
| svd_128x128            |     996.6 |    995.5 |    969.4 |     1837.4 |        — |   1018.4 | MLX |
| qr_128x128             |     186.5 |    222.4 |    198.1 |      334.4 |        — |    192.0 | Peregrine |
| eigh_128x128           |    1870.5 |    707.5 |    723.8 |      713.6 |        — |    731.6 | PyTorch |
| det_128x128            |      52.3 |     50.0 |        — |       81.6 |        — |     79.0 | PyTorch |
| solve_256x256          |     190.1 |    183.1 |    818.2 |      377.5 |        — |    265.9 | PyTorch |
| inv_256x256            |     462.8 |    288.7 |    254.5 |      866.6 |        — |    331.6 | MLX |
| cholesky_256x256       |     228.7 |     76.1 |     60.2 |      288.9 |        — |    120.4 | MLX |
| svd_256x256            |    6032.4 |   5838.2 |   5740.8 |     8205.6 |        — |   5871.2 | MLX |
| qr_256x256             |    1042.0 |   1013.8 |   1003.5 |     1727.9 |        — |   1006.0 | MLX |
| eigh_256x256           |    6108.9 |   3462.5 |   3575.2 |     4676.7 |        — |   3602.2 | PyTorch |
| det_256x256            |     212.4 |    202.2 |        — |      439.0 |        — |    205.9 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1135.0 |    953.5 |        — |     2392.7 |   1325.1 |   2351.7 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2194.2 |   1984.0 |        — |     3672.8 |   1316.7 |   4016.7 | tinygrad |
| add_layernorm_196x768  |     105.4 |    100.3 |        — |     1215.7 |   1221.0 |    240.2 | PyTorch |
| add_layernorm_196x1024 |     143.1 |    111.7 |        — |     1292.6 |   1213.2 |    304.5 | PyTorch |
| matmul_f32_196x768x3072 |     664.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14819.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1568.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   27145.2 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.80x** (Peregrine is faster)
- **Peregrine vs MLX: 0.61x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.44x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.08x** (Peregrine is faster)
- **Peregrine vs JAX: 0.55x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 74/141 ops
- PyTorch: 30/141 ops
- MLX: 16/141 ops
- TensorFlow: 10/141 ops
- JAX: 10/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
