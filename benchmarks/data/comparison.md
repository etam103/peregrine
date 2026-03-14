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
| matmul_128x128         |      31.7 |      6.0 |     22.8 |       97.8 |    414.6 |     78.5 | PyTorch |
| matmul_256x256         |      54.8 |     31.8 |     44.6 |      193.5 |    419.2 |    163.8 | PyTorch |
| matmul_512x512         |     242.9 |    143.0 |    164.1 |      679.5 |    437.9 |    484.2 | PyTorch |
| matmul_1024x1024       |    1023.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    8776.3 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     40.6 |     27.9 |       53.1 |    187.0 |     35.2 | Peregrine |
| add_500k               |      62.6 |     62.0 |     76.8 |       88.0 |    184.7 |     61.3 | JAX |
| add_1M                 |     126.4 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     547.8 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     832.0 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     36.6 |     28.6 |       45.3 |    186.2 |     30.5 | Peregrine |
| mul_500k               |      63.0 |     61.7 |     76.6 |       78.5 |    187.9 |     61.6 | JAX |
| mul_1M                 |     127.4 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     543.4 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     901.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      50.8 |     67.9 |     59.9 |       67.7 |    221.7 |     69.6 | Peregrine |
| exp_500k               |     163.7 |    140.6 |    228.3 |      105.8 |    220.5 |    123.8 | TensorFlow |
| exp_1M                 |     138.9 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     584.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     875.7 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       9.0 |     36.9 |     28.1 |       43.3 |    333.0 |     99.5 | Peregrine |
| relu_1M                |      85.1 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     34.5 |     16.1 |       11.8 |    628.4 |     44.4 | Peregrine |
| softmax_8x512          |       4.3 |     37.7 |     20.6 |       14.4 |    612.3 |     34.0 | Peregrine |
| mlp_fwd_64x784         |      32.0 |     27.6 |     50.9 |      251.8 |   1780.9 |    168.5 | PyTorch |
| mlp_fwd_256x784_wide   |     408.2 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     826.1 |   1342.7 |    779.1 |     8711.3 |  23223.8 |   5040.5 | MLX |
| train_step_256_wide    |    3426.2 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.9 |     39.3 |     28.6 |       47.2 |    162.0 |     28.8 | Peregrine |
| square_100k            |       8.7 |     39.8 |     23.6 |       16.3 |    174.7 |     31.2 | Peregrine |
| rsqrt_100k             |      22.2 |     43.2 |     30.7 |       50.8 |        — |     85.0 | Peregrine |
| floor_100k             |       8.9 |     38.4 |     24.1 |       18.5 |    409.7 |     30.2 | Peregrine |
| ceil_100k              |       8.9 |     40.9 |     27.4 |       18.5 |    344.8 |     31.0 | Peregrine |
| round_100k             |       8.9 |     43.0 |     26.0 |       42.7 |        — |     36.0 | Peregrine |
| sign_100k              |       8.9 |     38.8 |     27.8 |       44.1 |    787.8 |     36.5 | Peregrine |
| expm1_100k             |      65.1 |    113.5 |    104.9 |      156.3 |        — |     99.4 | Peregrine |
| log2_100k              |      57.3 |     89.6 |     98.2 |      143.8 |    162.2 |     56.6 | JAX |
| log10_100k             |      59.8 |     89.8 |    106.8 |      147.6 |        — |     56.4 | JAX |
| log1p_100k             |      77.8 |     83.6 |    127.7 |       97.7 |        — |    110.0 | Peregrine |
| erf_100k               |     103.8 |     54.6 |    102.3 |       58.8 |        — |     42.8 | JAX |
| sinh_100k              |      52.6 |    153.8 |     94.0 |      138.0 |    532.5 |    107.4 | Peregrine |
| cosh_100k              |      47.8 |    145.5 |     89.6 |      131.2 |    458.3 |     69.5 | Peregrine |
| arcsin_100k            |      53.7 |     90.2 |     94.7 |       55.4 |   2885.7 |    111.9 | Peregrine |
| arccos_100k            |      62.6 |     93.1 |    111.8 |       55.0 |        — |    209.2 | TensorFlow |
| arctan_100k            |      54.8 |     98.2 |     93.7 |       57.1 |   2993.2 |    211.5 | Peregrine |
| arcsinh_100k           |     211.2 |    159.9 |    336.1 |      159.0 |        — |    114.5 | JAX |
| maximum_100k           |      13.0 |     38.1 |     27.3 |       42.9 |    189.5 |     30.5 | Peregrine |
| minimum_100k           |      13.0 |     36.2 |     27.1 |       42.9 |    373.9 |     31.4 | Peregrine |
| power_100k             |     158.5 |    251.9 |    210.8 |      322.3 |        — |    142.7 | JAX |
| arctan2_100k           |      98.1 |    156.3 |    144.2 |       70.5 |        — |    316.1 | TensorFlow |
| logaddexp_100k         |     281.0 |    152.2 |    258.8 |      374.5 |        — |    149.7 | JAX |
| clip_100k              |       8.9 |     40.6 |     34.5 |       42.1 |    530.0 |     42.0 | Peregrine |
| where_100k             |      17.0 |     49.7 |     28.4 |       65.8 |    278.6 |     33.7 | Peregrine |
| greater_100k           |      13.0 |     43.8 |     23.9 |       49.2 |    188.6 |     26.8 | Peregrine |
| equal_100k             |      12.9 |     28.0 |     24.0 |       50.8 |    283.9 |     27.4 | Peregrine |
| sum_axis_256x512       |      19.4 |     37.9 |     22.8 |       48.9 |    204.8 |     45.1 | Peregrine |
| mean_axis_256x512      |      19.5 |     40.0 |     24.7 |       49.1 |    291.0 |     45.0 | Peregrine |
| max_axis_256x512       |      14.1 |     64.5 |     42.1 |       48.2 |    201.8 |     45.5 | Peregrine |
| min_axis_256x512       |      14.1 |     62.4 |     41.9 |       48.4 |    328.0 |     47.6 | Peregrine |
| var_256x512            |      47.1 |    275.5 |     62.4 |      222.3 |        — |     78.1 | Peregrine |
| prod_axis_256x512      |      24.9 |     42.9 |     25.2 |       46.1 |        — |     55.4 | Peregrine |
| logsumexp_256x512      |      98.3 |    207.8 |    106.7 |      337.4 |        — |    269.4 | Peregrine |
| cumsum_256x512         |     116.4 |     76.9 |    128.4 |      191.8 |    603.6 |    208.7 | PyTorch |
| argmax_axis_256x512    |      52.9 |     95.2 |    170.2 |       73.3 |   1338.3 |    164.9 | Peregrine |
| sum_axis_1024x1024     |     179.4 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     441.0 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       8.0 |     43.7 |     55.6 |       52.6 |   1868.1 |     40.0 | Peregrine |
| triu_256x256           |       7.9 |     43.1 |     56.1 |       53.4 |   1889.4 |     37.6 | Peregrine |
| repeat_64x128_2x3      |       7.0 |     45.6 |     30.3 |       73.8 |        — |     28.0 | Peregrine |
| pad_64x128             |       2.6 |      4.1 |     18.7 |       82.6 |     90.3 |     20.1 | Peregrine |
| stack_8x64x128         |       4.0 |      8.9 |     43.8 |       58.0 |    903.9 |    161.2 | Peregrine |
| diagonal_512x512       |       0.3 |      0.6 |     29.4 |       12.8 |        — |      8.1 | Peregrine |
| silu_100k              |      66.0 |     70.4 |     84.7 |      251.6 |    328.5 |     51.9 | JAX |
| softplus_100k          |     186.2 |    152.1 |    260.6 |      133.2 |    765.6 |    155.2 | TensorFlow |
| mish_100k              |     141.0 |    307.2 |    370.9 |      245.8 |   1148.0 |    228.9 | Peregrine |
| leaky_relu_100k        |       8.9 |     38.3 |     74.8 |       19.9 |        — |     36.5 | Peregrine |
| elu_100k               |      61.9 |    133.0 |    115.9 |      140.4 |    858.0 |     77.5 | Peregrine |
| hard_tanh_100k         |       8.9 |     37.9 |     35.0 |       43.9 |        — |     36.0 | Peregrine |
| relu6_100k             |       8.9 |     40.1 |     43.6 |       51.1 |    728.9 |    111.2 | Peregrine |
| hardswish_100k         |      10.4 |     38.6 |     69.2 |      198.0 |        — |     26.0 | Peregrine |
| gelu_100k              |      99.3 |     65.2 |    135.7 |      237.8 |    838.6 |    203.4 | PyTorch |
| selu_100k              |      65.7 |    133.2 |     85.4 |      140.9 |    732.7 |     81.9 | Peregrine |
| softsign_100k          |      39.5 |    114.9 |     44.0 |       45.0 |        — |     57.1 | Peregrine |
| cross_entropy_64x10    |       2.8 |     34.4 |     22.2 |      627.0 |   3565.3 |     51.6 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.6 |     18.7 |       44.2 |   1165.8 |     13.2 | Peregrine |
| mse_loss_64x10         |       3.8 |      5.1 |     23.3 |       40.0 |    455.7 |     23.5 | Peregrine |
| huber_loss_64x10       |       0.3 |      4.8 |     33.8 |      242.0 |        — |     48.0 | Peregrine |
| smooth_l1_loss_64x10   |       0.8 |      5.2 |     35.0 |      241.2 |        — |     48.4 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.2 |     18.0 |      393.5 |        — |     56.5 | Peregrine |
| cosine_sim_loss_64x64  |       1.9 |     10.2 |    111.1 |      244.6 |        — |     47.9 | Peregrine |
| rmsnorm_64x512         |      19.6 |     63.9 |     33.0 |      451.9 |        — |     66.1 | Peregrine |
| conv1d_1x32x128_k3     |      20.2 |     49.8 |     27.3 |      516.7 |        — |     75.3 | Peregrine |
| avgpool2d_1x16x32x32   |      25.8 |     43.4 |    263.7 |       63.5 |        — |     43.1 | Peregrine |
| groupnorm_4x64x16x16   |      22.4 |     50.3 |    221.8 |      750.8 |        — |    266.8 | Peregrine |
| rnn_seq32_128_256      |     189.2 |    267.8 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |     994.5 |    808.8 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     803.4 |    785.2 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     821.9 |   1334.7 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     935.9 |   1374.4 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     928.7 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1307.4 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      61.3 |    262.2 |    479.7 |      124.8 |   2385.0 |    518.0 | Peregrine |
| rand_normal_100k       |     241.1 |    990.2 |    689.4 |      331.4 |   3258.0 |    592.7 | Peregrine |
| rand_bernoulli_100k    |     120.8 |    254.8 |    449.0 |      210.2 |        — |    510.8 | Peregrine |
| rand_uniform_1M        |     618.3 |   2583.3 |   4558.2 |      450.1 |   2450.2 |   2217.3 | TensorFlow |
| rand_normal_1M         |    2417.1 |   9766.1 |   6635.1 |     2127.8 |   3445.4 |   2906.8 | TensorFlow |
| rfft_1k                |       2.1 |      4.4 |     23.4 |       44.2 |        — |     58.7 | Peregrine |
| rfft_4k                |       7.2 |     14.8 |     28.1 |       54.5 |        — |     64.7 | Peregrine |
| rfft_16k               |      29.1 |     65.5 |     79.2 |      108.1 |        — |    122.6 | Peregrine |
| fft_1k                 |       3.0 |      6.6 |     22.2 |        9.0 |        — |     28.8 | Peregrine |
| fft_4k                 |      11.9 |     26.4 |     38.4 |       17.9 |        — |     65.6 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     19.5 |       71.3 |        — |      3.9 | Peregrine |
| solve_64x64            |      11.7 |     24.9 |     97.5 |       25.2 |        — |     34.7 | Peregrine |
| inv_64x64              |      36.3 |     26.1 |     48.3 |       33.5 |        — |     37.0 | PyTorch |
| cholesky_64x64         |       6.0 |     49.8 |     21.7 |       20.0 |        — |     20.5 | Peregrine |
| svd_64x64              |     275.3 |    277.4 |    292.7 |      504.5 |        — |    297.6 | Peregrine |
| qr_64x64               |      40.1 |     77.1 |     56.2 |       86.3 |        — |     62.6 | Peregrine |
| eigh_64x64             |     381.3 |    218.6 |    234.9 |      144.5 |        — |    235.2 | TensorFlow |
| det_64x64              |      18.5 |     19.9 |        — |       23.6 |        — |     28.2 | Peregrine |
| solve_128x128          |      49.2 |     44.5 |    191.1 |       79.0 |        — |     85.2 | PyTorch |
| inv_128x128            |      95.7 |     59.8 |     87.7 |      143.0 |        — |     83.5 | PyTorch |
| cholesky_128x128       |      34.3 |     52.2 |     26.8 |       60.1 |        — |     35.8 | MLX |
| svd_128x128            |    1002.1 |    987.1 |    996.6 |     1885.3 |        — |   1012.1 | PyTorch |
| qr_128x128             |     189.1 |    210.4 |    192.0 |      336.5 |        — |    237.7 | Peregrine |
| eigh_128x128           |    1877.3 |    707.9 |    724.0 |      731.6 |        — |    781.2 | PyTorch |
| det_128x128            |      39.8 |     48.7 |        — |       84.6 |        — |     76.4 | Peregrine |
| solve_256x256          |     190.4 |    174.6 |    754.5 |      388.4 |        — |    266.5 | PyTorch |
| inv_256x256            |     471.8 |    283.3 |    252.9 |      875.3 |        — |    333.5 | MLX |
| cholesky_256x256       |     145.8 |     90.3 |     56.5 |      289.7 |        — |    120.5 | MLX |
| svd_256x256            |    5830.5 |   5791.1 |   5629.4 |     8272.2 |        — |   5896.2 | MLX |
| qr_256x256             |    1013.2 |   1001.8 |   1023.6 |     1726.3 |        — |    978.0 | JAX |
| eigh_256x256           |    6150.7 |   3903.6 |   3451.6 |     4642.3 |        — |   3583.1 | MLX |
| det_256x256            |     140.1 |    227.9 |        — |      436.1 |        — |    204.7 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1768.6 |   1042.7 |        — |     2336.5 |   1244.1 |   2102.5 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3406.2 |   2046.0 |        — |     3629.5 |   1255.0 |   3408.3 | tinygrad |
| add_layernorm_196x768  |     105.8 |    101.3 |        — |     1252.5 |   1138.1 |    228.4 | PyTorch |
| add_layernorm_196x1024 |     140.4 |    109.0 |        — |     1295.9 |   1144.9 |    276.3 | PyTorch |
| matmul_f32_196x768x3072 |     442.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14683.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1467.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   27037.3 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.52x** (Peregrine is faster)
- **Peregrine vs MLX: 0.39x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.28x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.36x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 100/141 ops
- PyTorch: 17/141 ops
- JAX: 10/141 ops
- TensorFlow: 7/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
