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
| matmul_128x128         |      25.0 |      5.8 |     20.4 |       87.2 |    426.6 |     56.6 | PyTorch |
| matmul_256x256         |      77.8 |     30.0 |     44.7 |      122.3 |    419.5 |    148.5 | PyTorch |
| matmul_512x512         |     220.8 |    128.3 |    157.2 |      657.7 |    425.6 |    496.6 | PyTorch |
| matmul_1024x1024       |     995.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9186.6 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     34.2 |     32.2 |       45.3 |    185.6 |     35.6 | Peregrine |
| add_500k               |      62.7 |     59.4 |     81.1 |       76.2 |    189.3 |     59.7 | PyTorch |
| add_1M                 |     125.4 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     528.0 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     905.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.7 |     38.9 |     29.5 |       38.6 |    187.7 |     26.1 | Peregrine |
| mul_500k               |      63.0 |     63.1 |     83.4 |       60.4 |    190.5 |     65.3 | TensorFlow |
| mul_1M                 |     125.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     521.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     918.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      50.7 |     43.9 |     56.7 |       63.8 |    224.0 |     46.7 | PyTorch |
| exp_500k               |     102.1 |    138.2 |    227.2 |       96.8 |    224.4 |    123.0 | TensorFlow |
| exp_1M                 |     147.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     445.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     828.5 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     30.2 |     27.3 |       35.6 |    345.0 |    108.2 | Peregrine |
| relu_1M                |      84.1 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     25.4 |     19.7 |       10.6 |    614.8 |     31.2 | Peregrine |
| softmax_8x512          |       4.3 |     27.7 |     19.2 |       14.0 |    606.1 |     33.6 | Peregrine |
| mlp_fwd_64x784         |      32.6 |     27.5 |     48.6 |      204.9 |   1769.2 |    179.5 | PyTorch |
| mlp_fwd_256x784_wide   |     390.8 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     810.5 |   1229.0 |    804.4 |     7902.1 |  23045.4 |   4951.6 | MLX |
| train_step_256_wide    |    3321.8 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     38.5 |     26.0 |       45.3 |    170.4 |     27.8 | Peregrine |
| square_100k            |       8.8 |     29.3 |     25.9 |       15.3 |    175.5 |     33.0 | Peregrine |
| rsqrt_100k             |      21.9 |     32.3 |     29.7 |       48.1 |        — |     94.2 | Peregrine |
| floor_100k             |       8.8 |     30.2 |     23.9 |       14.9 |    427.8 |     28.6 | Peregrine |
| ceil_100k              |       8.8 |     28.7 |     24.1 |       15.0 |    358.8 |     27.6 | Peregrine |
| round_100k             |       8.8 |     32.5 |     24.2 |       42.9 |        — |     33.6 | Peregrine |
| sign_100k              |       8.8 |     31.0 |     28.9 |       46.7 |    796.4 |     36.9 | Peregrine |
| expm1_100k             |      64.4 |     68.5 |    108.3 |      144.9 |        — |    100.8 | Peregrine |
| log2_100k              |      56.6 |     55.6 |     98.3 |      130.2 |    167.7 |     56.3 | PyTorch |
| log10_100k             |      59.0 |     55.2 |    107.9 |      137.4 |        — |     58.8 | PyTorch |
| log1p_100k             |      76.9 |     52.1 |    130.2 |       92.2 |        — |    110.4 | PyTorch |
| erf_100k               |     102.5 |     43.1 |    102.8 |       52.4 |        — |     54.5 | PyTorch |
| sinh_100k              |      52.0 |    105.0 |     94.6 |      126.2 |    532.5 |    115.4 | Peregrine |
| cosh_100k              |      47.2 |    100.6 |     89.6 |      123.5 |    462.2 |     69.7 | Peregrine |
| arcsin_100k            |      53.1 |     49.9 |     91.8 |       56.1 |   2919.8 |    112.4 | PyTorch |
| arccos_100k            |      61.9 |     55.5 |    108.8 |       54.1 |        — |    194.8 | TensorFlow |
| arctan_100k            |      54.1 |     60.5 |     93.0 |       54.4 |   2998.2 |    219.9 | Peregrine |
| arcsinh_100k           |     208.7 |    130.2 |    332.3 |      129.0 |        — |    112.4 | JAX |
| maximum_100k           |      12.7 |     29.8 |     24.4 |       41.9 |    193.7 |     27.1 | Peregrine |
| minimum_100k           |      12.7 |     29.7 |     27.0 |       41.6 |    381.4 |     30.2 | Peregrine |
| power_100k             |     156.7 |    211.0 |    215.1 |      264.0 |        — |    142.2 | JAX |
| arctan2_100k           |      96.9 |    110.0 |    149.5 |       53.5 |        — |    314.8 | TensorFlow |
| logaddexp_100k         |     277.7 |    125.3 |    266.9 |      347.0 |        — |    143.1 | PyTorch |
| clip_100k              |       8.8 |     31.0 |     35.1 |       39.7 |    529.0 |     41.2 | Peregrine |
| where_100k             |      16.7 |     33.3 |     27.9 |       64.7 |    276.8 |     34.6 | Peregrine |
| greater_100k           |      12.7 |     33.3 |     24.2 |       50.6 |    192.0 |     25.9 | Peregrine |
| equal_100k             |      12.7 |     34.9 |     20.4 |       57.2 |    287.7 |     26.2 | Peregrine |
| sum_axis_256x512       |      19.2 |     37.6 |     18.8 |       46.3 |    209.1 |     53.5 | MLX |
| mean_axis_256x512      |      19.2 |     40.1 |     20.6 |       47.6 |    291.9 |     47.3 | Peregrine |
| max_axis_256x512       |      13.9 |     41.0 |     38.9 |       48.1 |    201.0 |     48.2 | Peregrine |
| min_axis_256x512       |      13.9 |     38.6 |     36.5 |       45.6 |    326.7 |     48.2 | Peregrine |
| var_256x512            |      46.6 |    381.4 |     54.1 |      183.8 |        — |     82.2 | Peregrine |
| prod_axis_256x512      |      24.6 |     51.2 |     21.3 |       47.1 |        — |     55.6 | MLX |
| logsumexp_256x512      |      98.4 |    210.2 |    107.0 |      278.1 |        — |    283.9 | Peregrine |
| cumsum_256x512         |     112.8 |     74.9 |    128.2 |      161.2 |    642.6 |    190.8 | PyTorch |
| argmax_axis_256x512    |      51.8 |     80.2 |    170.2 |       52.3 |   1320.2 |    175.1 | Peregrine |
| sum_axis_1024x1024     |     177.4 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     451.4 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.9 |     42.2 |     55.9 |       49.9 |   1845.9 |     36.1 | Peregrine |
| triu_256x256           |       7.8 |     41.9 |     51.2 |       46.0 |   1848.8 |     36.1 | Peregrine |
| repeat_64x128_2x3      |       7.6 |     51.6 |     26.9 |       72.5 |        — |     28.5 | Peregrine |
| pad_64x128             |       2.6 |      4.8 |     15.2 |       81.5 |     92.8 |     18.2 | Peregrine |
| stack_8x64x128         |       4.1 |      9.5 |     42.2 |       53.5 |    931.8 |    157.3 | Peregrine |
| diagonal_512x512       |       0.3 |      0.7 |     24.0 |       11.1 |        — |      7.4 | Peregrine |
| silu_100k              |      65.2 |     60.4 |     84.2 |      178.6 |    330.5 |     53.5 | JAX |
| softplus_100k          |     184.2 |    130.8 |    265.9 |      104.2 |    785.6 |    156.0 | TensorFlow |
| mish_100k              |     139.4 |    301.9 |    376.8 |      223.2 |   1161.7 |    234.9 | Peregrine |
| leaky_relu_100k        |       8.8 |     40.1 |     79.7 |       18.8 |        — |     28.1 | Peregrine |
| elu_100k               |      61.1 |    103.8 |    117.5 |      127.1 |    878.7 |     77.9 | Peregrine |
| hard_tanh_100k         |       8.8 |     32.4 |     35.3 |       40.2 |        — |     37.4 | Peregrine |
| relu6_100k             |       8.8 |     32.0 |     45.5 |       51.8 |    740.1 |    111.8 | Peregrine |
| hardswish_100k         |      10.2 |     30.3 |     69.1 |      204.0 |        — |     25.5 | Peregrine |
| gelu_100k              |      98.4 |     46.5 |    137.4 |      222.3 |    854.2 |    203.3 | PyTorch |
| selu_100k              |      64.9 |    102.8 |     86.2 |      124.6 |    748.8 |     82.1 | Peregrine |
| softsign_100k          |      39.0 |    119.5 |     43.9 |       44.9 |        — |     56.1 | Peregrine |
| cross_entropy_64x10    |       2.7 |     37.7 |     23.0 |      580.4 |   3389.1 |     52.7 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.6 |     15.7 |       39.0 |   1123.4 |     13.1 | Peregrine |
| mse_loss_64x10         |       4.0 |      5.0 |     19.0 |       35.0 |    454.3 |     23.9 | Peregrine |
| huber_loss_64x10       |       0.3 |      4.9 |     32.5 |      219.0 |        — |     47.5 | Peregrine |
| smooth_l1_loss_64x10   |       0.8 |      5.2 |     30.2 |      218.3 |        — |     48.1 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     16.3 |      351.4 |        — |     57.1 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.4 |    111.7 |      216.7 |        — |     57.4 | Peregrine |
| rmsnorm_64x512         |      19.1 |     51.9 |     32.8 |      431.7 |        — |     80.4 | Peregrine |
| conv1d_1x32x128_k3     |      20.6 |     46.8 |     27.4 |      491.6 |        — |     71.8 | Peregrine |
| avgpool2d_1x16x32x32   |      25.6 |     32.2 |    265.3 |       58.5 |        — |     42.5 | Peregrine |
| groupnorm_4x64x16x16   |      22.2 |     37.8 |    222.6 |      687.3 |        — |    263.2 | Peregrine |
| rnn_seq32_128_256      |     188.9 |    257.1 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1029.2 |    802.3 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     710.6 |    762.8 |        — |          — |        — |        — | Peregrine |
| optim_adam_64          |     815.8 |   1253.8 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     928.9 |   1109.5 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     916.6 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1297.0 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      64.0 |    262.4 |    483.9 |      114.0 |   2411.6 |    515.0 | Peregrine |
| rand_normal_100k       |     236.4 |    990.6 |    687.5 |      318.7 |   3314.7 |    585.2 | Peregrine |
| rand_bernoulli_100k    |     119.5 |    255.0 |    454.0 |      204.2 |        — |    519.4 | Peregrine |
| rand_uniform_1M        |     600.5 |   2612.0 |   4558.0 |      412.7 |   2425.6 |   2261.2 | TensorFlow |
| rand_normal_1M         |    2412.7 |   9896.9 |   6622.9 |     2006.3 |   3338.9 |   2945.8 | TensorFlow |
| rfft_1k                |       2.1 |      4.4 |     21.8 |       39.3 |        — |     48.6 | Peregrine |
| rfft_4k                |       6.6 |     14.8 |     32.2 |       50.9 |        — |     69.4 | Peregrine |
| rfft_16k               |      29.5 |     67.0 |     78.1 |      102.8 |        — |    121.8 | Peregrine |
| fft_1k                 |       3.1 |      7.0 |     23.9 |        7.9 |        — |     17.6 | Peregrine |
| fft_4k                 |      11.9 |     26.4 |     40.8 |       16.4 |        — |     61.8 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     17.3 |       63.9 |        — |      3.7 | Peregrine |
| solve_64x64            |      11.8 |     23.8 |     87.5 |       23.5 |        — |     32.3 | Peregrine |
| inv_64x64              |      36.3 |     25.8 |     51.9 |       31.6 |        — |     36.8 | PyTorch |
| cholesky_64x64         |       5.9 |     29.6 |     21.7 |       18.8 |        — |     20.7 | Peregrine |
| svd_64x64              |     274.9 |    275.4 |    292.0 |      493.0 |        — |    298.5 | Peregrine |
| qr_64x64               |      40.0 |     72.6 |     56.5 |       83.8 |        — |     65.2 | Peregrine |
| eigh_64x64             |     381.3 |    217.4 |    231.2 |      145.9 |        — |    231.7 | TensorFlow |
| det_64x64              |      18.3 |     19.5 |        — |       21.6 |        — |     35.4 | Peregrine |
| solve_128x128          |      48.7 |     44.7 |    189.6 |       76.2 |        — |     84.2 | PyTorch |
| inv_128x128            |      91.8 |     59.7 |     88.2 |      139.8 |        — |     82.9 | PyTorch |
| cholesky_128x128       |      35.1 |     45.8 |     26.3 |       58.3 |        — |     36.5 | MLX |
| svd_128x128            |     984.8 |    985.1 |    989.0 |     1870.2 |        — |   1012.8 | Peregrine |
| qr_128x128             |     187.5 |    212.8 |    190.9 |      331.8 |        — |    190.1 | Peregrine |
| eigh_128x128           |    1852.6 |    700.7 |    720.3 |      723.7 |        — |    733.1 | PyTorch |
| det_128x128            |      39.5 |     49.5 |        — |       81.9 |        — |     76.0 | Peregrine |
| solve_256x256          |     189.5 |    175.3 |    741.7 |      375.6 |        — |    255.2 | PyTorch |
| inv_256x256            |     450.3 |    283.5 |    249.2 |      861.7 |        — |    329.4 | MLX |
| cholesky_256x256       |     145.2 |     81.0 |     56.6 |      285.8 |        — |    113.1 | MLX |
| svd_256x256            |    5999.8 |   5784.2 |   5764.8 |     8270.9 |        — |   5716.6 | JAX |
| qr_256x256             |     990.3 |    987.7 |    961.9 |     1726.2 |        — |    962.7 | MLX |
| eigh_256x256           |    6001.3 |   3402.8 |   3439.9 |     4686.8 |        — |   3557.7 | PyTorch |
| det_256x256            |     139.3 |    200.8 |        — |      437.8 |        — |    205.5 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1784.6 |    871.2 |        — |     2363.2 |   1249.3 |   2132.1 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3415.3 |   1986.8 |        — |     3654.7 |   1505.1 |   3844.5 | tinygrad |
| add_layernorm_196x768  |     108.0 |    108.1 |        — |     1123.5 |   1327.9 |    220.4 | Peregrine |
| add_layernorm_196x1024 |     143.2 |    107.9 |        — |     1209.1 |   1311.2 |    272.6 | PyTorch |
| matmul_f32_196x768x3072 |     699.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14741.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1591.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26531.8 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.58x** (Peregrine is faster)
- **Peregrine vs MLX: 0.39x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.30x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.37x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 98/141 ops
- PyTorch: 23/141 ops
- TensorFlow: 8/141 ops
- MLX: 7/141 ops
- JAX: 4/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
