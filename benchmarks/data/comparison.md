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
| matmul_128x128         |      20.9 |      5.7 |     19.0 |       99.7 |    441.0 |     83.1 | PyTorch |
| matmul_256x256         |      54.5 |     30.2 |     45.9 |      181.7 |    445.1 |    178.1 | PyTorch |
| matmul_512x512         |     239.3 |    142.6 |    172.7 |      686.2 |    433.6 |    868.0 | PyTorch |
| matmul_1024x1024       |     958.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    8985.7 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.8 |     42.2 |     32.3 |       52.9 |    191.8 |     42.8 | Peregrine |
| add_500k               |      62.7 |     57.9 |     80.4 |       84.7 |    190.5 |     69.3 | PyTorch |
| add_1M                 |     135.0 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |    1143.1 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     893.1 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.8 |     43.5 |     30.9 |       44.1 |    190.8 |     32.6 | Peregrine |
| mul_500k               |      52.7 |     57.5 |     72.7 |       83.0 |    191.8 |     60.9 | Peregrine |
| mul_1M                 |     121.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     618.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |    1083.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      50.3 |     66.3 |     58.6 |       72.0 |    229.2 |     47.5 | JAX |
| exp_500k               |     103.7 |    135.1 |    236.0 |       99.6 |    223.9 |    127.4 | TensorFlow |
| exp_1M                 |     156.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     543.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    1067.3 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       9.1 |     39.5 |     28.3 |       39.8 |    352.3 |     97.6 | Peregrine |
| relu_1M                |      84.7 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     29.5 |     23.1 |       11.4 |    640.4 |     32.4 | Peregrine |
| softmax_8x512          |       4.1 |     41.5 |     18.8 |       14.8 |    636.5 |     33.1 | Peregrine |
| mlp_fwd_64x784         |      31.9 |     26.4 |     47.7 |      263.4 |   1841.1 |    177.3 | PyTorch |
| mlp_fwd_256x784_wide   |     400.0 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     818.0 |   1329.6 |    870.7 |     9265.7 |  24810.2 |   5203.2 | Peregrine |
| train_step_256_wide    |    3346.2 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.6 |     38.8 |     29.4 |       50.3 |    166.3 |     29.9 | Peregrine |
| square_100k            |       8.8 |     39.7 |     26.2 |       16.4 |    177.2 |     31.8 | Peregrine |
| rsqrt_100k             |      21.9 |     41.9 |     32.5 |       50.3 |        — |     79.8 | Peregrine |
| floor_100k             |       8.8 |     41.5 |     27.5 |       17.9 |    417.5 |     35.0 | Peregrine |
| ceil_100k              |       8.8 |     40.5 |     29.6 |       18.0 |    364.0 |     36.1 | Peregrine |
| round_100k             |       8.8 |     40.0 |     26.4 |       53.6 |        — |     36.5 | Peregrine |
| sign_100k              |       8.8 |     41.1 |     29.2 |       55.0 |    821.0 |     37.3 | Peregrine |
| expm1_100k             |      64.4 |    108.5 |    117.2 |      157.5 |        — |    112.0 | Peregrine |
| log2_100k              |      56.7 |     88.5 |    109.2 |      155.4 |    169.6 |     61.7 | Peregrine |
| log10_100k             |      59.0 |     86.9 |    110.9 |      143.5 |        — |     56.8 | JAX |
| log1p_100k             |      77.9 |     87.3 |    134.6 |      105.5 |        — |    108.2 | Peregrine |
| erf_100k               |     102.6 |     56.0 |    106.7 |       61.1 |        — |     58.5 | PyTorch |
| sinh_100k              |      52.0 |    129.9 |    101.4 |      126.1 |    544.1 |    113.4 | Peregrine |
| cosh_100k              |      47.2 |    120.0 |     98.5 |      122.4 |    475.5 |     70.2 | Peregrine |
| arcsin_100k            |      53.1 |     76.9 |     97.6 |       55.6 |   3003.9 |    111.6 | Peregrine |
| arccos_100k            |      62.0 |     85.0 |    115.5 |       50.6 |        — |    199.4 | TensorFlow |
| arctan_100k            |      54.2 |     99.2 |     99.5 |       61.4 |   3077.9 |    222.4 | Peregrine |
| arcsinh_100k           |     209.3 |    149.2 |    350.4 |      143.2 |        — |    126.6 | JAX |
| maximum_100k           |      12.7 |     39.9 |     28.5 |       44.6 |    210.6 |     38.9 | Peregrine |
| minimum_100k           |      12.7 |     38.8 |     28.9 |       43.9 |    420.2 |     28.0 | Peregrine |
| power_100k             |     156.7 |    233.4 |    220.2 |      281.7 |        — |    143.8 | JAX |
| arctan2_100k           |      96.9 |    136.9 |    151.9 |       69.3 |        — |    314.4 | TensorFlow |
| logaddexp_100k         |     277.9 |    151.2 |    267.9 |      378.1 |        — |    151.2 | PyTorch |
| clip_100k              |       8.8 |     46.8 |     38.4 |       43.1 |    545.9 |     39.1 | Peregrine |
| where_100k             |      16.9 |     55.0 |     28.5 |       67.3 |    283.4 |     33.9 | Peregrine |
| greater_100k           |      12.8 |     47.8 |     25.9 |       57.5 |    193.8 |     25.0 | Peregrine |
| equal_100k             |      12.7 |     35.7 |     26.2 |       49.3 |    299.7 |     26.8 | Peregrine |
| sum_axis_256x512       |      19.2 |     47.2 |     24.1 |       51.5 |    211.1 |     47.5 | Peregrine |
| mean_axis_256x512      |      19.2 |     40.8 |     26.1 |       53.6 |    301.1 |     47.1 | Peregrine |
| max_axis_256x512       |      13.9 |     53.9 |     42.9 |       49.2 |    205.0 |     45.4 | Peregrine |
| min_axis_256x512       |      14.0 |     53.6 |     44.3 |       54.1 |    337.3 |     46.4 | Peregrine |
| var_256x512            |      46.5 |    273.5 |     64.3 |      215.9 |        — |     77.8 | Peregrine |
| prod_axis_256x512      |      24.7 |     39.6 |     25.9 |       48.5 |        — |     49.7 | Peregrine |
| logsumexp_256x512      |      97.4 |    210.5 |    111.1 |      335.8 |        — |    294.1 | Peregrine |
| cumsum_256x512         |     114.8 |     77.4 |    137.1 |      186.9 |    648.7 |    208.6 | PyTorch |
| argmax_axis_256x512    |      52.7 |     87.0 |    181.4 |       70.9 |   1348.7 |    178.0 | Peregrine |
| sum_axis_1024x1024     |     177.3 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     435.9 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.8 |     42.5 |     58.3 |       55.6 |   1874.8 |     36.6 | Peregrine |
| triu_256x256           |       7.8 |     40.9 |     55.5 |       53.1 |   1959.8 |     36.6 | Peregrine |
| repeat_64x128_2x3      |       6.1 |     46.6 |     32.2 |       79.5 |        — |     28.4 | Peregrine |
| pad_64x128             |       2.5 |      4.5 |     20.9 |       84.0 |     91.9 |     18.6 | Peregrine |
| stack_8x64x128         |       3.9 |      8.8 |     45.8 |       59.1 |   1002.0 |    161.9 | Peregrine |
| diagonal_512x512       |       0.3 |      0.6 |     30.0 |       12.2 |        — |      9.1 | Peregrine |
| silu_100k              |      65.3 |     75.9 |     90.5 |      225.2 |    368.8 |     52.0 | JAX |
| softplus_100k          |     173.3 |    147.0 |    273.2 |      129.2 |    828.2 |    156.0 | TensorFlow |
| mish_100k              |     139.4 |    306.8 |    395.0 |      250.4 |   1187.3 |    244.7 | Peregrine |
| leaky_relu_100k        |       8.8 |     42.5 |     79.5 |       19.8 |        — |     34.8 | Peregrine |
| elu_100k               |      61.2 |    125.9 |    125.9 |      132.8 |   1004.9 |     77.9 | Peregrine |
| hard_tanh_100k         |       8.8 |     40.3 |     35.7 |       43.3 |        — |     40.4 | Peregrine |
| relu6_100k             |       8.8 |     39.8 |     46.9 |       55.2 |    765.0 |    103.2 | Peregrine |
| hardswish_100k         |      10.2 |     39.8 |     68.1 |      224.8 |        — |     27.4 | Peregrine |
| gelu_100k              |      97.1 |     72.1 |    141.1 |      242.6 |    881.4 |    214.3 | PyTorch |
| selu_100k              |      64.9 |    122.8 |     89.8 |      139.0 |    764.1 |     83.7 | Peregrine |
| softsign_100k          |      39.0 |    123.9 |     45.5 |       44.7 |        — |     62.2 | Peregrine |
| cross_entropy_64x10    |       2.7 |     37.8 |     23.4 |      633.9 |   3936.8 |     62.5 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.4 |     18.5 |       44.9 |   1200.4 |     12.2 | Peregrine |
| mse_loss_64x10         |       3.7 |      5.1 |     25.7 |       40.6 |    476.8 |     25.1 | Peregrine |
| huber_loss_64x10       |       0.3 |      5.1 |     40.1 |      244.0 |        — |     48.5 | Peregrine |
| smooth_l1_loss_64x10   |       0.8 |      5.2 |     34.3 |      246.8 |        — |     48.7 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.5 |     18.1 |      385.8 |        — |     56.6 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.2 |    117.5 |      240.1 |        — |     49.3 | Peregrine |
| rmsnorm_64x512         |      18.8 |     64.0 |     34.2 |      470.1 |        — |     77.0 | Peregrine |
| conv1d_1x32x128_k3     |      20.7 |     57.0 |     34.5 |      649.6 |        — |     74.0 | Peregrine |
| avgpool2d_1x16x32x32   |      25.6 |     41.3 |    276.3 |       64.2 |        — |     45.1 | Peregrine |
| groupnorm_4x64x16x16   |      22.0 |     47.2 |    232.6 |      771.0 |        — |    267.8 | Peregrine |
| rnn_seq32_128_256      |     190.1 |    270.3 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |     993.2 |    796.8 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     715.3 |    783.2 |        — |          — |        — |        — | Peregrine |
| optim_adam_64          |     809.1 |   1637.0 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     935.6 |   1267.0 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     920.5 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1302.2 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      61.4 |    262.5 |    501.9 |      124.1 |   2507.2 |    550.7 | Peregrine |
| rand_normal_100k       |     241.2 |   1008.8 |    735.5 |      332.3 |   3402.1 |    600.7 | Peregrine |
| rand_bernoulli_100k    |     122.0 |    257.5 |    490.5 |      214.5 |        — |    553.1 | Peregrine |
| rand_uniform_1M        |     611.8 |   2641.9 |   4760.4 |      433.9 |   2536.4 |   2231.9 | TensorFlow |
| rand_normal_1M         |    2411.6 |  10022.2 |   7009.7 |     2189.7 |   3514.3 |   2895.5 | TensorFlow |
| rfft_1k                |       2.0 |      4.5 |     25.9 |       43.9 |        — |     48.8 | Peregrine |
| rfft_4k                |       6.6 |     15.0 |     32.5 |       54.5 |        — |     65.2 | Peregrine |
| rfft_16k               |      29.6 |     66.5 |     74.5 |      107.0 |        — |    124.0 | Peregrine |
| fft_1k                 |       3.0 |      7.0 |     24.0 |        9.4 |        — |     17.2 | Peregrine |
| fft_4k                 |      11.9 |     26.5 |     45.1 |       17.7 |        — |     60.3 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     21.0 |       69.8 |        — |      3.8 | Peregrine |
| solve_64x64            |      11.4 |     24.2 |    102.2 |       24.7 |        — |     32.7 | Peregrine |
| inv_64x64              |      35.7 |     26.0 |     46.7 |       33.0 |        — |     43.9 | PyTorch |
| cholesky_64x64         |       5.7 |     47.0 |     21.6 |       19.6 |        — |     20.8 | Peregrine |
| svd_64x64              |     274.4 |    277.2 |    288.0 |      504.6 |        — |    302.6 | Peregrine |
| qr_64x64               |      39.8 |     82.4 |     63.7 |       85.3 |        — |     67.3 | Peregrine |
| eigh_64x64             |     381.5 |    218.6 |    225.4 |      148.9 |        — |    231.9 | TensorFlow |
| det_64x64              |      18.4 |     19.9 |        — |       23.2 |        — |     29.8 | Peregrine |
| solve_128x128          |      48.7 |     44.5 |    219.4 |       77.9 |        — |     84.2 | PyTorch |
| inv_128x128            |      94.3 |     61.0 |     85.9 |      141.5 |        — |     87.5 | PyTorch |
| cholesky_128x128       |      35.1 |     52.0 |     27.3 |       60.4 |        — |     37.8 | MLX |
| svd_128x128            |     988.2 |   1002.0 |   1010.4 |     1918.0 |        — |   1022.2 | Peregrine |
| qr_128x128             |     187.8 |    221.8 |    203.0 |      339.8 |        — |    190.8 | Peregrine |
| eigh_128x128           |    1853.5 |    712.6 |    743.2 |      736.0 |        — |    733.0 | PyTorch |
| det_128x128            |      39.9 |     52.3 |        — |       84.3 |        — |     75.0 | Peregrine |
| solve_256x256          |     188.5 |    198.9 |    760.3 |      385.2 |        — |    255.5 | Peregrine |
| inv_256x256            |     485.5 |    318.7 |    258.3 |      892.7 |        — |    576.7 | MLX |
| cholesky_256x256       |     145.3 |     91.0 |     58.3 |      290.3 |        — |    152.4 | MLX |
| svd_256x256            |    6152.4 |   6020.5 |   6018.2 |     8557.4 |        — |   6290.6 | MLX |
| qr_256x256             |    1058.2 |    995.7 |   1023.9 |     1754.3 |        — |   1027.6 | PyTorch |
| eigh_256x256           |    6149.1 |   3696.1 |   3603.9 |     4771.9 |        — |   3659.6 | MLX |
| det_256x256            |     139.4 |    209.3 |        — |      489.2 |        — |    203.3 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1803.8 |   1454.9 |        — |     2739.7 |   1314.0 |   2144.2 | tinygrad |
| matmul_bias_gelu_196x1024x4096 |    3525.7 |   2758.2 |        — |     4094.7 |   1265.1 |   3403.2 | tinygrad |
| add_layernorm_196x768  |     108.0 |    132.1 |        — |     1365.0 |   1136.9 |    376.9 | Peregrine |
| add_layernorm_196x1024 |     143.5 |    256.1 |        — |     1334.1 |   1191.9 |    284.7 | Peregrine |
| matmul_f32_196x768x3072 |     506.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14929.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1628.4 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   27335.3 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.50x** (Peregrine is faster)
- **Peregrine vs MLX: 0.36x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.27x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.35x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 107/141 ops
- PyTorch: 15/141 ops
- TensorFlow: 7/141 ops
- MLX: 5/141 ops
- JAX: 5/141 ops
- tinygrad: 2/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
