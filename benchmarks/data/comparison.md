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
| matmul_128x128         |      13.4 |      6.6 |     21.4 |       53.0 |    424.9 |     79.0 | PyTorch |
| matmul_256x256         |      59.0 |     30.9 |     45.6 |      137.3 |    428.3 |    171.0 | PyTorch |
| matmul_512x512         |     218.4 |    132.1 |    147.1 |      628.6 |    423.3 |    514.4 | PyTorch |
| matmul_1024x1024       |    1051.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9849.4 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.8 |     29.2 |     28.7 |       53.4 |    190.5 |     38.1 | Peregrine |
| add_500k               |     111.4 |     61.5 |     81.9 |       87.0 |    188.5 |     65.1 | PyTorch |
| add_1M                 |     130.8 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     514.1 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     971.0 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     35.2 |     28.4 |       47.0 |    189.1 |     32.2 | Peregrine |
| mul_500k               |      91.2 |     60.8 |     81.6 |       73.9 |    191.0 |     60.9 | PyTorch |
| mul_1M                 |     137.3 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     626.1 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     870.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |     105.5 |     58.4 |     61.0 |       69.0 |    225.7 |     46.5 | JAX |
| exp_500k               |     196.0 |    136.9 |    226.3 |      109.9 |    225.5 |    118.3 | TensorFlow |
| exp_1M                 |     301.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1117.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2128.4 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     30.0 |     23.9 |       32.4 |    337.0 |    100.0 | Peregrine |
| relu_1M                |     102.0 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     27.1 |     16.2 |       12.2 |    638.8 |     30.8 | Peregrine |
| softmax_8x512          |       4.2 |     28.6 |     18.1 |       14.4 |    635.0 |     33.6 | Peregrine |
| mlp_fwd_64x784         |      34.4 |     26.7 |     45.3 |      249.0 |   1812.1 |    196.2 | PyTorch |
| mlp_fwd_256x784_wide   |     421.2 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     808.9 |   1316.8 |    778.6 |     8907.3 |  26141.4 |   5078.6 | MLX |
| train_step_256_wide    |    3317.0 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     28.2 |     24.6 |       50.8 |    171.1 |     29.5 | Peregrine |
| square_100k            |       8.8 |     33.9 |     23.0 |       15.3 |    183.7 |     31.4 | Peregrine |
| rsqrt_100k             |      79.3 |     33.4 |     30.6 |       50.4 |        — |     82.2 | MLX |
| floor_100k             |      46.6 |     33.9 |     23.7 |       16.2 |    437.5 |     30.1 | TensorFlow |
| ceil_100k              |      46.6 |     34.9 |     23.3 |       16.2 |    383.4 |     36.5 | TensorFlow |
| round_100k             |      46.6 |     36.1 |     23.5 |       42.5 |        — |     31.2 | MLX |
| sign_100k              |      54.4 |     33.0 |     31.6 |       45.6 |    845.6 |     35.5 | MLX |
| expm1_100k             |     192.9 |     96.7 |    109.5 |      154.8 |        — |     98.9 | PyTorch |
| log2_100k              |     142.9 |     66.7 |    103.2 |      154.1 |    171.8 |     56.6 | JAX |
| log10_100k             |     115.9 |     64.2 |    115.5 |      146.2 |        — |     65.3 | PyTorch |
| log1p_100k             |     135.8 |     61.6 |    129.4 |       96.4 |        — |    105.5 | PyTorch |
| erf_100k               |     148.6 |     37.2 |    103.9 |       61.9 |        — |     47.0 | PyTorch |
| sinh_100k              |      51.1 |    105.4 |     95.9 |      150.7 |    553.0 |    105.9 | Peregrine |
| cosh_100k              |      46.4 |    104.1 |     89.8 |      139.7 |    487.7 |     71.1 | Peregrine |
| arcsin_100k            |      54.4 |     55.7 |     93.7 |       58.0 |   3360.4 |    111.6 | Peregrine |
| arccos_100k            |     108.5 |     69.3 |    113.6 |       60.3 |        — |    205.2 | TensorFlow |
| arctan_100k            |      54.5 |     75.3 |     93.9 |       62.4 |   3663.2 |    215.7 | Peregrine |
| arcsinh_100k           |     150.8 |    130.6 |    336.8 |      165.3 |        — |    119.8 | JAX |
| maximum_100k           |      12.5 |     31.9 |     24.3 |       40.3 |    213.7 |     31.9 | Peregrine |
| minimum_100k           |      12.5 |     29.8 |     23.0 |       47.5 |    436.3 |     32.6 | Peregrine |
| power_100k             |     392.9 |    214.4 |    213.9 |      328.6 |        — |    148.6 | JAX |
| arctan2_100k           |    1122.1 |    117.2 |    147.4 |       61.4 |        — |    316.7 | TensorFlow |
| logaddexp_100k         |     414.6 |    145.7 |    260.6 |      391.7 |        — |    152.0 | PyTorch |
| clip_100k              |       8.1 |     40.1 |     35.5 |       46.1 |    624.9 |     36.1 | Peregrine |
| where_100k             |      95.0 |     35.3 |     28.1 |       69.9 |    319.7 |     29.5 | MLX |
| greater_100k           |      71.3 |     39.5 |     21.1 |       51.9 |    211.9 |     26.8 | MLX |
| equal_100k             |      71.4 |     32.8 |     21.7 |       59.3 |    339.8 |     31.0 | MLX |
| sum_axis_256x512       |     114.7 |     40.8 |     18.6 |       54.9 |    263.4 |     51.2 | MLX |
| mean_axis_256x512      |     114.9 |     44.8 |     21.1 |       53.8 |    329.3 |     48.3 | MLX |
| max_axis_256x512       |     154.3 |     47.8 |     37.5 |       52.5 |    232.8 |     48.6 | MLX |
| min_axis_256x512       |     157.7 |     51.1 |     37.4 |       49.0 |    375.7 |     49.9 | MLX |
| var_256x512            |     246.5 |    347.6 |     55.4 |      212.5 |        — |     81.2 | MLX |
| prod_axis_256x512      |     151.1 |     49.6 |     20.9 |       52.2 |        — |     50.6 | MLX |
| logsumexp_256x512      |     386.8 |    185.6 |    118.2 |      319.4 |        — |    282.6 | MLX |
| cumsum_256x512         |     123.2 |     65.1 |    132.6 |      180.5 |    724.3 |    215.8 | PyTorch |
| argmax_axis_256x512    |     154.6 |     78.8 |    177.7 |       53.9 |   1564.2 |    173.3 | TensorFlow |
| sum_axis_1024x1024     |     944.9 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    1933.8 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      34.7 |     48.1 |     58.1 |       47.3 |   2141.9 |     37.4 | Peregrine |
| triu_256x256           |      34.7 |     35.2 |     52.4 |       49.5 |   2158.9 |     39.8 | Peregrine |
| repeat_64x128_2x3      |     127.5 |     43.2 |     28.6 |       75.3 |        — |     30.0 | MLX |
| pad_64x128             |      16.9 |      4.7 |     17.6 |       86.6 |     96.0 |     18.8 | PyTorch |
| stack_8x64x128         |      16.2 |     10.3 |     42.1 |       56.8 |   1080.1 |    161.8 | PyTorch |
| diagonal_512x512       |       0.8 |      0.7 |     25.2 |       12.6 |        — |      9.0 | PyTorch |
| silu_100k              |      64.0 |     51.0 |     85.3 |      208.3 |    386.0 |     53.8 | PyTorch |
| softplus_100k          |     323.4 |    144.1 |    264.2 |      120.4 |    927.6 |    164.2 | TensorFlow |
| mish_100k              |     560.0 |    331.2 |    379.9 |      240.2 |   1355.0 |    241.8 | TensorFlow |
| leaky_relu_100k        |       8.0 |     48.7 |     79.4 |       19.7 |        — |     38.1 | Peregrine |
| elu_100k               |     179.4 |    120.0 |    121.0 |      143.1 |   1012.5 |     81.2 | JAX |
| hard_tanh_100k         |      50.6 |     33.8 |     34.8 |       43.0 |        — |     44.9 | PyTorch |
| relu6_100k             |      50.5 |     35.5 |     47.2 |       51.9 |    732.4 |    114.3 | PyTorch |
| hardswish_100k         |      83.5 |     43.6 |     68.3 |      219.8 |        — |     28.6 | JAX |
| gelu_100k              |      77.9 |     68.1 |    135.6 |      246.1 |    887.5 |    214.7 | PyTorch |
| selu_100k              |     169.8 |    126.4 |     88.2 |      125.5 |    742.3 |     94.3 | MLX |
| softsign_100k          |      34.9 |    156.5 |     46.9 |       50.1 |        — |     71.0 | Peregrine |
| cross_entropy_64x10    |       2.6 |     43.7 |     25.0 |      673.5 |   3482.9 |     53.6 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.8 |     19.3 |       45.5 |   1150.8 |     12.5 | Peregrine |
| mse_loss_64x10         |       3.9 |      5.3 |     21.9 |       40.7 |    471.1 |     23.5 | Peregrine |
| huber_loss_64x10       |       5.1 |      6.0 |     33.3 |      254.0 |        — |     47.6 | Peregrine |
| smooth_l1_loss_64x10   |       5.0 |      5.1 |     36.5 |      247.2 |        — |     47.7 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.7 |     18.3 |      397.4 |        — |     58.7 | Peregrine |
| cosine_sim_loss_64x64  |      13.6 |     12.2 |    111.9 |      239.2 |        — |     73.1 | PyTorch |
| rmsnorm_64x512         |      57.8 |     66.3 |     29.5 |      448.6 |        — |     72.2 | MLX |
| conv1d_1x32x128_k3     |      19.7 |     55.5 |     26.2 |      571.7 |        — |     75.5 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     39.6 |    275.4 |       66.2 |        — |     41.8 | Peregrine |
| groupnorm_4x64x16x16   |      72.6 |     44.5 |    225.9 |      807.6 |        — |    275.8 | PyTorch |
| rnn_seq32_128_256      |     197.1 |    381.1 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1143.6 |   1106.4 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     876.6 |   1052.8 |        — |          — |        — |        — | Peregrine |
| optim_adam_64          |     802.3 |   1628.5 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     951.1 |   1384.0 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     924.4 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1271.5 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     106.4 |    289.6 |    490.1 |      126.6 |   2529.6 |    540.0 | Peregrine |
| rand_normal_100k       |     773.3 |   1080.4 |    704.3 |      346.0 |   3472.9 |    615.6 | TensorFlow |
| rand_bernoulli_100k    |     303.9 |    269.9 |    454.1 |      220.8 |        — |    533.8 | TensorFlow |
| rand_uniform_1M        |    1073.7 |   2847.2 |   4628.2 |      432.9 |   2464.0 |   2332.3 | TensorFlow |
| rand_normal_1M         |    7895.5 |  10030.0 |   6697.0 |     2110.3 |   3591.0 |   2894.8 | TensorFlow |
| rfft_1k                |       2.1 |      4.6 |     21.6 |       43.5 |        — |     49.8 | Peregrine |
| rfft_4k                |       6.5 |     15.2 |     32.6 |       55.1 |        — |     63.2 | Peregrine |
| rfft_16k               |      30.5 |     67.2 |     79.2 |      106.9 |        — |    117.2 | Peregrine |
| fft_1k                 |       3.3 |      7.0 |     24.1 |        8.7 |        — |     25.1 | Peregrine |
| fft_4k                 |      12.1 |     27.0 |     40.2 |       17.4 |        — |     56.4 | Peregrine |
| norm_l2_1k             |       1.1 |      1.4 |     16.3 |       69.2 |        — |      3.9 | Peregrine |
| solve_64x64            |      12.0 |     18.0 |     95.5 |       24.8 |        — |     32.7 | Peregrine |
| inv_64x64              |      37.5 |     25.6 |     52.6 |       32.4 |        — |     40.0 | PyTorch |
| cholesky_64x64         |       9.5 |     39.2 |     22.5 |       19.8 |        — |     20.7 | Peregrine |
| svd_64x64              |     283.0 |    286.8 |    296.6 |      509.9 |        — |    305.2 | Peregrine |
| qr_64x64               |      41.3 |     77.8 |     61.2 |       86.8 |        — |     63.9 | Peregrine |
| eigh_64x64             |     381.4 |    215.1 |    235.6 |      149.5 |        — |    237.9 | TensorFlow |
| det_64x64              |      14.1 |     19.4 |        — |       24.4 |        — |     28.4 | Peregrine |
| solve_128x128          |      51.0 |     43.7 |    191.7 |       79.8 |        — |     85.0 | PyTorch |
| inv_128x128            |     103.7 |     58.6 |     97.4 |      148.9 |        — |     83.9 | PyTorch |
| cholesky_128x128       |      51.1 |     57.3 |     27.2 |       58.7 |        — |     35.8 | MLX |
| svd_128x128            |    1050.6 |   1001.2 |    969.0 |     1924.5 |        — |   1011.5 | MLX |
| qr_128x128             |     189.5 |    232.0 |    194.0 |      328.7 |        — |    193.8 | Peregrine |
| eigh_128x128           |    1846.0 |    723.2 |    773.8 |      718.4 |        — |    789.1 | TensorFlow |
| det_128x128            |      52.3 |     49.7 |        — |       82.2 |        — |     73.7 | PyTorch |
| solve_256x256          |     188.5 |    197.2 |    978.8 |      385.5 |        — |    275.4 | Peregrine |
| inv_256x256            |     465.2 |    302.6 |    292.1 |      847.7 |        — |    354.9 | MLX |
| cholesky_256x256       |     226.1 |     91.0 |     59.3 |      301.6 |        — |    135.2 | MLX |
| svd_256x256            |    5837.6 |   6330.7 |   5704.2 |     8373.1 |        — |   5898.5 | MLX |
| qr_256x256             |    1013.6 |   1021.7 |   1027.4 |     1708.8 |        — |   1133.1 | Peregrine |
| eigh_256x256           |    6063.1 |   3516.1 |   3476.6 |     4656.3 |        — |   3664.6 | MLX |
| det_256x256            |     212.0 |    206.5 |        — |      429.8 |        — |    206.8 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1109.4 |    909.6 |        — |     2435.3 |   1332.6 |   2177.8 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2138.2 |   2087.9 |        — |     3794.5 |   1262.4 |   4413.6 | tinygrad |
| add_layernorm_196x768  |     109.9 |    102.6 |        — |     1277.7 |   1158.2 |    235.9 | PyTorch |
| add_layernorm_196x1024 |     148.3 |    104.7 |        — |     1370.2 |   1154.9 |    293.7 | PyTorch |
| matmul_f32_196x768x3072 |     518.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14621.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1596.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26038.4 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.98x** (Peregrine is faster)
- **Peregrine vs MLX: 0.75x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.52x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.09x** (Peregrine is faster)
- **Peregrine vs JAX: 0.66x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 67/141 ops
- PyTorch: 30/141 ops
- MLX: 23/141 ops
- TensorFlow: 14/141 ops
- JAX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
