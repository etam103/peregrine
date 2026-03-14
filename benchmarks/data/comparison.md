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
| matmul_128x128         |      13.3 |      5.8 |     20.2 |       96.8 |    409.7 |     78.8 | PyTorch |
| matmul_256x256         |      58.9 |     29.9 |     45.7 |      193.6 |    417.7 |    146.9 | PyTorch |
| matmul_512x512         |     211.8 |    142.6 |    176.4 |      665.1 |    425.1 |    513.5 | PyTorch |
| matmul_1024x1024       |    1029.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9089.9 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.8 |     40.8 |     29.8 |       50.8 |    186.8 |     35.3 | Peregrine |
| add_500k               |      63.1 |     57.5 |     82.7 |       84.4 |    185.8 |     59.7 | PyTorch |
| add_1M                 |     130.6 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     510.4 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     868.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     41.3 |     29.7 |       45.4 |    189.5 |     29.2 | Peregrine |
| mul_500k               |      63.0 |     56.9 |     80.9 |       74.1 |    187.1 |     61.2 | PyTorch |
| mul_1M                 |     129.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     531.7 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     949.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.2 |     64.1 |     63.0 |       65.2 |    219.6 |     46.4 | JAX |
| exp_500k               |     187.7 |    139.5 |    229.7 |      107.0 |    217.8 |    117.0 | TensorFlow |
| exp_1M                 |     283.0 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1269.0 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2361.9 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     38.8 |     25.2 |       43.3 |    331.6 |    100.2 | Peregrine |
| relu_1M                |      83.8 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     36.0 |     16.6 |       11.6 |    610.3 |     31.4 | Peregrine |
| softmax_8x512          |       4.4 |     33.7 |     19.1 |       14.4 |    615.9 |     33.8 | Peregrine |
| mlp_fwd_64x784         |      33.0 |     27.9 |     52.5 |      242.2 |   1793.3 |    182.2 | PyTorch |
| mlp_fwd_256x784_wide   |     435.0 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     851.3 |   1261.2 |    791.9 |     8551.2 |  24179.7 |   5019.4 | MLX |
| train_step_256_wide    |    3324.3 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     38.4 |     25.4 |       47.9 |    163.5 |     29.5 | Peregrine |
| square_100k            |       8.8 |     37.0 |     24.2 |       14.3 |    176.8 |     29.4 | Peregrine |
| rsqrt_100k             |      21.9 |     41.6 |     32.5 |       52.0 |        — |     92.9 | Peregrine |
| floor_100k             |       8.8 |     43.1 |     24.0 |       15.8 |    405.4 |     28.5 | Peregrine |
| ceil_100k              |       8.8 |     39.4 |     24.0 |       15.7 |    351.6 |     28.4 | Peregrine |
| round_100k             |       8.8 |     41.0 |     24.2 |       43.0 |        — |     28.8 | Peregrine |
| sign_100k              |       8.8 |     38.2 |     28.1 |       50.3 |    801.6 |     36.5 | Peregrine |
| expm1_100k             |      64.3 |    109.2 |    119.0 |      152.5 |        — |     98.8 | Peregrine |
| log2_100k              |      56.6 |     83.9 |    112.5 |      156.9 |    166.8 |     56.4 | JAX |
| log10_100k             |      59.0 |     82.8 |    117.1 |      148.9 |        — |     55.9 | JAX |
| log1p_100k             |      76.9 |     82.8 |    131.7 |       94.4 |        — |    104.8 | Peregrine |
| erf_100k               |     102.6 |     57.2 |    103.8 |       57.8 |        — |     48.8 | JAX |
| sinh_100k              |      52.0 |    134.8 |     96.5 |      138.4 |    524.0 |    112.9 | Peregrine |
| cosh_100k              |      47.2 |    128.7 |     92.5 |      125.3 |    466.2 |     68.8 | Peregrine |
| arcsin_100k            |      53.1 |     80.4 |     97.2 |       54.4 |   2856.8 |    110.9 | Peregrine |
| arccos_100k            |      61.9 |     86.0 |    114.0 |       53.5 |        — |    202.7 | TensorFlow |
| arctan_100k            |      54.1 |     93.6 |     96.2 |       58.9 |   3009.5 |    215.7 | Peregrine |
| arcsinh_100k           |     208.7 |    159.0 |    343.4 |      136.2 |        — |    115.6 | JAX |
| maximum_100k           |      12.7 |     40.1 |     27.8 |       39.6 |    190.7 |     32.4 | Peregrine |
| minimum_100k           |      12.7 |     40.9 |     27.4 |       43.4 |    372.8 |     26.3 | Peregrine |
| power_100k             |     156.7 |    244.2 |    217.2 |      271.2 |        — |    139.8 | JAX |
| arctan2_100k           |      96.9 |    143.7 |    149.0 |       74.1 |        — |    314.7 | TensorFlow |
| logaddexp_100k         |     277.6 |    154.0 |    263.8 |      347.6 |        — |    143.6 | JAX |
| clip_100k              |       8.8 |     41.2 |     35.6 |       42.1 |    534.3 |     35.6 | Peregrine |
| where_100k             |      16.7 |     49.8 |     28.8 |       65.8 |    278.0 |     32.9 | Peregrine |
| greater_100k           |      12.7 |     47.2 |     24.5 |       46.6 |    191.0 |     28.7 | Peregrine |
| equal_100k             |      12.7 |     23.9 |     24.5 |       55.6 |    284.6 |     26.7 | Peregrine |
| sum_axis_256x512       |      19.2 |     39.4 |     23.1 |       50.2 |    205.8 |     53.3 | Peregrine |
| mean_axis_256x512      |      19.2 |     43.5 |     24.5 |       51.4 |    295.6 |     52.9 | Peregrine |
| max_axis_256x512       |      13.9 |     53.2 |     43.0 |       50.8 |    202.2 |     50.5 | Peregrine |
| min_axis_256x512       |      13.9 |     53.4 |     41.0 |       49.0 |    324.4 |     49.3 | Peregrine |
| var_256x512            |      46.5 |    275.7 |     60.4 |      211.5 |        — |     80.0 | Peregrine |
| prod_axis_256x512      |      24.5 |     38.0 |     27.2 |       50.2 |        — |     55.5 | Peregrine |
| logsumexp_256x512      |      97.3 |    195.4 |    117.7 |      338.4 |        — |    274.6 | Peregrine |
| cumsum_256x512         |     121.0 |     75.8 |    137.1 |      176.0 |    616.7 |    208.0 | PyTorch |
| argmax_axis_256x512    |      52.5 |     94.5 |    185.7 |       69.2 |   1275.0 |    169.0 | Peregrine |
| sum_axis_1024x1024     |     177.3 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     435.7 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      34.7 |     41.9 |     55.1 |       51.8 |   1821.4 |     38.1 | Peregrine |
| triu_256x256           |      34.6 |     41.3 |     57.6 |       52.9 |   1789.2 |     36.4 | Peregrine |
| repeat_64x128_2x3      |     125.1 |     47.4 |     31.5 |       75.8 |        — |     28.0 | JAX |
| pad_64x128             |      16.8 |      4.4 |     18.9 |       82.2 |     91.0 |     18.3 | PyTorch |
| stack_8x64x128         |      15.5 |      8.7 |     56.8 |       59.2 |    916.4 |    159.5 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     29.1 |       12.6 |        — |      9.2 | PyTorch |
| silu_100k              |      65.2 |     75.6 |     85.2 |      231.9 |    325.9 |     52.1 | JAX |
| softplus_100k          |     343.9 |    152.7 |    286.4 |      121.5 |    775.2 |    156.0 | TensorFlow |
| mish_100k              |     504.4 |    310.0 |    403.4 |      233.3 |   1133.7 |    233.8 | TensorFlow |
| leaky_relu_100k        |       8.8 |     40.1 |     87.0 |       19.3 |        — |     33.2 | Peregrine |
| elu_100k               |      60.6 |    136.5 |    130.3 |      130.4 |    860.1 |     80.4 | Peregrine |
| hard_tanh_100k         |       9.2 |     41.6 |     39.6 |       41.9 |        — |     45.6 | Peregrine |
| relu6_100k             |       9.2 |     40.4 |     53.0 |       53.6 |    730.3 |    110.8 | Peregrine |
| hardswish_100k         |      10.2 |     40.3 |     75.2 |      210.3 |        — |     28.3 | Peregrine |
| gelu_100k              |      95.3 |     72.9 |    148.7 |      236.3 |    849.5 |    210.0 | PyTorch |
| selu_100k              |      63.7 |    138.7 |     93.6 |      124.4 |    734.1 |     81.8 | Peregrine |
| softsign_100k          |      34.8 |    120.0 |     49.4 |       45.1 |        — |     56.8 | Peregrine |
| cross_entropy_64x10    |       2.6 |     40.2 |     27.9 |      603.9 |   3324.7 |     55.1 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.5 |     21.8 |       42.7 |   1115.7 |     12.2 | Peregrine |
| mse_loss_64x10         |       4.0 |      5.0 |     25.8 |       38.7 |    440.6 |     23.5 | Peregrine |
| huber_loss_64x10       |       5.5 |      5.0 |     47.9 |      234.2 |        — |     47.8 | PyTorch |
| smooth_l1_loss_64x10   |       5.3 |      5.2 |     45.1 |      233.0 |        — |     48.2 | PyTorch |
| kl_div_loss_64x10      |       2.5 |      6.3 |     20.6 |      371.1 |        — |     60.3 | Peregrine |
| cosine_sim_loss_64x64  |      13.6 |     10.6 |    124.5 |      238.7 |        — |     70.6 | PyTorch |
| rmsnorm_64x512         |      57.6 |     65.2 |     44.3 |      437.6 |        — |     71.5 | MLX |
| conv1d_1x32x128_k3     |      20.5 |     50.7 |     31.6 |      496.1 |        — |     73.4 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     47.1 |    283.6 |       65.3 |        — |     41.5 | Peregrine |
| groupnorm_4x64x16x16   |      72.8 |     54.1 |    238.4 |      751.7 |        — |    265.4 | PyTorch |
| rnn_seq32_128_256      |     198.7 |    275.6 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1034.2 |    824.6 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     820.0 |    796.7 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     808.4 |   1264.2 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     934.5 |   1122.2 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     922.4 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1296.4 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     108.3 |    257.5 |    502.2 |      117.2 |   2411.5 |    533.2 | Peregrine |
| rand_normal_100k       |     772.1 |    973.3 |    708.9 |      329.4 |   3337.5 |    612.5 | TensorFlow |
| rand_bernoulli_100k    |     309.2 |    250.3 |    463.7 |      204.6 |        — |    537.9 | TensorFlow |
| rand_uniform_1M        |    1084.7 |   2571.6 |   4679.6 |      410.0 |   2387.5 |   2244.8 | TensorFlow |
| rand_normal_1M         |    7698.2 |   9756.7 |   6590.9 |     2049.3 |   3287.6 |   2871.7 | TensorFlow |
| rfft_1k                |       2.1 |      4.7 |     23.3 |       43.7 |        — |     50.4 | Peregrine |
| rfft_4k                |       6.6 |     15.7 |     35.4 |       54.6 |        — |     63.8 | Peregrine |
| rfft_16k               |      29.5 |     69.3 |     75.8 |      106.4 |        — |    116.6 | Peregrine |
| fft_1k                 |       3.1 |      6.6 |     22.2 |        9.0 |        — |     43.7 | Peregrine |
| fft_4k                 |      11.9 |     26.4 |     39.0 |       17.6 |        — |     55.4 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     19.8 |       69.8 |        — |      4.1 | Peregrine |
| solve_64x64            |      11.8 |     18.6 |     95.1 |       24.9 |        — |     32.3 | Peregrine |
| inv_64x64              |      36.2 |     26.2 |     46.8 |       33.1 |        — |     37.0 | PyTorch |
| cholesky_64x64         |       9.0 |     40.6 |     21.5 |       19.9 |        — |     20.6 | Peregrine |
| svd_64x64              |     275.4 |    277.5 |    284.4 |      482.6 |        — |    299.6 | Peregrine |
| qr_64x64               |      40.1 |     82.2 |     55.2 |       85.5 |        — |     64.0 | Peregrine |
| eigh_64x64             |     384.0 |    215.1 |    230.5 |      147.6 |        — |    239.0 | TensorFlow |
| det_64x64              |      23.2 |     20.0 |        — |       23.5 |        — |     28.5 | PyTorch |
| solve_128x128          |      50.0 |     45.1 |    184.9 |       78.0 |        — |     84.8 | PyTorch |
| inv_128x128            |      92.6 |     62.0 |     87.5 |      141.2 |        — |     88.4 | PyTorch |
| cholesky_128x128       |      50.5 |     48.3 |     28.3 |       59.4 |        — |     36.1 | MLX |
| svd_128x128            |     986.1 |    988.9 |    998.6 |     1852.2 |        — |   1015.0 | Peregrine |
| qr_128x128             |     188.2 |    222.9 |    192.3 |      332.5 |        — |    189.8 | Peregrine |
| eigh_128x128           |    1866.8 |    700.5 |    723.3 |      725.2 |        — |    744.5 | PyTorch |
| det_128x128            |      50.7 |     49.8 |        — |       83.7 |        — |     77.8 | PyTorch |
| solve_256x256          |     188.5 |    179.7 |    726.6 |      378.9 |        — |    287.9 | PyTorch |
| inv_256x256            |     472.0 |    297.9 |    252.4 |      848.2 |        — |    338.8 | MLX |
| cholesky_256x256       |     226.2 |     75.2 |     56.6 |      286.7 |        — |    122.8 | MLX |
| svd_256x256            |    6021.6 |   5795.0 |   5563.4 |     8057.2 |        — |   5826.4 | MLX |
| qr_256x256             |    1020.1 |    983.0 |    983.0 |     1718.0 |        — |    991.4 | PyTorch |
| eigh_256x256           |    5982.6 |   3445.7 |   3443.1 |     4638.4 |        — |   3447.9 | MLX |
| det_256x256            |     211.9 |    206.4 |        — |      441.1 |        — |    211.8 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1463.8 |    811.5 |        — |     2368.5 |   1219.8 |   2256.7 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2608.4 |   2057.4 |        — |     3677.1 |   1253.9 |   3376.6 | tinygrad |
| add_layernorm_196x768  |     104.9 |    109.0 |        — |     1207.1 |   1119.7 |    230.6 | Peregrine |
| add_layernorm_196x1024 |     136.9 |    117.2 |        — |     1279.1 |   1124.6 |    289.5 | PyTorch |
| matmul_f32_196x768x3072 |     500.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14617.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1438.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26178.4 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.66x** (Peregrine is faster)
- **Peregrine vs MLX: 0.47x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.36x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.06x** (Peregrine is faster)
- **Peregrine vs JAX: 0.45x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 86/141 ops
- PyTorch: 28/141 ops
- TensorFlow: 10/141 ops
- JAX: 9/141 ops
- MLX: 7/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
