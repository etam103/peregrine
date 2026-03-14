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
| matmul_128x128         |      28.3 |      6.2 |     22.2 |       97.6 |    429.9 |     79.4 | PyTorch |
| matmul_256x256         |      74.4 |     31.7 |     44.8 |      209.5 |    425.6 |    164.6 | PyTorch |
| matmul_512x512         |     245.7 |    146.2 |    167.8 |      765.6 |    444.8 |    498.9 | PyTorch |
| matmul_1024x1024       |    1111.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |   10850.1 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     44.8 |     28.2 |       56.4 |    191.5 |     53.9 | Peregrine |
| add_500k               |     109.7 |     59.8 |     83.9 |       91.3 |    187.5 |     71.8 | PyTorch |
| add_1M                 |     121.8 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     534.0 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |    1039.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |       9.5 |     46.4 |     28.1 |       48.4 |    190.0 |     38.6 | Peregrine |
| mul_500k               |     108.5 |     65.3 |     68.8 |       78.8 |    190.6 |     70.4 | PyTorch |
| mul_1M                 |     126.3 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     530.4 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     916.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |     116.8 |     62.8 |     58.9 |       67.2 |    224.4 |     49.8 | JAX |
| exp_500k               |     204.8 |    158.1 |    220.5 |      116.3 |    223.5 |    124.4 | TensorFlow |
| exp_1M                 |     345.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1166.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2334.9 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.6 |     38.7 |     25.1 |       39.0 |    349.7 |    100.8 | Peregrine |
| relu_1M                |     105.7 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     36.4 |     18.0 |       10.6 |    651.5 |     31.9 | Peregrine |
| softmax_8x512          |       4.3 |     39.3 |     16.3 |       14.1 |    637.3 |     33.5 | Peregrine |
| mlp_fwd_64x784         |      32.6 |     27.5 |     49.6 |      260.1 |   1801.2 |    198.2 | PyTorch |
| mlp_fwd_256x784_wide   |     419.6 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     817.0 |   1283.6 |    759.5 |     8855.8 |  25308.8 |   5622.3 | MLX |
| train_step_256_wide    |    3380.6 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.6 |     41.3 |     23.9 |       45.5 |    164.8 |     32.2 | Peregrine |
| square_100k            |       8.6 |     41.1 |     21.8 |       14.3 |    179.4 |     31.0 | Peregrine |
| rsqrt_100k             |      71.5 |     42.4 |     30.4 |       55.7 |        — |    107.2 | MLX |
| floor_100k             |       8.8 |     40.4 |     23.0 |       15.8 |    431.8 |     41.1 | Peregrine |
| ceil_100k              |       8.8 |     42.1 |     22.5 |       15.7 |    370.8 |     28.8 | Peregrine |
| round_100k             |       8.8 |     42.2 |     23.0 |       43.9 |        — |     35.8 | Peregrine |
| sign_100k              |       8.6 |     40.9 |     27.1 |       48.8 |    874.8 |     37.1 | Peregrine |
| expm1_100k             |     169.5 |    110.4 |    103.6 |      252.4 |        — |     98.8 | JAX |
| log2_100k              |     111.8 |     89.0 |     97.8 |      178.4 |    165.7 |     57.8 | JAX |
| log10_100k             |     146.5 |     83.8 |    106.1 |      191.1 |        — |     57.1 | JAX |
| log1p_100k             |     115.2 |     86.6 |    127.5 |      115.4 |        — |    110.8 | PyTorch |
| erf_100k               |     110.4 |     58.5 |    100.7 |       61.2 |        — |     47.7 | JAX |
| sinh_100k              |      52.6 |    124.9 |     94.0 |      153.8 |    594.4 |    115.9 | Peregrine |
| cosh_100k              |      47.4 |    131.6 |     89.5 |      154.1 |    478.0 |     75.5 | Peregrine |
| arcsin_100k            |      53.1 |     82.8 |     94.0 |       65.0 |   3204.8 |    113.2 | Peregrine |
| arccos_100k            |     112.7 |    102.1 |    113.4 |       56.6 |        — |    207.0 | TensorFlow |
| arctan_100k            |      54.2 |    106.9 |     92.8 |       63.2 |   3374.8 |    216.4 | Peregrine |
| arcsinh_100k           |     135.0 |    164.4 |    332.9 |      162.5 |        — |    127.8 | JAX |
| maximum_100k           |      12.7 |     43.7 |     28.2 |       45.1 |    208.3 |     32.2 | Peregrine |
| minimum_100k           |      12.8 |     44.8 |     26.8 |       39.6 |    417.4 |     34.4 | Peregrine |
| power_100k             |     156.7 |    251.2 |    215.9 |      345.5 |        — |    159.0 | Peregrine |
| arctan2_100k           |      98.2 |    147.0 |    147.9 |       72.5 |        — |    325.5 | TensorFlow |
| logaddexp_100k         |     417.0 |    157.3 |    256.6 |      378.6 |        — |    161.2 | PyTorch |
| clip_100k              |       8.9 |     48.4 |     34.9 |       40.0 |    582.0 |     44.9 | Peregrine |
| where_100k             |      16.9 |     55.0 |     29.4 |       65.5 |    303.9 |     39.8 | Peregrine |
| greater_100k           |      12.8 |     56.4 |     24.1 |       50.4 |    190.3 |     36.3 | Peregrine |
| equal_100k             |      12.8 |     36.8 |     24.5 |       50.9 |    307.1 |     27.3 | Peregrine |
| sum_axis_256x512       |     114.6 |     40.3 |     22.7 |       55.1 |    217.8 |     61.1 | MLX |
| mean_axis_256x512      |     112.5 |     41.0 |     24.3 |       52.8 |    311.3 |     58.4 | MLX |
| max_axis_256x512       |     154.6 |     72.7 |     41.7 |       50.6 |    220.8 |     51.7 | MLX |
| min_axis_256x512       |     157.4 |     62.2 |     41.7 |       50.7 |    343.5 |     63.3 | MLX |
| var_256x512            |     236.0 |    402.7 |     62.4 |      227.4 |        — |     99.0 | MLX |
| prod_axis_256x512      |     151.2 |     45.5 |     26.4 |       49.0 |        — |     63.8 | MLX |
| logsumexp_256x512      |     380.8 |    224.7 |    106.8 |      327.2 |        — |    309.4 | MLX |
| cumsum_256x512         |     124.3 |     84.2 |    130.2 |      197.1 |    658.5 |    198.6 | PyTorch |
| argmax_axis_256x512    |     157.5 |     96.6 |    181.7 |       69.8 |   1296.6 |    170.7 | TensorFlow |
| sum_axis_1024x1024     |     955.8 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    1942.2 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      37.3 |     44.6 |     60.9 |       49.8 |   1961.8 |     19.3 | JAX |
| triu_256x256           |      36.0 |     42.1 |     60.9 |       53.8 |   1985.3 |     17.4 | JAX |
| repeat_64x128_2x3      |     125.1 |     52.0 |     31.1 |       71.8 |        — |     28.2 | JAX |
| pad_64x128             |      16.8 |      4.2 |     20.0 |       80.0 |     89.8 |     18.3 | PyTorch |
| stack_8x64x128         |      16.2 |      9.4 |     46.1 |       55.5 |    917.2 |    162.7 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     29.1 |       11.2 |        — |      9.6 | PyTorch |
| silu_100k              |      65.2 |     75.4 |     85.0 |      225.0 |    334.5 |     45.9 | JAX |
| softplus_100k          |     346.5 |    160.7 |    264.4 |      124.3 |    826.7 |    173.4 | TensorFlow |
| mish_100k              |     506.3 |    334.4 |    372.2 |      231.0 |   1161.2 |    215.1 | JAX |
| leaky_relu_100k        |       8.8 |     42.0 |     80.1 |       19.1 |        — |     28.4 | Peregrine |
| elu_100k               |     149.0 |    137.2 |    121.8 |      129.6 |    891.2 |     98.1 | JAX |
| hard_tanh_100k         |       8.1 |     50.7 |     35.9 |       39.8 |        — |     41.6 | Peregrine |
| relu6_100k             |       8.1 |     52.2 |     44.7 |       49.8 |    730.8 |    111.4 | Peregrine |
| hardswish_100k         |      10.2 |     43.7 |     69.2 |      208.5 |        — |     23.7 | Peregrine |
| gelu_100k              |      72.2 |     79.9 |    135.8 |      235.3 |    848.6 |    260.3 | Peregrine |
| selu_100k              |      65.0 |    129.2 |     85.2 |      136.1 |    736.1 |     97.1 | Peregrine |
| softsign_100k          |      35.6 |    127.3 |     44.0 |       47.5 |        — |     59.2 | Peregrine |
| cross_entropy_64x10    |       2.7 |     39.2 |     23.2 |      592.3 |   3325.1 |     45.2 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.2 |     19.2 |       39.0 |   1119.1 |     12.0 | Peregrine |
| mse_loss_64x10         |       3.7 |      4.8 |     22.1 |       34.9 |    444.8 |     26.6 | Peregrine |
| huber_loss_64x10       |       5.2 |      4.7 |     37.6 |      233.9 |        — |     49.8 | PyTorch |
| smooth_l1_loss_64x10   |       5.0 |      5.0 |     33.4 |      229.8 |        — |     48.0 | Peregrine |
| kl_div_loss_64x10      |       2.6 |      6.4 |     18.4 |      381.2 |        — |     64.8 | Peregrine |
| cosine_sim_loss_64x64  |      13.8 |     10.2 |    115.2 |      225.0 |        — |     71.0 | PyTorch |
| rmsnorm_64x512         |      59.5 |     67.0 |     32.7 |      439.5 |        — |     64.2 | MLX |
| conv1d_1x32x128_k3     |      19.8 |     53.9 |     28.3 |      679.6 |        — |     61.4 | Peregrine |
| avgpool2d_1x16x32x32   |      25.6 |     51.5 |    263.5 |       60.1 |        — |     49.0 | Peregrine |
| groupnorm_4x64x16x16   |      72.6 |     56.6 |    223.4 |      788.8 |        — |    285.7 | PyTorch |
| rnn_seq32_128_256      |     198.0 |    268.5 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1142.8 |    823.7 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     814.2 |    779.8 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     805.8 |   1486.2 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     964.6 |   1258.2 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     930.7 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1294.1 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     108.5 |    260.1 |    515.7 |      123.1 |   2375.5 |    565.2 | Peregrine |
| rand_normal_100k       |     767.7 |    989.1 |    746.0 |      326.1 |   3267.3 |    631.9 | TensorFlow |
| rand_bernoulli_100k    |     303.6 |    255.0 |    473.0 |      203.7 |        — |    590.7 | TensorFlow |
| rand_uniform_1M        |    1073.3 |   2662.5 |   4767.1 |      417.9 |   2401.3 |   2344.3 | TensorFlow |
| rand_normal_1M         |    7845.8 |  10070.2 |   6803.0 |     2154.2 |   3278.2 |   2972.8 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     20.4 |       40.9 |        — |     46.7 | Peregrine |
| rfft_4k                |       7.5 |     14.8 |     29.7 |       51.8 |        — |     65.0 | Peregrine |
| rfft_16k               |      30.3 |     65.3 |     77.0 |      105.0 |        — |    122.8 | Peregrine |
| fft_1k                 |       3.3 |      6.7 |     23.0 |        8.0 |        — |     38.3 | Peregrine |
| fft_4k                 |      12.2 |     26.2 |     39.3 |       16.4 |        — |     66.5 | Peregrine |
| norm_l2_1k             |       1.0 |      1.3 |     19.6 |       63.3 |        — |      3.8 | Peregrine |
| solve_64x64            |      11.7 |     24.1 |    103.3 |       23.5 |        — |     33.1 | Peregrine |
| inv_64x64              |      36.8 |     25.8 |     48.0 |       31.6 |        — |     45.6 | PyTorch |
| cholesky_64x64         |       9.0 |     46.2 |     21.5 |       18.7 |        — |     21.8 | Peregrine |
| svd_64x64              |     288.1 |    280.3 |    294.1 |      505.7 |        — |    307.8 | PyTorch |
| qr_64x64               |      41.4 |     85.5 |     55.6 |       84.2 |        — |     66.0 | Peregrine |
| eigh_64x64             |     408.4 |    219.6 |    232.4 |      148.2 |        — |    241.7 | TensorFlow |
| det_64x64              |      22.9 |     20.0 |        — |       22.1 |        — |     29.8 | PyTorch |
| solve_128x128          |      55.3 |     44.8 |    192.7 |       75.2 |        — |     85.2 | PyTorch |
| inv_128x128            |      97.4 |     61.4 |     89.2 |      137.1 |        — |     86.3 | PyTorch |
| cholesky_128x128       |      50.6 |     51.4 |     28.5 |       56.8 |        — |     36.0 | MLX |
| svd_128x128            |    1046.2 |   1000.6 |   1004.9 |     1873.2 |        — |   1031.2 | PyTorch |
| qr_128x128             |     192.8 |    222.2 |    203.0 |      332.0 |        — |    195.0 | Peregrine |
| eigh_128x128           |    1833.3 |    777.6 |    734.1 |      723.3 |        — |    740.1 | TensorFlow |
| det_128x128            |      52.1 |     54.9 |        — |       82.4 |        — |     76.0 | Peregrine |
| solve_256x256          |     192.8 |    200.2 |    750.7 |      378.6 |        — |    267.2 | Peregrine |
| inv_256x256            |     492.0 |    267.9 |    246.4 |      853.6 |        — |    339.6 | MLX |
| cholesky_256x256       |     227.2 |     86.5 |     57.4 |      285.6 |        — |    118.6 | MLX |
| svd_256x256            |    6251.3 |   6146.3 |   5744.1 |     8085.5 |        — |   5886.9 | MLX |
| qr_256x256             |    1039.0 |   1035.6 |   1039.1 |     1748.5 |        — |   1050.2 | PyTorch |
| eigh_256x256           |    6176.7 |   3502.9 |   3539.7 |     4660.9 |        — |   3681.6 | PyTorch |
| det_256x256            |     213.7 |    203.8 |        — |      434.3 |        — |    212.7 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1096.5 |    914.3 |        — |     2344.4 |   1279.7 |   2321.5 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2390.1 |   1973.1 |        — |     3694.5 |   1248.6 |   3865.0 | tinygrad |
| add_layernorm_196x768  |     108.5 |    106.3 |        — |     1234.7 |   1126.4 |    215.9 | PyTorch |
| add_layernorm_196x1024 |     308.8 |    106.4 |        — |     1254.1 |   1171.0 |    288.2 | PyTorch |
| matmul_f32_196x768x3072 |     629.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   15274.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1725.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   27067.7 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.76x** (Peregrine is faster)
- **Peregrine vs MLX: 0.61x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.44x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.08x** (Peregrine is faster)
- **Peregrine vs JAX: 0.54x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 74/141 ops
- PyTorch: 29/141 ops
- MLX: 14/141 ops
- JAX: 12/141 ops
- TensorFlow: 11/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
