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
| matmul_128x128         |      13.4 |      6.1 |     21.1 |       50.4 |    413.2 |     78.7 | PyTorch |
| matmul_256x256         |      58.7 |     31.7 |     46.4 |      189.3 |    418.2 |    147.9 | PyTorch |
| matmul_512x512         |     219.8 |    136.4 |    166.2 |      682.6 |    420.3 |    506.0 | PyTorch |
| matmul_1024x1024       |    1041.6 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9024.6 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     39.8 |     30.5 |       47.8 |    184.7 |     33.5 | Peregrine |
| add_500k               |      49.6 |     58.7 |     79.8 |       78.8 |    185.3 |     59.9 | Peregrine |
| add_1M                 |     128.7 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     542.8 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     906.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     39.0 |     27.8 |       43.6 |    186.9 |     33.5 | Peregrine |
| mul_500k               |      63.2 |     83.2 |     82.2 |       75.7 |    186.5 |     59.4 | JAX |
| mul_1M                 |     128.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     535.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     952.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.1 |     54.6 |     56.5 |       66.7 |    221.9 |     46.5 | JAX |
| exp_500k               |     246.1 |    140.8 |    222.9 |      107.2 |    220.5 |    119.6 | TensorFlow |
| exp_1M                 |     501.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1180.9 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2114.0 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     35.5 |     24.4 |       41.9 |    332.9 |    100.7 | Peregrine |
| relu_1M                |      84.0 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     25.2 |     16.1 |       11.5 |    603.9 |     30.1 | Peregrine |
| softmax_8x512          |       4.3 |     31.9 |     19.1 |       14.2 |    609.2 |     33.1 | Peregrine |
| mlp_fwd_64x784         |      33.1 |     27.9 |     51.3 |      248.2 |   1792.4 |    179.1 | PyTorch |
| mlp_fwd_256x784_wide   |     429.9 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     801.0 |   1164.4 |    766.1 |     8625.0 |  23587.2 |   5155.9 | MLX |
| train_step_256_wide    |    3298.9 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.7 |     35.2 |     24.6 |       47.7 |    163.1 |     38.1 | Peregrine |
| square_100k            |       8.6 |     36.9 |     23.6 |       16.3 |    172.4 |     28.8 | Peregrine |
| rsqrt_100k             |      21.5 |     38.0 |     36.1 |       50.8 |        — |     82.0 | Peregrine |
| floor_100k             |       8.6 |     35.1 |     23.6 |       18.0 |    412.0 |     28.4 | Peregrine |
| ceil_100k              |       8.7 |     38.1 |     23.6 |       17.8 |    348.0 |     31.0 | Peregrine |
| round_100k             |       8.7 |     39.8 |     23.6 |       46.1 |        — |     29.6 | Peregrine |
| sign_100k              |       8.6 |     37.3 |     27.6 |       45.5 |    794.1 |     36.2 | Peregrine |
| expm1_100k             |      63.1 |    111.7 |    107.7 |      157.3 |        — |    101.3 | Peregrine |
| log2_100k              |      56.6 |     84.5 |    101.9 |      156.5 |    163.0 |     64.4 | Peregrine |
| log10_100k             |      59.0 |     81.7 |    106.5 |      153.6 |        — |     52.1 | JAX |
| log1p_100k             |      76.9 |     81.4 |    127.5 |       99.4 |        — |    106.6 | Peregrine |
| erf_100k               |     102.6 |     49.3 |    100.8 |       54.2 |        — |     36.9 | JAX |
| sinh_100k              |      52.0 |    136.1 |     93.4 |      136.9 |    520.5 |    118.8 | Peregrine |
| cosh_100k              |      47.2 |    134.8 |     89.6 |      125.7 |    468.8 |     68.9 | Peregrine |
| arcsin_100k            |      53.1 |     89.3 |     94.0 |       55.4 |   2860.6 |    105.3 | Peregrine |
| arccos_100k            |      60.7 |     89.5 |    110.3 |       54.3 |        — |    204.3 | TensorFlow |
| arctan_100k            |      53.1 |    100.1 |     93.2 |       57.8 |   3031.9 |    175.9 | Peregrine |
| arcsinh_100k           |     204.8 |    162.9 |    332.1 |      133.7 |        — |    120.8 | JAX |
| maximum_100k           |      12.5 |     35.5 |     28.3 |       44.1 |    188.5 |     31.9 | Peregrine |
| minimum_100k           |      12.5 |     37.8 |     27.3 |       41.5 |    376.2 |     31.3 | Peregrine |
| power_100k             |     156.7 |    239.0 |    210.7 |      278.3 |        — |    139.9 | JAX |
| arctan2_100k           |      95.2 |    136.8 |    144.1 |       70.4 |        — |    311.7 | TensorFlow |
| logaddexp_100k         |     272.4 |    152.9 |    260.3 |      362.2 |        — |    148.2 | JAX |
| clip_100k              |       8.1 |     37.5 |     36.2 |       42.4 |    536.0 |     40.2 | Peregrine |
| where_100k             |      14.7 |     48.8 |     34.6 |       65.2 |    275.9 |     31.2 | Peregrine |
| greater_100k           |       9.8 |     44.7 |     34.1 |       47.9 |    193.0 |     27.9 | Peregrine |
| equal_100k             |       9.8 |     23.8 |     26.1 |       50.8 |    289.6 |     26.4 | Peregrine |
| sum_axis_256x512       |      19.2 |     37.2 |     24.2 |       49.0 |    203.7 |     53.2 | Peregrine |
| mean_axis_256x512      |      19.2 |     41.9 |     25.6 |       50.9 |    292.0 |     50.2 | Peregrine |
| max_axis_256x512       |      13.9 |     48.5 |     41.5 |       50.7 |    208.4 |     45.4 | Peregrine |
| min_axis_256x512       |      13.9 |     50.2 |     42.8 |       53.9 |    322.9 |     46.0 | Peregrine |
| var_256x512            |      46.5 |    274.8 |     64.9 |      226.3 |        — |     79.0 | Peregrine |
| prod_axis_256x512      |      24.1 |     37.0 |     26.6 |       49.7 |        — |     55.3 | Peregrine |
| logsumexp_256x512      |      95.5 |    201.0 |    106.7 |      337.0 |        — |    277.5 | Peregrine |
| cumsum_256x512         |     118.8 |     78.8 |    128.3 |      190.2 |    611.6 |    208.1 | PyTorch |
| argmax_axis_256x512    |      51.7 |     92.9 |    171.0 |       69.3 |   1306.7 |    172.3 | Peregrine |
| sum_axis_1024x1024     |     174.0 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.8 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      43.9 |     32.8 |     53.7 |       55.3 |   1812.8 |     36.4 | PyTorch |
| triu_256x256           |      34.7 |     35.2 |     55.8 |       56.9 |   1788.3 |     38.3 | Peregrine |
| repeat_64x128_2x3      |       6.0 |     41.2 |     31.0 |       77.5 |        — |     28.2 | Peregrine |
| pad_64x128             |       2.5 |      4.0 |     18.1 |       83.8 |     89.8 |     18.3 | Peregrine |
| stack_8x64x128         |      14.0 |      8.8 |     45.3 |       51.1 |    908.9 |    158.6 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     28.2 |       12.7 |        — |      9.0 | PyTorch |
| silu_100k              |      64.0 |     72.8 |     84.2 |      228.8 |    329.5 |     52.1 | JAX |
| softplus_100k          |     180.7 |    146.7 |    263.1 |      129.5 |    765.2 |    154.9 | TensorFlow |
| mish_100k              |     288.8 |    307.9 |    370.6 |      244.5 |   1141.2 |    235.8 | JAX |
| leaky_relu_100k        |       8.6 |     37.2 |     78.2 |       19.5 |        — |     28.5 | Peregrine |
| elu_100k               |      61.1 |    137.2 |    115.3 |      129.4 |    870.2 |     78.0 | Peregrine |
| hard_tanh_100k         |       8.8 |     39.8 |     34.7 |       43.8 |        — |     36.3 | Peregrine |
| relu6_100k             |       8.8 |     37.3 |     44.4 |       52.2 |    722.2 |    113.1 | Peregrine |
| hardswish_100k         |      10.2 |     38.2 |     69.0 |      211.5 |        — |     26.9 | Peregrine |
| gelu_100k              |      97.1 |     77.0 |    135.4 |      249.4 |    842.2 |    207.5 | PyTorch |
| selu_100k              |      64.9 |    133.8 |     81.8 |      135.8 |    729.9 |     82.4 | Peregrine |
| softsign_100k          |      39.1 |    114.0 |     43.8 |       47.9 |        — |     56.3 | Peregrine |
| cross_entropy_64x10    |       2.6 |     36.5 |     22.9 |      618.7 |   3314.3 |     56.2 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.4 |     18.7 |       43.4 |   1103.7 |     12.2 | Peregrine |
| mse_loss_64x10         |       3.6 |      4.9 |     21.5 |       39.5 |    441.2 |     23.5 | Peregrine |
| huber_loss_64x10       |       5.1 |      4.8 |     33.1 |      237.0 |        — |     47.8 | PyTorch |
| smooth_l1_loss_64x10   |       4.9 |      5.1 |     33.6 |      233.7 |        — |     48.1 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     17.5 |      371.6 |        — |     62.1 | Peregrine |
| cosine_sim_loss_64x64  |      13.8 |     10.3 |    111.2 |      236.5 |        — |     45.1 | PyTorch |
| rmsnorm_64x512         |      58.7 |     64.0 |     33.0 |      440.1 |        — |     71.1 | MLX |
| conv1d_1x32x128_k3     |      20.5 |     48.6 |     27.6 |      509.3 |        — |     73.9 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     40.8 |    260.9 |       62.1 |        — |     45.4 | Peregrine |
| groupnorm_4x64x16x16   |      72.8 |     49.1 |    223.9 |      772.6 |        — |    264.5 | PyTorch |
| rnn_seq32_128_256      |     194.8 |    269.3 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1142.2 |    804.3 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     841.6 |    778.3 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     802.9 |   1180.0 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     934.7 |   1073.5 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     921.4 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1270.5 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     106.3 |    257.2 |    480.3 |      117.8 |   2329.9 |    534.0 | Peregrine |
| rand_normal_100k       |     239.3 |    985.9 |    685.6 |      341.7 |   3246.0 |    611.8 | Peregrine |
| rand_bernoulli_100k    |     304.8 |    250.0 |    448.8 |      220.7 |        — |    529.1 | TensorFlow |
| rand_uniform_1M        |    1064.0 |   2564.5 |   4539.0 |      422.3 |   2340.0 |   2285.9 | TensorFlow |
| rand_normal_1M         |    2410.2 |   9700.1 |   6782.6 |     2067.2 |   3208.5 |   2872.8 | TensorFlow |
| rfft_1k                |       2.1 |      4.4 |     20.4 |       43.3 |        — |     47.8 | Peregrine |
| rfft_4k                |       6.5 |     14.9 |     29.2 |       55.1 |        — |     64.2 | Peregrine |
| rfft_16k               |      29.2 |     65.1 |     82.0 |      105.5 |        — |    119.5 | Peregrine |
| fft_1k                 |       3.0 |      6.6 |     24.0 |        8.9 |        — |     16.9 | Peregrine |
| fft_4k                 |      11.9 |     26.1 |     45.3 |       17.4 |        — |     54.6 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     19.9 |       69.8 |        — |      3.8 | Peregrine |
| solve_64x64            |      12.0 |     24.5 |    101.4 |       24.5 |        — |     32.6 | Peregrine |
| inv_64x64              |      36.3 |     26.1 |     52.2 |       32.4 |        — |     37.7 | PyTorch |
| cholesky_64x64         |       9.1 |     42.3 |     20.9 |       19.4 |        — |     19.2 | Peregrine |
| svd_64x64              |     276.0 |    276.5 |    292.2 |      504.0 |        — |    304.9 | Peregrine |
| qr_64x64               |      41.2 |     85.6 |     59.6 |       85.1 |        — |     62.8 | Peregrine |
| eigh_64x64             |     385.3 |    214.6 |    232.3 |      145.2 |        — |    237.7 | TensorFlow |
| det_64x64              |      22.5 |     20.0 |        — |       23.2 |        — |     29.0 | PyTorch |
| solve_128x128          |      48.9 |     44.8 |    191.8 |       78.1 |        — |     84.7 | PyTorch |
| inv_128x128            |      92.1 |     60.8 |     88.2 |      141.4 |        — |     82.7 | PyTorch |
| cholesky_128x128       |      50.6 |     51.1 |     26.7 |       59.6 |        — |     35.5 | MLX |
| svd_128x128            |     986.5 |    984.9 |    970.2 |     1881.6 |        — |   1012.3 | MLX |
| qr_128x128             |     188.1 |    220.7 |    196.7 |      332.2 |        — |    190.3 | Peregrine |
| eigh_128x128           |    1839.6 |    706.5 |    722.1 |      723.1 |        — |    743.6 | PyTorch |
| det_128x128            |      52.2 |     49.7 |        — |       81.8 |        — |     76.8 | PyTorch |
| solve_256x256          |     188.7 |    181.2 |    738.9 |      374.4 |        — |    265.5 | PyTorch |
| inv_256x256            |     469.5 |    293.2 |    246.0 |      845.4 |        — |    332.2 | MLX |
| cholesky_256x256       |     226.1 |     75.3 |     58.5 |      286.0 |        — |    118.6 | MLX |
| svd_256x256            |    6046.1 |   5860.1 |   5779.5 |     7990.1 |        — |   5776.3 | JAX |
| qr_256x256             |    1026.7 |    987.2 |   1001.7 |     1725.4 |        — |    980.2 | JAX |
| eigh_256x256           |    5993.1 |   3468.0 |   3541.4 |     4590.5 |        — |   3554.2 | PyTorch |
| det_256x256            |     212.7 |    201.4 |        — |      441.2 |        — |    206.2 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1491.1 |    859.5 |        — |     2373.6 |   1223.0 |   2099.6 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2666.8 |   1910.9 |        — |     3676.5 |   1252.7 |   3702.5 | tinygrad |
| add_layernorm_196x768  |     106.1 |     99.9 |        — |     1215.4 |   1112.7 |    250.3 | PyTorch |
| add_layernorm_196x1024 |     139.0 |    106.3 |        — |     1301.2 |   1102.1 |    287.0 | PyTorch |
| matmul_f32_196x768x3072 |     653.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14702.4 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1458.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26822.4 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.62x** (Peregrine is faster)
- **Peregrine vs MLX: 0.45x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.33x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.06x** (Peregrine is faster)
- **Peregrine vs JAX: 0.43x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 89/141 ops
- PyTorch: 26/141 ops
- JAX: 11/141 ops
- TensorFlow: 8/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
