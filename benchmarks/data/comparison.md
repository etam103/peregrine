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
| matmul_128x128         |      10.5 |      6.1 |     19.9 |       92.5 |    417.8 |     79.1 | PyTorch |
| matmul_256x256         |      54.5 |     31.7 |     47.8 |      198.8 |    444.0 |    149.0 | PyTorch |
| matmul_512x512         |     188.1 |    140.7 |    164.2 |      687.9 |    434.1 |    499.5 | PyTorch |
| matmul_1024x1024       |     948.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    8983.2 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     39.4 |     27.4 |       44.5 |    185.7 |     32.7 | Peregrine |
| add_500k               |      61.7 |     57.5 |     81.8 |       78.8 |    187.4 |     63.0 | PyTorch |
| add_1M                 |     126.0 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     561.7 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     985.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.8 |     40.0 |     30.1 |       43.1 |    188.7 |     31.4 | Peregrine |
| mul_500k               |      61.8 |     56.6 |     75.8 |       72.4 |    191.6 |     59.1 | PyTorch |
| mul_1M                 |     126.3 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     623.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     879.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.3 |     66.0 |     61.2 |       62.4 |    224.1 |     46.1 | JAX |
| exp_500k               |     133.1 |    139.1 |    226.1 |       96.0 |    222.9 |    122.0 | TensorFlow |
| exp_1M                 |     143.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     412.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     777.8 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.7 |     39.2 |     27.4 |       43.0 |    340.2 |     97.1 | Peregrine |
| relu_1M                |      82.5 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     25.9 |     17.3 |       10.5 |    625.2 |     30.7 | Peregrine |
| softmax_8x512          |       4.4 |     34.3 |     18.8 |       13.1 |    622.7 |     33.3 | Peregrine |
| mlp_fwd_64x784         |      33.0 |     27.8 |     51.5 |      241.9 |   1798.0 |    183.1 | PyTorch |
| mlp_fwd_256x784_wide   |     399.1 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     812.6 |   1328.8 |    799.3 |     8481.2 |  24513.3 |   5132.1 | MLX |
| train_step_256_wide    |    3296.4 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.6 |     38.0 |     24.9 |       47.9 |    169.6 |     29.6 | Peregrine |
| square_100k            |       8.6 |     39.0 |     23.7 |       13.4 |    182.5 |     28.2 | Peregrine |
| rsqrt_100k             |      21.5 |     41.1 |     36.2 |       48.2 |        — |     92.4 | Peregrine |
| floor_100k             |       8.6 |     44.3 |     23.8 |       14.9 |    423.5 |     28.0 | Peregrine |
| ceil_100k              |       8.6 |     41.2 |     23.5 |       14.9 |    365.7 |     30.3 | Peregrine |
| round_100k             |       8.6 |     40.5 |     24.6 |       44.2 |        — |     30.3 | Peregrine |
| sign_100k              |       8.6 |     39.3 |     27.6 |       42.8 |    855.4 |     36.2 | Peregrine |
| expm1_100k             |      63.2 |    109.0 |    107.7 |      144.0 |        — |     99.8 | Peregrine |
| log2_100k              |      55.6 |     86.3 |    102.0 |      147.6 |    177.3 |     56.4 | Peregrine |
| log10_100k             |      58.0 |     84.5 |    106.0 |      146.2 |        — |     56.2 | JAX |
| log1p_100k             |      75.5 |     86.6 |    127.7 |       88.4 |        — |    105.4 | Peregrine |
| erf_100k               |     100.7 |     56.6 |    100.7 |       59.7 |        — |     55.3 | JAX |
| sinh_100k              |      51.0 |    136.9 |     96.0 |      134.5 |    550.4 |    120.2 | Peregrine |
| cosh_100k              |      46.3 |    128.5 |     90.0 |      119.2 |    486.6 |     76.9 | Peregrine |
| arcsin_100k            |      52.1 |     79.7 |     95.7 |       53.9 |   3015.0 |    113.5 | Peregrine |
| arccos_100k            |      60.7 |     88.6 |    110.2 |       55.3 |        — |    203.1 | TensorFlow |
| arctan_100k            |      53.2 |     92.5 |     93.8 |       59.6 |   3094.3 |    209.1 | Peregrine |
| arcsinh_100k           |     204.8 |    147.6 |    334.0 |      131.9 |        — |    132.9 | TensorFlow |
| maximum_100k           |      12.5 |     38.5 |     27.4 |       38.4 |    189.9 |     34.0 | Peregrine |
| minimum_100k           |      12.5 |     40.5 |     28.3 |       41.1 |    375.6 |     34.4 | Peregrine |
| power_100k             |     153.8 |    244.3 |    214.2 |      277.0 |        — |    148.0 | JAX |
| arctan2_100k           |      95.1 |    137.9 |    150.7 |       73.9 |        — |    312.8 | TensorFlow |
| logaddexp_100k         |     272.8 |    154.2 |    260.5 |      355.6 |        — |    149.8 | JAX |
| clip_100k              |       8.5 |     40.2 |     37.6 |       39.9 |    542.8 |     42.0 | Peregrine |
| where_100k             |      16.5 |     50.5 |     29.2 |       64.6 |    274.2 |     32.9 | Peregrine |
| greater_100k           |      12.5 |     48.0 |     25.3 |       56.1 |    192.1 |     27.0 | Peregrine |
| equal_100k             |      12.5 |     32.8 |     24.0 |       58.9 |    287.8 |     26.6 | Peregrine |
| sum_axis_256x512       |      18.8 |     40.8 |     23.5 |       52.8 |    206.1 |     52.8 | Peregrine |
| mean_axis_256x512      |      18.8 |     42.3 |     25.1 |       50.9 |    293.4 |     49.7 | Peregrine |
| max_axis_256x512       |      13.7 |     54.5 |     41.3 |       49.4 |    205.0 |     50.9 | Peregrine |
| min_axis_256x512       |      13.7 |     54.6 |     40.1 |       47.3 |    334.4 |     47.0 | Peregrine |
| var_256x512            |      45.7 |    277.4 |     63.4 |      218.2 |        — |     78.6 | Peregrine |
| prod_axis_256x512      |      24.1 |     39.4 |     26.3 |       47.9 |        — |     55.8 | Peregrine |
| logsumexp_256x512      |      95.4 |    199.8 |    106.7 |      355.1 |        — |    275.9 | Peregrine |
| cumsum_256x512         |     117.4 |     79.5 |    132.4 |      204.1 |    614.8 |    219.4 | PyTorch |
| argmax_axis_256x512    |      51.9 |     92.8 |    173.1 |       81.1 |   1306.3 |    179.5 | Peregrine |
| sum_axis_1024x1024     |     177.6 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.8 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.7 |     37.9 |     56.5 |       57.4 |   1768.4 |     39.0 | Peregrine |
| triu_256x256           |       7.6 |     39.8 |     56.2 |       51.3 |   1807.6 |     38.0 | Peregrine |
| repeat_64x128_2x3      |       6.2 |     49.5 |     30.2 |       72.4 |        — |     27.9 | Peregrine |
| pad_64x128             |       2.5 |      4.7 |     19.8 |       81.1 |     91.2 |     18.1 | Peregrine |
| stack_8x64x128         |       3.8 |      8.7 |     47.2 |       53.4 |    924.8 |    165.8 | Peregrine |
| diagonal_512x512       |       1.0 |      0.6 |     31.3 |       11.0 |        — |     10.7 | PyTorch |
| silu_100k              |      64.0 |     68.8 |     85.0 |      218.1 |    328.3 |     51.9 | JAX |
| softplus_100k          |     180.7 |    155.0 |    260.8 |      133.3 |    825.0 |    157.3 | TensorFlow |
| mish_100k              |     286.1 |    308.3 |    392.3 |      244.3 |   1170.2 |    233.2 | JAX |
| leaky_relu_100k        |       8.7 |     40.6 |     87.0 |       18.8 |        — |     32.0 | Peregrine |
| elu_100k               |      60.0 |    132.8 |    126.6 |      132.0 |    886.1 |     87.3 | Peregrine |
| hard_tanh_100k         |       8.7 |     39.4 |     36.1 |       39.2 |        — |     42.2 | Peregrine |
| relu6_100k             |       8.7 |     41.0 |     47.5 |       51.2 |    730.6 |    115.4 | Peregrine |
| hardswish_100k         |      10.0 |     40.4 |     67.5 |      203.7 |        — |     28.0 | Peregrine |
| gelu_100k              |      95.8 |     73.2 |    135.6 |      217.7 |    852.0 |    217.2 | PyTorch |
| selu_100k              |      63.7 |    131.8 |     85.1 |      131.8 |    740.7 |     90.6 | Peregrine |
| softsign_100k          |      38.2 |    127.0 |     44.1 |       45.9 |        — |     65.4 | Peregrine |
| cross_entropy_64x10    |       2.6 |     40.6 |     22.2 |      592.1 |   3473.0 |     53.4 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.3 |     18.4 |       38.2 |   1148.9 |     12.2 | Peregrine |
| mse_loss_64x10         |       3.6 |      4.9 |     22.5 |       34.4 |    458.0 |     23.7 | Peregrine |
| huber_loss_64x10       |       5.0 |      4.8 |     36.2 |      218.3 |        — |     47.5 | PyTorch |
| smooth_l1_loss_64x10   |       4.9 |      5.0 |     36.3 |      221.7 |        — |     47.6 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.3 |     17.4 |      378.5 |        — |     63.6 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.2 |    115.7 |      218.2 |        — |     67.1 | Peregrine |
| rmsnorm_64x512         |      18.7 |     66.5 |     32.7 |      442.2 |        — |     68.1 | Peregrine |
| conv1d_1x32x128_k3     |      19.7 |     54.8 |     30.1 |      503.9 |        — |     73.9 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     41.7 |    263.9 |       59.0 |        — |     42.2 | Peregrine |
| groupnorm_4x64x16x16   |      22.9 |     54.3 |    226.4 |      775.5 |        — |    271.5 | Peregrine |
| rnn_seq32_128_256      |     198.5 |    266.9 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1014.2 |    806.5 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     801.2 |    779.4 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     805.0 |   1263.2 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     915.5 |   1145.7 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     903.4 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1292.5 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      60.2 |    257.5 |    492.0 |      150.7 |   2353.6 |    532.8 | Peregrine |
| rand_normal_100k       |     236.5 |    983.1 |    694.4 |      346.8 |   3285.5 |    614.2 | Peregrine |
| rand_bernoulli_100k    |     118.7 |    251.2 |    453.0 |      210.6 |        — |    531.6 | Peregrine |
| rand_uniform_1M        |     603.1 |   2601.0 |   4636.5 |      426.8 |   2414.1 |   2274.2 | TensorFlow |
| rand_normal_1M         |    2371.2 |   9831.5 |   6672.8 |     2066.8 |   3252.8 |   2928.3 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     20.2 |       39.1 |        — |     48.2 | Peregrine |
| rfft_4k                |       6.4 |     14.8 |     30.0 |       50.1 |        — |     65.3 | Peregrine |
| rfft_16k               |      30.2 |     65.0 |     79.1 |      100.4 |        — |    116.5 | Peregrine |
| fft_1k                 |       3.3 |      6.6 |     23.4 |        8.1 |        — |     19.2 | Peregrine |
| fft_4k                 |      12.2 |     26.4 |     44.0 |       15.9 |        — |     56.4 | Peregrine |
| norm_l2_1k             |       1.1 |      1.2 |     19.8 |       62.6 |        — |      4.0 | Peregrine |
| solve_64x64            |      11.8 |     18.3 |    103.7 |       23.3 |        — |     33.0 | Peregrine |
| inv_64x64              |      36.9 |     26.1 |     49.2 |       31.0 |        — |     36.7 | PyTorch |
| cholesky_64x64         |       6.1 |     42.2 |     21.9 |       18.4 |        — |     20.5 | Peregrine |
| svd_64x64              |     276.0 |    281.8 |    293.2 |      476.4 |        — |    308.5 | Peregrine |
| qr_64x64               |      41.1 |     77.8 |     58.8 |       82.4 |        — |     63.5 | Peregrine |
| eigh_64x64             |     378.3 |    220.2 |    235.2 |      140.5 |        — |    239.0 | TensorFlow |
| det_64x64              |      19.0 |     19.7 |        — |       21.7 |        — |     28.4 | Peregrine |
| solve_128x128          |      50.1 |     45.0 |    191.2 |       77.9 |        — |     84.3 | PyTorch |
| inv_128x128            |      93.7 |     62.3 |     90.5 |      139.3 |        — |     82.7 | PyTorch |
| cholesky_128x128       |      35.1 |     48.2 |     26.5 |       57.2 |        — |     36.0 | MLX |
| svd_128x128            |     986.4 |   1007.9 |    967.9 |     1831.8 |        — |   1019.4 | MLX |
| qr_128x128             |     189.0 |    222.7 |    195.9 |      326.9 |        — |    193.2 | Peregrine |
| eigh_128x128           |    1827.2 |    705.0 |    726.6 |      711.1 |        — |    757.1 | PyTorch |
| det_128x128            |      41.0 |     49.8 |        — |       80.6 |        — |     76.1 | Peregrine |
| solve_256x256          |     188.3 |    183.4 |    742.6 |      376.1 |        — |    270.9 | PyTorch |
| inv_256x256            |     471.6 |    307.9 |    259.0 |      849.0 |        — |    334.6 | MLX |
| cholesky_256x256       |     145.1 |     76.5 |     63.4 |      280.0 |        — |    123.0 | MLX |
| svd_256x256            |    5906.4 |   5672.1 |   5735.1 |     8080.4 |        — |   5842.6 | PyTorch |
| qr_256x256             |    1028.6 |   1025.2 |   1014.0 |     1720.1 |        — |    988.3 | JAX |
| eigh_256x256           |    5939.2 |   3511.1 |   3488.1 |     4658.2 |        — |   3580.3 | MLX |
| det_256x256            |     140.7 |    208.2 |        — |      429.2 |        — |    217.2 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1864.9 |    926.3 |        — |     2390.1 |   1285.1 |   2154.8 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3293.3 |   2011.5 |        — |     3728.0 |   1260.2 |   3463.2 | tinygrad |
| add_layernorm_196x768  |     109.0 |    109.8 |        — |     1207.7 |   1119.2 |    236.2 | Peregrine |
| add_layernorm_196x1024 |     149.7 |    109.4 |        — |     1259.8 |   1130.6 |    292.2 | PyTorch |
| matmul_f32_196x768x3072 |     618.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14786.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1650.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26080.4 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.54x** (Peregrine is faster)
- **Peregrine vs MLX: 0.39x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.30x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.37x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 98/141 ops
- PyTorch: 20/141 ops
- TensorFlow: 8/141 ops
- JAX: 8/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
