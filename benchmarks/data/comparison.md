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
| matmul_128x128         |      13.3 |      5.8 |     21.3 |       94.2 |    421.9 |     78.5 | PyTorch |
| matmul_256x256         |      58.9 |     30.1 |     46.6 |      194.5 |    424.2 |    161.7 | PyTorch |
| matmul_512x512         |     209.8 |    141.9 |    178.2 |      673.2 |    429.8 |    511.9 | PyTorch |
| matmul_1024x1024       |    1057.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9045.1 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     40.8 |     30.4 |       47.1 |    184.2 |     34.5 | Peregrine |
| add_500k               |      62.7 |     56.9 |     71.1 |       76.5 |    186.5 |     61.5 | PyTorch |
| add_1M                 |     129.3 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     501.9 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     847.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.7 |     43.2 |     33.8 |       41.1 |    187.8 |     31.7 | Peregrine |
| mul_500k               |      62.8 |     56.4 |     76.0 |       73.2 |    188.7 |     59.9 | PyTorch |
| mul_1M                 |     133.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     558.7 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |    1004.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.2 |     64.1 |     63.1 |       61.6 |    217.4 |     44.5 | JAX |
| exp_500k               |     172.6 |    138.8 |    232.1 |      103.6 |    221.2 |    116.0 | TensorFlow |
| exp_1M                 |     282.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1238.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2361.0 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     39.7 |     28.3 |       37.0 |    339.5 |     99.5 | Peregrine |
| relu_1M                |      84.1 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     34.1 |     18.2 |       10.4 |    613.2 |     31.2 | Peregrine |
| softmax_8x512          |       4.3 |     41.1 |     20.7 |       13.1 |    617.0 |     33.5 | Peregrine |
| mlp_fwd_64x784         |      33.4 |     27.6 |     53.3 |      233.8 |   1804.4 |    184.3 | PyTorch |
| mlp_fwd_256x784_wide   |     422.5 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     812.8 |   1291.1 |    854.3 |     8266.7 |  22824.0 |   5081.6 | Peregrine |
| train_step_256_wide    |    3371.9 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     39.0 |     25.9 |       47.6 |    163.0 |     30.6 | Peregrine |
| square_100k            |       8.8 |     38.2 |     23.8 |       13.2 |    174.9 |     32.3 | Peregrine |
| rsqrt_100k             |      21.9 |     44.0 |     32.1 |       51.5 |        — |     92.3 | Peregrine |
| floor_100k             |       8.8 |     39.2 |     25.1 |       14.8 |    418.5 |     28.3 | Peregrine |
| ceil_100k              |       8.8 |     41.2 |     26.0 |       14.7 |    350.0 |     27.8 | Peregrine |
| round_100k             |       8.8 |     43.1 |     24.1 |       41.9 |        — |     30.2 | Peregrine |
| sign_100k              |       8.8 |     39.1 |     28.0 |       43.4 |    789.2 |     35.0 | Peregrine |
| expm1_100k             |      64.3 |    110.6 |    105.6 |      145.5 |        — |     94.1 | Peregrine |
| log2_100k              |      56.7 |     85.8 |    100.5 |      146.7 |    166.2 |     56.0 | JAX |
| log10_100k             |      59.1 |     86.4 |    115.6 |      149.0 |        — |     56.1 | JAX |
| log1p_100k             |      76.9 |     81.4 |    135.0 |       92.1 |        — |    116.5 | Peregrine |
| erf_100k               |     102.5 |     62.9 |    103.8 |       55.8 |        — |     54.5 | JAX |
| sinh_100k              |      52.0 |    130.7 |     93.8 |      130.9 |    524.9 |    113.1 | Peregrine |
| cosh_100k              |      47.2 |    131.6 |     89.7 |      130.5 |    464.9 |     68.8 | Peregrine |
| arcsin_100k            |      53.0 |     77.2 |     94.8 |       53.1 |   2894.2 |    111.4 | Peregrine |
| arccos_100k            |      61.8 |     88.3 |    112.4 |       53.6 |        — |    208.2 | TensorFlow |
| arctan_100k            |      54.1 |     92.2 |     94.9 |       55.4 |   3198.3 |    221.2 | Peregrine |
| arcsinh_100k           |     208.7 |    155.4 |    346.7 |      135.2 |        — |    142.5 | TensorFlow |
| maximum_100k           |      12.7 |     38.8 |     28.2 |       41.1 |    191.9 |     34.8 | Peregrine |
| minimum_100k           |      12.7 |     40.7 |     27.2 |       41.9 |    374.7 |     35.9 | Peregrine |
| power_100k             |     156.6 |    239.8 |    213.2 |      273.3 |        — |    145.8 | JAX |
| arctan2_100k           |      96.9 |    128.7 |    151.5 |       70.2 |        — |    320.5 | TensorFlow |
| logaddexp_100k         |     276.9 |    157.2 |    276.1 |      347.1 |        — |    157.1 | JAX |
| clip_100k              |       8.8 |     39.5 |     36.6 |       40.2 |    537.3 |     48.1 | Peregrine |
| where_100k             |      16.7 |     52.4 |     30.4 |       64.6 |    276.8 |     33.8 | Peregrine |
| greater_100k           |      12.7 |     48.4 |     26.1 |       49.8 |    190.7 |     27.8 | Peregrine |
| equal_100k             |      12.7 |     32.5 |     27.1 |       52.6 |    289.7 |     31.7 | Peregrine |
| sum_axis_256x512       |      19.2 |     39.4 |     25.5 |       48.2 |    207.0 |     53.1 | Peregrine |
| mean_axis_256x512      |      19.2 |     42.0 |     39.0 |       47.8 |    294.5 |     53.8 | Peregrine |
| max_axis_256x512       |      13.9 |     54.9 |     48.5 |       48.0 |    204.0 |     48.5 | Peregrine |
| min_axis_256x512       |      13.9 |     53.9 |     48.0 |       47.0 |    332.2 |     48.7 | Peregrine |
| var_256x512            |      46.5 |    275.9 |     67.6 |      213.6 |        — |     77.0 | Peregrine |
| prod_axis_256x512      |      24.6 |     39.8 |     27.2 |       47.9 |        — |     54.5 | Peregrine |
| logsumexp_256x512      |      97.3 |    194.8 |    118.6 |      325.5 |        — |    294.6 | Peregrine |
| cumsum_256x512         |     121.0 |     82.9 |    148.7 |      190.6 |    617.2 |    214.1 | PyTorch |
| argmax_axis_256x512    |      52.5 |     95.9 |    179.7 |       73.8 |   1288.9 |    188.6 | Peregrine |
| sum_axis_1024x1024     |     177.4 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     435.9 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      35.4 |     37.9 |     57.4 |       50.5 |   1807.8 |     41.8 | Peregrine |
| triu_256x256           |      35.2 |     37.5 |     55.1 |       55.9 |   1832.4 |     37.0 | Peregrine |
| repeat_64x128_2x3      |     127.0 |     43.4 |     31.7 |       72.7 |        — |     27.9 | JAX |
| pad_64x128             |      17.1 |      4.1 |     18.5 |       82.4 |     91.6 |     18.8 | PyTorch |
| stack_8x64x128         |      15.8 |      8.6 |     44.6 |       56.1 |    920.6 |    161.8 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     29.2 |       11.2 |        — |      9.8 | PyTorch |
| silu_100k              |      65.2 |     71.2 |     84.5 |      227.9 |    331.2 |     52.5 | JAX |
| softplus_100k          |     184.1 |    151.0 |    267.2 |      124.8 |    784.4 |    154.8 | TensorFlow |
| mish_100k              |     291.5 |    308.7 |    377.9 |      248.4 |   1160.0 |    234.3 | JAX |
| leaky_relu_100k        |       8.8 |     39.2 |     77.4 |       18.1 |        — |     31.7 | Peregrine |
| elu_100k               |      61.2 |    127.5 |    121.2 |      132.1 |    857.0 |     87.6 | Peregrine |
| hard_tanh_100k         |       8.8 |     39.3 |     36.0 |       40.7 |        — |     47.0 | Peregrine |
| relu6_100k             |       8.8 |     38.8 |     45.0 |       51.2 |    737.6 |    114.7 | Peregrine |
| hardswish_100k         |      10.2 |     38.6 |     67.3 |      213.5 |        — |     26.0 | Peregrine |
| gelu_100k              |      97.1 |     73.3 |    139.7 |      238.0 |    843.8 |    210.8 | PyTorch |
| selu_100k              |      64.9 |    136.0 |     87.3 |      137.0 |    741.4 |     82.2 | Peregrine |
| softsign_100k          |      38.8 |    123.3 |     42.8 |       46.8 |        — |     58.2 | Peregrine |
| cross_entropy_64x10    |       2.6 |     34.4 |     23.4 |      602.7 |   3356.3 |     55.5 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.3 |     19.0 |       39.8 |   1148.2 |     12.3 | Peregrine |
| mse_loss_64x10         |       4.1 |      4.8 |     22.0 |       35.3 |    451.1 |     24.7 | Peregrine |
| huber_loss_64x10       |       5.5 |      4.8 |     34.5 |      224.1 |        — |     48.3 | PyTorch |
| smooth_l1_loss_64x10   |       5.0 |      5.1 |     33.8 |      219.0 |        — |     48.6 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     18.3 |      357.9 |        — |     62.7 | Peregrine |
| cosine_sim_loss_64x64  |      13.8 |     10.1 |    111.3 |      220.8 |        — |     71.7 | PyTorch |
| rmsnorm_64x512         |      58.9 |     66.8 |     33.1 |      431.6 |        — |     81.5 | MLX |
| conv1d_1x32x128_k3     |      20.0 |     53.6 |     28.2 |      496.2 |        — |     75.2 | Peregrine |
| avgpool2d_1x16x32x32   |      25.5 |     45.5 |    265.4 |       63.0 |        — |     42.8 | Peregrine |
| groupnorm_4x64x16x16   |      74.0 |     51.5 |    223.3 |      699.2 |        — |    265.7 | PyTorch |
| rnn_seq32_128_256      |     187.8 |    265.5 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1136.9 |    799.7 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     855.3 |    757.1 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     815.0 |   1282.5 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     938.1 |   1122.9 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     923.1 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1287.4 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     108.3 |    257.3 |    481.0 |      122.5 |   2412.5 |    535.5 | Peregrine |
| rand_normal_100k       |     240.8 |    990.3 |    684.1 |      327.5 |   3287.7 |    618.2 | Peregrine |
| rand_bernoulli_100k    |     309.2 |    252.5 |    450.7 |      211.5 |        — |    533.5 | TensorFlow |
| rand_uniform_1M        |    1084.4 |   2626.2 |   4583.2 |      422.4 |   2409.5 |   2289.9 | TensorFlow |
| rand_normal_1M         |    2415.5 |   9985.7 |   6612.5 |     2000.3 |   3296.5 |   2923.6 | TensorFlow |
| rfft_1k                |       2.0 |      4.5 |     21.7 |       39.7 |        — |     60.7 | Peregrine |
| rfft_4k                |       7.0 |     15.0 |     30.7 |       50.9 |        — |     67.9 | Peregrine |
| rfft_16k               |      29.0 |     66.6 |     79.3 |      102.4 |        — |    116.2 | Peregrine |
| fft_1k                 |       3.0 |      6.7 |     23.2 |        8.0 |        — |     43.2 | Peregrine |
| fft_4k                 |      11.9 |     26.7 |     44.1 |       16.5 |        — |     64.2 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     20.7 |       64.0 |        — |      3.9 | Peregrine |
| solve_64x64            |      11.8 |     18.5 |    101.4 |       24.1 |        — |     32.3 | Peregrine |
| inv_64x64              |      36.2 |     26.4 |     51.8 |       31.7 |        — |     43.5 | PyTorch |
| cholesky_64x64         |       9.1 |     40.5 |     21.9 |       18.8 |        — |     20.7 | Peregrine |
| svd_64x64              |     274.7 |    279.4 |    290.6 |      506.6 |        — |    296.5 | Peregrine |
| qr_64x64               |      40.1 |     84.2 |     58.3 |       83.8 |        — |     65.3 | Peregrine |
| eigh_64x64             |     385.2 |    217.2 |    231.7 |      140.9 |        — |    237.8 | TensorFlow |
| det_64x64              |      22.5 |     20.8 |        — |       22.0 |        — |     28.7 | PyTorch |
| solve_128x128          |      48.6 |     45.1 |    190.1 |       76.7 |        — |     85.1 | PyTorch |
| inv_128x128            |      91.7 |     62.1 |     90.5 |      140.1 |        — |     83.9 | PyTorch |
| cholesky_128x128       |      49.8 |     53.0 |     26.5 |       58.1 |        — |     35.5 | MLX |
| svd_128x128            |     987.6 |    984.3 |    991.1 |     1887.2 |        — |   1011.0 | PyTorch |
| qr_128x128             |     187.7 |    223.0 |    193.9 |      336.3 |        — |    190.6 | Peregrine |
| eigh_128x128           |    1866.1 |    702.3 |    711.5 |      724.9 |        — |    735.1 | PyTorch |
| det_128x128            |      50.9 |     49.8 |        — |       82.9 |        — |     75.7 | PyTorch |
| solve_256x256          |     188.5 |    179.8 |    745.5 |      384.4 |        — |    259.3 | PyTorch |
| inv_256x256            |     461.4 |    296.3 |    250.8 |      861.1 |        — |    333.4 | MLX |
| cholesky_256x256       |     226.5 |     77.5 |     53.4 |      286.8 |        — |    117.0 | MLX |
| svd_256x256            |    5895.0 |   5879.6 |   5836.1 |     8261.4 |        — |   5877.8 | MLX |
| qr_256x256             |     991.8 |    993.0 |   1018.3 |     1724.4 |        — |    960.9 | JAX |
| eigh_256x256           |    6051.6 |   3490.1 |   3482.7 |     4640.1 |        — |   3552.2 | MLX |
| det_256x256            |     212.0 |    204.7 |        — |      436.9 |        — |    206.2 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1430.2 |    875.9 |        — |     2356.8 |   1238.6 |   2110.3 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2588.8 |   2007.0 |        — |     3636.8 |   1256.8 |   3387.5 | tinygrad |
| add_layernorm_196x768  |     105.2 |    101.0 |        — |     1184.7 |   1169.3 |    233.0 | PyTorch |
| add_layernorm_196x1024 |     139.5 |    102.4 |        — |     1246.9 |   1167.9 |    289.0 | PyTorch |
| matmul_f32_196x768x3072 |     539.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14728.0 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1431.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26523.4 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.64x** (Peregrine is faster)
- **Peregrine vs MLX: 0.46x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.36x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.06x** (Peregrine is faster)
- **Peregrine vs JAX: 0.43x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 87/141 ops
- PyTorch: 28/141 ops
- JAX: 10/141 ops
- TensorFlow: 9/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
