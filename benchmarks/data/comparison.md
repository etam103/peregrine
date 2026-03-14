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
| matmul_128x128         |      25.1 |      6.1 |     20.1 |       52.0 |    442.4 |     79.0 | PyTorch |
| matmul_256x256         |      28.7 |     31.8 |     44.8 |      194.3 |    426.1 |    156.8 | Peregrine |
| matmul_512x512         |     277.5 |    142.3 |    164.6 |      676.1 |    436.5 |    514.5 | PyTorch |
| matmul_1024x1024       |    1003.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9228.2 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.9 |     37.0 |     29.4 |       51.2 |    191.5 |     35.0 | Peregrine |
| add_500k               |      63.8 |     61.2 |     71.1 |       86.4 |    186.8 |     64.3 | PyTorch |
| add_1M                 |     117.6 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     533.2 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     932.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |       9.5 |     41.4 |     28.6 |       49.2 |    190.8 |     23.6 | Peregrine |
| mul_500k               |      47.8 |     58.6 |     77.1 |       76.1 |    212.7 |     61.9 | Peregrine |
| mul_1M                 |     105.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     604.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |    1040.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      50.5 |     64.5 |     60.4 |       68.2 |    219.8 |     45.9 | JAX |
| exp_500k               |      95.1 |    136.5 |    224.1 |      104.3 |    222.2 |    122.3 | Peregrine |
| exp_1M                 |     140.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     434.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     749.0 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.7 |     37.9 |     28.8 |       41.0 |    338.2 |     99.8 | Peregrine |
| relu_1M                |      82.3 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     31.4 |     18.2 |       11.5 |    622.0 |     31.1 | Peregrine |
| softmax_8x512          |       4.0 |     41.8 |     16.5 |       14.4 |    626.7 |     33.8 | Peregrine |
| mlp_fwd_64x784         |      33.2 |     27.7 |     52.1 |      254.0 |   1857.6 |    175.6 | PyTorch |
| mlp_fwd_256x784_wide   |     396.0 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     804.5 |   1265.8 |    778.3 |     8678.4 |  24626.4 |   4976.0 | MLX |
| train_step_256_wide    |    3430.1 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.6 |     34.9 |     28.5 |       50.0 |    163.5 |     30.4 | Peregrine |
| square_100k            |       8.6 |     38.9 |     25.6 |       16.3 |    181.8 |     29.1 | Peregrine |
| rsqrt_100k             |      21.5 |     44.3 |     32.4 |       52.4 |        — |     92.9 | Peregrine |
| floor_100k             |       8.6 |     39.6 |     23.2 |       16.5 |    426.7 |     32.0 | Peregrine |
| ceil_100k              |       8.7 |     39.7 |     24.7 |       16.4 |    355.6 |     28.5 | Peregrine |
| round_100k             |       8.9 |     40.2 |     23.3 |       49.2 |        — |     28.8 | Peregrine |
| sign_100k              |       8.9 |     41.4 |     28.2 |       47.0 |    826.6 |     35.3 | Peregrine |
| expm1_100k             |      65.2 |    110.9 |    109.5 |      134.5 |        — |     99.1 | Peregrine |
| log2_100k              |      57.4 |     89.5 |     97.8 |      148.8 |    166.9 |     56.4 | JAX |
| log10_100k             |      59.9 |     85.0 |    107.2 |      145.0 |        — |     57.3 | JAX |
| log1p_100k             |      78.0 |     85.8 |    127.9 |       98.9 |        — |    104.7 | Peregrine |
| erf_100k               |     103.8 |     56.9 |    101.1 |       58.5 |        — |     55.5 | JAX |
| sinh_100k              |      52.7 |    128.4 |     97.0 |      126.1 |    537.8 |    111.0 | Peregrine |
| cosh_100k              |      47.8 |    122.4 |     89.9 |      134.7 |    476.0 |     69.5 | Peregrine |
| arcsin_100k            |      53.8 |     75.5 |     94.2 |       53.6 |   3062.6 |    111.9 | TensorFlow |
| arccos_100k            |      64.5 |     88.4 |    113.7 |       53.8 |        — |    204.6 | TensorFlow |
| arctan_100k            |      56.6 |     93.6 |     97.2 |       57.8 |   3120.8 |    213.6 | Peregrine |
| arcsinh_100k           |     211.0 |    143.2 |    359.1 |      142.2 |        — |    112.1 | JAX |
| maximum_100k           |      13.4 |     37.2 |     27.5 |       45.0 |    188.9 |     28.9 | Peregrine |
| minimum_100k           |      12.9 |     40.0 |     26.1 |       40.6 |    373.7 |     24.7 | Peregrine |
| power_100k             |     165.2 |    222.1 |    212.0 |      271.0 |        — |    156.4 | JAX |
| arctan2_100k           |     102.3 |    133.5 |    145.0 |       73.0 |        — |    316.0 | TensorFlow |
| logaddexp_100k         |     288.0 |    153.4 |    259.8 |      369.1 |        — |    155.0 | PyTorch |
| clip_100k              |       8.7 |     44.0 |     35.0 |       42.3 |    550.7 |     39.1 | Peregrine |
| where_100k             |      16.6 |     53.5 |     28.4 |       66.0 |    277.1 |     33.2 | Peregrine |
| greater_100k           |      12.9 |     49.6 |     24.6 |       55.4 |    189.3 |     27.1 | Peregrine |
| equal_100k             |      12.8 |     30.1 |     24.8 |       55.7 |    287.7 |     26.9 | Peregrine |
| sum_axis_256x512       |      19.3 |     41.4 |     23.4 |       52.8 |    205.5 |     51.6 | Peregrine |
| mean_axis_256x512      |      19.4 |     43.7 |     31.6 |       51.5 |    294.4 |     51.5 | Peregrine |
| max_axis_256x512       |      14.0 |     55.8 |     43.3 |       47.1 |    208.3 |     45.6 | Peregrine |
| min_axis_256x512       |      14.0 |     55.2 |     42.2 |       51.2 |    334.9 |     45.4 | Peregrine |
| var_256x512            |      46.6 |    273.7 |     61.1 |      219.0 |        — |     80.5 | Peregrine |
| prod_axis_256x512      |      24.9 |     39.0 |     26.1 |       51.4 |        — |     58.9 | Peregrine |
| logsumexp_256x512      |     101.5 |    197.0 |    108.7 |      328.9 |        — |    269.7 | Peregrine |
| cumsum_256x512         |     119.8 |     78.4 |    129.1 |      179.5 |    627.5 |    207.3 | PyTorch |
| argmax_axis_256x512    |      51.9 |     94.6 |    170.7 |       74.8 |   1305.6 |    170.2 | Peregrine |
| sum_axis_1024x1024     |     174.1 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.8 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.8 |     38.5 |     58.3 |       61.1 |   1849.2 |     37.0 | Peregrine |
| triu_256x256           |       7.6 |     37.8 |     56.4 |       53.9 |   1835.0 |     36.5 | Peregrine |
| repeat_64x128_2x3      |       6.8 |     46.4 |     31.0 |       75.0 |        — |     28.1 | Peregrine |
| pad_64x128             |       2.6 |      4.3 |     18.0 |       84.5 |     92.1 |     18.1 | Peregrine |
| stack_8x64x128         |       3.8 |      8.7 |     47.1 |       58.0 |    913.4 |    160.2 | Peregrine |
| diagonal_512x512       |       0.4 |      0.6 |     31.4 |       12.4 |        — |      8.6 | Peregrine |
| silu_100k              |      64.0 |     80.6 |     85.0 |      215.3 |    328.1 |     52.8 | JAX |
| softplus_100k          |     180.8 |    144.1 |    267.1 |      126.0 |    799.9 |    155.7 | TensorFlow |
| mish_100k              |     286.2 |    305.7 |    392.1 |      253.2 |   1171.5 |    236.0 | JAX |
| leaky_relu_100k        |       8.7 |     40.3 |     81.2 |       19.5 |        — |     20.7 | Peregrine |
| elu_100k               |      60.0 |    130.5 |    119.0 |      131.8 |    883.6 |     78.8 | Peregrine |
| hard_tanh_100k         |       8.7 |     39.2 |     31.8 |       42.5 |        — |     32.5 | Peregrine |
| relu6_100k             |       8.7 |     42.0 |     45.6 |       51.6 |    733.6 |    113.4 | Peregrine |
| hardswish_100k         |      10.0 |     44.4 |     70.8 |      212.7 |        — |     28.6 | Peregrine |
| gelu_100k              |      97.8 |     73.8 |    139.5 |      243.0 |    848.9 |    212.2 | PyTorch |
| selu_100k              |      63.8 |    132.5 |     86.8 |      130.7 |    754.0 |     82.0 | Peregrine |
| softsign_100k          |      38.4 |    123.3 |     53.8 |       45.1 |        — |     67.7 | Peregrine |
| cross_entropy_64x10    |       2.8 |     39.0 |     21.7 |      621.9 |   3367.7 |     53.3 | Peregrine |
| l1_loss_64x10          |       1.1 |      5.4 |     15.7 |       43.9 |   1141.4 |     12.4 | Peregrine |
| mse_loss_64x10         |       3.8 |      4.8 |     20.2 |       39.2 |    448.6 |     24.4 | Peregrine |
| huber_loss_64x10       |       0.3 |      4.9 |     31.1 |      240.1 |        — |     47.6 | Peregrine |
| smooth_l1_loss_64x10   |       5.0 |      5.0 |     31.8 |      242.9 |        — |     48.1 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     17.6 |      379.4 |        — |     60.7 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.5 |    110.3 |      240.8 |        — |     51.0 | Peregrine |
| rmsnorm_64x512         |      18.6 |     63.6 |     45.1 |      440.3 |        — |     65.9 | Peregrine |
| conv1d_1x32x128_k3     |      20.2 |     50.4 |     28.3 |      556.7 |        — |     74.6 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     43.8 |    265.1 |       66.5 |        — |     44.4 | Peregrine |
| groupnorm_4x64x16x16   |      21.6 |    116.8 |    223.5 |      810.5 |        — |    262.2 | Peregrine |
| rnn_seq32_128_256      |     192.8 |    517.0 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1011.8 |    816.9 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     876.0 |    786.7 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     801.1 |   1349.8 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     924.5 |   1152.0 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     909.7 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1271.9 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      60.2 |    257.7 |    490.6 |      133.8 |   2390.4 |    513.7 | Peregrine |
| rand_normal_100k       |     237.9 |    989.5 |    697.4 |      352.9 |   3267.1 |    586.2 | Peregrine |
| rand_bernoulli_100k    |     121.5 |    254.2 |    461.4 |      232.3 |        — |    526.4 | Peregrine |
| rand_uniform_1M        |     603.7 |   2587.2 |   4644.0 |      448.6 |   2385.8 |   2254.6 | TensorFlow |
| rand_normal_1M         |    2382.5 |   9768.6 |   6698.3 |     2127.0 |   3418.3 |   2850.2 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     21.8 |       43.8 |        — |     56.7 | Peregrine |
| rfft_4k                |       6.5 |     14.8 |     30.3 |       54.5 |        — |     62.6 | Peregrine |
| rfft_16k               |      30.6 |     65.5 |     95.3 |      107.1 |        — |    118.8 | Peregrine |
| fft_1k                 |       3.3 |      6.6 |     29.5 |        8.7 |        — |     41.6 | Peregrine |
| fft_4k                 |      12.3 |     26.1 |     49.0 |       17.2 |        — |     55.8 | Peregrine |
| norm_l2_1k             |       1.1 |      1.2 |     19.1 |       69.6 |        — |      3.9 | Peregrine |
| solve_64x64            |      11.7 |     24.2 |     96.6 |       24.3 |        — |     32.3 | Peregrine |
| inv_64x64              |      36.8 |     26.0 |     46.0 |       32.2 |        — |     36.8 | PyTorch |
| cholesky_64x64         |       6.2 |     42.4 |     19.6 |       19.4 |        — |     20.9 | Peregrine |
| svd_64x64              |     275.6 |    278.8 |    297.8 |      510.1 |        — |    297.8 | Peregrine |
| qr_64x64               |      41.0 |     79.9 |     58.6 |       84.0 |        — |     64.3 | Peregrine |
| eigh_64x64             |     376.3 |    215.6 |    234.3 |      145.7 |        — |    236.9 | TensorFlow |
| det_64x64              |      11.9 |     19.9 |        — |       23.0 |        — |     29.3 | Peregrine |
| solve_128x128          |      50.5 |     45.1 |    347.7 |       76.7 |        — |     85.0 | PyTorch |
| inv_128x128            |      94.9 |     61.9 |     87.5 |      139.0 |        — |     84.1 | PyTorch |
| cholesky_128x128       |      35.3 |     56.7 |     33.4 |       58.5 |        — |     38.6 | MLX |
| svd_128x128            |     989.4 |    998.7 |   1012.5 |     1864.8 |        — |   1014.5 | Peregrine |
| qr_128x128             |     190.3 |    212.3 |    197.4 |      327.5 |        — |    190.5 | Peregrine |
| eigh_128x128           |    1831.7 |    705.4 |    738.7 |      714.7 |        — |    736.7 | PyTorch |
| det_128x128            |      41.1 |     49.7 |        — |       82.0 |        — |     75.7 | Peregrine |
| solve_256x256          |     188.8 |    175.3 |    770.3 |      378.9 |        — |    264.8 | PyTorch |
| inv_256x256            |     459.4 |    308.3 |    258.1 |      852.3 |        — |    336.9 | MLX |
| cholesky_256x256       |     145.4 |     95.2 |     58.8 |      283.8 |        — |    117.1 | MLX |
| svd_256x256            |    5887.5 |   5874.7 |   5975.3 |     8710.1 |        — |   5881.5 | PyTorch |
| qr_256x256             |     971.0 |    991.0 |   1038.6 |     1713.0 |        — |    965.0 | JAX |
| eigh_256x256           |    5993.6 |   3465.0 |   3368.5 |     4660.1 |        — |   3569.1 | MLX |
| det_256x256            |     140.8 |    201.8 |        — |      428.1 |        — |    206.8 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1824.8 |    856.1 |        — |     2427.1 |   1274.3 |   2121.5 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3323.1 |   1957.0 |        — |     3772.6 |   1236.4 |   3510.6 | tinygrad |
| add_layernorm_196x768  |     105.9 |    102.3 |        — |     1210.2 |   1118.3 |    246.5 | PyTorch |
| add_layernorm_196x1024 |     141.0 |    106.6 |        — |     1276.8 |   1118.0 |    282.7 | PyTorch |
| matmul_f32_196x768x3072 |     581.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14428.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1624.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26262.4 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.52x** (Peregrine is faster)
- **Peregrine vs MLX: 0.38x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.29x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.37x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 101/141 ops
- PyTorch: 18/141 ops
- JAX: 9/141 ops
- TensorFlow: 7/141 ops
- MLX: 5/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
