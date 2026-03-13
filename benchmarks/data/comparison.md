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
| matmul_128x128         |      35.8 |      6.5 |     21.2 |       52.5 |    442.5 |     58.4 | PyTorch |
| matmul_256x256         |     174.9 |     34.1 |     48.4 |      183.5 |    428.5 |    209.2 | PyTorch |
| matmul_512x512         |     277.5 |    146.5 |    197.0 |      702.2 |    428.8 |    556.6 | PyTorch |
| matmul_1024x1024       |    1014.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9952.6 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      10.2 |     48.8 |     33.8 |       60.3 |    194.6 |     34.3 | Peregrine |
| add_500k               |     112.8 |     57.8 |     80.7 |       87.0 |    191.8 |     63.0 | PyTorch |
| add_1M                 |     136.1 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     518.4 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     927.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.7 |     42.6 |     33.5 |       48.0 |    197.2 |     38.1 | Peregrine |
| mul_500k               |      94.7 |     65.5 |     78.1 |       86.7 |    196.4 |     60.5 | JAX |
| mul_1M                 |     110.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     531.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     957.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |     120.7 |     69.8 |     72.2 |       71.4 |    226.9 |     55.8 | JAX |
| exp_500k               |     228.7 |    173.5 |    246.7 |      120.3 |    226.8 |    132.3 | TensorFlow |
| exp_1M                 |     313.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1197.9 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2226.0 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     45.8 |     36.2 |       39.5 |    354.2 |    100.5 | Peregrine |
| relu_1M                |     135.8 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     37.0 |     23.5 |       12.4 |    642.1 |     33.0 | Peregrine |
| softmax_8x512          |       4.3 |     33.8 |     21.2 |       14.8 |    648.2 |     35.1 | Peregrine |
| mlp_fwd_64x784         |      33.7 |     26.7 |     74.8 |      233.2 |   1967.4 |    188.5 | PyTorch |
| mlp_fwd_256x784_wide   |     424.7 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     821.1 |   1543.1 |   1029.1 |     9305.1 |  25607.6 |   5264.5 | Peregrine |
| train_step_256_wide    |    3308.3 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     43.6 |     49.0 |       48.6 |    170.2 |     20.6 | Peregrine |
| square_100k            |       8.8 |     45.8 |     42.2 |       16.5 |    184.0 |     32.6 | Peregrine |
| rsqrt_100k             |      88.2 |     52.3 |     52.3 |       50.4 |        — |     92.5 | TensorFlow |
| floor_100k             |      46.6 |     44.0 |     35.9 |       18.2 |    428.4 |     30.0 | TensorFlow |
| ceil_100k              |      47.5 |     47.1 |     32.9 |       18.1 |    362.9 |     30.1 | TensorFlow |
| round_100k             |      47.5 |     45.8 |     48.1 |       44.8 |        — |     26.8 | JAX |
| sign_100k              |      55.4 |     45.0 |     47.3 |       47.6 |    835.1 |     35.9 | JAX |
| expm1_100k             |     171.8 |    117.5 |    135.7 |      148.7 |        — |     99.4 | JAX |
| log2_100k              |     104.7 |     93.3 |    136.6 |      154.7 |    167.9 |     56.6 | JAX |
| log10_100k             |     114.6 |     97.2 |    140.3 |      146.6 |        — |     56.9 | JAX |
| log1p_100k             |     110.5 |     92.4 |    163.7 |      101.0 |        — |    104.8 | PyTorch |
| erf_100k               |     142.8 |     70.4 |    137.6 |       59.4 |        — |     56.0 | JAX |
| sinh_100k              |      54.3 |    138.8 |    122.3 |      126.4 |    562.2 |    120.2 | Peregrine |
| cosh_100k              |      47.2 |    136.9 |    122.8 |      127.1 |    482.8 |     70.1 | Peregrine |
| arcsin_100k            |      53.0 |     86.0 |     98.2 |       62.2 |   3079.4 |    125.8 | Peregrine |
| arccos_100k            |     113.6 |     91.3 |    116.9 |       60.4 |        — |    205.0 | TensorFlow |
| arctan_100k            |      54.1 |    101.6 |    105.2 |       67.4 |   3308.5 |    220.6 | Peregrine |
| arcsinh_100k           |     150.4 |    162.9 |    364.4 |      166.9 |        — |    119.9 | JAX |
| maximum_100k           |      12.7 |     51.2 |     28.8 |       47.1 |    198.6 |     32.1 | Peregrine |
| minimum_100k           |      12.7 |     47.1 |     28.2 |       49.7 |    383.5 |     32.9 | Peregrine |
| power_100k             |     393.1 |    251.2 |    217.5 |      325.0 |        — |    156.2 | JAX |
| arctan2_100k           |    1119.8 |    142.7 |    156.8 |       69.8 |        — |    320.7 | TensorFlow |
| logaddexp_100k         |     419.1 |    161.6 |    278.6 |      440.9 |        — |    170.8 | PyTorch |
| clip_100k              |       8.1 |     44.8 |     39.9 |       43.2 |    570.4 |     40.6 | Peregrine |
| where_100k             |      95.0 |     36.7 |     30.2 |       68.2 |    293.7 |     34.8 | MLX |
| greater_100k           |      71.4 |     55.2 |     25.3 |       46.1 |    197.3 |     28.1 | MLX |
| equal_100k             |      71.4 |     36.6 |     30.0 |       53.8 |    296.5 |     38.8 | MLX |
| sum_axis_256x512       |     114.8 |     36.5 |     24.4 |       54.4 |    215.2 |     54.3 | MLX |
| mean_axis_256x512      |     116.1 |     51.1 |     26.6 |       48.9 |    304.1 |     53.0 | MLX |
| max_axis_256x512       |     154.7 |     62.0 |     50.4 |       48.4 |    211.3 |     45.5 | JAX |
| min_axis_256x512       |     154.3 |     59.4 |     42.8 |       47.8 |    335.6 |     47.7 | MLX |
| var_256x512            |     235.8 |    429.6 |     70.8 |      180.2 |        — |     79.5 | MLX |
| prod_axis_256x512      |     149.3 |     44.6 |     27.0 |       48.9 |        — |     56.1 | MLX |
| logsumexp_256x512      |     382.1 |    229.2 |    116.6 |      294.4 |        — |    296.6 | MLX |
| cumsum_256x512         |     122.0 |     84.3 |    133.0 |      168.8 |    645.6 |    211.1 | PyTorch |
| argmax_axis_256x512    |     154.6 |     97.6 |    179.0 |       62.1 |   1342.5 |    179.5 | TensorFlow |
| sum_axis_1024x1024     |     957.6 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    1928.0 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      36.1 |     42.2 |     57.0 |       52.1 |   1910.5 |     37.3 | Peregrine |
| triu_256x256           |      34.7 |     38.5 |     53.7 |       55.4 |   1887.3 |     44.3 | Peregrine |
| repeat_64x128_2x3      |     125.1 |     51.3 |     27.5 |       78.7 |        — |     28.1 | MLX |
| pad_64x128             |      16.8 |      4.2 |     16.5 |       87.4 |     96.8 |     19.0 | PyTorch |
| stack_8x64x128         |      15.9 |      8.4 |     45.7 |       55.6 |   1016.0 |    161.5 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     26.4 |       13.2 |        — |      7.7 | PyTorch |
| silu_100k              |      64.5 |     75.8 |     87.0 |      226.5 |    351.9 |     74.5 | Peregrine |
| softplus_100k          |     353.6 |    159.8 |    278.6 |      124.6 |    812.6 |    159.8 | TensorFlow |
| mish_100k              |     471.9 |    322.4 |    396.6 |      245.4 |   1232.2 |    238.5 | JAX |
| leaky_relu_100k        |       8.1 |     44.7 |     86.2 |       20.2 |        — |     31.5 | Peregrine |
| elu_100k               |     145.1 |    135.9 |    124.0 |      142.8 |    902.3 |     86.8 | JAX |
| hard_tanh_100k         |      51.5 |     52.6 |     34.6 |       43.7 |        — |     47.5 | MLX |
| relu6_100k             |      51.5 |     45.6 |     46.1 |       53.3 |    784.0 |    110.9 | PyTorch |
| hardswish_100k         |      85.3 |     49.2 |     67.8 |      221.0 |        — |     34.8 | JAX |
| gelu_100k              |      83.2 |     83.7 |    143.1 |      249.7 |    895.4 |    222.3 | Peregrine |
| selu_100k              |     175.3 |    140.2 |     85.4 |      131.6 |    804.0 |     84.6 | JAX |
| softsign_100k          |      36.0 |    149.6 |     42.0 |       49.0 |        — |     77.6 | Peregrine |
| cross_entropy_64x10    |       2.7 |     47.2 |     24.3 |      635.8 |   3823.9 |     58.2 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.5 |     15.8 |       44.5 |   1199.5 |     12.1 | Peregrine |
| mse_loss_64x10         |       3.8 |      5.0 |     19.5 |       40.3 |    479.6 |     24.1 | Peregrine |
| huber_loss_64x10       |       5.4 |      4.9 |     33.8 |      248.9 |        — |     49.3 | PyTorch |
| smooth_l1_loss_64x10   |       5.3 |      5.2 |     31.0 |      246.1 |        — |     46.9 | PyTorch |
| kl_div_loss_64x10      |       2.5 |      6.6 |     19.2 |      397.0 |        — |     64.1 | Peregrine |
| cosine_sim_loss_64x64  |      13.9 |     11.0 |    116.2 |      245.1 |        — |     68.5 | PyTorch |
| rmsnorm_64x512         |      58.8 |     79.0 |     48.2 |      441.7 |        — |     70.9 | MLX |
| conv1d_1x32x128_k3     |      20.3 |     66.1 |     33.0 |      515.9 |        — |     74.4 | Peregrine |
| avgpool2d_1x16x32x32   |      25.6 |     50.7 |    292.1 |       66.4 |        — |     43.1 | Peregrine |
| groupnorm_4x64x16x16   |      74.0 |     63.2 |    238.4 |      802.5 |        — |    279.0 | PyTorch |
| rnn_seq32_128_256      |     185.2 |    278.7 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1126.9 |    821.5 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     753.2 |    785.6 |        — |          — |        — |        — | Peregrine |
| optim_adam_64          |     819.4 |   1286.0 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     935.8 |   1214.0 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     930.3 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1306.1 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     109.7 |    271.3 |    532.1 |      127.2 |   2619.6 |    553.9 | Peregrine |
| rand_normal_100k       |     781.6 |   1037.2 |    817.5 |      342.6 |   3519.1 |    625.5 | TensorFlow |
| rand_bernoulli_100k    |     308.8 |    278.8 |    533.6 |      220.0 |        — |    544.8 | TensorFlow |
| rand_uniform_1M        |    1087.7 |   2871.5 |   5178.0 |      429.7 |   2513.4 |   2330.8 | TensorFlow |
| rand_normal_1M         |    7800.7 |  10803.1 |   7665.3 |     2171.3 |   3524.0 |   2956.9 | TensorFlow |
| rfft_1k                |       2.0 |      4.5 |     30.6 |       46.1 |        — |     23.0 | Peregrine |
| rfft_4k                |       6.7 |     15.2 |     30.2 |       56.2 |        — |     63.6 | Peregrine |
| rfft_16k               |      29.5 |     66.9 |     85.3 |      107.7 |        — |    115.9 | Peregrine |
| fft_1k                 |       3.2 |      7.0 |     35.9 |        8.9 |        — |     17.6 | Peregrine |
| fft_4k                 |      11.9 |     27.9 |     56.0 |       17.6 |        — |     56.5 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     63.2 |       71.7 |        — |      3.9 | Peregrine |
| solve_64x64            |      11.8 |     17.9 |    152.9 |       25.4 |        — |     34.8 | Peregrine |
| inv_64x64              |      36.3 |     25.8 |     60.1 |       34.7 |        — |     40.3 | PyTorch |
| cholesky_64x64         |       9.1 |     31.8 |     21.6 |       20.8 |        — |     21.0 | Peregrine |
| svd_64x64              |     277.0 |    291.4 |    311.9 |      521.1 |        — |    297.7 | Peregrine |
| qr_64x64               |      41.3 |     85.9 |     57.9 |       85.7 |        — |     63.5 | Peregrine |
| eigh_64x64             |     387.3 |    219.0 |    255.0 |      148.4 |        — |    237.7 | TensorFlow |
| det_64x64              |      22.5 |     19.7 |        — |       23.0 |        — |     28.6 | PyTorch |
| solve_128x128          |      50.1 |     49.0 |    202.1 |       77.8 |        — |     85.1 | PyTorch |
| inv_128x128            |      93.8 |     59.3 |    105.5 |      141.6 |        — |     88.2 | PyTorch |
| cholesky_128x128       |      50.4 |     52.7 |     35.1 |       61.9 |        — |     36.2 | MLX |
| svd_128x128            |     983.7 |   1002.7 |   1059.1 |     1879.2 |        — |   1036.5 | Peregrine |
| qr_128x128             |     186.6 |    233.0 |    211.1 |      333.7 |        — |    196.6 | Peregrine |
| eigh_128x128           |    1868.7 |    708.4 |    770.1 |      726.0 |        — |    741.3 | PyTorch |
| det_128x128            |      52.2 |     48.6 |        — |       82.1 |        — |     75.6 | PyTorch |
| solve_256x256          |     189.9 |    165.3 |    985.7 |      385.0 |        — |    265.0 | PyTorch |
| inv_256x256            |     469.6 |    303.3 |    255.2 |      867.8 |        — |    354.1 | MLX |
| cholesky_256x256       |     227.9 |     95.0 |     69.5 |      286.3 |        — |    117.2 | MLX |
| svd_256x256            |    6251.3 |   5981.7 |   7058.1 |     8467.9 |        — |   6149.6 | PyTorch |
| qr_256x256             |    1024.4 |   1031.4 |   1126.5 |     1779.9 |        — |   1033.7 | Peregrine |
| eigh_256x256           |    6198.3 |   3515.4 |   3885.3 |     4711.5 |        — |   3622.1 | PyTorch |
| det_256x256            |     217.4 |    208.0 |        — |      441.6 |        — |    206.2 | JAX |
| matmul_bias_gelu_196x768x3072 |    1207.2 |   1099.6 |        — |     2463.9 |   1314.8 |   2155.1 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2230.3 |   2312.2 |        — |     3898.9 |   1322.6 |   3776.7 | tinygrad |
| add_layernorm_196x768  |     107.5 |    109.7 |        — |     1280.5 |   1159.9 |    237.2 | Peregrine |
| add_layernorm_196x1024 |     140.2 |    121.1 |        — |     1356.9 |   1183.2 |    297.9 | PyTorch |
| matmul_f32_196x768x3072 |     687.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14865.4 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1607.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26825.4 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.90x** (Peregrine is faster)
- **Peregrine vs MLX: 0.66x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.53x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.10x** (Peregrine is faster)
- **Peregrine vs JAX: 0.67x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 68/141 ops
- PyTorch: 28/141 ops
- JAX: 16/141 ops
- MLX: 15/141 ops
- TensorFlow: 13/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
