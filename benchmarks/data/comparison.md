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
| matmul_128x128         |      25.1 |      6.1 |     20.2 |       95.3 |    417.7 |     77.3 | PyTorch |
| matmul_256x256         |      78.1 |     31.7 |     43.4 |      192.0 |    422.4 |    145.0 | PyTorch |
| matmul_512x512         |     239.0 |    138.9 |    164.4 |      660.4 |    426.1 |    497.5 | PyTorch |
| matmul_1024x1024       |     951.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    8793.6 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     38.4 |     28.2 |       52.1 |    185.6 |     34.5 | Peregrine |
| add_500k               |      61.6 |     56.4 |     76.7 |       77.2 |    185.0 |     59.7 | PyTorch |
| add_1M                 |     126.2 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     534.9 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     911.1 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.6 |     38.6 |     30.5 |       43.9 |    187.2 |     30.6 | Peregrine |
| mul_500k               |      61.9 |     56.9 |     76.7 |       75.2 |    185.4 |     58.7 | PyTorch |
| mul_1M                 |     126.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     532.7 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     904.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.1 |     59.3 |     55.5 |       62.6 |    219.8 |     46.4 | JAX |
| exp_500k               |     154.8 |    148.6 |    218.0 |       98.5 |    218.9 |    123.2 | TensorFlow |
| exp_1M                 |     185.0 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     443.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     820.1 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.7 |     39.4 |     24.9 |       36.3 |    331.2 |     98.6 | Peregrine |
| relu_1M                |      82.2 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.1 |     27.0 |     19.2 |       11.4 |    621.6 |     30.6 | Peregrine |
| softmax_8x512          |       4.0 |     28.8 |     19.8 |       14.3 |    608.4 |     32.9 | Peregrine |
| mlp_fwd_64x784         |      33.1 |     27.6 |     50.2 |      236.2 |   1755.4 |    169.2 | PyTorch |
| mlp_fwd_256x784_wide   |     395.6 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     814.7 |   1346.7 |    758.6 |     8063.3 |  22467.7 |   4901.2 | MLX |
| train_step_256_wide    |    3281.2 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.7 |     41.0 |     24.8 |       48.6 |    162.0 |     30.7 | Peregrine |
| square_100k            |       8.7 |     40.2 |     24.0 |       14.5 |    175.3 |     28.9 | Peregrine |
| rsqrt_100k             |      21.5 |     45.1 |     30.8 |       51.5 |        — |     84.0 | Peregrine |
| floor_100k             |       8.6 |     43.4 |     24.0 |       16.2 |    413.5 |     28.8 | Peregrine |
| ceil_100k              |       8.6 |     44.0 |     23.8 |       16.0 |    349.4 |     28.3 | Peregrine |
| round_100k             |       8.6 |     43.2 |     24.5 |       43.6 |        — |     28.2 | Peregrine |
| sign_100k              |       8.7 |     41.6 |     29.3 |       45.1 |    782.8 |     35.6 | Peregrine |
| expm1_100k             |      63.1 |    111.3 |    103.9 |      140.6 |        — |     99.3 | Peregrine |
| log2_100k              |      55.5 |     85.9 |     98.7 |      148.8 |    162.1 |     56.1 | Peregrine |
| log10_100k             |      57.9 |     86.8 |    107.2 |      130.3 |        — |     55.8 | JAX |
| log1p_100k             |      75.4 |     82.1 |    127.4 |       95.4 |        — |    104.4 | Peregrine |
| erf_100k               |     100.7 |     57.3 |    100.1 |       55.9 |        — |     55.2 | JAX |
| sinh_100k              |      51.1 |    127.6 |     93.2 |      129.2 |    521.6 |    108.8 | Peregrine |
| cosh_100k              |      46.3 |    124.8 |     89.3 |      127.7 |    455.4 |     69.2 | Peregrine |
| arcsin_100k            |      52.1 |     80.8 |     93.9 |       54.4 |   2872.7 |    110.8 | Peregrine |
| arccos_100k            |      60.7 |     87.4 |    110.3 |       53.6 |        — |    205.2 | TensorFlow |
| arctan_100k            |      53.1 |     91.8 |     91.8 |       62.9 |   2961.3 |    211.0 | Peregrine |
| arcsinh_100k           |     204.8 |    152.2 |    331.2 |      131.5 |        — |    112.7 | JAX |
| maximum_100k           |      12.5 |     38.9 |     27.9 |       43.5 |    189.4 |     29.4 | Peregrine |
| minimum_100k           |      12.5 |     39.1 |     27.1 |       42.8 |    370.2 |     31.9 | Peregrine |
| power_100k             |     153.8 |    221.9 |    210.2 |      270.0 |        — |    141.1 | JAX |
| arctan2_100k           |      95.1 |    129.5 |    144.0 |       66.4 |        — |    314.6 | TensorFlow |
| logaddexp_100k         |     272.4 |    151.6 |    254.5 |      353.9 |        — |    142.8 | JAX |
| clip_100k              |       8.7 |     42.5 |     34.5 |       42.5 |    533.4 |     39.5 | Peregrine |
| where_100k             |      16.4 |     50.9 |     28.2 |       65.9 |    272.6 |     32.0 | Peregrine |
| greater_100k           |      12.5 |     47.5 |     23.9 |       51.8 |    187.7 |     27.4 | Peregrine |
| equal_100k             |      12.5 |     32.1 |     24.1 |       46.1 |    284.5 |     26.7 | Peregrine |
| sum_axis_256x512       |      18.8 |     38.0 |     23.1 |       49.0 |    203.5 |     53.6 | Peregrine |
| mean_axis_256x512      |      18.9 |     41.6 |     24.3 |       48.0 |    290.0 |     45.8 | Peregrine |
| max_axis_256x512       |      13.7 |     54.6 |     39.6 |       46.5 |    200.6 |     44.9 | Peregrine |
| min_axis_256x512       |      13.7 |     53.1 |     41.0 |       46.5 |    321.3 |     45.9 | Peregrine |
| var_256x512            |      45.7 |    272.3 |     57.0 |      211.8 |        — |     74.6 | Peregrine |
| prod_axis_256x512      |      24.1 |     39.3 |     25.4 |       50.4 |        — |     56.2 | Peregrine |
| logsumexp_256x512      |      95.5 |    192.8 |    105.8 |      328.4 |        — |    264.1 | Peregrine |
| cumsum_256x512         |     112.7 |     78.6 |    128.1 |      179.9 |    604.9 |    206.3 | PyTorch |
| argmax_axis_256x512    |      51.7 |     91.9 |    170.3 |       70.1 |   1270.3 |    163.8 | Peregrine |
| sum_axis_1024x1024     |     174.0 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.9 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.7 |     38.2 |     53.8 |       53.9 |   1770.6 |     36.2 | Peregrine |
| triu_256x256           |       7.6 |     37.3 |     54.7 |       52.4 |   1772.4 |     34.1 | Peregrine |
| repeat_64x128_2x3      |       6.5 |     46.2 |     30.4 |       75.0 |        — |     27.8 | Peregrine |
| pad_64x128             |       2.6 |      4.1 |     18.1 |       83.9 |     90.0 |     18.0 | Peregrine |
| stack_8x64x128         |       3.8 |      8.7 |     44.7 |       56.4 |    897.5 |    160.2 | Peregrine |
| diagonal_512x512       |       0.3 |      0.6 |     28.9 |       12.4 |        — |      9.3 | Peregrine |
| silu_100k              |      64.0 |     71.1 |     84.8 |      214.9 |    325.4 |     51.7 | JAX |
| softplus_100k          |     180.7 |    150.9 |    260.1 |      118.5 |    763.2 |    154.9 | TensorFlow |
| mish_100k              |     286.2 |    304.0 |    369.2 |      236.8 |   1139.0 |    237.6 | TensorFlow |
| leaky_relu_100k        |       8.7 |     37.7 |     74.8 |       19.3 |        — |     20.0 | Peregrine |
| elu_100k               |      60.0 |    123.8 |    114.8 |      122.7 |    858.9 |     77.3 | Peregrine |
| hard_tanh_100k         |       8.7 |     41.1 |     34.2 |       42.0 |        — |     36.9 | Peregrine |
| relu6_100k             |       8.7 |     38.2 |     44.2 |       49.3 |    725.7 |    108.9 | Peregrine |
| hardswish_100k         |      10.0 |     39.8 |     69.2 |      197.1 |        — |     27.0 | Peregrine |
| gelu_100k              |      95.7 |     73.2 |    135.2 |      229.4 |    833.0 |    199.7 | PyTorch |
| selu_100k              |      63.7 |    132.7 |     85.2 |      126.7 |    730.5 |     81.5 | Peregrine |
| softsign_100k          |      38.2 |    122.8 |     43.8 |       47.9 |        — |     53.6 | Peregrine |
| cross_entropy_64x10    |       2.6 |     38.0 |     22.4 |      613.9 |   3310.8 |     50.9 | Peregrine |
| l1_loss_64x10          |       0.9 |      5.3 |     18.8 |       48.5 |   1110.2 |     11.8 | Peregrine |
| mse_loss_64x10         |       4.1 |      4.8 |     21.7 |       44.0 |    442.7 |     23.6 | Peregrine |
| huber_loss_64x10       |       0.3 |      4.8 |     33.1 |      233.8 |        — |     47.3 | Peregrine |
| smooth_l1_loss_64x10   |       0.8 |      5.1 |     32.7 |      233.8 |        — |     47.8 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.2 |     17.7 |      374.6 |        — |     55.2 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.2 |    111.8 |      238.4 |        — |     48.2 | Peregrine |
| rmsnorm_64x512         |      18.4 |     65.6 |     32.9 |      437.2 |        — |     65.3 | Peregrine |
| conv1d_1x32x128_k3     |      20.5 |     58.4 |     27.6 |      501.4 |        — |     73.7 | Peregrine |
| avgpool2d_1x16x32x32   |      25.1 |     41.0 |    261.8 |       62.1 |        — |     42.1 | Peregrine |
| groupnorm_4x64x16x16   |      21.3 |     54.2 |    221.2 |      732.5 |        — |    276.2 | Peregrine |
| rnn_seq32_128_256      |     197.5 |    266.0 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1048.6 |    796.7 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     842.0 |    772.3 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     804.0 |   1234.2 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     934.4 |   1117.1 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     912.2 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1277.9 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      60.2 |    257.5 |    477.9 |      119.3 |   2329.0 |    515.2 | Peregrine |
| rand_normal_100k       |     236.5 |    970.1 |    681.9 |      314.4 |   3223.8 |    593.5 | Peregrine |
| rand_bernoulli_100k    |     118.4 |    250.0 |    445.4 |      206.7 |        — |    504.8 | Peregrine |
| rand_uniform_1M        |     600.5 |   2562.0 |   4536.3 |      408.7 |   2341.8 |   2217.0 | TensorFlow |
| rand_normal_1M         |    2367.6 |   9697.3 |   6577.4 |     2045.4 |   3238.8 |   2834.2 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     27.8 |       42.9 |        — |     61.4 | Peregrine |
| rfft_4k                |       6.5 |     14.7 |     36.7 |       53.6 |        — |     67.7 | Peregrine |
| rfft_16k               |      30.4 |     65.2 |     77.3 |      104.8 |        — |    119.7 | Peregrine |
| fft_1k                 |       3.3 |      6.6 |     24.3 |        8.6 |        — |     18.0 | Peregrine |
| fft_4k                 |      12.2 |     26.0 |     41.1 |       17.4 |        — |     55.1 | Peregrine |
| norm_l2_1k             |       1.1 |      1.2 |     21.0 |       69.3 |        — |      3.8 | Peregrine |
| solve_64x64            |      12.0 |     24.4 |     94.7 |       24.4 |        — |     32.4 | Peregrine |
| inv_64x64              |      37.3 |     26.3 |     47.0 |       32.4 |        — |     37.2 | PyTorch |
| cholesky_64x64         |       6.4 |     47.1 |     21.7 |       19.5 |        — |     20.1 | Peregrine |
| svd_64x64              |     712.2 |    275.0 |    286.8 |      499.1 |        — |    298.3 | PyTorch |
| qr_64x64               |      41.3 |     79.8 |     54.7 |       83.7 |        — |     61.6 | Peregrine |
| eigh_64x64             |     376.4 |    214.3 |    227.2 |      141.0 |        — |    236.1 | TensorFlow |
| det_64x64              |      19.1 |     20.1 |        — |       22.8 |        — |     28.6 | Peregrine |
| solve_128x128          |      50.1 |     44.9 |    184.4 |       76.5 |        — |     84.3 | PyTorch |
| inv_128x128            |      92.7 |     62.0 |     86.4 |      138.8 |        — |     82.6 | PyTorch |
| cholesky_128x128       |      35.5 |     53.0 |     26.3 |       58.5 |        — |     36.0 | MLX |
| svd_128x128            |     985.1 |    988.9 |    997.1 |     1787.1 |        — |   1004.3 | Peregrine |
| qr_128x128             |     188.0 |    208.9 |    192.1 |      326.4 |        — |    189.4 | Peregrine |
| eigh_128x128           |    1823.5 |    702.1 |    709.9 |      701.9 |        — |    735.8 | TensorFlow |
| det_128x128            |      41.1 |     49.8 |        — |       82.1 |        — |     76.1 | Peregrine |
| solve_256x256          |     188.5 |    175.4 |    723.3 |      373.1 |        — |    265.9 | PyTorch |
| inv_256x256            |     470.9 |    278.2 |    249.2 |      846.5 |        — |    332.1 | MLX |
| cholesky_256x256       |     145.0 |     74.0 |     56.5 |      281.4 |        — |    116.9 | MLX |
| svd_256x256            |    5870.5 |   5678.7 |   5647.8 |     8037.6 |        — |   5763.2 | MLX |
| qr_256x256             |     976.6 |    957.6 |    961.4 |     1689.7 |        — |    957.6 | PyTorch |
| eigh_256x256           |    5882.1 |   3403.7 |   3374.2 |     4551.4 |        — |   3529.0 | MLX |
| det_256x256            |     140.7 |    202.6 |        — |      430.4 |        — |    204.8 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1807.8 |    820.9 |        — |     2315.3 |   1216.3 |   2084.7 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3184.5 |   1882.5 |        — |     3577.7 |   1227.2 |   3320.0 | tinygrad |
| add_layernorm_196x768  |     105.8 |     99.8 |        — |     1162.0 |   1098.9 |    217.2 | PyTorch |
| add_layernorm_196x1024 |     148.8 |    103.8 |        — |     1223.5 |   1112.7 |    279.2 | PyTorch |
| matmul_f32_196x768x3072 |     548.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14395.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1394.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   25934.7 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.53x** (Peregrine is faster)
- **Peregrine vs MLX: 0.39x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.29x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.38x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 99/141 ops
- PyTorch: 19/141 ops
- TensorFlow: 9/141 ops
- JAX: 7/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
