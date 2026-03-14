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
| matmul_128x128         |      31.8 |      6.1 |     20.1 |       48.7 |    414.8 |     59.5 | PyTorch |
| matmul_256x256         |     169.2 |     31.7 |     43.1 |      172.1 |    435.2 |    149.9 | PyTorch |
| matmul_512x512         |     242.0 |    128.7 |    153.5 |      615.0 |    440.0 |    496.2 | PyTorch |
| matmul_1024x1024       |     971.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9380.6 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     38.8 |     32.1 |       50.1 |    189.7 |     36.3 | Peregrine |
| add_500k               |      63.0 |     59.5 |     76.7 |       74.8 |    191.0 |     79.7 | PyTorch |
| add_1M                 |     133.6 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     500.8 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     847.3 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     42.2 |     31.3 |       41.6 |    190.1 |     31.6 | Peregrine |
| mul_500k               |      62.7 |     56.1 |     76.6 |       77.2 |    198.0 |     65.0 | PyTorch |
| mul_1M                 |     134.4 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     636.1 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     926.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.2 |     64.5 |     55.2 |       63.9 |    227.0 |     30.5 | JAX |
| exp_500k               |     140.8 |    136.7 |    223.3 |       99.7 |    226.8 |    124.1 | TensorFlow |
| exp_1M                 |     158.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     429.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     818.1 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     39.7 |     27.6 |       37.7 |    346.6 |    100.4 | Peregrine |
| relu_1M                |      83.8 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.1 |     32.9 |     18.8 |       10.5 |    635.5 |     33.5 | Peregrine |
| softmax_8x512          |       4.2 |     34.8 |     16.3 |       13.1 |    626.4 |     39.3 | Peregrine |
| mlp_fwd_64x784         |      31.8 |     27.7 |     44.5 |      231.3 |   1771.2 |    170.7 | PyTorch |
| mlp_fwd_256x784_wide   |     391.9 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     812.8 |   1209.7 |    758.3 |     8340.0 |  22762.0 |   4956.2 | MLX |
| train_step_256_wide    |    3298.4 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       9.3 |     40.0 |     29.4 |       46.6 |    161.0 |     30.9 | Peregrine |
| square_100k            |       9.7 |     36.6 |     25.1 |       15.0 |    173.8 |     27.2 | Peregrine |
| rsqrt_100k             |      23.4 |     42.0 |     32.8 |       49.9 |        — |     90.9 | Peregrine |
| floor_100k             |       8.8 |     39.8 |     23.4 |       16.9 |    408.2 |     29.0 | Peregrine |
| ceil_100k              |       8.8 |     41.4 |     23.2 |       16.5 |    348.7 |     29.6 | Peregrine |
| round_100k             |       8.8 |     42.4 |     23.3 |       38.9 |        — |     28.8 | Peregrine |
| sign_100k              |       8.8 |     42.5 |     27.2 |       45.9 |    787.4 |     37.5 | Peregrine |
| expm1_100k             |      64.3 |    108.5 |    107.3 |      143.6 |        — |     99.3 | Peregrine |
| log2_100k              |      56.6 |     86.5 |    101.3 |      157.6 |    160.2 |     56.0 | JAX |
| log10_100k             |      59.0 |     86.5 |    106.8 |      142.7 |        — |     56.0 | JAX |
| log1p_100k             |      76.8 |     81.3 |    130.2 |       95.5 |        — |    104.5 | Peregrine |
| erf_100k               |     102.6 |     56.8 |    100.4 |       55.9 |        — |     42.5 | JAX |
| sinh_100k              |      52.0 |    127.2 |     93.4 |      122.4 |    526.4 |    107.2 | Peregrine |
| cosh_100k              |      47.2 |    125.4 |     89.7 |      126.1 |    461.2 |     69.1 | Peregrine |
| arcsin_100k            |      53.0 |     80.4 |     92.0 |       53.1 |   2861.2 |    111.3 | Peregrine |
| arccos_100k            |      61.8 |     89.2 |    107.4 |       51.5 |        — |    190.2 | TensorFlow |
| arctan_100k            |      54.1 |     93.9 |     92.9 |       55.2 |   2963.4 |    210.6 | Peregrine |
| arcsinh_100k           |     208.8 |    151.8 |    331.9 |      135.5 |        — |    109.2 | JAX |
| maximum_100k           |      12.7 |     38.9 |     26.9 |       39.4 |    185.7 |     32.9 | Peregrine |
| minimum_100k           |      12.8 |     38.1 |     23.4 |       43.4 |    372.3 |     30.7 | Peregrine |
| power_100k             |     156.6 |    231.4 |    220.7 |      279.9 |        — |    139.0 | JAX |
| arctan2_100k           |      96.9 |    132.3 |    147.5 |       67.7 |        — |    315.9 | TensorFlow |
| logaddexp_100k         |     277.6 |    150.7 |    265.9 |      369.4 |        — |    151.4 | PyTorch |
| clip_100k              |       8.8 |     39.8 |     35.6 |       40.1 |    529.4 |     45.2 | Peregrine |
| where_100k             |      16.7 |     49.3 |     29.9 |       64.7 |    273.1 |     35.2 | Peregrine |
| greater_100k           |      12.7 |     47.7 |     21.8 |       49.1 |    187.1 |     27.8 | Peregrine |
| equal_100k             |      12.7 |     26.9 |     21.1 |       51.6 |    281.6 |     27.0 | Peregrine |
| sum_axis_256x512       |      19.2 |     41.2 |     19.8 |       48.5 |    202.4 |     50.1 | Peregrine |
| mean_axis_256x512      |      19.2 |     42.0 |     22.0 |       48.9 |    290.4 |     45.5 | Peregrine |
| max_axis_256x512       |      13.9 |     54.0 |     38.1 |       49.9 |    200.5 |     44.7 | Peregrine |
| min_axis_256x512       |      13.9 |     56.6 |     37.9 |       47.4 |    325.0 |     51.8 | Peregrine |
| var_256x512            |      45.8 |    273.0 |     53.5 |      208.2 |        — |     79.1 | Peregrine |
| prod_axis_256x512      |      24.2 |     37.9 |     21.1 |       48.2 |        — |     52.6 | MLX |
| logsumexp_256x512      |      95.5 |    196.7 |    107.2 |      316.5 |        — |    279.1 | Peregrine |
| cumsum_256x512         |     112.7 |     73.9 |    125.9 |      188.4 |    613.7 |    183.2 | PyTorch |
| argmax_axis_256x512    |      51.7 |     92.7 |    172.4 |       66.4 |   1273.6 |    161.6 | Peregrine |
| sum_axis_1024x1024     |     174.0 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.8 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.8 |     38.4 |     52.5 |       52.6 |   1793.4 |     36.4 | Peregrine |
| triu_256x256           |       7.6 |     35.4 |     50.8 |       53.4 |   1785.8 |     35.9 | Peregrine |
| repeat_64x128_2x3      |       7.4 |     48.1 |     31.8 |       72.4 |        — |     27.8 | Peregrine |
| pad_64x128             |       2.5 |      4.1 |     19.6 |       81.8 |     89.5 |     17.9 | Peregrine |
| stack_8x64x128         |       3.8 |      8.7 |     42.2 |       57.2 |    904.2 |    158.0 | Peregrine |
| diagonal_512x512       |       0.3 |      0.6 |     24.2 |       11.2 |        — |      9.2 | Peregrine |
| silu_100k              |      64.1 |     76.2 |     81.0 |      226.2 |    331.9 |     51.8 | JAX |
| softplus_100k          |     180.7 |    153.2 |    262.2 |      124.5 |    782.1 |    170.5 | TensorFlow |
| mish_100k              |     291.3 |    304.7 |    369.6 |      237.2 |   1169.7 |    229.2 | JAX |
| leaky_relu_100k        |       8.8 |     40.3 |     75.5 |       18.2 |        — |     29.2 | Peregrine |
| elu_100k               |      61.1 |    130.4 |    116.1 |      131.9 |    874.3 |     77.3 | Peregrine |
| hard_tanh_100k         |       9.0 |     41.2 |     34.6 |       40.8 |        — |     38.8 | Peregrine |
| relu6_100k             |       9.0 |     40.5 |     43.7 |       51.0 |    728.7 |    112.1 | Peregrine |
| hardswish_100k         |      10.3 |     40.4 |     69.4 |      211.1 |        — |     28.7 | Peregrine |
| gelu_100k              |      97.7 |     75.7 |    135.5 |      240.7 |    840.9 |    198.8 | PyTorch |
| selu_100k              |      64.9 |    123.0 |     85.5 |      131.3 |    742.7 |     81.7 | Peregrine |
| softsign_100k          |      38.9 |    119.2 |     46.7 |       47.2 |        — |     55.5 | Peregrine |
| cross_entropy_64x10    |       2.6 |     39.9 |     21.0 |      592.8 |   3471.2 |     55.4 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.5 |     17.0 |       38.2 |   1121.1 |     12.6 | Peregrine |
| mse_loss_64x10         |       3.7 |      5.0 |     19.9 |       34.4 |    443.5 |     23.6 | Peregrine |
| huber_loss_64x10       |       0.3 |      4.9 |     31.2 |      220.0 |        — |     47.5 | Peregrine |
| smooth_l1_loss_64x10   |       0.8 |      5.2 |     30.3 |      225.7 |        — |     48.0 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     17.5 |      359.9 |        — |     57.1 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.5 |    102.2 |      222.9 |        — |     49.4 | Peregrine |
| rmsnorm_64x512         |      19.5 |     66.3 |     27.9 |      432.0 |        — |     64.5 | Peregrine |
| conv1d_1x32x128_k3     |      20.1 |     53.3 |     27.0 |      500.3 |        — |     78.5 | Peregrine |
| avgpool2d_1x16x32x32   |      25.8 |     45.5 |    271.2 |       61.1 |        — |     43.0 | Peregrine |
| groupnorm_4x64x16x16   |      21.8 |     55.1 |    223.6 |      726.9 |        — |    274.6 | Peregrine |
| rnn_seq32_128_256      |     189.2 |    269.3 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |     994.5 |    821.3 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     753.7 |    785.0 |        — |          — |        — |        — | Peregrine |
| optim_adam_64          |     808.5 |   1258.9 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     928.9 |   1096.5 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     913.9 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1284.3 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      61.3 |    257.4 |    479.9 |      119.5 |   2393.5 |    563.3 | Peregrine |
| rand_normal_100k       |     241.0 |    972.4 |    686.9 |      333.2 |   3265.8 |    609.3 | Peregrine |
| rand_bernoulli_100k    |     118.4 |    250.0 |    449.5 |      207.0 |        — |    526.1 | Peregrine |
| rand_uniform_1M        |     600.5 |   2597.3 |   4611.3 |      427.5 |   2547.0 |   2235.8 | TensorFlow |
| rand_normal_1M         |    2412.7 |   9856.1 |   6615.1 |     2066.3 |   3445.8 |   2839.4 | TensorFlow |
| rfft_1k                |       2.0 |      4.4 |     19.6 |       39.5 |        — |     22.4 | Peregrine |
| rfft_4k                |       6.6 |     14.8 |     29.4 |       50.4 |        — |     67.9 | Peregrine |
| rfft_16k               |      29.2 |     65.7 |     74.9 |      102.6 |        — |    119.9 | Peregrine |
| fft_1k                 |       3.1 |      6.6 |     22.3 |        8.0 |        — |     18.5 | Peregrine |
| fft_4k                 |      12.0 |     26.2 |     38.9 |       16.5 |        — |     57.2 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     16.8 |       65.2 |        — |      4.0 | Peregrine |
| solve_64x64            |      11.8 |     23.8 |     87.6 |       23.6 |        — |     32.2 | Peregrine |
| inv_64x64              |      36.2 |     26.2 |     47.0 |       31.7 |        — |     37.0 | PyTorch |
| cholesky_64x64         |       6.0 |     43.6 |     21.3 |       18.9 |        — |     22.4 | Peregrine |
| svd_64x64              |     274.5 |    277.1 |    286.3 |      493.2 |        — |    306.2 | Peregrine |
| qr_64x64               |      41.0 |     82.8 |     55.8 |       81.8 |        — |     65.5 | Peregrine |
| eigh_64x64             |     376.3 |    217.6 |    231.3 |      141.2 |        — |    239.8 | TensorFlow |
| det_64x64              |      19.1 |     19.9 |        — |       22.2 |        — |     29.2 | Peregrine |
| solve_128x128          |      50.0 |     45.0 |    185.5 |       74.7 |        — |     85.6 | PyTorch |
| inv_128x128            |      92.3 |     62.7 |     86.8 |      140.0 |        — |     83.7 | PyTorch |
| cholesky_128x128       |      35.3 |     50.8 |     25.4 |       56.8 |        — |     35.7 | MLX |
| svd_128x128            |     983.5 |   1000.3 |    946.6 |     1900.4 |        — |   1011.6 | MLX |
| qr_128x128             |     186.2 |    218.7 |    192.6 |      325.1 |        — |    191.5 | Peregrine |
| eigh_128x128           |    1823.6 |    702.5 |    718.8 |      713.5 |        — |    794.1 | PyTorch |
| det_128x128            |      41.1 |     49.6 |        — |       82.0 |        — |     77.4 | Peregrine |
| solve_256x256          |     188.8 |    181.3 |    732.3 |      375.8 |        — |    273.5 | PyTorch |
| inv_256x256            |     463.8 |    290.5 |    252.3 |      860.2 |        — |    340.6 | MLX |
| cholesky_256x256       |     145.0 |     81.2 |     54.7 |      285.3 |        — |    127.6 | MLX |
| svd_256x256            |    6049.6 |   5629.7 |   5688.3 |     8206.2 |        — |   5699.8 | PyTorch |
| qr_256x256             |    1002.8 |    960.9 |    969.3 |     1721.8 |        — |    957.7 | JAX |
| eigh_256x256           |    5935.8 |   3474.4 |   3466.4 |     4603.9 |        — |   3479.5 | MLX |
| det_256x256            |     141.0 |    202.0 |        — |      439.2 |        — |    204.6 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1829.8 |    876.5 |        — |     2291.6 |   1290.7 |   2056.6 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3239.0 |   2109.1 |        — |     3643.8 |   1264.2 |   3856.6 | tinygrad |
| add_layernorm_196x768  |     107.7 |    101.7 |        — |     1113.9 |   1124.6 |    229.9 | PyTorch |
| add_layernorm_196x1024 |     143.2 |    107.3 |        — |     1194.8 |   1147.7 |    279.5 | PyTorch |
| matmul_f32_196x768x3072 |     617.6 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14649.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1435.0 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26371.6 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.53x** (Peregrine is faster)
- **Peregrine vs MLX: 0.40x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.30x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.37x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 98/141 ops
- PyTorch: 19/141 ops
- JAX: 9/141 ops
- MLX: 7/141 ops
- TensorFlow: 7/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
