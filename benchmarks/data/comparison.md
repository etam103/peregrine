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
| matmul_128x128         |      13.4 |      6.2 |     23.4 |       91.7 |    420.2 |     78.0 | PyTorch |
| matmul_256x256         |      59.1 |     31.9 |     47.0 |      187.4 |    425.8 |    150.2 | PyTorch |
| matmul_512x512         |     218.6 |    139.6 |    167.4 |      666.9 |    427.3 |    512.6 | PyTorch |
| matmul_1024x1024       |    1058.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9489.4 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.6 |     39.8 |     31.5 |       47.8 |    186.6 |     37.2 | Peregrine |
| add_500k               |      61.7 |     56.6 |     81.1 |       81.9 |    186.0 |     61.1 | PyTorch |
| add_1M                 |     129.3 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     534.6 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     856.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     38.4 |     30.7 |       45.5 |    187.6 |     34.8 | Peregrine |
| mul_500k               |      61.7 |     58.6 |     81.6 |       74.7 |    193.5 |     62.0 | PyTorch |
| mul_1M                 |     127.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     519.1 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     959.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      48.9 |     65.7 |     62.1 |       62.5 |    273.6 |     46.0 | JAX |
| exp_500k               |     244.7 |    138.5 |    222.8 |      100.9 |    220.5 |    122.3 | TensorFlow |
| exp_1M                 |     489.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    2457.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    4909.9 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.6 |     40.2 |     28.6 |       38.6 |    336.5 |     98.1 | Peregrine |
| relu_1M                |      83.8 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     34.4 |     17.4 |       10.2 |    614.3 |     30.6 | Peregrine |
| softmax_8x512          |       4.2 |     35.4 |     18.8 |       13.2 |    624.5 |     34.7 | Peregrine |
| mlp_fwd_64x784         |      34.5 |     27.4 |     50.4 |      226.2 |   1748.7 |    163.7 | PyTorch |
| mlp_fwd_256x784_wide   |     431.7 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     807.9 |   1262.2 |    761.4 |     8098.9 |  22597.3 |   5011.6 | MLX |
| train_step_256_wide    |    3329.2 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.7 |     39.8 |     27.8 |       47.1 |    163.7 |     29.3 | Peregrine |
| square_100k            |       8.7 |     39.5 |     23.9 |       15.1 |    176.8 |     28.2 | Peregrine |
| rsqrt_100k             |      21.6 |     43.2 |     32.6 |       52.4 |        — |     93.3 | Peregrine |
| floor_100k             |       8.7 |     38.4 |     26.2 |       14.7 |    412.4 |     28.3 | Peregrine |
| ceil_100k              |       8.7 |     41.0 |     23.8 |       14.7 |    347.4 |     28.7 | Peregrine |
| round_100k             |       8.7 |     41.5 |     25.0 |       41.4 |        — |     27.5 | Peregrine |
| sign_100k              |       8.7 |     41.4 |     29.2 |       47.5 |    798.2 |     36.1 | Peregrine |
| expm1_100k             |      63.2 |    109.7 |    105.2 |      150.9 |        — |     99.1 | Peregrine |
| log2_100k              |      55.7 |     85.4 |     96.8 |      146.6 |    165.6 |     56.2 | Peregrine |
| log10_100k             |      58.1 |     84.9 |    106.1 |      134.8 |        — |     56.5 | JAX |
| log1p_100k             |      75.6 |     81.1 |    127.4 |       92.5 |        — |    105.0 | Peregrine |
| erf_100k               |     101.2 |     56.7 |    100.9 |       53.7 |        — |     44.3 | JAX |
| sinh_100k              |      51.2 |    122.8 |     93.3 |      130.0 |    527.2 |    119.2 | Peregrine |
| cosh_100k              |      46.5 |    128.4 |     89.8 |      125.1 |    453.8 |     70.0 | Peregrine |
| arcsin_100k            |      52.2 |     76.6 |     93.7 |       52.0 |   3063.9 |    111.5 | TensorFlow |
| arccos_100k            |      60.8 |     88.1 |    110.5 |       49.7 |        — |    193.6 | TensorFlow |
| arctan_100k            |      53.2 |     92.4 |     93.5 |       56.9 |   2998.0 |    210.7 | Peregrine |
| arcsinh_100k           |     205.2 |    158.9 |    332.6 |      128.8 |        — |    114.5 | JAX |
| maximum_100k           |      12.5 |     40.2 |     27.4 |       42.0 |    190.3 |     33.1 | Peregrine |
| minimum_100k           |      12.5 |     36.9 |     27.4 |       40.7 |    375.2 |     31.0 | Peregrine |
| power_100k             |     153.9 |    244.9 |    210.7 |      272.1 |        — |    140.9 | JAX |
| arctan2_100k           |      95.2 |    138.7 |    144.2 |       68.0 |        — |    310.9 | TensorFlow |
| logaddexp_100k         |     272.5 |    153.5 |    253.8 |      344.0 |        — |    142.7 | JAX |
| clip_100k              |       8.7 |     36.8 |     34.7 |       40.0 |    523.8 |     33.8 | Peregrine |
| where_100k             |      16.4 |     50.4 |     28.5 |       64.9 |    275.2 |     32.5 | Peregrine |
| greater_100k           |      12.5 |     49.2 |     24.9 |       52.7 |    188.8 |     27.9 | Peregrine |
| equal_100k             |      12.5 |     28.4 |     23.9 |       56.3 |    288.8 |     28.9 | Peregrine |
| sum_axis_256x512       |      18.8 |     41.4 |     22.6 |       47.5 |    205.1 |     48.0 | Peregrine |
| mean_axis_256x512      |      18.9 |     44.7 |     24.8 |       49.7 |    290.0 |     44.8 | Peregrine |
| max_axis_256x512       |      13.8 |     59.6 |     40.8 |       50.3 |    201.8 |     45.4 | Peregrine |
| min_axis_256x512       |      13.7 |     53.8 |     41.4 |       45.9 |    334.2 |     47.0 | Peregrine |
| var_256x512            |      45.7 |    275.7 |     57.3 |      210.1 |        — |     82.4 | Peregrine |
| prod_axis_256x512      |      24.1 |     40.0 |     25.8 |       46.0 |        — |     54.9 | Peregrine |
| logsumexp_256x512      |      95.5 |    195.2 |    106.2 |      322.0 |        — |    273.6 | Peregrine |
| cumsum_256x512         |     117.3 |     80.0 |    127.9 |      180.8 |    611.2 |    217.3 | PyTorch |
| argmax_axis_256x512    |      52.3 |     92.4 |    170.4 |       69.9 |   1296.0 |    170.5 | Peregrine |
| sum_axis_1024x1024     |     174.7 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     428.9 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.8 |     39.5 |     56.1 |       53.0 |   1832.0 |     37.9 | Peregrine |
| triu_256x256           |       7.7 |     39.7 |     55.8 |       49.4 |   1839.4 |     37.2 | Peregrine |
| repeat_64x128_2x3      |       6.0 |     46.6 |     31.7 |       74.0 |        — |     28.1 | Peregrine |
| pad_64x128             |       2.5 |      4.1 |     19.4 |       80.8 |     90.5 |     17.9 | Peregrine |
| stack_8x64x128         |       3.8 |      8.7 |     50.7 |       55.8 |    922.0 |    158.3 | Peregrine |
| diagonal_512x512       |       0.8 |      0.6 |     33.5 |       11.0 |        — |      7.9 | PyTorch |
| silu_100k              |      64.5 |     77.0 |     89.5 |      223.5 |    328.6 |     52.0 | JAX |
| softplus_100k          |     181.2 |    152.8 |    264.3 |      129.6 |    765.8 |    155.3 | TensorFlow |
| mish_100k              |     286.9 |    311.4 |    376.9 |      240.3 |   1151.2 |    240.5 | TensorFlow |
| leaky_relu_100k        |       8.7 |     42.4 |     79.2 |       18.2 |        — |     35.1 | Peregrine |
| elu_100k               |      60.7 |    137.9 |    128.3 |      126.5 |    854.4 |     77.8 | Peregrine |
| hard_tanh_100k         |       8.7 |     45.9 |     34.9 |       39.9 |        — |     39.0 | Peregrine |
| relu6_100k             |       8.7 |     39.2 |     44.0 |       49.1 |    743.8 |    113.0 | Peregrine |
| hardswish_100k         |      10.0 |     40.4 |     69.0 |      199.1 |        — |     27.0 | Peregrine |
| gelu_100k              |      95.4 |     66.6 |    141.8 |      226.0 |    841.1 |    208.6 | PyTorch |
| selu_100k              |      64.1 |    123.3 |     86.2 |      123.0 |    737.5 |     82.2 | Peregrine |
| softsign_100k          |      38.6 |    121.4 |     42.4 |       44.6 |        — |     55.9 | Peregrine |
| cross_entropy_64x10    |       2.6 |     38.1 |     22.8 |      578.2 |   3281.2 |     56.4 | Peregrine |
| l1_loss_64x10          |       0.9 |      5.3 |     19.5 |       37.9 |   1108.7 |     12.4 | Peregrine |
| mse_loss_64x10         |       3.7 |      4.7 |     21.7 |       34.2 |    443.3 |     23.9 | Peregrine |
| huber_loss_64x10       |       5.0 |      4.7 |     33.2 |      219.0 |        — |     47.3 | PyTorch |
| smooth_l1_loss_64x10   |       4.9 |      4.9 |     33.6 |      214.2 |        — |     47.8 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     17.7 |      348.1 |        — |     57.9 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.1 |    110.2 |      215.3 |        — |     69.1 | Peregrine |
| rmsnorm_64x512         |      58.2 |     66.0 |     33.2 |      430.8 |        — |     68.5 | MLX |
| conv1d_1x32x128_k3     |      20.1 |     53.3 |     28.1 |      486.7 |        — |     74.7 | Peregrine |
| avgpool2d_1x16x32x32   |      25.2 |     45.0 |    261.3 |       58.1 |        — |     42.2 | Peregrine |
| groupnorm_4x64x16x16   |      72.8 |     52.0 |    222.3 |      723.4 |        — |    262.1 | PyTorch |
| rnn_seq32_128_256      |     195.0 |    267.5 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1120.9 |    799.4 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     827.9 |    772.4 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     808.0 |   1270.5 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     925.1 |   1093.9 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     916.3 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1288.6 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      60.2 |    257.5 |    479.4 |      117.8 |   2379.2 |    532.7 | Peregrine |
| rand_normal_100k       |     237.2 |    971.4 |    685.0 |      323.9 |   3227.7 |    608.8 | Peregrine |
| rand_bernoulli_100k    |     119.3 |    250.5 |    445.1 |      205.2 |        — |    523.0 | Peregrine |
| rand_uniform_1M        |     604.4 |   2567.8 |   4534.2 |      414.2 |   2452.2 |   2273.5 | TensorFlow |
| rand_normal_1M         |    2368.6 |   9701.4 |   6584.7 |     2059.5 |   3246.5 |   2904.0 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     24.3 |       38.1 |        — |     61.3 | Peregrine |
| rfft_4k                |       7.5 |     14.8 |     32.2 |       49.0 |        — |     69.0 | Peregrine |
| rfft_16k               |      30.4 |     65.5 |     80.9 |      100.5 |        — |    116.2 | Peregrine |
| fft_1k                 |       3.3 |      6.5 |     25.4 |        7.5 |        — |     17.4 | Peregrine |
| fft_4k                 |      12.2 |     26.1 |     42.7 |       15.9 |        — |     57.7 | Peregrine |
| norm_l2_1k             |       1.1 |      1.2 |     20.5 |       62.5 |        — |      3.9 | Peregrine |
| solve_64x64            |      12.1 |     24.2 |     96.7 |       23.0 |        — |     32.1 | Peregrine |
| inv_64x64              |      37.5 |     26.6 |     47.4 |       31.0 |        — |     37.1 | PyTorch |
| cholesky_64x64         |       9.6 |     45.0 |     21.8 |       18.3 |        — |     20.5 | Peregrine |
| svd_64x64              |     275.6 |    279.9 |    287.5 |      483.0 |        — |    299.0 | Peregrine |
| qr_64x64               |      41.5 |     80.3 |     54.9 |       82.2 |        — |     64.2 | Peregrine |
| eigh_64x64             |     380.0 |    217.3 |    229.9 |      143.9 |        — |    237.5 | TensorFlow |
| det_64x64              |      23.3 |     19.9 |        — |       21.7 |        — |     28.9 | PyTorch |
| solve_128x128          |      50.2 |     44.9 |    186.5 |       75.0 |        — |     88.8 | PyTorch |
| inv_128x128            |      93.6 |     61.8 |     88.3 |      137.7 |        — |     83.3 | PyTorch |
| cholesky_128x128       |      50.8 |     50.6 |     27.8 |       57.3 |        — |     37.3 | MLX |
| svd_128x128            |     988.2 |    978.6 |    952.8 |     1815.3 |        — |   1021.0 | MLX |
| qr_128x128             |     189.8 |    219.8 |    191.2 |      324.5 |        — |    191.5 | Peregrine |
| eigh_128x128           |    1841.0 |    712.7 |    722.7 |      711.9 |        — |    737.0 | TensorFlow |
| det_128x128            |      52.4 |     50.0 |        — |       80.3 |        — |     76.9 | PyTorch |
| solve_256x256          |     189.5 |    178.4 |    724.4 |      375.2 |        — |    265.0 | PyTorch |
| inv_256x256            |     495.5 |    298.3 |    247.9 |      846.2 |        — |    344.9 | MLX |
| cholesky_256x256       |     226.7 |     75.4 |     54.0 |      280.1 |        — |    124.3 | MLX |
| svd_256x256            |    5897.1 |   5572.9 |   5629.7 |     7973.5 |        — |   5697.0 | PyTorch |
| qr_256x256             |     988.5 |    977.0 |   1005.6 |     1691.3 |        — |    990.5 | PyTorch |
| eigh_256x256           |    6005.7 |   3453.8 |   3427.5 |     4551.0 |        — |   3588.6 | MLX |
| det_256x256            |     212.6 |    202.3 |        — |      429.0 |        — |    206.2 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1803.8 |    820.7 |        — |     2355.2 |   1219.7 |   2149.1 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3221.0 |   1895.2 |        — |     3668.8 |   1226.9 |   3353.6 | tinygrad |
| add_layernorm_196x768  |     106.0 |     98.3 |        — |     1190.7 |   1107.2 |    232.6 | PyTorch |
| add_layernorm_196x1024 |     140.8 |    103.6 |        — |     1231.1 |   1114.2 |    292.2 | PyTorch |
| matmul_f32_196x768x3072 |     534.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14460.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1421.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26022.7 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.57x** (Peregrine is faster)
- **Peregrine vs MLX: 0.41x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.32x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.40x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 91/141 ops
- PyTorch: 25/141 ops
- TensorFlow: 10/141 ops
- MLX: 7/141 ops
- JAX: 7/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
