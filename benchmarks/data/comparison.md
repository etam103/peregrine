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
| matmul_128x128         |      13.3 |      6.2 |     20.4 |       93.0 |    415.6 |     78.9 | PyTorch |
| matmul_256x256         |      58.9 |     31.9 |     47.1 |      196.3 |    426.1 |    148.0 | PyTorch |
| matmul_512x512         |     220.0 |    142.2 |    169.2 |      596.5 |    428.6 |    509.9 | PyTorch |
| matmul_1024x1024       |    1022.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9475.4 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     40.6 |     30.6 |       47.6 |    186.6 |     33.6 | Peregrine |
| add_500k               |      61.7 |     57.1 |     80.2 |       80.9 |    185.9 |     60.1 | PyTorch |
| add_1M                 |     124.5 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     512.9 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     912.0 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     41.2 |     28.2 |       45.9 |    186.5 |     31.0 | Peregrine |
| mul_500k               |      61.9 |     58.3 |     82.7 |       72.3 |    186.7 |     58.4 | PyTorch |
| mul_1M                 |     125.1 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     534.7 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     906.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.2 |     59.4 |     61.2 |       65.0 |    216.9 |     46.0 | JAX |
| exp_500k               |     246.2 |    140.2 |    223.1 |       98.1 |    217.7 |    121.2 | TensorFlow |
| exp_1M                 |     492.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1116.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2608.2 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     40.5 |     24.8 |       36.8 |    333.3 |     98.3 | Peregrine |
| relu_1M                |      85.1 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     34.0 |     16.0 |       10.3 |    607.6 |     30.3 | Peregrine |
| softmax_8x512          |       4.3 |     33.7 |     18.3 |       13.2 |    618.7 |     33.5 | Peregrine |
| mlp_fwd_64x784         |      33.0 |     28.7 |     51.6 |      235.2 |   1791.3 |    181.2 | PyTorch |
| mlp_fwd_256x784_wide   |     444.0 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     802.4 |   1286.5 |    773.1 |     8146.6 |  24148.4 |   5105.2 | MLX |
| train_step_256_wide    |    3616.4 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       9.4 |     35.9 |     24.6 |       46.8 |    164.7 |     31.2 | Peregrine |
| square_100k            |       9.4 |     38.4 |     23.6 |       13.4 |    176.6 |     29.2 | Peregrine |
| rsqrt_100k             |      22.9 |     40.4 |     36.3 |       48.2 |        — |     93.0 | Peregrine |
| floor_100k             |       9.4 |     43.0 |     23.8 |       15.2 |    409.7 |     29.0 | Peregrine |
| ceil_100k              |       9.4 |     44.6 |     23.6 |       15.1 |    351.9 |     28.3 | Peregrine |
| round_100k             |       8.7 |     45.6 |     23.7 |       41.4 |        — |     28.5 | Peregrine |
| sign_100k              |       8.7 |     41.3 |     27.6 |       45.8 |    796.7 |     35.4 | Peregrine |
| expm1_100k             |      65.9 |    109.3 |    107.7 |      140.3 |        — |     98.6 | Peregrine |
| log2_100k              |      56.2 |     86.6 |    101.9 |      147.4 |    164.1 |     56.1 | JAX |
| log10_100k             |      59.2 |     85.3 |    106.2 |      138.8 |        — |     56.3 | JAX |
| log1p_100k             |      77.0 |     86.9 |    127.5 |       94.5 |        — |    104.3 | Peregrine |
| erf_100k               |     103.1 |     55.5 |    100.6 |       54.2 |        — |     49.7 | JAX |
| sinh_100k              |      51.5 |    136.1 |     93.3 |      130.9 |    524.6 |    116.9 | Peregrine |
| cosh_100k              |      46.4 |    131.5 |     89.5 |      120.6 |    459.8 |     68.9 | Peregrine |
| arcsin_100k            |      52.2 |     77.3 |     93.6 |       54.6 |   2891.2 |    111.2 | Peregrine |
| arccos_100k            |      60.8 |     88.6 |    110.4 |       53.6 |        — |    199.7 | TensorFlow |
| arctan_100k            |      53.3 |     91.0 |     92.8 |       56.3 |   3000.1 |    213.6 | Peregrine |
| arcsinh_100k           |     205.0 |    164.6 |    331.5 |      132.4 |        — |    120.1 | JAX |
| maximum_100k           |      12.8 |     39.7 |     28.2 |       41.8 |    188.9 |     34.6 | Peregrine |
| minimum_100k           |      12.5 |     38.5 |     27.1 |       38.5 |    373.0 |     30.8 | Peregrine |
| power_100k             |     153.9 |    240.0 |    210.4 |      266.6 |        — |    140.0 | JAX |
| arctan2_100k           |      95.2 |    133.5 |    143.8 |       70.4 |        — |    312.8 | TensorFlow |
| logaddexp_100k         |     278.0 |    158.2 |    259.0 |      357.6 |        — |    147.5 | JAX |
| clip_100k              |       8.9 |     40.6 |     34.9 |       39.6 |    547.4 |     41.5 | Peregrine |
| where_100k             |      17.0 |     50.2 |     27.1 |       64.6 |    279.8 |     32.1 | Peregrine |
| greater_100k           |      13.0 |     46.5 |     25.6 |       50.0 |    190.2 |     26.2 | Peregrine |
| equal_100k             |      13.0 |     28.0 |     24.4 |       60.4 |    286.6 |     27.7 | Peregrine |
| sum_axis_256x512       |      19.4 |     39.8 |     24.8 |       49.5 |    204.0 |     52.9 | Peregrine |
| mean_axis_256x512      |      20.0 |     41.4 |     25.1 |       51.5 |    300.1 |     46.3 | Peregrine |
| max_axis_256x512       |      14.5 |     55.9 |     41.5 |       52.1 |    212.9 |     46.2 | Peregrine |
| min_axis_256x512       |      14.1 |     53.4 |     41.3 |       47.5 |    326.0 |     47.5 | Peregrine |
| var_256x512            |      47.2 |    272.8 |     59.8 |      207.8 |        — |     78.9 | Peregrine |
| prod_axis_256x512      |      24.8 |     37.5 |     26.1 |       47.6 |        — |     53.4 | Peregrine |
| logsumexp_256x512      |      95.6 |    198.8 |    106.7 |      318.2 |        — |    280.8 | Peregrine |
| cumsum_256x512         |     120.8 |     76.7 |    128.2 |      192.8 |    617.2 |    207.7 | PyTorch |
| argmax_axis_256x512    |      51.9 |     91.6 |    170.3 |       69.0 |   1269.7 |    171.2 | Peregrine |
| sum_axis_1024x1024     |     174.2 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.8 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      43.3 |     37.5 |     54.9 |       52.1 |   1797.2 |     36.4 | JAX |
| triu_256x256           |      34.7 |     38.4 |     55.4 |       51.6 |   1802.0 |     36.4 | Peregrine |
| repeat_64x128_2x3      |       6.1 |     45.7 |     29.9 |       73.4 |        — |     27.9 | Peregrine |
| pad_64x128             |       2.6 |      4.4 |     18.8 |       79.2 |     89.9 |     17.9 | Peregrine |
| stack_8x64x128         |      15.7 |      8.8 |     44.6 |       50.0 |    914.2 |    157.3 | PyTorch |
| diagonal_512x512       |       0.8 |      0.7 |     29.1 |       10.9 |        — |      9.1 | PyTorch |
| silu_100k              |      64.0 |     73.6 |     89.9 |      226.5 |    328.8 |     52.7 | JAX |
| softplus_100k          |     180.8 |    153.8 |    260.5 |      125.2 |    772.9 |    155.2 | TensorFlow |
| mish_100k              |     286.2 |    308.0 |    370.6 |      246.7 |   1140.3 |    231.6 | JAX |
| leaky_relu_100k        |       8.7 |     38.5 |     74.9 |       18.0 |        — |     36.2 | Peregrine |
| elu_100k               |      60.0 |    134.1 |    116.3 |      130.9 |    862.6 |     77.4 | Peregrine |
| hard_tanh_100k         |       8.7 |     41.5 |     34.4 |       39.8 |        — |     36.1 | Peregrine |
| relu6_100k             |       8.7 |     41.8 |     43.5 |       50.2 |    738.2 |    110.5 | Peregrine |
| hardswish_100k         |      10.0 |     40.8 |     65.5 |      211.1 |        — |     28.1 | Peregrine |
| gelu_100k              |      95.3 |     75.0 |    135.2 |      237.6 |    846.9 |    211.2 | PyTorch |
| selu_100k              |      63.7 |    132.2 |     85.4 |      132.8 |    747.7 |     82.8 | Peregrine |
| softsign_100k          |      38.0 |    121.0 |     43.8 |       46.6 |        — |     54.9 | Peregrine |
| cross_entropy_64x10    |       2.6 |     40.0 |     22.6 |      573.2 |   3345.8 |     53.9 | Peregrine |
| l1_loss_64x10          |       0.9 |      5.5 |     18.4 |       38.0 |   1115.4 |     12.0 | Peregrine |
| mse_loss_64x10         |       3.6 |      5.0 |     21.1 |       34.2 |    448.1 |     23.6 | Peregrine |
| huber_loss_64x10       |       5.0 |      5.0 |     33.7 |      216.7 |        — |     47.2 | Peregrine |
| smooth_l1_loss_64x10   |       4.8 |      5.3 |     32.8 |      213.8 |        — |     47.4 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.6 |     17.6 |      348.5 |        — |     63.0 | Peregrine |
| cosine_sim_loss_64x64  |      13.5 |     11.0 |    108.5 |      215.8 |        — |     71.5 | PyTorch |
| rmsnorm_64x512         |      57.5 |     65.9 |     33.0 |      431.1 |        — |     71.2 | MLX |
| conv1d_1x32x128_k3     |      20.4 |     52.9 |     27.8 |      487.8 |        — |     74.8 | Peregrine |
| avgpool2d_1x16x32x32   |      25.1 |     45.8 |    263.0 |       62.9 |        — |     42.5 | Peregrine |
| groupnorm_4x64x16x16   |      72.7 |     53.3 |    223.4 |      712.4 |        — |    264.0 | PyTorch |
| rnn_seq32_128_256      |     195.5 |    286.6 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1124.7 |    850.4 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     856.1 |    822.4 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     807.6 |   1278.3 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     932.2 |   1108.5 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     913.5 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1276.4 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     106.2 |    257.7 |    480.5 |      120.2 |   2365.4 |    530.5 | Peregrine |
| rand_normal_100k       |     236.4 |    972.3 |    686.0 |      332.0 |   3226.4 |    614.6 | Peregrine |
| rand_bernoulli_100k    |     303.4 |    250.2 |    448.4 |      207.8 |        — |    529.7 | TensorFlow |
| rand_uniform_1M        |    1064.2 |   2564.6 |   4535.9 |      414.3 |   2372.4 |   2331.8 | TensorFlow |
| rand_normal_1M         |    2371.3 |   9700.1 |   6577.8 |     2046.6 |   3260.6 |   2838.6 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     25.9 |       39.2 |        — |     61.0 | Peregrine |
| rfft_4k                |       7.4 |     14.8 |     32.8 |       49.8 |        — |     70.0 | Peregrine |
| rfft_16k               |      30.3 |     65.5 |     80.1 |      100.1 |        — |    116.5 | Peregrine |
| fft_1k                 |       3.3 |      6.7 |     29.0 |        7.7 |        — |     18.2 | Peregrine |
| fft_4k                 |      12.2 |     26.2 |     43.6 |       16.0 |        — |     63.4 | Peregrine |
| norm_l2_1k             |       1.1 |      1.4 |     19.8 |       62.2 |        — |      3.9 | Peregrine |
| solve_64x64            |      12.0 |     25.5 |    101.1 |       23.0 |        — |     32.7 | Peregrine |
| inv_64x64              |      37.4 |     27.0 |     51.9 |       31.0 |        — |     37.0 | PyTorch |
| cholesky_64x64         |       9.7 |     45.1 |     21.5 |       18.1 |        — |     19.7 | Peregrine |
| svd_64x64              |     277.4 |    279.2 |    291.7 |      504.8 |        — |    300.3 | Peregrine |
| qr_64x64               |      41.2 |     81.0 |     58.5 |       82.0 |        — |     63.5 | Peregrine |
| eigh_64x64             |     379.7 |    218.7 |    231.5 |      144.0 |        — |    237.7 | TensorFlow |
| det_64x64              |      23.2 |     20.7 |        — |       21.7 |        — |     28.5 | PyTorch |
| solve_128x128          |      50.0 |     45.7 |    187.8 |       74.9 |        — |     85.0 | PyTorch |
| inv_128x128            |      91.8 |     62.8 |     90.6 |      137.0 |        — |     82.9 | PyTorch |
| cholesky_128x128       |      50.2 |     49.2 |     26.6 |       57.0 |        — |     35.6 | MLX |
| svd_128x128            |     983.3 |    989.2 |    957.9 |     1791.0 |        — |   1010.2 | MLX |
| qr_128x128             |     188.2 |    228.6 |    190.1 |      323.6 |        — |    190.3 | Peregrine |
| eigh_128x128           |    1840.3 |    703.2 |    718.4 |      711.8 |        — |    741.0 | PyTorch |
| det_128x128            |      52.0 |     50.1 |        — |       80.6 |        — |     76.3 | PyTorch |
| solve_256x256          |     188.5 |    178.6 |    723.7 |      375.2 |        — |    265.0 | PyTorch |
| inv_256x256            |     470.1 |    290.1 |    248.3 |      845.0 |        — |    334.1 | MLX |
| cholesky_256x256       |     226.6 |     82.0 |     55.5 |      279.8 |        — |    116.8 | MLX |
| svd_256x256            |    5862.5 |   5578.0 |   5719.2 |     8057.8 |        — |   5897.7 | PyTorch |
| qr_256x256             |     995.9 |    999.5 |   1004.9 |     1691.2 |        — |    978.4 | JAX |
| eigh_256x256           |    5988.4 |   3470.5 |   3473.4 |     4577.3 |        — |   3579.9 | PyTorch |
| det_256x256            |     212.7 |    200.9 |        — |      426.9 |        — |    205.6 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1485.0 |    866.7 |        — |     2352.5 |   1221.7 |   2118.5 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2601.7 |   1892.5 |        — |     3628.7 |   1229.4 |   3386.1 | tinygrad |
| add_layernorm_196x768  |     105.4 |    104.1 |        — |     1152.0 |   1111.2 |    235.1 | PyTorch |
| add_layernorm_196x1024 |     137.1 |    107.6 |        — |     1245.2 |   1129.1 |    283.2 | PyTorch |
| matmul_f32_196x768x3072 |     550.0 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14475.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1570.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26415.8 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.61x** (Peregrine is faster)
- **Peregrine vs MLX: 0.46x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.35x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.06x** (Peregrine is faster)
- **Peregrine vs JAX: 0.43x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 88/141 ops
- PyTorch: 27/141 ops
- JAX: 11/141 ops
- TensorFlow: 8/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
