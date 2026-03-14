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
| matmul_128x128         |       6.0 |      6.2 |     20.4 |       51.6 |    414.3 |     79.1 | Peregrine |
| matmul_256x256         |     131.1 |     31.8 |     41.8 |      192.5 |    425.1 |    161.2 | PyTorch |
| matmul_512x512         |     296.3 |    127.6 |    151.1 |      564.1 |    434.5 |    516.5 | PyTorch |
| matmul_1024x1024       |    1611.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |   11589.2 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.8 |     41.2 |     31.1 |       54.0 |    187.2 |     34.1 | Peregrine |
| add_500k               |      63.6 |     57.2 |     80.5 |       80.2 |    186.7 |     65.5 | PyTorch |
| add_1M                 |     167.6 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     587.0 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     926.1 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      14.0 |     39.5 |     30.7 |       41.4 |    189.9 |     27.7 | Peregrine |
| mul_500k               |      64.4 |     57.5 |     78.0 |       73.5 |    187.4 |     59.0 | PyTorch |
| mul_1M                 |     127.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     619.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |    1122.9 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      54.2 |     62.4 |     55.4 |       63.1 |    218.5 |     46.4 | JAX |
| exp_500k               |     268.5 |    139.0 |    222.2 |      107.6 |    220.2 |    116.9 | TensorFlow |
| exp_1M                 |     487.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    2444.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    5133.3 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.9 |     37.7 |     27.1 |       42.1 |    339.3 |    101.3 | Peregrine |
| relu_1M                |      82.6 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     35.4 |     14.2 |       11.6 |    613.6 |     31.4 | Peregrine |
| softmax_8x512          |       4.4 |     36.8 |     18.8 |       14.5 |    614.7 |     34.5 | Peregrine |
| mlp_fwd_64x784         |      34.4 |     27.8 |     44.8 |      248.3 |   1760.2 |    176.2 | PyTorch |
| mlp_fwd_256x784_wide   |     506.0 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     853.8 |   1243.0 |    760.7 |     8706.4 |  22939.1 |   5098.5 | MLX |
| train_step_256_wide    |    3535.3 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       9.7 |     41.9 |     27.1 |       50.8 |    162.5 |     33.1 | Peregrine |
| square_100k            |       9.0 |     43.2 |     23.1 |       16.5 |    176.4 |     33.0 | Peregrine |
| rsqrt_100k             |      21.5 |     46.3 |     31.5 |       47.5 |        — |     92.5 | Peregrine |
| floor_100k             |       8.7 |     45.2 |     23.2 |       16.4 |    405.4 |     36.1 | Peregrine |
| ceil_100k              |       8.7 |     42.7 |     23.1 |       16.2 |    346.8 |     28.1 | Peregrine |
| round_100k             |       8.7 |     45.4 |     24.7 |       43.3 |        — |     26.3 | Peregrine |
| sign_100k              |       8.7 |     43.5 |     28.5 |       46.9 |    787.2 |     36.2 | Peregrine |
| expm1_100k             |      63.2 |    115.0 |    105.4 |      148.8 |        — |     99.1 | Peregrine |
| log2_100k              |      55.6 |     85.9 |     99.0 |      154.2 |    164.4 |     65.4 | Peregrine |
| log10_100k             |      59.9 |     88.4 |    106.2 |      151.8 |        — |     59.7 | JAX |
| log1p_100k             |      78.8 |     86.9 |    127.4 |       98.3 |        — |    105.4 | Peregrine |
| erf_100k               |     103.8 |     62.0 |    100.6 |       61.3 |        — |     55.1 | JAX |
| sinh_100k              |      54.6 |    131.2 |     93.5 |      133.2 |    525.3 |    122.8 | Peregrine |
| cosh_100k              |      46.9 |    129.2 |     89.5 |      130.3 |    466.2 |     77.8 | Peregrine |
| arcsin_100k            |      55.6 |     80.8 |     92.8 |       59.7 |   2871.8 |    125.3 | Peregrine |
| arccos_100k            |      64.6 |     87.5 |    108.1 |       52.8 |        — |    206.2 | TensorFlow |
| arctan_100k            |      54.2 |     94.3 |     92.8 |       59.6 |   2998.8 |    214.0 | Peregrine |
| arcsinh_100k           |     212.9 |    158.6 |    331.2 |      139.0 |        — |    118.8 | JAX |
| maximum_100k           |      13.5 |     42.8 |     26.9 |       44.3 |    187.1 |     32.1 | Peregrine |
| minimum_100k           |      12.5 |     40.1 |     23.1 |       43.6 |    366.6 |     28.8 | Peregrine |
| power_100k             |     158.1 |    237.7 |    210.7 |      261.9 |        — |    140.5 | JAX |
| arctan2_100k           |      98.6 |    140.9 |    142.5 |       71.7 |        — |    321.9 | TensorFlow |
| logaddexp_100k         |     287.1 |    153.2 |    256.5 |      360.5 |        — |    154.6 | PyTorch |
| clip_100k              |       8.9 |     43.2 |     34.5 |       43.2 |    533.7 |     26.4 | Peregrine |
| where_100k             |      16.5 |     54.4 |     24.7 |       66.3 |    276.6 |     32.7 | Peregrine |
| greater_100k           |      12.5 |     50.7 |     20.5 |       49.6 |    187.5 |     26.5 | Peregrine |
| equal_100k             |      12.5 |     38.2 |     20.4 |       52.4 |    282.1 |     27.3 | Peregrine |
| sum_axis_256x512       |      18.9 |     40.5 |     19.0 |       51.9 |    204.8 |     54.4 | Peregrine |
| mean_axis_256x512      |      19.6 |     44.0 |     20.5 |       53.6 |    291.6 |     53.7 | Peregrine |
| max_axis_256x512       |      13.7 |     62.6 |     36.8 |       52.1 |    201.6 |     46.3 | Peregrine |
| min_axis_256x512       |      14.1 |     57.4 |     36.5 |       52.3 |    322.8 |     47.4 | Peregrine |
| var_256x512            |      45.7 |    282.7 |     55.1 |      217.1 |        — |     78.2 | Peregrine |
| prod_axis_256x512      |      24.2 |     38.3 |     26.8 |       50.8 |        — |     57.5 | Peregrine |
| logsumexp_256x512      |      97.7 |    209.1 |    122.1 |      339.2 |        — |    291.5 | Peregrine |
| cumsum_256x512         |     120.6 |     83.8 |    129.0 |      190.4 |    605.2 |    207.3 | PyTorch |
| argmax_axis_256x512    |      52.4 |     95.7 |    176.2 |       73.2 |   1311.8 |    172.0 | Peregrine |
| sum_axis_1024x1024     |     178.8 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     436.7 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.8 |     39.4 |     55.1 |       55.7 |   1783.8 |     42.8 | Peregrine |
| triu_256x256           |       8.7 |     39.4 |     55.3 |       54.4 |   1776.7 |     38.4 | Peregrine |
| repeat_64x128_2x3      |       6.0 |     45.6 |     33.4 |       75.7 |        — |     28.9 | Peregrine |
| pad_64x128             |       2.6 |      4.4 |     17.6 |       85.5 |     88.8 |     18.5 | Peregrine |
| stack_8x64x128         |       4.5 |      8.8 |     42.8 |       56.9 |    926.5 |    170.5 | Peregrine |
| diagonal_512x512       |       0.8 |      0.7 |     24.1 |       13.0 |        — |      7.3 | PyTorch |
| silu_100k              |      66.1 |     66.3 |     83.4 |      234.3 |    325.4 |     57.3 | JAX |
| softplus_100k          |     190.7 |    145.1 |    261.4 |      126.2 |    767.3 |    156.2 | TensorFlow |
| mish_100k              |     302.4 |    304.3 |    370.5 |      243.4 |   1147.6 |    240.3 | JAX |
| leaky_relu_100k        |       9.0 |     41.1 |     76.4 |       19.9 |        — |     28.8 | Peregrine |
| elu_100k               |      62.1 |    122.3 |    115.7 |      133.2 |    860.6 |     87.8 | Peregrine |
| hard_tanh_100k         |       8.7 |     44.9 |     34.6 |       43.1 |        — |     33.0 | Peregrine |
| relu6_100k             |       8.7 |     42.5 |     43.6 |       50.7 |    724.5 |    115.7 | Peregrine |
| hardswish_100k         |      10.4 |     45.4 |     69.7 |      207.8 |        — |     25.8 | Peregrine |
| gelu_100k              |      96.9 |     78.3 |    136.8 |      242.2 |    838.7 |    234.5 | PyTorch |
| selu_100k              |      65.1 |    134.5 |     83.6 |      133.3 |    736.9 |     91.4 | Peregrine |
| softsign_100k          |      40.4 |    119.8 |     40.7 |       48.4 |        — |     67.9 | Peregrine |
| cross_entropy_64x10    |       2.5 |     38.8 |     20.6 |      630.7 |   3291.3 |     59.6 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.4 |     15.0 |       43.9 |   1101.6 |     13.6 | Peregrine |
| mse_loss_64x10         |       4.2 |      5.0 |     18.3 |       39.8 |    436.1 |     26.5 | Peregrine |
| huber_loss_64x10       |       5.8 |      4.9 |     29.8 |      242.2 |        — |     51.4 | PyTorch |
| smooth_l1_loss_64x10   |       4.8 |      5.1 |     30.1 |      239.8 |        — |     49.2 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.5 |     17.6 |      378.4 |        — |     57.4 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.5 |    107.8 |      239.7 |        — |     54.1 | Peregrine |
| rmsnorm_64x512         |      58.3 |     69.2 |     29.1 |      439.0 |        — |     75.2 | MLX |
| conv1d_1x32x128_k3     |      20.8 |     55.1 |     27.1 |      510.0 |        — |     80.6 | Peregrine |
| avgpool2d_1x16x32x32   |      28.0 |     43.4 |    263.5 |       62.8 |        — |     43.5 | Peregrine |
| groupnorm_4x64x16x16   |      74.2 |     54.3 |    220.9 |      764.7 |        — |    269.0 | PyTorch |
| rnn_seq32_128_256      |     225.2 |    273.7 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1244.2 |    819.9 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |    1075.1 |    792.5 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     881.1 |   1249.1 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     999.7 |   1171.4 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     961.7 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1341.5 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     111.5 |    257.4 |    479.2 |      125.0 |   2327.5 |    531.1 | Peregrine |
| rand_normal_100k       |     244.0 |    971.6 |    685.3 |      337.8 |   3220.2 |    604.2 | Peregrine |
| rand_bernoulli_100k    |     124.4 |    250.0 |    448.9 |      229.8 |        — |    539.6 | Peregrine |
| rand_uniform_1M        |    1124.5 |   2564.7 |   4537.3 |      421.1 |   2354.8 |   2464.4 | TensorFlow |
| rand_normal_1M         |    2481.7 |   9701.1 |   6581.0 |     2070.2 |   3216.2 |   2958.2 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     20.3 |       42.7 |        — |     51.1 | Peregrine |
| rfft_4k                |       7.5 |     15.2 |     29.3 |       54.9 |        — |     64.4 | Peregrine |
| rfft_16k               |      31.6 |     65.2 |     78.8 |      105.7 |        — |    118.5 | Peregrine |
| fft_1k                 |       3.3 |      6.7 |     22.4 |        9.0 |        — |     18.0 | Peregrine |
| fft_4k                 |      13.4 |     26.2 |     37.0 |       17.8 |        — |     57.4 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     16.7 |       70.6 |        — |      3.9 | Peregrine |
| solve_64x64            |      12.0 |     18.7 |     89.3 |       24.6 |        — |     32.5 | Peregrine |
| inv_64x64              |      37.4 |     26.5 |     47.6 |       32.6 |        — |     44.4 | PyTorch |
| cholesky_64x64         |       8.8 |     42.2 |     21.3 |       19.7 |        — |     20.7 | Peregrine |
| svd_64x64              |     279.6 |    276.8 |    290.4 |      477.2 |        — |    315.6 | PyTorch |
| qr_64x64               |      41.4 |     81.5 |     58.4 |       83.8 |        — |     66.0 | Peregrine |
| eigh_64x64             |     386.8 |    212.7 |    232.5 |      142.1 |        — |    241.7 | TensorFlow |
| det_64x64              |      22.9 |     20.0 |        — |       23.1 |        — |     32.8 | PyTorch |
| solve_128x128          |      49.7 |     45.2 |    188.4 |       76.8 |        — |     83.6 | PyTorch |
| inv_128x128            |     105.9 |     62.1 |     90.5 |      138.8 |        — |     87.7 | PyTorch |
| cholesky_128x128       |      50.6 |     50.4 |     26.2 |       58.6 |        — |     37.0 | MLX |
| svd_128x128            |    1012.6 |    996.7 |    964.5 |     1822.5 |        — |   1050.6 | MLX |
| qr_128x128             |     195.3 |    225.5 |    194.2 |      326.2 |        — |    201.3 | MLX |
| eigh_128x128           |    1923.2 |    705.1 |    721.5 |      719.2 |        — |    758.8 | PyTorch |
| det_128x128            |      52.2 |     50.0 |        — |       81.9 |        — |     72.2 | PyTorch |
| solve_256x256          |     194.0 |    181.4 |    731.0 |      379.6 |        — |    283.9 | PyTorch |
| inv_256x256            |     502.8 |    288.9 |    251.0 |      848.2 |        — |    347.4 | MLX |
| cholesky_256x256       |     249.3 |     73.1 |     56.6 |      281.8 |        — |    116.5 | MLX |
| svd_256x256            |    6290.0 |   5682.4 |   5680.0 |     8037.4 |        — |   6097.4 | MLX |
| qr_256x256             |    1107.3 |   1006.1 |    990.2 |     1691.8 |        — |    996.9 | MLX |
| eigh_256x256           |    6339.4 |   3448.3 |   3456.8 |     4576.9 |        — |   3759.7 | PyTorch |
| det_256x256            |     219.3 |    202.4 |        — |      430.6 |        — |    208.0 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    2242.2 |    811.5 |        — |     2370.8 |   1223.0 |   2208.6 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3422.4 |   1880.0 |        — |     3676.3 |   1234.0 |   3528.9 | tinygrad |
| add_layernorm_196x768  |     112.9 |     96.4 |        — |     1211.8 |   1099.8 |    260.4 | PyTorch |
| add_layernorm_196x1024 |     147.2 |    108.2 |        — |     1303.7 |   1106.7 |    290.0 | PyTorch |
| matmul_f32_196x768x3072 |    1056.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14497.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1602.0 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26304.5 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.59x** (Peregrine is faster)
- **Peregrine vs MLX: 0.45x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.32x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.06x** (Peregrine is faster)
- **Peregrine vs JAX: 0.41x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 91/141 ops
- PyTorch: 26/141 ops
- MLX: 9/141 ops
- TensorFlow: 7/141 ops
- JAX: 7/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
