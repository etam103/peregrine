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
| matmul_128x128         |      13.3 |      6.1 |     20.0 |       50.6 |    417.3 |     79.7 | PyTorch |
| matmul_256x256         |      59.0 |     30.7 |     47.9 |      197.8 |    424.7 |    147.1 | PyTorch |
| matmul_512x512         |     219.5 |    127.8 |    171.4 |      668.4 |    428.7 |    514.9 | PyTorch |
| matmul_1024x1024       |    1051.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9452.8 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     40.5 |     29.6 |       48.9 |    185.8 |     36.5 | Peregrine |
| add_500k               |      48.0 |     56.9 |     82.3 |       86.2 |    185.3 |     59.8 | Peregrine |
| add_1M                 |     128.0 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     512.2 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     961.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.7 |     38.2 |     28.5 |       47.4 |    186.4 |     32.9 | Peregrine |
| mul_500k               |      62.4 |     56.2 |     81.3 |       75.7 |    188.9 |     58.7 | PyTorch |
| mul_1M                 |     127.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     530.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     922.9 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.2 |     58.9 |     60.2 |       64.3 |    220.8 |     45.6 | JAX |
| exp_500k               |     187.1 |    138.8 |    222.8 |      102.3 |    220.8 |    118.3 | TensorFlow |
| exp_1M                 |     292.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1087.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2127.2 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.6 |     39.3 |     24.7 |       36.8 |    335.4 |    100.0 | Peregrine |
| relu_1M                |      84.1 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     31.9 |     16.0 |       11.8 |    613.1 |     30.5 | Peregrine |
| softmax_8x512          |       4.2 |     35.1 |     20.5 |       14.6 |    617.5 |     33.0 | Peregrine |
| mlp_fwd_64x784         |      33.1 |     27.6 |     51.1 |      247.2 |   1799.6 |    181.8 | PyTorch |
| mlp_fwd_256x784_wide   |     430.9 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     808.5 |   1208.3 |    767.4 |     8511.1 |  22845.9 |   5037.2 | MLX |
| train_step_256_wide    |    3272.1 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.7 |     37.5 |     24.6 |       48.6 |    165.0 |     30.5 | Peregrine |
| square_100k            |       8.7 |     40.3 |     23.8 |       16.3 |    174.2 |     30.9 | Peregrine |
| rsqrt_100k             |      21.5 |     42.2 |     36.0 |       51.7 |        — |     93.2 | Peregrine |
| floor_100k             |       8.7 |     38.2 |     23.6 |       17.9 |    404.6 |     29.3 | Peregrine |
| ceil_100k              |       8.7 |     40.0 |     23.6 |       17.9 |    348.8 |     31.2 | Peregrine |
| round_100k             |       8.7 |     40.4 |     23.6 |       41.0 |        — |     28.1 | Peregrine |
| sign_100k              |       8.6 |     39.8 |     27.6 |       45.8 |    811.4 |     36.4 | Peregrine |
| expm1_100k             |      63.1 |    109.4 |    107.7 |      146.8 |        — |     97.5 | Peregrine |
| log2_100k              |      56.6 |     84.2 |    101.8 |      148.3 |    165.3 |     58.8 | Peregrine |
| log10_100k             |      59.0 |     83.8 |    109.5 |      148.4 |        — |     56.5 | JAX |
| log1p_100k             |      76.9 |     81.8 |    127.5 |       95.0 |        — |    104.4 | Peregrine |
| erf_100k               |     102.6 |     57.3 |    100.7 |       58.5 |        — |     43.4 | JAX |
| sinh_100k              |      52.0 |    132.1 |     93.5 |      129.4 |    527.8 |    111.1 | Peregrine |
| cosh_100k              |      47.2 |    127.5 |     89.3 |      133.2 |    459.5 |     75.8 | Peregrine |
| arcsin_100k            |      53.1 |     77.3 |     93.9 |       55.3 |   2870.2 |    111.9 | Peregrine |
| arccos_100k            |      61.8 |     87.8 |    110.3 |       53.2 |        — |    192.5 | TensorFlow |
| arctan_100k            |      54.1 |     93.9 |     92.8 |       59.1 |   2972.2 |    212.3 | Peregrine |
| arcsinh_100k           |     204.9 |    160.3 |    332.9 |      143.4 |        — |    118.7 | JAX |
| maximum_100k           |      12.5 |     38.9 |     29.6 |       41.3 |    188.2 |     33.6 | Peregrine |
| minimum_100k           |      12.5 |     39.3 |     27.9 |       39.7 |    369.6 |     31.6 | Peregrine |
| power_100k             |     153.7 |    236.0 |    210.5 |      275.1 |        — |    140.2 | JAX |
| arctan2_100k           |      95.1 |    126.0 |    143.9 |       75.2 |        — |    311.4 | TensorFlow |
| logaddexp_100k         |     272.5 |    153.4 |    255.2 |      360.3 |        — |    142.1 | JAX |
| clip_100k              |       8.7 |     40.2 |     34.2 |       43.3 |    533.1 |     35.1 | Peregrine |
| where_100k             |      16.7 |     52.8 |     28.2 |       66.7 |    275.8 |     32.0 | Peregrine |
| greater_100k           |      12.8 |     49.0 |     23.8 |       55.0 |    188.0 |     27.4 | Peregrine |
| equal_100k             |      12.7 |     29.6 |     24.1 |       54.9 |    287.0 |     29.9 | Peregrine |
| sum_axis_256x512       |      19.2 |     38.7 |     23.1 |       50.4 |    209.8 |     52.4 | Peregrine |
| mean_axis_256x512      |      19.2 |     43.8 |     24.2 |       53.6 |    291.3 |     49.3 | Peregrine |
| max_axis_256x512       |      13.9 |     56.3 |     39.8 |       51.9 |    207.5 |     45.6 | Peregrine |
| min_axis_256x512       |      13.9 |     54.5 |     39.5 |       53.1 |    323.4 |     45.2 | Peregrine |
| var_256x512            |      45.7 |    270.3 |     59.3 |      207.7 |        — |     80.9 | Peregrine |
| prod_axis_256x512      |      24.1 |     38.6 |     25.7 |       53.1 |        — |     55.4 | Peregrine |
| logsumexp_256x512      |      95.5 |    200.7 |    106.6 |      341.3 |        — |    277.1 | Peregrine |
| cumsum_256x512         |     118.7 |     69.4 |    128.2 |      186.8 |    609.6 |    211.9 | PyTorch |
| argmax_axis_256x512    |      51.7 |     92.8 |    170.4 |       73.3 |   1299.2 |    171.8 | Peregrine |
| sum_axis_1024x1024     |     174.1 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.8 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      36.1 |     38.9 |     56.3 |       53.2 |   1775.8 |     38.2 | Peregrine |
| triu_256x256           |      34.7 |     39.2 |     55.6 |       55.0 |   1782.2 |     39.6 | Peregrine |
| repeat_64x128_2x3      |     124.7 |     47.4 |     30.5 |       76.5 |        — |     27.9 | JAX |
| pad_64x128             |      16.8 |      4.1 |     18.8 |       84.6 |     88.6 |     17.9 | PyTorch |
| stack_8x64x128         |      15.1 |      8.7 |     44.6 |       55.1 |    904.2 |    157.4 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     29.0 |       12.7 |        — |      7.6 | PyTorch |
| silu_100k              |      64.0 |     67.0 |     85.5 |      225.1 |    327.7 |     54.2 | JAX |
| softplus_100k          |     180.7 |    148.2 |    262.0 |      134.2 |    777.3 |    155.6 | TensorFlow |
| mish_100k              |     286.1 |    308.5 |    371.0 |      247.1 |   1175.8 |    231.0 | JAX |
| leaky_relu_100k        |       8.7 |     39.5 |     77.7 |       19.8 |        — |     30.0 | Peregrine |
| elu_100k               |      60.0 |    123.3 |    116.0 |      136.0 |    883.7 |     77.3 | Peregrine |
| hard_tanh_100k         |       8.7 |     40.1 |     34.2 |       42.6 |        — |     36.1 | Peregrine |
| relu6_100k             |       8.7 |     36.9 |     44.6 |       52.5 |    734.0 |    113.4 | Peregrine |
| hardswish_100k         |      10.3 |     38.4 |     69.3 |      209.0 |        — |     27.1 | Peregrine |
| gelu_100k              |      95.3 |     74.1 |    135.2 |      249.9 |    849.5 |    209.0 | PyTorch |
| selu_100k              |      63.7 |    123.2 |     82.6 |      133.0 |    735.0 |     81.5 | Peregrine |
| softsign_100k          |      38.0 |    124.2 |     43.8 |       47.7 |        — |     57.7 | Peregrine |
| cross_entropy_64x10    |       2.6 |     37.4 |     22.0 |      625.6 |   3352.0 |     53.3 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.4 |     19.0 |       43.8 |   1118.5 |     12.3 | Peregrine |
| mse_loss_64x10         |       3.7 |      5.0 |     21.8 |       41.8 |    443.4 |     23.5 | Peregrine |
| huber_loss_64x10       |       5.0 |      4.8 |     32.8 |      263.4 |        — |     46.8 | PyTorch |
| smooth_l1_loss_64x10   |       4.9 |      5.2 |     34.6 |      260.3 |        — |     47.0 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     17.5 |      378.4 |        — |     59.4 | Peregrine |
| cosine_sim_loss_64x64  |      13.5 |     10.4 |    111.1 |      240.3 |        — |     69.5 | PyTorch |
| rmsnorm_64x512         |      57.6 |     64.8 |     33.0 |      442.3 |        — |     72.4 | MLX |
| conv1d_1x32x128_k3     |      21.4 |     55.0 |     27.9 |      509.5 |        — |     74.3 | Peregrine |
| avgpool2d_1x16x32x32   |      26.5 |     44.2 |    261.5 |       63.5 |        — |     42.3 | Peregrine |
| groupnorm_4x64x16x16   |      72.6 |     55.2 |    223.5 |      739.3 |        — |    261.9 | PyTorch |
| rnn_seq32_128_256      |     194.2 |    268.3 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1126.1 |    801.7 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     851.3 |    779.5 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     801.1 |   1241.3 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     921.8 |   1091.3 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     908.0 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1276.9 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     106.3 |    257.2 |    485.6 |      124.2 |   2366.6 |    525.9 | Peregrine |
| rand_normal_100k       |     236.4 |    971.6 |    687.2 |      334.8 |   3233.0 |    602.2 | Peregrine |
| rand_bernoulli_100k    |     303.3 |    250.1 |    449.7 |      224.2 |        — |    518.6 | TensorFlow |
| rand_uniform_1M        |    1064.0 |   2647.8 |   4548.8 |      416.0 |   2370.5 |   2250.2 | TensorFlow |
| rand_normal_1M         |    2369.4 |   9704.2 |   6584.4 |     2061.4 |   3232.0 |   2835.1 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     25.4 |       45.1 |        — |     60.1 | Peregrine |
| rfft_4k                |       6.5 |     15.0 |     33.5 |       55.2 |        — |     69.9 | Peregrine |
| rfft_16k               |      30.2 |     65.4 |     80.1 |      107.5 |        — |    116.3 | Peregrine |
| fft_1k                 |       3.4 |      6.6 |     24.4 |        9.0 |        — |     42.2 | Peregrine |
| fft_4k                 |      12.2 |     26.1 |     43.7 |       17.5 |        — |     59.0 | Peregrine |
| norm_l2_1k             |       1.1 |      1.2 |     20.6 |       70.2 |        — |      3.9 | Peregrine |
| solve_64x64            |      12.0 |     25.2 |    100.5 |       25.0 |        — |     32.3 | Peregrine |
| inv_64x64              |      37.4 |     26.0 |     51.6 |       32.8 |        — |     37.1 | PyTorch |
| cholesky_64x64         |       9.6 |     46.9 |     21.7 |       19.9 |        — |     20.5 | Peregrine |
| svd_64x64              |     276.9 |    279.1 |    296.5 |      482.9 |        — |    305.0 | Peregrine |
| qr_64x64               |      41.2 |     82.7 |     58.3 |       83.8 |        — |     65.1 | Peregrine |
| eigh_64x64             |     379.8 |    218.5 |    236.0 |      145.2 |        — |    237.8 | TensorFlow |
| det_64x64              |      23.2 |     20.0 |        — |       23.4 |        — |     28.7 | PyTorch |
| solve_128x128          |      50.1 |     45.1 |    188.2 |       77.2 |        — |     85.7 | PyTorch |
| inv_128x128            |      92.0 |     62.1 |     90.5 |      139.2 |        — |     82.7 | PyTorch |
| cholesky_128x128       |      50.6 |     49.3 |     26.4 |       58.9 |        — |     35.7 | MLX |
| svd_128x128            |     985.9 |    984.3 |    970.8 |     1823.4 |        — |   1016.7 | MLX |
| qr_128x128             |     188.0 |    223.2 |    197.8 |      326.5 |        — |    190.6 | Peregrine |
| eigh_128x128           |    1841.2 |    701.1 |    725.5 |      699.2 |        — |    749.4 | TensorFlow |
| det_128x128            |      52.5 |     49.8 |        — |       82.7 |        — |     76.8 | PyTorch |
| solve_256x256          |     188.3 |    179.4 |    726.4 |      377.6 |        — |    266.9 | PyTorch |
| inv_256x256            |     453.8 |    296.7 |    247.4 |      848.4 |        — |    335.3 | MLX |
| cholesky_256x256       |     226.0 |     78.2 |     56.8 |      281.5 |        — |    117.5 | MLX |
| svd_256x256            |    5877.4 |   5905.1 |   5774.6 |     8253.8 |        — |   5918.1 | MLX |
| qr_256x256             |     989.6 |    991.6 |    996.1 |     1691.7 |        — |    976.4 | JAX |
| eigh_256x256           |    6006.3 |   3464.7 |   3480.4 |     4671.1 |        — |   3601.6 | PyTorch |
| det_256x256            |     212.4 |    202.5 |        — |      440.2 |        — |    205.6 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1481.2 |    976.2 |        — |     2391.8 |   1226.5 |   2087.7 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2645.3 |   1907.0 |        — |     3703.0 |   1224.8 |   3336.8 | tinygrad |
| add_layernorm_196x768  |     105.6 |    101.5 |        — |     1241.2 |   1112.7 |    236.9 | PyTorch |
| add_layernorm_196x1024 |     134.4 |    105.8 |        — |     1275.5 |   1109.8 |    274.3 | PyTorch |
| matmul_f32_196x768x3072 |     567.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14515.6 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1468.4 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26140.2 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.64x** (Peregrine is faster)
- **Peregrine vs MLX: 0.47x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.34x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.06x** (Peregrine is faster)
- **Peregrine vs JAX: 0.44x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 88/141 ops
- PyTorch: 26/141 ops
- JAX: 10/141 ops
- TensorFlow: 9/141 ops
- MLX: 7/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
