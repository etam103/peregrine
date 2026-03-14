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
| matmul_128x128         |       5.7 |      6.2 |     22.6 |       98.3 |    419.9 |     79.0 | Peregrine |
| matmul_256x256         |      69.0 |     30.4 |     51.7 |      192.4 |    434.4 |    151.3 | PyTorch |
| matmul_512x512         |     218.7 |    138.5 |    177.4 |      675.1 |    431.5 |    499.0 | PyTorch |
| matmul_1024x1024       |    1037.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9720.6 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     40.2 |     32.8 |       51.8 |    187.3 |     33.9 | Peregrine |
| add_500k               |     100.8 |     58.6 |     71.4 |       83.7 |    189.6 |     60.8 | PyTorch |
| add_1M                 |     134.9 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     533.4 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     929.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      13.0 |     39.4 |     32.6 |       43.1 |    189.0 |     33.7 | Peregrine |
| mul_500k               |     151.7 |     58.8 |     76.9 |       79.0 |    187.5 |     59.4 | PyTorch |
| mul_1M                 |     159.7 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     570.0 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     936.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |     146.7 |     61.9 |     60.4 |       67.1 |    218.5 |     48.5 | JAX |
| exp_500k               |     207.1 |    137.8 |    247.0 |      107.7 |    223.2 |    117.8 | TensorFlow |
| exp_1M                 |     331.9 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1286.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2472.3 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.9 |     39.9 |     27.9 |       40.6 |    338.9 |     97.8 | Peregrine |
| relu_1M                |     108.4 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     33.5 |     16.3 |       11.5 |    629.0 |     31.3 | Peregrine |
| softmax_8x512          |       4.3 |     34.7 |     20.2 |       14.8 |    626.9 |     33.4 | Peregrine |
| mlp_fwd_64x784         |      32.8 |     28.1 |     56.1 |      258.0 |   1813.5 |    186.3 | PyTorch |
| mlp_fwd_256x784_wide   |     430.1 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     820.4 |   1417.9 |    879.3 |     8811.9 |  23995.6 |   5201.2 | Peregrine |
| train_step_256_wide    |    3381.3 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.9 |     44.3 |     29.0 |       48.3 |    163.5 |     27.8 | Peregrine |
| square_100k            |       9.0 |     40.6 |     24.7 |       16.8 |    177.1 |     40.4 | Peregrine |
| rsqrt_100k             |      94.5 |     48.0 |     31.3 |       56.2 |        — |     82.5 | MLX |
| floor_100k             |       9.1 |     45.8 |     25.8 |       18.4 |    410.3 |     34.3 | Peregrine |
| ceil_100k              |       9.3 |     44.1 |     29.1 |       18.4 |    354.8 |     36.3 | Peregrine |
| round_100k             |       8.7 |     44.9 |     26.0 |       46.3 |        — |     41.4 | Peregrine |
| sign_100k              |       8.5 |     42.1 |     31.4 |       45.2 |    802.7 |     35.8 | Peregrine |
| expm1_100k             |     173.8 |    122.7 |    117.9 |      162.0 |        — |     98.6 | JAX |
| log2_100k              |     108.9 |     90.4 |    101.4 |      163.8 |    162.9 |     56.7 | JAX |
| log10_100k             |     109.5 |     93.6 |    116.0 |      149.1 |        — |     56.7 | JAX |
| log1p_100k             |     112.6 |     89.8 |    143.0 |      100.8 |        — |    104.4 | PyTorch |
| erf_100k               |     111.8 |     63.1 |    110.0 |       61.1 |        — |     46.0 | JAX |
| sinh_100k              |      52.6 |    130.1 |    101.5 |      156.3 |    533.1 |    108.7 | Peregrine |
| cosh_100k              |      46.3 |    129.3 |    103.6 |      137.0 |    462.4 |     70.8 | Peregrine |
| arcsin_100k            |      52.1 |     79.9 |    103.5 |       58.0 |   2910.3 |    112.1 | Peregrine |
| arccos_100k            |     115.7 |     94.7 |    119.5 |       55.4 |        — |    198.9 | TensorFlow |
| arctan_100k            |      54.1 |    104.4 |    104.4 |       59.9 |   3055.8 |    215.1 | Peregrine |
| arcsinh_100k           |     150.6 |    160.8 |    388.9 |      160.7 |        — |    127.5 | JAX |
| maximum_100k           |      12.9 |     42.8 |     49.2 |       44.9 |    193.2 |     33.3 | Peregrine |
| minimum_100k           |      12.9 |     40.3 |     49.0 |       45.4 |    376.4 |     32.9 | Peregrine |
| power_100k             |     158.5 |    226.9 |    280.6 |      334.6 |        — |    145.9 | JAX |
| arctan2_100k           |      98.0 |    131.4 |    189.7 |       75.0 |        — |    321.2 | TensorFlow |
| logaddexp_100k         |     420.9 |    154.8 |    280.2 |      390.9 |        — |    159.1 | PyTorch |
| clip_100k              |       8.9 |     46.7 |     41.4 |       43.9 |    536.2 |     39.2 | Peregrine |
| where_100k             |      17.0 |     55.3 |     33.2 |       68.0 |    282.4 |     32.4 | Peregrine |
| greater_100k           |      12.9 |     46.6 |     28.6 |       50.6 |    190.1 |     25.9 | Peregrine |
| equal_100k             |      13.0 |     38.5 |     31.7 |       60.8 |    282.9 |     26.2 | Peregrine |
| sum_axis_256x512       |      19.6 |     39.1 |     26.8 |       54.5 |    209.0 |     58.8 | Peregrine |
| mean_axis_256x512      |      18.9 |     42.1 |     32.1 |       56.1 |    304.9 |     53.1 | Peregrine |
| max_axis_256x512       |      13.7 |     54.1 |     55.8 |       53.6 |    205.7 |     47.4 | Peregrine |
| min_axis_256x512       |      13.7 |     56.8 |     53.0 |       51.9 |    328.2 |     44.5 | Peregrine |
| var_256x512            |      45.7 |    273.1 |     86.7 |      235.8 |        — |     79.7 | Peregrine |
| prod_axis_256x512      |      24.2 |     44.1 |     29.3 |       48.8 |        — |     56.0 | Peregrine |
| logsumexp_256x512      |      95.5 |    206.9 |    124.3 |      346.9 |        — |    283.3 | Peregrine |
| cumsum_256x512         |     124.0 |     85.1 |    154.0 |      202.1 |    614.5 |    213.1 | PyTorch |
| argmax_axis_256x512    |      53.3 |     93.7 |    195.4 |       74.2 |   1310.3 |    170.9 | Peregrine |
| sum_axis_1024x1024     |     179.5 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     441.4 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      35.9 |     41.6 |     62.4 |       56.4 |   1807.2 |     38.8 | Peregrine |
| triu_256x256           |      35.6 |     42.1 |     68.3 |       56.5 |   1847.9 |     37.4 | Peregrine |
| repeat_64x128_2x3      |     128.8 |     50.6 |     36.4 |       79.7 |        — |     28.2 | JAX |
| pad_64x128             |      17.3 |      4.7 |     23.0 |       86.9 |     89.3 |     18.2 | PyTorch |
| stack_8x64x128         |      15.0 |      8.5 |     52.2 |       59.6 |    929.0 |    159.8 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     30.4 |       12.4 |        — |      8.9 | PyTorch |
| silu_100k              |      66.0 |     68.8 |     94.8 |      248.4 |    330.6 |     52.1 | JAX |
| softplus_100k          |     347.9 |    153.4 |    329.4 |      141.0 |    787.1 |    155.8 | TensorFlow |
| mish_100k              |     512.4 |    320.1 |    435.3 |      246.8 |   1183.7 |    228.8 | JAX |
| leaky_relu_100k        |       8.2 |     45.4 |     97.0 |       19.5 |        — |     32.9 | Peregrine |
| elu_100k               |     172.4 |    137.3 |    137.9 |      137.0 |    873.5 |     77.9 | JAX |
| hard_tanh_100k         |       8.2 |     46.0 |     40.9 |       43.6 |        — |     39.8 | Peregrine |
| relu6_100k             |       8.2 |     44.1 |     49.3 |       50.7 |    744.7 |    112.8 | Peregrine |
| hardswish_100k         |      10.3 |     46.2 |     92.6 |      209.8 |        — |     30.0 | Peregrine |
| gelu_100k              |      71.6 |     78.3 |    160.3 |      252.7 |    859.6 |    214.1 | Peregrine |
| selu_100k              |      65.8 |    129.2 |    102.8 |      141.6 |    733.2 |     82.7 | Peregrine |
| softsign_100k          |      36.0 |    131.2 |     49.6 |       51.0 |        — |     58.6 | Peregrine |
| cross_entropy_64x10    |       2.7 |     40.6 |     26.2 |      635.9 |   3349.9 |     55.3 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.4 |     20.3 |       43.2 |   1119.6 |     12.2 | Peregrine |
| mse_loss_64x10         |       3.9 |      4.9 |     21.2 |       41.4 |    447.4 |     23.9 | Peregrine |
| huber_loss_64x10       |       5.4 |      4.8 |     35.2 |      247.1 |        — |     48.1 | PyTorch |
| smooth_l1_loss_64x10   |       5.1 |      5.0 |     33.6 |      247.1 |        — |     47.4 | PyTorch |
| kl_div_loss_64x10      |       2.6 |      6.5 |     20.8 |      389.3 |        — |     60.0 | Peregrine |
| cosine_sim_loss_64x64  |      14.0 |     10.3 |    117.2 |      245.2 |        — |     66.2 | PyTorch |
| rmsnorm_64x512         |      59.5 |     71.0 |     36.6 |      450.6 |        — |     72.5 | MLX |
| conv1d_1x32x128_k3     |      20.6 |     53.7 |     30.3 |      527.6 |        — |     74.6 | Peregrine |
| avgpool2d_1x16x32x32   |      25.1 |     41.0 |    270.4 |       62.8 |        — |     44.2 | Peregrine |
| groupnorm_4x64x16x16   |      72.6 |     56.1 |    228.6 |      777.2 |        — |    274.9 | PyTorch |
| rnn_seq32_128_256      |     190.7 |    267.2 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1133.7 |    816.3 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     814.5 |    779.3 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     803.0 |   1327.8 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     924.6 |   1314.9 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     910.2 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1282.3 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     106.3 |    263.0 |    484.6 |      138.0 |   2416.4 |    533.6 | Peregrine |
| rand_normal_100k       |     771.6 |    993.1 |    693.6 |      355.7 |   3289.1 |    606.5 | TensorFlow |
| rand_bernoulli_100k    |     303.8 |    253.0 |    450.2 |      225.2 |        — |    529.8 | TensorFlow |
| rand_uniform_1M        |    1067.9 |   2658.1 |   4595.4 |      455.5 |   2402.3 |   2241.3 | TensorFlow |
| rand_normal_1M         |    7718.6 |  10002.7 |   6598.9 |     2280.0 |   3305.4 |   2867.4 | TensorFlow |
| rfft_1k                |       2.0 |      4.9 |     28.1 |       43.7 |        — |     60.8 | Peregrine |
| rfft_4k                |       6.6 |     15.1 |     35.9 |       55.0 |        — |     68.2 | Peregrine |
| rfft_16k               |      29.3 |     65.5 |     77.6 |      108.6 |        — |    117.7 | Peregrine |
| fft_1k                 |       3.3 |      6.6 |     24.0 |        9.1 |        — |     40.0 | Peregrine |
| fft_4k                 |      12.1 |     26.6 |     40.5 |       17.3 |        — |     61.1 | Peregrine |
| norm_l2_1k             |       1.0 |      1.3 |     20.2 |       69.8 |        — |      3.7 | Peregrine |
| solve_64x64            |      12.1 |     24.6 |     96.1 |       24.5 |        — |     32.1 | Peregrine |
| inv_64x64              |      37.3 |     26.5 |     47.6 |       32.7 |        — |     45.7 | PyTorch |
| cholesky_64x64         |       9.5 |     52.0 |     21.3 |       19.7 |        — |     21.7 | Peregrine |
| svd_64x64              |     276.6 |    279.4 |    291.5 |      510.0 |        — |    309.0 | Peregrine |
| qr_64x64               |      42.2 |     83.6 |     55.5 |       86.2 |        — |     62.6 | Peregrine |
| eigh_64x64             |     379.9 |    217.9 |    231.8 |      146.8 |        — |    238.2 | TensorFlow |
| det_64x64              |      23.0 |     19.8 |        — |       23.4 |        — |     28.4 | PyTorch |
| solve_128x128          |      49.7 |     44.9 |    185.8 |       76.7 |        — |     86.3 | PyTorch |
| inv_128x128            |      93.4 |     61.5 |     86.9 |      141.6 |        — |     84.0 | PyTorch |
| cholesky_128x128       |      50.4 |     39.1 |     26.5 |       60.2 |        — |     35.8 | MLX |
| svd_128x128            |     991.3 |   1012.2 |    957.2 |     1859.2 |        — |   1017.0 | MLX |
| qr_128x128             |     192.5 |    224.5 |    190.5 |      332.0 |        — |    190.3 | JAX |
| eigh_128x128           |    1843.9 |    701.8 |    723.6 |      715.6 |        — |    746.0 | PyTorch |
| det_128x128            |      52.3 |     52.3 |        — |       83.6 |        — |     76.2 | Peregrine |
| solve_256x256          |     189.3 |    159.8 |    735.4 |      377.6 |        — |    265.0 | PyTorch |
| inv_256x256            |     456.4 |    318.0 |    249.6 |      851.3 |        — |    335.7 | MLX |
| cholesky_256x256       |     226.7 |    143.7 |     55.7 |      288.6 |        — |    117.1 | MLX |
| svd_256x256            |    5937.0 |   6231.6 |   5813.1 |     8431.4 |        — |   5801.0 | JAX |
| qr_256x256             |    1005.0 |   1076.1 |    990.0 |     1700.2 |        — |    995.3 | MLX |
| eigh_256x256           |    6016.5 |   3620.9 |   3428.8 |     4614.9 |        — |   3583.2 | MLX |
| det_256x256            |     212.5 |    215.1 |        — |      438.9 |        — |    204.3 | JAX |
| matmul_bias_gelu_196x768x3072 |    1308.3 |   1222.7 |        — |     2385.7 |   1245.3 |   2131.1 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2220.9 |   2465.1 |        — |     3722.9 |   1248.8 |   3528.0 | tinygrad |
| add_layernorm_196x768  |     107.5 |    106.9 |        — |     1233.2 |   1118.3 |    227.7 | PyTorch |
| add_layernorm_196x1024 |     294.1 |    117.7 |        — |     1288.4 |   1136.4 |    277.8 | PyTorch |
| matmul_f32_196x768x3072 |     650.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14550.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1647.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26481.9 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.67x** (Peregrine is faster)
- **Peregrine vs MLX: 0.48x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.37x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.07x** (Peregrine is faster)
- **Peregrine vs JAX: 0.48x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 83/141 ops
- PyTorch: 26/141 ops
- JAX: 14/141 ops
- TensorFlow: 9/141 ops
- MLX: 8/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
