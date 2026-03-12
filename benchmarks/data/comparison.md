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
| matmul_128x128         |       6.7 |      6.2 |     28.4 |       69.1 |    457.9 |     71.4 | PyTorch |
| matmul_256x256         |      36.0 |     31.8 |     81.4 |      214.6 |    447.6 |    162.5 | PyTorch |
| matmul_512x512         |     216.8 |    142.0 |    221.2 |      724.8 |    462.0 |    621.9 | PyTorch |
| matmul_1024x1024       |    1310.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |   10422.2 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      13.5 |     41.0 |     32.4 |       46.1 |    195.2 |     45.0 | Peregrine |
| add_500k               |     216.4 |     58.9 |     84.4 |       85.6 |    209.2 |     78.7 | PyTorch |
| add_1M                 |     378.1 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     561.0 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |    1017.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      13.6 |     39.5 |     33.0 |       43.6 |    205.2 |     47.4 | Peregrine |
| mul_500k               |     135.5 |     57.4 |     88.0 |       83.8 |    204.0 |     70.2 | PyTorch |
| mul_1M                 |     173.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     597.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |    1300.8 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |     279.3 |     58.8 |     73.6 |       61.6 |    246.7 |     59.8 | PyTorch |
| exp_500k               |     432.5 |    139.7 |    241.9 |      111.8 |    242.3 |    144.0 | TensorFlow |
| exp_1M                 |     552.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1746.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    3682.4 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       9.2 |     39.2 |     27.0 |       41.2 |    375.3 |    109.0 | Peregrine |
| relu_1M                |     156.7 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.3 |     33.2 |     19.0 |       10.4 |    795.8 |     33.3 | Peregrine |
| softmax_8x512          |       4.4 |     36.6 |     20.1 |       14.1 |    781.5 |     36.1 | Peregrine |
| mlp_fwd_64x784         |      33.1 |     27.8 |     54.6 |      275.5 |   1908.4 |    207.3 | PyTorch |
| mlp_fwd_256x784_wide   |     479.9 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     874.0 |   1316.3 |    869.1 |     8918.2 |  27402.5 |   5820.6 | MLX |
| train_step_256_wide    |    3612.0 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       9.3 |     40.4 |     28.5 |       48.7 |    185.2 |     49.7 | Peregrine |
| square_100k            |       9.3 |     41.4 |     24.7 |       15.3 |    196.9 |     48.4 | Peregrine |
| rsqrt_100k             |     114.6 |     45.8 |     39.2 |       52.3 |        — |     83.3 | MLX |
| floor_100k             |      46.7 |     44.2 |     25.7 |       17.0 |    479.0 |     36.4 | TensorFlow |
| ceil_100k              |      48.7 |     44.8 |     30.8 |       17.0 |    379.9 |     52.6 | TensorFlow |
| round_100k             |      48.7 |     43.3 |     28.2 |       45.6 |        — |     43.4 | MLX |
| sign_100k              |      57.8 |     40.2 |     32.8 |       46.5 |    892.1 |     59.8 | MLX |
| expm1_100k             |     243.8 |    121.8 |    119.8 |      163.2 |        — |    105.8 | JAX |
| log2_100k              |     175.8 |     92.5 |    111.0 |      158.3 |    173.7 |     97.9 | PyTorch |
| log10_100k             |     153.9 |     90.0 |    122.9 |      185.6 |        — |    105.1 | PyTorch |
| log1p_100k             |     175.7 |     93.7 |    138.3 |      145.1 |        — |    117.8 | PyTorch |
| erf_100k               |     179.0 |     62.6 |    109.0 |       65.0 |        — |     42.1 | JAX |
| sinh_100k              |      54.5 |    131.1 |    105.1 |      159.4 |    575.7 |    154.2 | Peregrine |
| cosh_100k              |      49.3 |    131.2 |    102.8 |      148.3 |    500.7 |    119.9 | Peregrine |
| arcsin_100k            |      55.4 |     91.5 |    102.6 |       63.1 |   3343.9 |    165.5 | Peregrine |
| arccos_100k            |     194.2 |     90.1 |    119.4 |       59.1 |        — |    209.3 | TensorFlow |
| arctan_100k            |      58.0 |     96.8 |    101.9 |       61.6 |   3462.2 |    218.4 | Peregrine |
| arcsinh_100k           |     255.0 |    156.0 |    357.5 |      159.2 |        — |    685.7 | PyTorch |
| maximum_100k           |      10.5 |     42.3 |     26.6 |       43.1 |    197.6 |     44.5 | Peregrine |
| minimum_100k           |      10.3 |     42.5 |     30.8 |       43.1 |    401.6 |     51.5 | Peregrine |
| power_100k             |     395.8 |    237.4 |    238.0 |      342.3 |        — |    123.6 | JAX |
| arctan2_100k           |    1195.2 |    137.4 |    157.4 |       77.1 |        — |    329.9 | TensorFlow |
| logaddexp_100k         |     433.1 |    155.7 |    280.5 |      397.5 |        — |    194.9 | PyTorch |
| clip_100k              |       8.4 |     42.4 |     39.2 |       41.4 |    574.6 |     52.0 | Peregrine |
| where_100k             |      99.0 |     53.2 |     29.4 |       67.2 |    300.8 |     57.1 | MLX |
| greater_100k           |      86.8 |     50.4 |     21.5 |       59.5 |    201.2 |     43.6 | MLX |
| equal_100k             |      86.1 |     32.7 |     24.4 |       61.9 |    304.5 |     53.1 | MLX |
| sum_axis_256x512       |     120.0 |     44.8 |     20.8 |       46.7 |    226.6 |     59.5 | MLX |
| mean_axis_256x512      |     119.8 |     45.0 |     25.5 |       65.9 |    309.9 |     27.8 | MLX |
| max_axis_256x512       |     164.2 |     57.8 |     47.3 |       50.3 |    215.8 |     31.9 | JAX |
| min_axis_256x512       |     164.2 |     58.5 |     49.4 |       52.3 |    324.9 |     35.7 | JAX |
| var_256x512            |     250.6 |    306.3 |     60.9 |      244.7 |        — |     97.4 | MLX |
| prod_axis_256x512      |     153.7 |     42.8 |     26.3 |       50.6 |        — |     57.2 | MLX |
| logsumexp_256x512      |     401.3 |    201.8 |    119.9 |      375.7 |        — |    350.0 | MLX |
| cumsum_256x512         |     130.9 |     79.3 |    144.5 |      222.5 |    661.5 |    280.6 | PyTorch |
| argmax_axis_256x512    |     164.4 |     96.7 |    184.3 |       84.4 |   1426.2 |    199.0 | TensorFlow |
| sum_axis_1024x1024     |    1001.1 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    2041.3 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      37.3 |     42.8 |     63.6 |       64.2 |   2065.8 |     49.8 | Peregrine |
| triu_256x256           |      36.9 |     40.6 |     59.8 |       59.4 |   2044.3 |     48.0 | Peregrine |
| repeat_64x128_2x3      |     133.8 |     47.0 |     32.4 |       77.5 |        — |     30.2 | JAX |
| pad_64x128             |      18.0 |      4.5 |     21.8 |       84.0 |     95.0 |     21.5 | PyTorch |
| stack_8x64x128         |      20.4 |      9.2 |     47.1 |       63.8 |   1023.8 |    176.8 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     28.4 |       11.8 |        — |      9.3 | PyTorch |
| silu_100k              |      68.0 |     67.5 |     89.8 |      278.4 |    391.8 |     80.0 | PyTorch |
| softplus_100k          |     359.7 |    151.0 |    286.7 |      163.3 |    946.3 |    224.6 | PyTorch |
| mish_100k              |     574.4 |    316.7 |    408.3 |      289.1 |   1259.3 |    258.7 | JAX |
| leaky_relu_100k        |       8.5 |     42.9 |     85.1 |       19.6 |        — |     48.5 | Peregrine |
| elu_100k               |     221.0 |    137.3 |    125.8 |      159.6 |    922.7 |     91.1 | JAX |
| hard_tanh_100k         |      53.5 |     46.6 |     37.7 |       40.9 |        — |     43.7 | MLX |
| relu6_100k             |      53.7 |     56.5 |     54.0 |       62.1 |    761.2 |    132.7 | Peregrine |
| hardswish_100k         |      91.9 |     43.8 |     75.5 |      291.5 |        — |     50.6 | PyTorch |
| gelu_100k              |      66.2 |     77.7 |    148.3 |      331.8 |    903.6 |    248.9 | Peregrine |
| selu_100k              |     245.1 |    133.8 |     88.5 |      167.1 |    806.4 |    111.4 | MLX |
| softsign_100k          |      37.9 |    144.6 |     50.9 |       50.9 |        — |     84.4 | Peregrine |
| cross_entropy_64x10    |       2.8 |     44.3 |     24.0 |      627.2 |   3812.0 |     66.7 | Peregrine |
| l1_loss_64x10          |       1.1 |      5.5 |     19.2 |       41.5 |   1228.3 |     12.4 | Peregrine |
| mse_loss_64x10         |       4.0 |      5.2 |     20.7 |       37.4 |    499.9 |     23.4 | Peregrine |
| huber_loss_64x10       |       5.6 |      5.2 |     39.6 |      233.7 |        — |     47.9 | PyTorch |
| smooth_l1_loss_64x10   |       5.4 |      5.6 |     40.5 |      240.4 |        — |     50.9 | Peregrine |
| kl_div_loss_64x10      |       2.7 |      6.5 |     23.5 |      373.8 |        — |     67.8 | Peregrine |
| cosine_sim_loss_64x64  |      14.6 |     10.5 |    125.2 |      254.8 |        — |     63.5 | PyTorch |
| rmsnorm_64x512         |      63.3 |     71.8 |     39.0 |      514.6 |        — |     80.7 | MLX |
| conv1d_1x32x128_k3     |      21.7 |     46.5 |     29.9 |      757.5 |        — |    108.3 | Peregrine |
| avgpool2d_1x16x32x32   |      27.4 |     60.5 |    283.6 |       69.9 |        — |     77.4 | Peregrine |
| groupnorm_4x64x16x16   |     116.5 |     69.4 |    235.1 |      974.3 |        — |    288.6 | PyTorch |
| rnn_seq32_128_256      |     196.7 |    266.6 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1209.2 |    808.5 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     846.4 |    775.7 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     868.6 |   1327.0 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     974.1 |   1261.3 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |    1365.3 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1381.7 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     113.0 |    273.5 |    520.5 |      204.9 |   2764.3 |    567.5 | Peregrine |
| rand_normal_100k       |     815.5 |   1030.9 |    748.1 |      409.2 |   3626.1 |    628.4 | TensorFlow |
| rand_bernoulli_100k    |     321.5 |    265.7 |    490.7 |      266.6 |        — |    566.1 | PyTorch |
| rand_uniform_1M        |    1132.8 |   2724.9 |   4873.0 |      634.1 |   2700.6 |   2262.5 | TensorFlow |
| rand_normal_1M         |    8188.5 |  10326.9 |   7038.8 |     2753.4 |   3531.6 |   2965.9 | TensorFlow |
| rfft_1k                |       2.1 |      4.7 |     20.6 |       41.2 |        — |     51.1 | Peregrine |
| rfft_4k                |       7.3 |     15.8 |     29.7 |       52.9 |        — |     56.0 | Peregrine |
| rfft_16k               |      30.4 |     69.5 |     83.8 |      113.1 |        — |    108.9 | Peregrine |
| fft_1k                 |       3.2 |      7.0 |     24.4 |        8.7 |        — |     46.8 | Peregrine |
| fft_4k                 |      12.2 |     27.3 |     44.4 |       18.7 |        — |     49.6 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     20.8 |       72.4 |        — |      4.1 | Peregrine |
| solve_64x64            |      18.4 |     18.1 |    100.3 |       25.0 |        — |     33.0 | PyTorch |
| inv_64x64              |      48.1 |     26.0 |     50.5 |       33.5 |        — |     50.5 | PyTorch |
| cholesky_64x64         |      13.5 |     46.9 |     22.0 |       18.9 |        — |     21.2 | Peregrine |
| svd_64x64              |     286.9 |    277.9 |    305.3 |      499.3 |        — |    313.7 | PyTorch |
| qr_64x64               |      41.7 |     85.6 |     62.4 |       87.1 |        — |     65.8 | Peregrine |
| eigh_64x64             |     390.2 |    212.8 |    236.4 |      146.1 |        — |    243.8 | TensorFlow |
| det_64x64              |      22.6 |     19.2 |        — |       23.3 |        — |     30.7 | PyTorch |
| solve_128x128          |      49.0 |     43.5 |    209.5 |       77.6 |        — |     85.2 | PyTorch |
| inv_128x128            |      98.4 |     58.3 |     97.1 |      146.0 |        — |     83.2 | PyTorch |
| cholesky_128x128       |      51.6 |     64.4 |     31.5 |       60.9 |        — |     46.4 | MLX |
| svd_128x128            |    1027.6 |   1042.9 |   1025.6 |     1975.8 |        — |   1050.4 | MLX |
| qr_128x128             |     192.2 |    245.4 |    205.6 |      345.7 |        — |    209.3 | Peregrine |
| eigh_128x128           |    1941.7 |    706.2 |    747.8 |      752.8 |        — |    826.6 | PyTorch |
| det_128x128            |      52.5 |     54.8 |        — |       86.2 |        — |     86.3 | Peregrine |
| solve_256x256          |     191.0 |    157.9 |    747.4 |      386.7 |        — |    290.9 | PyTorch |
| inv_256x256            |     575.0 |    276.9 |    250.6 |      875.0 |        — |   1052.5 | MLX |
| cholesky_256x256       |     226.8 |     87.4 |     53.3 |      371.1 |        — |    405.0 | MLX |
| svd_256x256            |    6834.9 |   5989.1 |   6162.9 |     8777.0 |        — |   9683.6 | PyTorch |
| qr_256x256             |    1109.2 |   1093.9 |   1065.7 |     1809.4 |        — |   1588.1 | MLX |
| eigh_256x256           |    6490.9 |   3426.9 |   3615.8 |     4727.3 |        — |   5506.2 | PyTorch |
| det_256x256            |     220.3 |    201.0 |        — |      445.4 |        — |    619.1 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1680.2 |   1089.0 |        — |     3659.8 |   1481.7 |   2375.9 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2871.7 |   2550.9 |        — |     5697.0 |   1327.5 |   3490.5 | tinygrad |
| add_layernorm_196x768  |     117.6 |    140.8 |        — |     1315.4 |   1182.5 |    217.9 | Peregrine |
| add_layernorm_196x1024 |     155.8 |    143.6 |        — |     1293.1 |   1176.8 |    288.6 | PyTorch |
| matmul_f32_196x768x3072 |     935.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   15394.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1897.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   27750.9 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_128x128     |       4.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_256x256     |       4.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_512x512     |       4.7 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_1024x1024   |       5.2 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_2048x2048   |       8.5 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_100k           |       4.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_500k           |       4.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_1M             |       4.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_5M             |       6.1 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_10M            |      39.7 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_100k           |       5.7 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_500k           |       5.3 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_1M             |       5.6 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_5M             |       7.1 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_10M            |       9.7 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_100k           |       5.0 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_500k           |       5.0 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_1M             |       5.0 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_5M             |       6.0 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_10M            |       8.2 |        — |        — |          — |        — |        — | Peregrine |
| gpu_relu_100k          |       5.1 |        — |        — |          — |        — |        — | Peregrine |
| gpu_relu_1M            |       5.3 |        — |        — |          — |        — |        — | Peregrine |
| gpu_softmax_8x128      |       2.4 |        — |        — |          — |        — |        — | Peregrine |
| gpu_softmax_8x512      |       4.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mlp_fwd_64x784     |      32.5 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mlp_fwd_256x784_wide |      40.9 |        — |        — |          — |        — |        — | Peregrine |
| gpu_train_step_64      |    1441.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_train_step_256_wide |    4448.6 |        — |        — |          — |        — |        — | Peregrine |
| gpu_train_fused_64     |    1370.3 |        — |        — |          — |        — |        — | Peregrine |
| gpu_train_fused_256_wide |    4741.6 |        — |        — |          — |        — |        — | Peregrine |
| het_sequential_gpu_gpu |    2775.8 |        — |        — |          — |        — |        — | Peregrine |
| het_pipelined_gpu_cpu  |    1533.4 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.99x** (Peregrine is faster)
- **Peregrine vs MLX: 0.75x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.54x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.10x** (Peregrine is faster)
- **Peregrine vs JAX: 0.60x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 96/173 ops
- PyTorch: 38/173 ops
- MLX: 20/173 ops
- TensorFlow: 10/173 ops
- JAX: 8/173 ops
- tinygrad: 1/173 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
