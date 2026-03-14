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
| matmul_128x128         |      13.3 |      6.1 |     20.6 |       50.3 |    419.0 |     77.8 | PyTorch |
| matmul_256x256         |      58.9 |     31.8 |     48.1 |      201.5 |    422.5 |    146.5 | PyTorch |
| matmul_512x512         |     218.8 |    132.7 |    166.7 |      658.5 |    423.5 |    498.7 | PyTorch |
| matmul_1024x1024       |    1035.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9424.2 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.5 |     43.4 |     29.2 |       52.8 |    185.5 |     36.6 | Peregrine |
| add_500k               |      61.8 |     58.2 |     79.5 |       84.9 |    186.7 |     58.4 | PyTorch |
| add_1M                 |     126.5 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     510.4 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     846.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.9 |     43.5 |     28.0 |       40.9 |    188.1 |     31.5 | Peregrine |
| mul_500k               |      63.7 |     55.9 |     77.2 |       77.7 |    190.8 |     58.0 | PyTorch |
| mul_1M                 |     130.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     579.0 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     979.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.3 |     61.7 |     56.3 |       73.2 |    222.2 |     46.4 | JAX |
| exp_500k               |     132.8 |    139.2 |    221.0 |      114.6 |    225.9 |    117.1 | TensorFlow |
| exp_1M                 |     143.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     479.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     880.9 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       9.0 |     40.0 |     28.1 |       43.1 |    336.2 |     98.6 | Peregrine |
| relu_1M                |      84.8 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     33.9 |     17.6 |       11.7 |    609.6 |     30.4 | Peregrine |
| softmax_8x512          |       4.3 |     34.6 |     20.8 |       14.5 |    619.3 |     33.0 | Peregrine |
| mlp_fwd_64x784         |      34.0 |     27.7 |     50.5 |      256.4 |   1784.8 |    183.5 | PyTorch |
| mlp_fwd_256x784_wide   |     478.8 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     813.2 |   1246.5 |    776.7 |     8535.6 |  23291.2 |   5086.2 | MLX |
| train_step_256_wide    |    3535.8 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       9.4 |     39.4 |     25.0 |       48.4 |    163.8 |     29.9 | Peregrine |
| square_100k            |       9.3 |     37.9 |     23.8 |       15.7 |    177.1 |     28.5 | Peregrine |
| rsqrt_100k             |      23.0 |     41.3 |     36.4 |       51.0 |        — |     92.9 | Peregrine |
| floor_100k             |       9.4 |     40.7 |     23.5 |       16.3 |    415.1 |     29.0 | Peregrine |
| ceil_100k              |       9.4 |     39.8 |     23.5 |       16.3 |    355.4 |     29.3 | Peregrine |
| round_100k             |       9.4 |     40.0 |     23.4 |       46.8 |        — |     29.1 | Peregrine |
| sign_100k              |       9.4 |     38.3 |     27.6 |       46.2 |    806.5 |     35.7 | Peregrine |
| expm1_100k             |      67.5 |    108.8 |    107.6 |      143.3 |        — |     98.8 | Peregrine |
| log2_100k              |      59.1 |     89.0 |    101.9 |      150.5 |    165.3 |     55.9 | JAX |
| log10_100k             |      58.2 |     85.3 |    105.6 |      152.3 |        — |     56.0 | JAX |
| log1p_100k             |      78.1 |     82.2 |    127.4 |       93.5 |        — |    106.1 | Peregrine |
| erf_100k               |     104.2 |     57.6 |    100.1 |       56.9 |        — |     50.6 | JAX |
| sinh_100k              |      52.7 |    126.8 |     93.4 |      128.8 |    528.5 |    114.0 | Peregrine |
| cosh_100k              |      47.9 |    125.7 |     89.5 |      132.3 |    469.9 |     68.9 | Peregrine |
| arcsin_100k            |      53.8 |     75.7 |     94.2 |       59.0 |   2878.5 |    111.1 | Peregrine |
| arccos_100k            |      62.7 |     87.9 |    110.3 |       54.3 |        — |    207.5 | TensorFlow |
| arctan_100k            |      54.9 |     91.8 |     92.9 |       58.6 |   3002.8 |    218.3 | Peregrine |
| arcsinh_100k           |     211.3 |    164.1 |    332.1 |      136.8 |        — |    121.2 | JAX |
| maximum_100k           |      12.8 |     38.9 |     27.5 |       42.0 |    189.5 |     30.2 | Peregrine |
| minimum_100k           |      12.8 |     36.3 |     27.4 |       41.7 |    370.6 |     31.5 | Peregrine |
| power_100k             |     158.5 |    229.2 |    217.9 |      265.2 |        — |    142.4 | JAX |
| arctan2_100k           |      98.0 |    132.7 |    151.9 |       69.3 |        — |    315.5 | TensorFlow |
| logaddexp_100k         |     280.9 |    150.2 |    262.1 |      354.6 |        — |    143.0 | JAX |
| clip_100k              |       8.0 |     40.4 |     35.6 |       42.1 |    534.0 |     41.4 | Peregrine |
| where_100k             |      14.6 |     52.5 |     28.4 |       65.3 |    274.5 |     33.2 | Peregrine |
| greater_100k           |       9.7 |     49.5 |     25.5 |       52.1 |    187.0 |     27.0 | Peregrine |
| equal_100k             |       9.6 |     25.6 |     25.5 |       53.0 |    284.1 |     25.1 | Peregrine |
| sum_axis_256x512       |      18.8 |     40.9 |     24.0 |       49.4 |    206.1 |     52.5 | Peregrine |
| mean_axis_256x512      |      18.9 |     41.5 |     25.5 |       51.0 |    295.2 |     51.7 | Peregrine |
| max_axis_256x512       |      13.7 |     53.7 |     41.0 |       50.3 |    202.0 |     44.6 | Peregrine |
| min_axis_256x512       |      13.8 |     54.3 |     41.3 |       49.7 |    323.5 |     45.0 | Peregrine |
| var_256x512            |      47.1 |    304.8 |     63.4 |      224.8 |        — |     82.9 | Peregrine |
| prod_axis_256x512      |      24.8 |     40.2 |     26.1 |       52.5 |        — |     53.9 | Peregrine |
| logsumexp_256x512      |      98.4 |    197.6 |    108.0 |      324.6 |        — |    278.8 | Peregrine |
| cumsum_256x512         |     120.6 |     72.1 |    128.7 |      197.8 |    611.1 |    209.5 | PyTorch |
| argmax_axis_256x512    |      53.3 |     97.2 |    170.2 |       74.6 |   1297.8 |    176.8 | Peregrine |
| sum_axis_1024x1024     |     179.4 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     441.2 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       8.0 |     39.3 |     56.0 |       57.0 |   1800.0 |     37.1 | Peregrine |
| triu_256x256           |       7.9 |     39.2 |     54.9 |       51.7 |   1811.3 |     36.2 | Peregrine |
| repeat_64x128_2x3      |       6.1 |     46.9 |     31.1 |       75.1 |        — |     27.9 | Peregrine |
| pad_64x128             |       2.6 |      4.1 |     18.2 |       83.8 |     89.6 |     18.0 | Peregrine |
| stack_8x64x128         |       3.9 |      8.7 |     44.8 |       55.8 |    907.1 |    158.1 | Peregrine |
| diagonal_512x512       |       0.8 |      0.6 |     29.1 |       12.5 |        — |      8.4 | PyTorch |
| silu_100k              |      66.0 |     74.5 |     84.5 |      223.5 |    327.1 |     52.3 | JAX |
| softplus_100k          |     186.3 |    151.1 |    261.2 |      122.3 |    787.9 |    155.2 | TensorFlow |
| mish_100k              |     294.9 |    309.9 |    371.0 |      241.8 |   1143.0 |    225.7 | JAX |
| leaky_relu_100k        |       9.0 |     41.7 |     75.8 |       19.7 |        — |     27.6 | Peregrine |
| elu_100k               |      61.8 |    129.6 |    114.6 |      129.9 |    855.0 |     79.0 | Peregrine |
| hard_tanh_100k         |       9.0 |     40.1 |     34.6 |       42.0 |        — |     41.2 | Peregrine |
| relu6_100k             |       9.0 |     39.6 |     47.8 |       51.2 |    735.5 |    111.0 | Peregrine |
| hardswish_100k         |      10.3 |     41.1 |     68.5 |      206.9 |        — |     28.2 | Peregrine |
| gelu_100k              |      98.2 |     66.4 |    135.5 |      248.3 |    848.1 |    203.8 | PyTorch |
| selu_100k              |      65.7 |    128.3 |     85.2 |      137.3 |    747.0 |     82.4 | Peregrine |
| softsign_100k          |      39.2 |    121.5 |     38.4 |       48.4 |        — |     56.6 | MLX |
| cross_entropy_64x10    |       2.7 |     37.7 |     22.5 |      626.1 |   3331.0 |     53.7 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.3 |     19.8 |       42.6 |   1113.4 |     12.0 | Peregrine |
| mse_loss_64x10         |       3.7 |      4.9 |     21.4 |       38.8 |    445.1 |     23.2 | Peregrine |
| huber_loss_64x10       |       5.2 |      4.7 |     35.0 |      242.0 |        — |     47.1 | PyTorch |
| smooth_l1_loss_64x10   |       5.1 |      5.1 |     33.6 |      234.7 |        — |     46.9 | PyTorch |
| kl_div_loss_64x10      |       2.5 |      6.3 |     17.8 |      375.4 |        — |     61.4 | Peregrine |
| cosine_sim_loss_64x64  |       1.9 |     10.1 |    110.7 |      236.1 |        — |     68.3 | Peregrine |
| rmsnorm_64x512         |      19.1 |     67.6 |     33.9 |      439.6 |        — |     71.2 | Peregrine |
| conv1d_1x32x128_k3     |      20.2 |     54.5 |     28.6 |      511.4 |        — |     74.7 | Peregrine |
| avgpool2d_1x16x32x32   |      25.8 |     43.7 |    261.6 |       63.0 |        — |     41.7 | Peregrine |
| groupnorm_4x64x16x16   |      22.5 |     56.0 |    221.5 |      740.5 |        — |    271.5 | Peregrine |
| rnn_seq32_128_256      |     186.5 |    267.7 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1086.8 |    804.4 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     848.7 |    780.8 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     807.6 |   1275.9 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     940.8 |   1116.4 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     938.7 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1273.6 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      60.2 |    257.3 |    481.1 |      124.3 |   2364.4 |    531.0 | Peregrine |
| rand_normal_100k       |     236.4 |    971.4 |    686.2 |      327.9 |   3246.6 |    605.8 | Peregrine |
| rand_bernoulli_100k    |     118.3 |    250.0 |    449.7 |      202.5 |        — |    528.3 | Peregrine |
| rand_uniform_1M        |     600.5 |   2568.6 |   4542.5 |      417.6 |   2375.6 |   2257.4 | TensorFlow |
| rand_normal_1M         |    2370.5 |   9699.6 |   6576.1 |     2072.3 |   3257.7 |   2887.1 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     24.9 |       42.8 |        — |     60.9 | Peregrine |
| rfft_4k                |       6.5 |     14.8 |     32.9 |       53.8 |        — |     65.7 | Peregrine |
| rfft_16k               |      30.3 |     65.4 |     77.8 |      104.1 |        — |    116.0 | Peregrine |
| fft_1k                 |       3.3 |      6.6 |     24.9 |        8.6 |        — |     16.8 | Peregrine |
| fft_4k                 |      12.1 |     26.1 |     45.3 |       17.2 |        — |     58.6 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     20.4 |       68.2 |        — |      3.8 | Peregrine |
| solve_64x64            |      12.0 |     24.3 |    101.0 |       24.4 |        — |     32.8 | Peregrine |
| inv_64x64              |      37.4 |     26.2 |     51.7 |       32.4 |        — |     44.3 | PyTorch |
| cholesky_64x64         |       9.5 |     44.6 |     21.6 |       19.4 |        — |     20.7 | Peregrine |
| svd_64x64              |     274.8 |    281.5 |    293.4 |      484.2 |        — |    300.7 | Peregrine |
| qr_64x64               |      41.4 |     80.5 |     58.5 |       83.7 |        — |     63.9 | Peregrine |
| eigh_64x64             |     379.8 |    213.5 |    234.1 |      150.6 |        — |    237.8 | TensorFlow |
| det_64x64              |      23.0 |     19.9 |        — |       22.9 |        — |     36.2 | PyTorch |
| solve_128x128          |      50.0 |     45.2 |    189.2 |       76.3 |        — |     85.7 | PyTorch |
| inv_128x128            |      92.3 |     61.8 |     90.7 |      138.9 |        — |     83.9 | PyTorch |
| cholesky_128x128       |      50.2 |     50.1 |     26.0 |       60.4 |        — |     35.5 | MLX |
| svd_128x128            |     986.1 |    988.8 |    961.1 |     1855.2 |        — |   1008.3 | MLX |
| qr_128x128             |     188.2 |    226.0 |    197.9 |      351.7 |        — |    190.7 | Peregrine |
| eigh_128x128           |    1841.2 |    704.9 |    727.2 |      732.1 |        — |    745.2 | PyTorch |
| det_128x128            |      52.2 |     49.7 |        — |       81.7 |        — |     76.3 | PyTorch |
| solve_256x256          |     188.5 |    170.4 |    730.9 |      379.4 |        — |    265.7 | PyTorch |
| inv_256x256            |     471.5 |    296.0 |    251.8 |      847.2 |        — |    332.3 | MLX |
| cholesky_256x256       |     226.5 |     74.8 |     55.8 |      281.1 |        — |    116.6 | MLX |
| svd_256x256            |    6010.1 |   5691.0 |   5699.8 |     8110.6 |        — |   5842.6 | PyTorch |
| qr_256x256             |    1000.5 |   1001.9 |    987.6 |     1693.1 |        — |    991.8 | MLX |
| eigh_256x256           |    5990.4 |   3487.3 |   3470.6 |     4502.1 |        — |   3563.1 | MLX |
| det_256x256            |     212.2 |    206.5 |        — |      431.6 |        — |    205.5 | JAX |
| matmul_bias_gelu_196x768x3072 |    1882.2 |    859.5 |        — |     2388.7 |   1219.7 |   2124.7 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3243.1 |   1915.5 |        — |     3662.5 |   1236.4 |   3374.1 | tinygrad |
| add_layernorm_196x768  |     105.7 |    104.2 |        — |     1302.7 |   1102.4 |    235.7 | PyTorch |
| add_layernorm_196x1024 |     140.5 |    106.1 |        — |     1312.6 |   1110.9 |    283.2 | PyTorch |
| matmul_f32_196x768x3072 |     594.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14570.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1470.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26039.0 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.56x** (Peregrine is faster)
- **Peregrine vs MLX: 0.41x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.31x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.39x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 91/141 ops
- PyTorch: 24/141 ops
- JAX: 10/141 ops
- MLX: 8/141 ops
- TensorFlow: 7/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
