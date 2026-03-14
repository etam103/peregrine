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
| matmul_128x128         |       5.7 |      5.9 |     19.9 |       94.0 |    417.4 |     79.8 | Peregrine |
| matmul_256x256         |      69.2 |     30.1 |     47.8 |      202.0 |    422.9 |    147.4 | PyTorch |
| matmul_512x512         |     222.2 |    141.8 |    166.7 |      689.8 |    444.5 |    536.2 | PyTorch |
| matmul_1024x1024       |    1062.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    8972.1 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     42.2 |     28.9 |       48.1 |    188.5 |     36.9 | Peregrine |
| add_500k               |     104.9 |     57.1 |     82.7 |       80.5 |    186.2 |     62.6 | PyTorch |
| add_1M                 |     120.2 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     529.3 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     983.4 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     37.7 |     29.0 |       44.2 |    186.1 |     30.9 | Peregrine |
| mul_500k               |     114.6 |     57.6 |     81.7 |       73.9 |    188.2 |     63.1 | PyTorch |
| mul_1M                 |     133.3 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     535.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     815.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      89.6 |     64.8 |     60.7 |       63.7 |    221.5 |     46.4 | JAX |
| exp_500k               |     203.1 |    140.5 |    223.3 |      100.2 |    221.5 |    117.0 | TensorFlow |
| exp_1M                 |     284.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1090.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2103.7 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     39.3 |     24.9 |       41.8 |    341.5 |     97.5 | Peregrine |
| relu_1M                |     104.4 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.3 |     32.8 |     15.9 |       11.5 |    633.1 |     30.9 | Peregrine |
| softmax_8x512          |       4.4 |     32.6 |     18.7 |       14.3 |    619.0 |     33.7 | Peregrine |
| mlp_fwd_64x784         |      33.5 |     27.7 |     55.2 |      247.3 |   1795.9 |    176.3 | PyTorch |
| mlp_fwd_256x784_wide   |     425.4 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     818.1 |   1200.7 |    775.3 |     8619.5 |  22984.0 |   5055.2 | MLX |
| train_step_256_wide    |    3318.2 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.8 |     37.0 |     24.5 |       49.1 |    163.6 |     28.1 | Peregrine |
| square_100k            |       8.8 |     37.9 |     23.4 |       16.4 |    175.3 |     27.3 | Peregrine |
| rsqrt_100k             |      70.9 |     41.6 |     36.2 |       52.4 |        — |     92.1 | MLX |
| floor_100k             |       8.8 |     40.9 |     23.5 |       16.2 |    416.0 |     25.0 | Peregrine |
| ceil_100k              |       8.8 |     40.9 |     23.5 |       17.7 |    355.5 |     30.6 | Peregrine |
| round_100k             |       8.8 |     40.8 |     23.5 |       45.6 |        — |     28.0 | Peregrine |
| sign_100k              |       8.8 |     39.0 |     27.5 |       45.8 |    795.7 |     36.6 | Peregrine |
| expm1_100k             |     167.2 |    108.9 |    107.7 |      141.0 |        — |     99.2 | JAX |
| log2_100k              |     122.8 |     86.5 |    101.2 |      153.1 |    161.9 |     56.4 | JAX |
| log10_100k             |     109.4 |     85.5 |    108.2 |      150.1 |        — |     56.1 | JAX |
| log1p_100k             |     105.9 |     81.3 |    130.5 |       95.5 |        — |    104.7 | PyTorch |
| erf_100k               |      97.2 |     54.5 |    100.8 |       58.8 |        — |     52.9 | JAX |
| sinh_100k              |      52.0 |    133.2 |     93.4 |      135.3 |    531.3 |    113.8 | Peregrine |
| cosh_100k              |      47.2 |    128.5 |     89.6 |      126.5 |    464.8 |     68.8 | Peregrine |
| arcsin_100k            |      53.1 |     75.5 |     93.9 |       56.0 |   2898.9 |    111.1 | Peregrine |
| arccos_100k            |     146.8 |     88.3 |    110.2 |       52.6 |        — |    200.4 | TensorFlow |
| arctan_100k            |      54.1 |     93.8 |     92.8 |       59.1 |   3029.3 |    213.4 | Peregrine |
| arcsinh_100k           |     154.2 |    141.4 |    333.0 |      141.9 |        — |    119.2 | JAX |
| maximum_100k           |      12.7 |     36.9 |     27.7 |       44.6 |    193.0 |     31.8 | Peregrine |
| minimum_100k           |      12.7 |     39.0 |     27.5 |       41.2 |    380.2 |     30.2 | Peregrine |
| power_100k             |     392.0 |    245.0 |    210.5 |      271.4 |        — |    141.1 | JAX |
| arctan2_100k           |      96.9 |    125.4 |    144.4 |       77.4 |        — |    314.5 | TensorFlow |
| logaddexp_100k         |     414.8 |    149.1 |    255.5 |      366.9 |        — |    140.0 | JAX |
| clip_100k              |       8.8 |     39.4 |     35.1 |       42.4 |    542.0 |     36.0 | Peregrine |
| where_100k             |      16.7 |     50.9 |     28.4 |       65.8 |    281.6 |     30.8 | Peregrine |
| greater_100k           |      12.7 |     48.8 |     24.6 |       53.3 |    187.7 |     30.4 | Peregrine |
| equal_100k             |      12.7 |     27.9 |     24.1 |       58.5 |    290.4 |     26.7 | Peregrine |
| sum_axis_256x512       |     114.5 |     38.2 |     23.2 |       50.9 |    207.7 |     49.9 | MLX |
| mean_axis_256x512      |     114.9 |     41.9 |     24.7 |       52.8 |    291.6 |     48.2 | MLX |
| max_axis_256x512       |     154.5 |     53.3 |     39.7 |       49.4 |    203.9 |     45.0 | MLX |
| min_axis_256x512       |     154.5 |     54.8 |     40.8 |       50.6 |    325.6 |     46.7 | MLX |
| var_256x512            |     235.6 |    274.0 |     57.1 |      210.9 |        — |     80.6 | MLX |
| prod_axis_256x512      |     149.5 |     38.2 |     25.1 |       52.3 |        — |     55.7 | MLX |
| logsumexp_256x512      |     387.9 |    180.9 |    106.7 |      345.4 |        — |    276.5 | MLX |
| cumsum_256x512         |     123.8 |     64.6 |    128.3 |      189.4 |    618.5 |    210.8 | PyTorch |
| argmax_axis_256x512    |     157.5 |     89.5 |    174.1 |       75.5 |   1308.7 |    169.5 | TensorFlow |
| sum_axis_1024x1024     |     941.0 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    1919.6 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      34.6 |     39.2 |     57.5 |       57.0 |   1807.9 |     38.0 | Peregrine |
| triu_256x256           |      34.7 |     37.5 |     56.3 |       55.5 |   1788.7 |     39.5 | Peregrine |
| repeat_64x128_2x3      |     124.8 |     45.6 |     31.4 |       74.7 |        — |     28.0 | JAX |
| pad_64x128             |      17.3 |      4.1 |     18.6 |       82.3 |     91.2 |     18.3 | PyTorch |
| stack_8x64x128         |      16.3 |      8.6 |     53.2 |       58.5 |    919.5 |    159.3 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     32.3 |       12.7 |        — |      9.7 | PyTorch |
| silu_100k              |      65.2 |     69.4 |     85.3 |      228.7 |    326.3 |     52.0 | JAX |
| softplus_100k          |     305.4 |    150.6 |    266.8 |      122.7 |    779.5 |    154.7 | TensorFlow |
| mish_100k              |     525.5 |    308.2 |    373.3 |      246.4 |   1168.2 |    231.8 | JAX |
| leaky_relu_100k        |       8.8 |     40.5 |     79.2 |       19.4 |        — |     30.6 | Peregrine |
| elu_100k               |     167.1 |    123.8 |    116.0 |      133.9 |    878.2 |     77.3 | JAX |
| hard_tanh_100k         |      51.5 |     39.5 |     34.6 |       42.1 |        — |     38.7 | MLX |
| relu6_100k             |      51.5 |     40.0 |     44.0 |       56.1 |    736.1 |    113.3 | PyTorch |
| hardswish_100k         |      85.2 |     38.9 |     69.2 |      220.5 |        — |     28.8 | JAX |
| gelu_100k              |      77.7 |     73.8 |    135.6 |      243.9 |    847.8 |    211.7 | PyTorch |
| selu_100k              |     162.2 |    131.2 |     85.5 |      129.8 |    737.5 |     81.7 | JAX |
| softsign_100k          |      35.4 |    119.7 |     44.5 |       45.4 |        — |     60.4 | Peregrine |
| cross_entropy_64x10    |       2.6 |     39.2 |     23.8 |      622.6 |   3348.2 |     53.7 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.2 |     18.6 |       44.0 |   1120.8 |     12.5 | Peregrine |
| mse_loss_64x10         |       3.7 |      4.8 |     22.1 |       40.3 |    445.5 |     24.1 | Peregrine |
| huber_loss_64x10       |       5.2 |      4.7 |     33.6 |      245.5 |        — |     48.0 | PyTorch |
| smooth_l1_loss_64x10   |       5.0 |      5.0 |     33.3 |      242.0 |        — |     48.0 | PyTorch |
| kl_div_loss_64x10      |       2.6 |      6.3 |     17.6 |      384.4 |        — |     58.1 | Peregrine |
| cosine_sim_loss_64x64  |      13.8 |     10.3 |    112.5 |      245.0 |        — |     70.8 | PyTorch |
| rmsnorm_64x512         |      58.7 |     67.1 |     32.9 |      446.8 |        — |     72.8 | MLX |
| conv1d_1x32x128_k3     |      20.7 |     55.4 |     28.5 |      505.0 |        — |     74.3 | Peregrine |
| avgpool2d_1x16x32x32   |      25.5 |     44.5 |    262.5 |       62.7 |        — |     43.4 | Peregrine |
| groupnorm_4x64x16x16   |      72.5 |     54.6 |    222.0 |      761.4 |        — |    268.3 | PyTorch |
| rnn_seq32_128_256      |     185.4 |    267.5 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1129.2 |    801.1 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     851.1 |    776.3 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     806.4 |   1518.1 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     931.6 |   1145.2 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     918.6 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1293.7 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     108.3 |    259.7 |    482.0 |      124.9 |   2376.4 |    539.6 | Peregrine |
| rand_normal_100k       |     777.0 |    988.9 |    686.2 |      341.3 |   3256.9 |    613.3 | TensorFlow |
| rand_bernoulli_100k    |     308.1 |    250.1 |    449.0 |      216.0 |        — |    513.4 | TensorFlow |
| rand_uniform_1M        |    1084.3 |   2567.9 |   4534.9 |      432.1 |   2407.4 |   2259.2 | TensorFlow |
| rand_normal_1M         |    7760.1 |   9877.9 |   6602.2 |     2050.3 |   3245.2 |   2861.2 | TensorFlow |
| rfft_1k                |       2.0 |      4.5 |     23.3 |       44.1 |        — |     60.8 | Peregrine |
| rfft_4k                |       6.6 |     15.0 |     32.4 |       55.8 |        — |     67.6 | Peregrine |
| rfft_16k               |      29.5 |     66.6 |     77.9 |      107.8 |        — |    116.8 | Peregrine |
| fft_1k                 |       3.1 |      6.7 |     23.1 |        8.8 |        — |     17.4 | Peregrine |
| fft_4k                 |      11.9 |     26.6 |     44.1 |       17.6 |        — |     55.9 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     20.4 |       70.3 |        — |      3.9 | Peregrine |
| solve_64x64            |      11.4 |     23.6 |    100.6 |       24.8 |        — |     32.2 | Peregrine |
| inv_64x64              |      35.6 |     25.2 |     51.6 |       32.9 |        — |     37.0 | PyTorch |
| cholesky_64x64         |       8.5 |     48.2 |     21.8 |       19.8 |        — |     21.0 | Peregrine |
| svd_64x64              |     274.2 |    276.0 |    293.0 |      501.1 |        — |    297.3 | Peregrine |
| qr_64x64               |      40.9 |     81.2 |     58.6 |       85.3 |        — |     65.2 | Peregrine |
| eigh_64x64             |     385.2 |    214.5 |    229.4 |      151.4 |        — |    239.2 | TensorFlow |
| det_64x64              |      22.5 |     19.8 |        — |       23.2 |        — |     28.5 | PyTorch |
| solve_128x128          |      48.8 |     44.7 |    188.9 |       77.9 |        — |     85.5 | PyTorch |
| inv_128x128            |     102.8 |     61.5 |     90.5 |      141.3 |        — |     83.7 | PyTorch |
| cholesky_128x128       |      50.4 |     49.5 |     26.1 |       59.7 |        — |     35.8 | MLX |
| svd_128x128            |     986.4 |    992.2 |    993.0 |     1864.2 |        — |   1017.6 | Peregrine |
| qr_128x128             |     189.3 |    224.0 |    190.2 |      332.4 |        — |    190.8 | Peregrine |
| eigh_128x128           |    1840.2 |    703.1 |    714.5 |      715.6 |        — |    739.4 | PyTorch |
| det_128x128            |      52.1 |     49.7 |        — |       83.6 |        — |     75.8 | PyTorch |
| solve_256x256          |     188.5 |    182.1 |    728.0 |      379.9 |        — |    288.9 | PyTorch |
| inv_256x256            |     488.9 |    295.7 |    250.5 |      862.8 |        — |    334.3 | MLX |
| cholesky_256x256       |     226.3 |     77.1 |     55.5 |      287.0 |        — |    112.7 | MLX |
| svd_256x256            |    5924.6 |   5622.1 |   5638.0 |     8210.3 |        — |   5754.8 | PyTorch |
| qr_256x256             |    1005.6 |    989.9 |    992.0 |     1723.6 |        — |    978.9 | JAX |
| eigh_256x256           |    6013.2 |   3469.7 |   3423.3 |     4672.1 |        — |   3529.3 | MLX |
| det_256x256            |     212.2 |    204.9 |        — |      438.0 |        — |    206.2 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1059.8 |    852.6 |        — |     2356.8 |   1231.9 |   2115.6 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2028.2 |   1928.9 |        — |     3712.0 |   1253.0 |   3399.7 | tinygrad |
| add_layernorm_196x768  |     107.6 |    102.0 |        — |     1212.0 |   1138.7 |    231.6 | PyTorch |
| add_layernorm_196x1024 |     136.3 |    106.9 |        — |     1275.9 |   1136.2 |    274.9 | PyTorch |
| matmul_f32_196x768x3072 |     534.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14723.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1453.3 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26436.6 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.84x** (Peregrine is faster)
- **Peregrine vs MLX: 0.64x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.45x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.08x** (Peregrine is faster)
- **Peregrine vs JAX: 0.59x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 70/141 ops
- PyTorch: 30/141 ops
- MLX: 15/141 ops
- JAX: 15/141 ops
- TensorFlow: 10/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
