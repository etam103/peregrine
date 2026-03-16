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
| matmul_128x128         |       4.7 |      6.2 |     24.0 |       95.7 |    418.9 |     80.4 | Peregrine |
| matmul_256x256         |      29.5 |     31.8 |     47.8 |      197.7 |    423.8 |    171.3 | Peregrine |
| matmul_512x512         |     134.8 |    140.0 |    172.1 |      682.6 |    433.2 |    518.9 | Peregrine |
| matmul_1024x1024       |     962.4 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    8790.7 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.5 |     41.1 |     32.3 |       50.1 |    190.5 |     33.4 | Peregrine |
| add_500k               |      61.6 |     57.4 |     85.8 |       84.9 |    192.2 |     61.6 | PyTorch |
| add_1M                 |     125.8 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     605.6 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     904.0 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     39.0 |     32.0 |       40.7 |    192.3 |     28.4 | Peregrine |
| mul_500k               |      61.6 |     57.1 |     83.5 |       74.8 |    195.4 |     59.0 | PyTorch |
| mul_1M                 |     125.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     528.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |    1049.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      43.5 |     62.2 |     61.1 |       63.9 |    225.5 |     46.9 | Peregrine |
| exp_500k               |      89.5 |    137.4 |    232.4 |      102.2 |    225.8 |    117.8 | Peregrine |
| exp_1M                 |     116.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     408.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     730.3 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.5 |     38.5 |     30.5 |       39.7 |    338.9 |     97.7 | Peregrine |
| relu_1M                |      82.4 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.1 |     35.2 |     18.4 |       11.3 |    609.4 |     30.9 | Peregrine |
| softmax_8x512          |       4.0 |     32.1 |     20.7 |       14.2 |    637.5 |     33.4 | Peregrine |
| mlp_fwd_64x784         |      25.9 |     27.9 |     54.7 |      248.9 |   1753.0 |    182.1 | Peregrine |
| mlp_fwd_256x784_wide   |     309.8 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     746.5 |   1254.5 |    810.0 |     8443.6 |  24421.2 |   5208.6 | Peregrine |
| train_step_256_wide    |    3016.9 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.5 |     41.0 |     25.2 |       48.1 |    167.2 |     32.2 | Peregrine |
| square_100k            |       8.4 |     38.2 |     24.6 |       16.3 |    183.7 |     33.6 | Peregrine |
| rsqrt_100k             |      21.5 |     40.7 |     36.7 |       51.5 |        — |     82.5 | Peregrine |
| floor_100k             |       8.7 |     38.3 |     24.1 |       16.3 |    430.7 |     29.3 | Peregrine |
| ceil_100k              |       8.6 |     41.2 |     23.4 |       16.2 |    368.2 |     28.0 | Peregrine |
| round_100k             |       8.7 |     40.9 |     22.8 |       43.0 |        — |     30.0 | Peregrine |
| sign_100k              |       8.7 |     39.7 |     28.2 |       46.2 |    847.3 |     36.1 | Peregrine |
| expm1_100k             |      63.2 |    109.6 |    107.4 |      149.3 |        — |     98.5 | Peregrine |
| log2_100k              |      55.6 |     85.6 |     97.4 |      153.1 |    163.8 |     65.7 | Peregrine |
| log10_100k             |      58.0 |     83.6 |    105.7 |      154.3 |        — |     64.4 | Peregrine |
| log1p_100k             |      75.5 |     81.2 |    128.2 |       94.1 |        — |    110.9 | Peregrine |
| erf_100k               |      49.1 |     57.2 |    101.0 |       57.0 |        — |     57.0 | Peregrine |
| sinh_100k              |      51.1 |    135.6 |     93.6 |      133.9 |    542.5 |    124.7 | Peregrine |
| cosh_100k              |      46.4 |    128.8 |     89.5 |      135.3 |    454.5 |     71.6 | Peregrine |
| arcsin_100k            |      52.1 |     72.3 |     94.1 |       56.2 |   2981.3 |    126.1 | Peregrine |
| arccos_100k            |      60.8 |     87.9 |    110.6 |       54.8 |        — |    216.9 | TensorFlow |
| arctan_100k            |      53.2 |     91.8 |     93.5 |       59.6 |   3292.2 |    220.2 | Peregrine |
| arcsinh_100k           |     117.7 |    156.5 |    332.4 |      136.6 |        — |    114.6 | JAX |
| maximum_100k           |      12.5 |     36.0 |     27.8 |       44.9 |    190.8 |     33.4 | Peregrine |
| minimum_100k           |      12.5 |     37.7 |     27.7 |       40.4 |    382.3 |     32.1 | Peregrine |
| power_100k             |     153.8 |    235.7 |    211.3 |      282.3 |        — |    140.7 | JAX |
| arctan2_100k           |      58.1 |    125.5 |    145.6 |       71.8 |        — |    317.1 | Peregrine |
| logaddexp_100k         |     148.2 |    154.3 |    256.2 |      366.3 |        — |    151.5 | Peregrine |
| clip_100k              |       8.7 |     42.6 |     35.5 |       42.6 |    545.7 |     42.5 | Peregrine |
| where_100k             |      16.4 |     49.2 |     27.2 |       66.1 |    277.9 |     34.5 | Peregrine |
| greater_100k           |      12.5 |     49.3 |     25.0 |       51.6 |    191.3 |     32.3 | Peregrine |
| equal_100k             |      12.5 |     34.9 |     24.8 |       61.2 |    296.0 |     26.2 | Peregrine |
| sum_axis_256x512       |      18.8 |     39.7 |     23.6 |       55.8 |    208.5 |     58.4 | Peregrine |
| mean_axis_256x512      |      18.9 |     41.6 |     24.8 |       52.3 |    299.8 |     22.2 | Peregrine |
| max_axis_256x512       |      13.7 |     55.9 |     41.9 |       50.2 |    202.2 |     48.4 | Peregrine |
| min_axis_256x512       |      13.7 |     56.2 |     42.0 |       48.6 |    331.0 |     51.6 | Peregrine |
| var_256x512            |      45.7 |    271.5 |     63.1 |      209.8 |        — |     89.1 | Peregrine |
| prod_axis_256x512      |      24.2 |     36.9 |     27.1 |       53.3 |        — |     54.6 | Peregrine |
| logsumexp_256x512      |      95.5 |    194.0 |    106.8 |      335.7 |        — |    287.5 | Peregrine |
| cumsum_256x512         |      48.8 |     76.4 |    129.5 |      189.3 |    640.8 |    211.3 | Peregrine |
| argmax_axis_256x512    |      51.8 |     96.8 |    171.6 |       67.4 |   1317.4 |    168.8 | Peregrine |
| sum_axis_1024x1024     |     174.1 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.7 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.7 |     37.1 |     56.8 |       56.1 |   1965.2 |     37.5 | Peregrine |
| triu_256x256           |       7.6 |     39.0 |     57.1 |       54.7 |   1915.1 |     36.3 | Peregrine |
| repeat_64x128_2x3      |       7.2 |     46.7 |     33.6 |       75.7 |        — |     28.7 | Peregrine |
| pad_64x128             |       2.6 |      4.3 |     19.2 |       81.8 |     89.6 |     19.8 | Peregrine |
| stack_8x64x128         |       3.8 |      8.6 |     46.9 |       56.0 |   1376.9 |    176.4 | Peregrine |
| diagonal_512x512       |       0.4 |      0.6 |     29.6 |       12.3 |        — |      9.6 | Peregrine |
| silu_100k              |      47.0 |     68.8 |     89.8 |      226.0 |    370.4 |     78.6 | Peregrine |
| softplus_100k          |     133.5 |    151.1 |    269.7 |      131.2 |    817.7 |    191.3 | TensorFlow |
| mish_100k              |     136.7 |    307.9 |    387.4 |      244.6 |   1187.1 |    258.7 | Peregrine |
| leaky_relu_100k        |       8.6 |     38.9 |     78.9 |       19.5 |        — |     34.4 | Peregrine |
| elu_100k               |      60.0 |    133.5 |    123.0 |      136.9 |    869.3 |     98.3 | Peregrine |
| hard_tanh_100k         |       8.6 |     41.5 |     33.5 |       41.8 |        — |     43.5 | Peregrine |
| relu6_100k             |       8.6 |     40.2 |     41.8 |       50.2 |    742.3 |    119.2 | Peregrine |
| hardswish_100k         |      10.0 |     40.7 |     65.0 |      220.3 |        — |     37.4 | Peregrine |
| gelu_100k              |      56.3 |     74.0 |    146.2 |      249.2 |    844.8 |    219.3 | Peregrine |
| selu_100k              |      63.7 |    132.9 |     86.5 |      137.3 |    794.2 |     92.9 | Peregrine |
| softsign_100k          |      38.3 |    118.9 |     41.9 |       45.8 |        — |     63.1 | Peregrine |
| cross_entropy_64x10    |       2.5 |     40.1 |     24.8 |      614.7 |   3622.5 |     56.7 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.4 |     20.6 |       42.7 |   1152.8 |     12.2 | Peregrine |
| mse_loss_64x10         |       3.5 |      4.9 |     22.9 |       38.7 |    452.9 |     24.1 | Peregrine |
| huber_loss_64x10       |       0.3 |      4.7 |     34.8 |      238.2 |        — |     48.8 | Peregrine |
| smooth_l1_loss_64x10   |       0.8 |      5.1 |     35.0 |      236.7 |        — |     48.2 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.3 |     20.0 |      372.4 |        — |     60.6 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.3 |    104.3 |      236.0 |        — |     55.8 | Peregrine |
| rmsnorm_64x512         |      18.7 |     63.5 |     32.0 |      439.4 |        — |     78.2 | Peregrine |
| conv1d_1x32x128_k3     |      20.5 |     54.6 |     32.6 |      504.2 |        — |     74.6 | Peregrine |
| avgpool2d_1x16x32x32   |      25.1 |     44.0 |    274.9 |       65.4 |        — |     46.3 | Peregrine |
| groupnorm_4x64x16x16   |      21.4 |     52.1 |    231.0 |      754.5 |        — |    271.6 | Peregrine |
| rnn_seq32_128_256      |     180.4 |    269.0 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |     929.6 |    810.3 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     761.5 |    781.9 |        — |          — |        — |        — | Peregrine |
| optim_adam_64          |     744.0 |   1269.1 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     881.2 |   1114.9 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     867.1 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1232.2 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      60.2 |    257.6 |    522.1 |      126.2 |   2494.9 |    593.4 | Peregrine |
| rand_normal_100k       |     236.8 |    970.0 |    725.8 |      326.1 |   3338.5 |    651.5 | Peregrine |
| rand_bernoulli_100k    |     118.5 |    250.3 |    481.8 |      216.5 |        — |    570.3 | Peregrine |
| rand_uniform_1M        |     441.6 |   2562.5 |   4684.3 |      421.1 |   2454.8 |   2448.5 | TensorFlow |
| rand_normal_1M         |     714.5 |   9694.8 |   6785.7 |     2071.0 |   3278.4 |   3002.3 | Peregrine |
| rfft_1k                |       2.2 |      4.4 |     19.0 |       42.7 |        — |     60.5 | Peregrine |
| rfft_4k                |       7.3 |     14.8 |     29.5 |       54.2 |        — |     64.1 | Peregrine |
| rfft_16k               |      30.3 |     65.2 |     76.6 |      104.5 |        — |    121.4 | Peregrine |
| fft_1k                 |       3.3 |      6.6 |     21.9 |        8.8 |        — |     44.3 | Peregrine |
| fft_4k                 |      12.2 |     26.4 |     39.8 |       17.3 |        — |     63.4 | Peregrine |
| norm_l2_1k             |       1.1 |      1.2 |     17.7 |       68.3 |        — |      4.0 | Peregrine |
| solve_64x64            |       9.2 |     18.4 |     94.5 |       24.5 |        — |     40.6 | Peregrine |
| inv_64x64              |      15.2 |     26.6 |     47.4 |       32.8 |        — |     45.4 | Peregrine |
| cholesky_64x64         |       7.2 |     43.1 |     21.3 |       19.5 |        — |     24.5 | Peregrine |
| svd_64x64              |     274.5 |    277.0 |    289.4 |      495.5 |        — |    324.1 | Peregrine |
| qr_64x64               |      41.5 |     80.8 |     56.1 |       83.7 |        — |     71.5 | Peregrine |
| eigh_64x64             |     163.7 |    214.8 |    228.3 |      141.5 |        — |    259.7 | TensorFlow |
| det_64x64              |      11.2 |     20.8 |        — |       23.1 |        — |     35.2 | Peregrine |
| solve_128x128          |      36.2 |     44.8 |    187.2 |       76.7 |        — |     85.8 | Peregrine |
| inv_128x128            |      48.5 |     62.0 |     84.6 |      139.1 |        — |     89.5 | Peregrine |
| cholesky_128x128       |      14.2 |     46.4 |     26.7 |       58.5 |        — |     37.7 | Peregrine |
| svd_128x128            |     984.5 |    987.9 |    973.1 |     1803.9 |        — |   1048.9 | MLX |
| qr_128x128             |     188.6 |    218.5 |    192.0 |      326.1 |        — |    195.1 | Peregrine |
| eigh_128x128           |     526.2 |    706.4 |    719.5 |      708.8 |        — |    764.0 | Peregrine |
| det_128x128            |      41.1 |     49.5 |        — |       81.8 |        — |     76.8 | Peregrine |
| solve_256x256          |     111.8 |    173.1 |    732.6 |      378.7 |        — |    266.0 | Peregrine |
| inv_256x256            |     177.8 |    292.0 |    240.5 |      850.1 |        — |    347.5 | Peregrine |
| cholesky_256x256       |      46.4 |     73.8 |     54.1 |      281.4 |        — |    121.3 | Peregrine |
| svd_256x256            |    5854.6 |   5663.9 |   5649.0 |     8041.9 |        — |   6033.3 | MLX |
| qr_256x256             |     981.3 |   1002.2 |   1007.3 |     1694.0 |        — |   1025.5 | Peregrine |
| eigh_256x256           |    2767.6 |   3434.2 |   3451.1 |     4567.8 |        — |   3621.0 | Peregrine |
| det_256x256            |     141.3 |    202.5 |        — |      430.7 |        — |    207.0 | Peregrine |
| matmul_bias_gelu_196x768x3072 |     985.6 |    849.2 |        — |     2367.4 |   1316.6 |   2198.5 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    1927.9 |   1898.7 |        — |     3694.0 |   1252.7 |   3954.3 | tinygrad |
| add_layernorm_196x768  |      72.0 |    101.0 |        — |     1224.3 |   1137.2 |    259.0 | Peregrine |
| add_layernorm_196x1024 |      97.2 |    105.9 |        — |     1278.3 |   1240.4 |    287.6 | Peregrine |
| matmul_f32_196x768x3072 |     500.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   12060.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1420.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   21728.7 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.44x** (Peregrine is faster)
- **Peregrine vs MLX: 0.32x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.24x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.04x** (Peregrine is faster)
- **Peregrine vs JAX: 0.29x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 128/141 ops
- PyTorch: 4/141 ops
- TensorFlow: 4/141 ops
- MLX: 2/141 ops
- JAX: 2/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
