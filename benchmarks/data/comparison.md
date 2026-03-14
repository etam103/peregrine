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
| matmul_128x128         |      10.2 |      6.1 |     19.9 |       95.3 |    412.2 |     79.7 | PyTorch |
| matmul_256x256         |      54.3 |     31.7 |     48.0 |      194.9 |    415.8 |    158.9 | PyTorch |
| matmul_512x512         |     190.3 |    141.8 |    169.8 |      659.1 |    423.1 |    507.6 | PyTorch |
| matmul_1024x1024       |     969.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    8825.0 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.8 |     39.5 |     30.1 |       53.5 |    185.3 |     34.0 | Peregrine |
| add_500k               |      61.5 |     57.5 |     82.2 |       80.2 |    185.8 |     60.6 | PyTorch |
| add_1M                 |     125.8 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     497.0 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     855.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     39.2 |     28.3 |       42.3 |    187.1 |     32.5 | Peregrine |
| mul_500k               |      61.4 |     56.8 |     81.2 |       74.2 |    187.3 |     60.7 | PyTorch |
| mul_1M                 |     125.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     563.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     864.9 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.2 |     60.7 |     60.4 |       65.0 |    220.5 |     45.8 | JAX |
| exp_500k               |     155.7 |    144.2 |    219.2 |      100.9 |    220.6 |    119.7 | TensorFlow |
| exp_1M                 |     167.0 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     376.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     734.1 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.7 |     39.4 |     31.1 |       39.5 |    331.5 |     98.6 | Peregrine |
| relu_1M                |      82.2 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     30.3 |     18.4 |       11.8 |    608.1 |     30.6 | Peregrine |
| softmax_8x512          |       4.2 |     33.5 |     21.1 |       14.6 |    621.4 |     33.1 | Peregrine |
| mlp_fwd_64x784         |      33.1 |     27.5 |     51.3 |      245.4 |   1751.2 |    181.4 | PyTorch |
| mlp_fwd_256x784_wide   |     398.6 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     801.1 |   1281.4 |    778.2 |     8688.9 |  22883.8 |   5055.7 | MLX |
| train_step_256_wide    |    3282.2 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.7 |     38.4 |     24.5 |       48.2 |    164.7 |     29.1 | Peregrine |
| square_100k            |       8.7 |     41.2 |     23.6 |       16.7 |    174.7 |     28.0 | Peregrine |
| rsqrt_100k             |      21.5 |     43.4 |     36.5 |       47.6 |        — |     92.4 | Peregrine |
| floor_100k             |       8.7 |     39.1 |     25.3 |       16.3 |    408.9 |     28.8 | Peregrine |
| ceil_100k              |       8.7 |     41.5 |     25.2 |       16.1 |    352.1 |     28.5 | Peregrine |
| round_100k             |       8.7 |     44.2 |     25.1 |       42.6 |        — |     28.2 | Peregrine |
| sign_100k              |       8.7 |     40.2 |     29.5 |       53.0 |    785.6 |     33.6 | Peregrine |
| expm1_100k             |      63.1 |    127.0 |    108.7 |      141.3 |        — |     98.5 | Peregrine |
| log2_100k              |      55.6 |     82.1 |    101.9 |      148.7 |    165.2 |     55.8 | Peregrine |
| log10_100k             |      58.0 |     79.6 |    111.3 |      150.3 |        — |     56.0 | JAX |
| log1p_100k             |      75.4 |     78.3 |    127.8 |       94.2 |        — |    104.2 | Peregrine |
| erf_100k               |     100.7 |     53.6 |    100.9 |       57.3 |        — |     42.3 | JAX |
| sinh_100k              |      51.0 |    119.1 |     93.4 |      145.7 |    526.5 |    111.9 | Peregrine |
| cosh_100k              |      46.3 |    120.0 |     89.6 |      129.7 |    461.9 |     68.7 | Peregrine |
| arcsin_100k            |      52.1 |     71.2 |     94.0 |       56.0 |   2829.0 |    111.1 | Peregrine |
| arccos_100k            |      60.7 |     82.6 |    110.2 |       55.0 |        — |    210.7 | TensorFlow |
| arctan_100k            |      53.1 |     89.7 |     92.8 |       58.3 |   2991.1 |    213.0 | Peregrine |
| arcsinh_100k           |     204.8 |    146.7 |    332.1 |      136.0 |        — |    123.5 | JAX |
| maximum_100k           |      12.5 |     38.9 |     27.4 |       40.2 |    193.3 |     30.6 | Peregrine |
| minimum_100k           |      12.5 |     39.3 |     27.5 |       41.2 |    371.1 |     29.3 | Peregrine |
| power_100k             |     153.7 |    231.0 |    210.2 |      285.5 |        — |    141.8 | JAX |
| arctan2_100k           |      95.1 |    126.7 |    144.2 |       74.1 |        — |    316.7 | TensorFlow |
| logaddexp_100k         |     272.6 |    145.0 |    265.9 |      364.7 |        — |    148.3 | PyTorch |
| clip_100k              |       8.7 |     38.5 |     34.8 |       43.5 |    522.2 |     44.2 | Peregrine |
| where_100k             |      16.4 |     49.1 |     28.1 |       65.8 |    279.2 |     34.4 | Peregrine |
| greater_100k           |      12.5 |     45.7 |     23.8 |       52.2 |    188.0 |     34.4 | Peregrine |
| equal_100k             |      12.5 |     31.1 |     23.8 |       54.7 |    285.2 |     32.9 | Peregrine |
| sum_axis_256x512       |      18.8 |     40.8 |     25.1 |       50.4 |    203.3 |     53.9 | Peregrine |
| mean_axis_256x512      |      18.8 |     43.2 |     24.6 |       52.6 |    293.6 |     53.7 | Peregrine |
| max_axis_256x512       |      13.7 |     51.9 |     41.7 |       52.9 |    202.6 |     45.4 | Peregrine |
| min_axis_256x512       |      13.7 |     52.4 |     41.5 |       49.4 |    324.8 |     44.5 | Peregrine |
| var_256x512            |      45.7 |    261.3 |     57.5 |      207.6 |        — |     81.5 | Peregrine |
| prod_axis_256x512      |      24.9 |     42.6 |     25.7 |       50.6 |        — |     55.5 | Peregrine |
| logsumexp_256x512      |      98.6 |    189.9 |    106.7 |      351.0 |        — |    275.4 | Peregrine |
| cumsum_256x512         |     120.0 |     65.6 |    128.3 |      189.8 |    608.4 |    213.2 | PyTorch |
| argmax_axis_256x512    |      51.7 |     78.5 |    170.2 |       73.7 |   1299.5 |    175.9 | Peregrine |
| sum_axis_1024x1024     |     174.2 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.9 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       8.0 |     40.5 |     55.6 |       53.1 |   1783.4 |     36.2 | Peregrine |
| triu_256x256           |       8.0 |     38.9 |     55.4 |       52.6 |   1764.2 |     36.1 | Peregrine |
| repeat_64x128_2x3      |       7.1 |     48.6 |     30.5 |       78.3 |        — |     27.7 | Peregrine |
| pad_64x128             |       2.6 |      4.1 |     18.6 |       85.1 |     88.3 |     17.9 | Peregrine |
| stack_8x64x128         |       4.0 |      8.7 |     43.9 |       55.6 |    914.3 |    160.3 | Peregrine |
| diagonal_512x512       |       0.4 |      0.7 |     29.1 |       13.1 |        — |      9.3 | Peregrine |
| silu_100k              |      66.0 |     64.3 |     84.1 |      239.7 |    324.5 |     52.9 | JAX |
| softplus_100k          |     186.3 |    142.7 |    261.2 |      125.9 |    783.0 |    154.9 | TensorFlow |
| mish_100k              |     286.2 |    310.3 |    371.2 |      244.4 |   1143.0 |    232.6 | JAX |
| leaky_relu_100k        |       8.6 |     40.7 |     75.3 |       19.7 |        — |     35.8 | Peregrine |
| elu_100k               |      60.0 |    124.1 |    117.9 |      137.3 |    851.9 |     78.0 | Peregrine |
| hard_tanh_100k         |       8.6 |     38.8 |     34.2 |       42.8 |        — |     35.9 | Peregrine |
| relu6_100k             |       8.6 |     41.7 |     43.4 |       52.6 |    721.2 |    111.6 | Peregrine |
| hardswish_100k         |      10.0 |     41.6 |     69.2 |      223.6 |        — |     28.0 | Peregrine |
| gelu_100k              |      95.3 |     66.8 |    135.3 |      260.0 |    853.1 |    219.0 | PyTorch |
| selu_100k              |      65.7 |    123.5 |     84.8 |      131.6 |    738.1 |     81.6 | Peregrine |
| softsign_100k          |      38.5 |    118.9 |     43.9 |       49.2 |        — |     57.3 | Peregrine |
| cross_entropy_64x10    |       2.6 |     36.8 |     22.5 |      634.1 |   3267.5 |     51.5 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.2 |     19.1 |       46.1 |   1104.8 |     12.4 | Peregrine |
| mse_loss_64x10         |       3.9 |      4.8 |     21.3 |       40.7 |    443.8 |     23.5 | Peregrine |
| huber_loss_64x10       |       5.3 |      4.8 |     33.2 |      244.2 |        — |     46.9 | PyTorch |
| smooth_l1_loss_64x10   |       5.0 |      5.0 |     32.8 |      241.9 |        — |     47.7 | Peregrine |
| kl_div_loss_64x10      |       2.6 |      6.5 |     17.8 |      383.6 |        — |     60.1 | Peregrine |
| cosine_sim_loss_64x64  |       1.9 |     10.4 |    110.3 |      243.6 |        — |     61.0 | Peregrine |
| rmsnorm_64x512         |      19.3 |     65.4 |     32.5 |      440.2 |        — |     82.8 | Peregrine |
| conv1d_1x32x128_k3     |      20.1 |     55.1 |     27.8 |      512.6 |        — |     71.2 | Peregrine |
| avgpool2d_1x16x32x32   |      25.9 |     46.5 |    261.6 |       63.8 |        — |     41.9 | Peregrine |
| groupnorm_4x64x16x16   |      22.0 |     52.7 |    221.3 |      770.4 |        — |    263.4 | Peregrine |
| rnn_seq32_128_256      |     191.1 |    267.1 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |     986.0 |    802.4 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     735.8 |    779.0 |        — |          — |        — |        — | Peregrine |
| optim_adam_64          |     816.4 |   1292.8 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     945.9 |   1208.4 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     931.1 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1316.6 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      62.6 |    265.7 |    479.2 |      128.2 |   2343.8 |    541.4 | Peregrine |
| rand_normal_100k       |     269.4 |   1085.4 |    692.7 |      338.0 |   3197.9 |    615.7 | Peregrine |
| rand_bernoulli_100k    |     122.4 |    250.7 |    449.0 |      216.5 |        — |    528.0 | Peregrine |
| rand_uniform_1M        |     684.9 |   2721.9 |   4538.8 |      417.8 |   2341.3 |   2245.7 | TensorFlow |
| rand_normal_1M         |    2738.3 |   9984.0 |   6579.1 |     2057.0 |   3223.9 |   2880.0 | TensorFlow |
| rfft_1k                |       2.1 |      4.4 |     24.0 |       45.0 |        — |     46.7 | Peregrine |
| rfft_4k                |       7.3 |     19.4 |     32.8 |       55.0 |        — |     69.8 | Peregrine |
| rfft_16k               |      29.5 |     65.2 |     80.3 |      107.2 |        — |    116.3 | Peregrine |
| fft_1k                 |       3.1 |      6.6 |     24.2 |        9.2 |        — |     19.2 | Peregrine |
| fft_4k                 |      12.0 |     26.2 |     43.8 |       17.8 |        — |     58.8 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     19.6 |       70.6 |        — |      4.0 | Peregrine |
| solve_64x64            |      12.0 |     24.8 |    101.5 |       25.0 |        — |     33.4 | Peregrine |
| inv_64x64              |      36.6 |     26.1 |     47.3 |       32.8 |        — |     37.7 | PyTorch |
| cholesky_64x64         |       6.2 |     46.2 |     21.5 |       19.8 |        — |     20.0 | Peregrine |
| svd_64x64              |     297.4 |    281.7 |    297.1 |      495.5 |        — |    307.5 | PyTorch |
| qr_64x64               |      42.5 |     85.2 |     55.9 |       84.2 |        — |     64.2 | Peregrine |
| eigh_64x64             |     385.6 |    217.1 |    234.8 |      142.7 |        — |    235.8 | TensorFlow |
| det_64x64              |      18.5 |     20.2 |        — |       23.3 |        — |     28.8 | Peregrine |
| solve_128x128          |      49.5 |     45.2 |    188.9 |       77.3 |        — |     84.6 | PyTorch |
| inv_128x128            |      92.4 |     62.4 |     86.7 |      139.6 |        — |     82.8 | PyTorch |
| cholesky_128x128       |      34.3 |     48.0 |     26.1 |       59.3 |        — |     35.7 | MLX |
| svd_128x128            |    1046.3 |    987.1 |   1014.2 |     1797.1 |        — |   1009.6 | PyTorch |
| qr_128x128             |     192.1 |    221.8 |    192.0 |      326.7 |        — |    190.4 | JAX |
| eigh_128x128           |    2067.5 |    698.1 |    740.5 |      722.4 |        — |    741.0 | PyTorch |
| det_128x128            |      40.6 |     49.9 |        — |       82.3 |        — |     75.9 | Peregrine |
| solve_256x256          |     194.5 |    179.5 |    765.6 |      378.4 |        — |    265.4 | PyTorch |
| inv_256x256            |     451.3 |    299.1 |    240.4 |      849.4 |        — |    333.8 | MLX |
| cholesky_256x256       |     145.3 |     74.3 |     54.9 |      281.7 |        — |    113.4 | MLX |
| svd_256x256            |    6248.6 |   5627.4 |   5820.7 |     8095.7 |        — |   5822.1 | PyTorch |
| qr_256x256             |    1036.6 |   1006.9 |   1012.2 |     1693.7 |        — |    980.9 | JAX |
| eigh_256x256           |    6067.6 |   3405.2 |   3452.1 |     4541.6 |        — |   3533.0 | PyTorch |
| det_256x256            |     140.9 |    208.2 |        — |      432.2 |        — |    205.4 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1771.2 |    825.5 |        — |     2387.4 |   1223.1 |   2108.9 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3183.0 |   1917.7 |        — |     3658.1 |   1214.3 |   3360.6 | tinygrad |
| add_layernorm_196x768  |     106.0 |     98.3 |        — |     1220.9 |   1110.9 |    221.2 | PyTorch |
| add_layernorm_196x1024 |     140.6 |    110.2 |        — |     1296.0 |   1108.2 |    267.6 | PyTorch |
| matmul_f32_196x768x3072 |     536.6 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14385.4 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1572.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   25986.4 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.55x** (Peregrine is faster)
- **Peregrine vs MLX: 0.40x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.29x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.38x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 97/141 ops
- PyTorch: 23/141 ops
- JAX: 9/141 ops
- TensorFlow: 7/141 ops
- MLX: 4/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
