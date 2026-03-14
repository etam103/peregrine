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
| matmul_128x128         |      16.6 |      5.8 |     20.9 |       94.1 |    416.4 |     77.7 | PyTorch |
| matmul_256x256         |      69.2 |     31.8 |     47.5 |      196.0 |    417.6 |    149.5 | PyTorch |
| matmul_512x512         |     217.7 |    142.3 |    168.8 |      671.2 |    429.1 |    511.4 | PyTorch |
| matmul_1024x1024       |    1039.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9209.1 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.7 |     41.6 |     29.1 |       51.0 |    189.0 |     34.0 | Peregrine |
| add_500k               |      48.4 |     56.5 |     80.2 |       72.8 |    185.5 |     60.0 | Peregrine |
| add_1M                 |     125.7 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     545.4 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     969.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.5 |     36.2 |     28.0 |       44.1 |    187.0 |     28.4 | Peregrine |
| mul_500k               |      62.4 |     61.8 |     79.5 |       69.6 |    188.2 |     61.0 | JAX |
| mul_1M                 |     127.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     646.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     855.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.1 |     56.0 |     59.2 |       62.4 |    220.1 |     45.9 | JAX |
| exp_500k               |     187.7 |    141.0 |    222.8 |      100.9 |    222.4 |    119.8 | TensorFlow |
| exp_1M                 |     317.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1078.9 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2117.1 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     37.4 |     24.8 |       36.0 |    334.3 |     98.9 | Peregrine |
| relu_1M                |      84.2 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     25.1 |     15.9 |       10.6 |    606.8 |     31.0 | Peregrine |
| softmax_8x512          |       4.4 |     28.5 |     18.3 |       13.2 |    609.1 |     33.3 | Peregrine |
| mlp_fwd_64x784         |      33.1 |     27.9 |     53.1 |      234.4 |   1755.8 |    181.6 | PyTorch |
| mlp_fwd_256x784_wide   |     440.6 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     827.6 |   1193.2 |    775.2 |     8181.5 |  23655.9 |   5043.9 | MLX |
| train_step_256_wide    |    3286.9 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.6 |     36.4 |     24.5 |       48.2 |    162.2 |     29.0 | Peregrine |
| square_100k            |       8.7 |     35.3 |     23.6 |       14.0 |    173.4 |     28.3 | Peregrine |
| rsqrt_100k             |      21.5 |     37.6 |     31.5 |       51.7 |        — |     85.0 | Peregrine |
| floor_100k             |       8.6 |     38.6 |     23.4 |       15.6 |    407.9 |     22.8 | Peregrine |
| ceil_100k              |       8.7 |     36.2 |     23.7 |       15.5 |    359.4 |     43.1 | Peregrine |
| round_100k             |       8.5 |     40.0 |     23.7 |       43.9 |        — |     30.9 | Peregrine |
| sign_100k              |       8.6 |     37.0 |     27.5 |       46.1 |    785.9 |     36.5 | Peregrine |
| expm1_100k             |      63.1 |    114.3 |    107.7 |      143.9 |        — |     98.8 | Peregrine |
| log2_100k              |      55.6 |     83.1 |    101.7 |      142.2 |    162.3 |     56.4 | Peregrine |
| log10_100k             |      58.0 |     81.7 |    109.6 |      144.3 |        — |     57.8 | JAX |
| log1p_100k             |      75.5 |     81.3 |    127.4 |       94.8 |        — |    105.0 | Peregrine |
| erf_100k               |     100.7 |     51.3 |    100.5 |       53.6 |        — |     55.7 | PyTorch |
| sinh_100k              |      51.1 |    133.4 |     93.5 |      132.4 |    533.1 |    113.3 | Peregrine |
| cosh_100k              |      46.3 |    130.7 |     89.4 |      128.2 |    459.2 |     69.5 | Peregrine |
| arcsin_100k            |      52.1 |     91.1 |     94.0 |       56.4 |   2885.0 |    114.2 | Peregrine |
| arccos_100k            |      60.7 |     87.6 |    110.3 |       54.7 |        — |    194.3 | TensorFlow |
| arctan_100k            |      53.3 |     96.4 |     92.7 |       57.2 |   3023.6 |    213.3 | Peregrine |
| arcsinh_100k           |     204.8 |    164.0 |    332.0 |      143.8 |        — |    115.4 | JAX |
| maximum_100k           |      12.5 |     36.2 |     27.5 |       36.6 |    191.0 |     31.9 | Peregrine |
| minimum_100k           |      12.5 |     37.3 |     26.9 |       42.3 |    378.2 |     30.5 | Peregrine |
| power_100k             |     153.8 |    240.0 |    210.1 |      264.3 |        — |    140.5 | JAX |
| arctan2_100k           |      95.1 |    135.8 |    144.1 |       67.6 |        — |    312.1 | TensorFlow |
| logaddexp_100k         |     272.5 |    153.2 |    255.0 |      350.8 |        — |    142.7 | JAX |
| clip_100k              |       8.7 |     35.0 |     36.7 |       40.0 |    532.5 |     35.2 | Peregrine |
| where_100k             |      16.4 |     50.6 |     28.7 |       65.0 |    273.1 |     33.6 | Peregrine |
| greater_100k           |      12.5 |     46.2 |     23.7 |       57.6 |    186.4 |     27.5 | Peregrine |
| equal_100k             |      12.5 |     31.4 |     23.9 |       58.9 |    284.2 |     27.2 | Peregrine |
| sum_axis_256x512       |      18.8 |     37.2 |     24.4 |       49.5 |    206.8 |     48.3 | Peregrine |
| mean_axis_256x512      |      18.9 |     42.4 |     24.8 |       52.7 |    296.8 |     45.4 | Peregrine |
| max_axis_256x512       |      13.7 |     60.5 |     42.9 |       47.7 |    205.2 |     46.8 | Peregrine |
| min_axis_256x512       |      13.7 |     58.2 |     41.5 |       50.2 |    323.8 |     45.1 | Peregrine |
| var_256x512            |      45.8 |    274.9 |     62.7 |      201.2 |        — |     78.6 | Peregrine |
| prod_axis_256x512      |      24.1 |     38.4 |     26.9 |       48.6 |        — |     55.2 | Peregrine |
| logsumexp_256x512      |      95.4 |    193.0 |    108.7 |      314.4 |        — |    281.7 | Peregrine |
| cumsum_256x512         |     118.8 |     78.2 |    128.7 |      180.0 |    608.3 |    213.2 | PyTorch |
| argmax_axis_256x512    |      51.7 |     93.9 |    173.5 |       70.2 |   1264.1 |    175.4 | Peregrine |
| sum_axis_1024x1024     |     174.1 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     428.0 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      34.7 |     30.7 |     56.2 |       52.9 |   1828.4 |     37.6 | PyTorch |
| triu_256x256           |      34.1 |     35.0 |     56.2 |       50.5 |   1806.0 |     36.7 | Peregrine |
| repeat_64x128_2x3      |     124.6 |     43.9 |     32.1 |       72.4 |        — |     27.8 | JAX |
| pad_64x128             |      17.0 |      4.3 |     19.0 |       79.4 |     89.8 |     18.1 | PyTorch |
| stack_8x64x128         |      15.3 |      8.7 |     45.3 |       45.4 |    903.6 |    158.8 | PyTorch |
| diagonal_512x512       |       0.7 |      0.6 |     28.7 |       11.2 |        — |      8.5 | PyTorch |
| silu_100k              |      64.0 |     72.3 |     84.6 |      227.9 |    328.0 |     51.9 | JAX |
| softplus_100k          |     180.7 |    155.4 |    260.4 |      122.3 |    776.2 |    155.3 | TensorFlow |
| mish_100k              |     286.1 |    307.6 |    371.1 |      242.0 |   1152.5 |    237.3 | JAX |
| leaky_relu_100k        |       8.7 |     38.6 |     75.3 |       18.3 |        — |     30.7 | Peregrine |
| elu_100k               |      60.0 |    137.2 |    118.0 |      131.9 |    872.3 |     77.7 | Peregrine |
| hard_tanh_100k         |       8.7 |     37.8 |     34.7 |       39.8 |        — |     40.8 | Peregrine |
| relu6_100k             |       8.7 |     38.5 |     44.9 |       51.7 |    728.9 |    110.3 | Peregrine |
| hardswish_100k         |      10.0 |     37.0 |     62.9 |      208.7 |        — |     28.0 | Peregrine |
| gelu_100k              |      95.3 |     63.3 |    135.4 |      235.5 |    859.7 |    208.2 | PyTorch |
| selu_100k              |      63.7 |    128.0 |     85.2 |      126.7 |    733.0 |     81.7 | Peregrine |
| softsign_100k          |      38.1 |    111.5 |     41.8 |       46.7 |        — |     58.4 | Peregrine |
| cross_entropy_64x10    |       2.6 |     35.7 |     23.1 |      580.7 |   3380.7 |     54.9 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.3 |     19.0 |       38.7 |   1109.2 |     12.2 | Peregrine |
| mse_loss_64x10         |       3.7 |      4.8 |     21.5 |       34.8 |    446.4 |     24.0 | Peregrine |
| huber_loss_64x10       |       5.1 |      4.8 |     33.9 |      220.5 |        — |     47.8 | PyTorch |
| smooth_l1_loss_64x10   |       5.0 |      5.1 |     32.9 |      217.7 |        — |     47.4 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.2 |     17.6 |      354.4 |        — |     57.5 | Peregrine |
| cosine_sim_loss_64x64  |      13.5 |     10.6 |    110.6 |      220.2 |        — |     56.6 | PyTorch |
| rmsnorm_64x512         |      57.6 |     65.8 |     32.7 |      433.8 |        — |     73.4 | MLX |
| conv1d_1x32x128_k3     |      20.5 |     48.4 |     27.7 |      493.4 |        — |     75.2 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     38.5 |    261.9 |       59.3 |        — |     42.2 | Peregrine |
| groupnorm_4x64x16x16   |      72.6 |     48.9 |    227.8 |      697.6 |        — |    266.9 | PyTorch |
| rnn_seq32_128_256      |     194.4 |    270.7 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1127.0 |    807.3 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     857.6 |    781.7 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     804.0 |   1223.5 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     927.8 |   1052.2 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     911.6 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1276.2 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     106.2 |    257.4 |    480.3 |      126.6 |   2355.5 |    534.5 | Peregrine |
| rand_normal_100k       |     236.5 |    971.3 |    686.5 |      335.6 |   3214.8 |    611.4 | Peregrine |
| rand_bernoulli_100k    |     303.3 |    250.0 |    449.0 |      209.6 |        — |    519.3 | TensorFlow |
| rand_uniform_1M        |    1064.1 |   2566.0 |   4540.5 |      413.0 |   2376.2 |   2267.3 | TensorFlow |
| rand_normal_1M         |    2367.0 |   9715.4 |   6578.1 |     2016.6 |   3207.5 |   2847.6 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     25.4 |       39.0 |        — |     61.1 | Peregrine |
| rfft_4k                |       6.5 |     15.0 |     32.8 |       49.5 |        — |     64.3 | Peregrine |
| rfft_16k               |      30.2 |     64.9 |     77.8 |      100.3 |        — |    119.7 | Peregrine |
| fft_1k                 |       3.3 |      6.5 |     24.7 |        7.9 |        — |     17.8 | Peregrine |
| fft_4k                 |      12.2 |     26.3 |     43.2 |       16.2 |        — |     71.2 | Peregrine |
| norm_l2_1k             |       1.0 |      1.2 |     20.5 |       63.0 |        — |      3.9 | Peregrine |
| solve_64x64            |      11.8 |     24.3 |    100.8 |       23.2 |        — |     31.8 | Peregrine |
| inv_64x64              |      36.9 |     25.8 |     51.9 |       31.0 |        — |     37.0 | PyTorch |
| cholesky_64x64         |       9.0 |     47.0 |     21.9 |       18.2 |        — |     19.5 | Peregrine |
| svd_64x64              |     276.4 |    278.7 |    296.5 |      493.0 |        — |    304.9 | Peregrine |
| qr_64x64               |      41.0 |     79.6 |     58.5 |       82.1 |        — |     63.1 | Peregrine |
| eigh_64x64             |     379.8 |    218.5 |    233.3 |      137.2 |        — |    236.0 | TensorFlow |
| det_64x64              |      23.3 |     19.9 |        — |       21.9 |        — |     28.8 | PyTorch |
| solve_128x128          |      50.0 |     44.9 |    189.0 |       75.1 |        — |     84.9 | PyTorch |
| inv_128x128            |      92.3 |     61.6 |     90.8 |      137.5 |        — |     82.8 | PyTorch |
| cholesky_128x128       |      50.7 |     50.4 |     26.1 |       57.4 |        — |     35.7 | MLX |
| svd_128x128            |     985.9 |    992.3 |    995.9 |     1874.7 |        — |   1011.5 | Peregrine |
| qr_128x128             |     188.3 |    220.2 |    190.3 |      324.3 |        — |    191.3 | Peregrine |
| eigh_128x128           |    1841.3 |    700.8 |    722.0 |      714.8 |        — |    749.4 | PyTorch |
| det_128x128            |      52.3 |     49.3 |        — |       80.4 |        — |     77.5 | PyTorch |
| solve_256x256          |     189.1 |    177.3 |    733.0 |      376.1 |        — |    270.1 | PyTorch |
| inv_256x256            |     486.7 |    286.1 |    251.2 |      846.0 |        — |    339.2 | MLX |
| cholesky_256x256       |     226.2 |     76.3 |     55.4 |      279.7 |        — |    118.0 | MLX |
| svd_256x256            |    5875.1 |   5750.0 |   5693.9 |     8114.7 |        — |   5789.0 | MLX |
| qr_256x256             |     985.2 |    998.7 |   1009.4 |     1689.8 |        — |    978.4 | JAX |
| eigh_256x256           |    6005.1 |   3421.9 |   3436.0 |     4540.7 |        — |   3548.7 | PyTorch |
| det_256x256            |     212.3 |    202.4 |        — |      434.7 |        — |    205.9 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1507.2 |    822.2 |        — |     2364.3 |   1257.0 |   2096.8 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2614.2 |   1899.1 |        — |     3636.1 |   1273.0 |   3414.1 | tinygrad |
| add_layernorm_196x768  |     103.0 |     98.7 |        — |     1195.8 |   1155.3 |    231.5 | PyTorch |
| add_layernorm_196x1024 |     133.5 |    107.0 |        — |     1244.6 |   1145.3 |    283.9 | PyTorch |
| matmul_f32_196x768x3072 |     566.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14477.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1472.1 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26014.5 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.65x** (Peregrine is faster)
- **Peregrine vs MLX: 0.47x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.36x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.06x** (Peregrine is faster)
- **Peregrine vs JAX: 0.44x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 88/141 ops
- PyTorch: 28/141 ops
- JAX: 10/141 ops
- TensorFlow: 8/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
