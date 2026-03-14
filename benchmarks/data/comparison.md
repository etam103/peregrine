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
| matmul_128x128         |      10.2 |      6.1 |     22.1 |       94.8 |    412.9 |     77.3 | PyTorch |
| matmul_256x256         |      54.3 |     31.8 |     47.4 |      198.5 |    422.0 |    146.6 | PyTorch |
| matmul_512x512         |     189.6 |    144.1 |    166.4 |      627.2 |    426.8 |    512.5 | PyTorch |
| matmul_1024x1024       |    1000.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    8810.5 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.5 |     42.2 |     31.1 |       47.8 |    187.7 |     33.0 | Peregrine |
| add_500k               |      61.6 |     57.9 |     80.5 |       83.6 |    186.3 |     60.0 | PyTorch |
| add_1M                 |     103.9 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     511.9 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     824.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.6 |     40.6 |     28.3 |       45.3 |    187.1 |     32.3 | Peregrine |
| mul_500k               |      61.6 |     59.1 |     80.2 |       71.8 |    187.1 |     61.2 | PyTorch |
| mul_1M                 |     125.3 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     513.9 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     978.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.4 |     63.9 |     61.1 |       63.9 |    218.4 |     46.0 | JAX |
| exp_500k               |     103.6 |    137.2 |    223.0 |       99.0 |    221.1 |    117.2 | TensorFlow |
| exp_1M                 |     130.5 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     416.4 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     782.1 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.9 |     38.8 |     24.8 |       37.7 |    338.1 |     98.0 | Peregrine |
| relu_1M                |      82.7 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     29.3 |     16.1 |       11.7 |    611.2 |     30.2 | Peregrine |
| softmax_8x512          |       4.2 |     34.8 |     19.1 |       14.4 |    607.7 |     33.0 | Peregrine |
| mlp_fwd_64x784         |      33.2 |     27.8 |     52.4 |      245.0 |   1775.0 |    179.3 | PyTorch |
| mlp_fwd_256x784_wide   |     398.6 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     795.4 |   1248.8 |    775.1 |     8607.3 |  23389.0 |   5041.0 | MLX |
| train_step_256_wide    |    3245.6 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.7 |     39.1 |     24.5 |       48.9 |    165.7 |     37.6 | Peregrine |
| square_100k            |       8.7 |     40.8 |     23.6 |       14.8 |    174.2 |     25.3 | Peregrine |
| rsqrt_100k             |      21.5 |     41.1 |     31.9 |       51.8 |        — |     81.3 | Peregrine |
| floor_100k             |       8.7 |     36.6 |     23.4 |       15.9 |    408.1 |     26.6 | Peregrine |
| ceil_100k              |       8.7 |     40.6 |     23.6 |       15.9 |    349.3 |     39.4 | Peregrine |
| round_100k             |       8.7 |     41.5 |     23.4 |       42.1 |        — |     33.7 | Peregrine |
| sign_100k              |       8.7 |     38.2 |     27.1 |       53.0 |    788.3 |     36.4 | Peregrine |
| expm1_100k             |      63.1 |    110.2 |    102.7 |      145.8 |        — |     98.8 | Peregrine |
| log2_100k              |      55.5 |     86.0 |     97.7 |      153.7 |    162.1 |     57.4 | Peregrine |
| log10_100k             |      58.0 |     81.7 |    106.7 |      150.8 |        — |     58.0 | Peregrine |
| log1p_100k             |      75.5 |     82.1 |    127.5 |       90.2 |        — |    104.3 | Peregrine |
| erf_100k               |     100.7 |     57.2 |    100.2 |       59.8 |        — |     42.4 | JAX |
| sinh_100k              |      51.1 |    132.6 |     93.4 |      136.7 |    526.8 |    109.5 | Peregrine |
| cosh_100k              |      46.4 |    129.0 |     89.4 |      133.4 |    467.7 |     68.8 | Peregrine |
| arcsin_100k            |      52.1 |     73.5 |     93.9 |       57.3 |   2852.0 |    111.5 | Peregrine |
| arccos_100k            |      60.7 |     88.2 |    110.2 |       53.5 |        — |    193.5 | TensorFlow |
| arctan_100k            |      53.1 |     93.0 |     92.9 |       58.5 |   3011.3 |    212.4 | Peregrine |
| arcsinh_100k           |     204.8 |    151.3 |    331.5 |      135.6 |        — |    113.6 | JAX |
| maximum_100k           |      12.5 |     39.9 |     27.5 |       41.9 |    192.1 |     33.8 | Peregrine |
| minimum_100k           |      12.5 |     41.3 |     27.7 |       40.4 |    368.6 |     32.8 | Peregrine |
| power_100k             |     153.8 |    240.6 |    210.3 |      282.8 |        — |    142.1 | JAX |
| arctan2_100k           |      95.1 |    134.1 |    144.3 |       72.7 |        — |    311.9 | TensorFlow |
| logaddexp_100k         |     272.5 |    156.0 |    255.8 |      370.6 |        — |    142.3 | JAX |
| clip_100k              |       8.7 |     38.3 |     34.3 |       42.8 |    526.1 |     46.0 | Peregrine |
| where_100k             |      16.4 |     50.1 |     28.2 |       65.9 |    279.1 |     45.1 | Peregrine |
| greater_100k           |      12.5 |     48.1 |     23.8 |       52.6 |    188.4 |     38.6 | Peregrine |
| equal_100k             |      12.5 |     32.2 |     23.8 |       60.4 |    286.0 |     36.3 | Peregrine |
| sum_axis_256x512       |      20.4 |     42.9 |     23.2 |       51.9 |    205.9 |     47.6 | Peregrine |
| mean_axis_256x512      |      20.4 |     44.7 |     24.4 |       51.6 |    292.2 |     44.9 | Peregrine |
| max_axis_256x512       |      14.8 |     53.4 |     41.2 |       50.6 |    200.7 |     46.8 | Peregrine |
| min_axis_256x512       |      14.8 |     54.6 |     39.5 |       47.9 |    324.6 |     46.6 | Peregrine |
| var_256x512            |      45.7 |    274.1 |     61.0 |      223.0 |        — |     80.9 | Peregrine |
| prod_axis_256x512      |      24.1 |     38.8 |     25.8 |       47.8 |        — |     56.2 | Peregrine |
| logsumexp_256x512      |      95.4 |    198.2 |    106.5 |      347.4 |        — |    275.2 | Peregrine |
| cumsum_256x512         |     117.0 |     79.2 |    128.3 |      188.2 |    618.9 |    215.7 | PyTorch |
| argmax_axis_256x512    |      51.8 |     94.5 |    170.1 |       71.8 |   1311.5 |    179.8 | Peregrine |
| sum_axis_1024x1024     |     174.0 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     427.7 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       7.7 |     37.8 |     55.2 |       53.5 |   1799.8 |     36.4 | Peregrine |
| triu_256x256           |       7.6 |     38.8 |     53.1 |       53.3 |   1769.4 |     37.9 | Peregrine |
| repeat_64x128_2x3      |       6.1 |     51.5 |     30.2 |       77.4 |        — |     27.8 | Peregrine |
| pad_64x128             |       2.6 |      4.3 |     18.2 |       84.4 |     89.1 |     18.2 | Peregrine |
| stack_8x64x128         |       3.8 |      8.6 |     43.9 |       53.5 |    930.5 |    161.0 | Peregrine |
| diagonal_512x512       |       0.8 |      0.6 |     28.9 |       12.7 |        — |      7.5 | PyTorch |
| silu_100k              |      64.0 |     69.7 |     84.4 |      221.3 |    327.4 |     51.5 | JAX |
| softplus_100k          |     180.7 |    153.6 |    260.2 |      133.5 |    767.8 |    155.0 | TensorFlow |
| mish_100k              |     286.2 |    311.1 |    378.0 |      247.8 |   1124.4 |    229.7 | JAX |
| leaky_relu_100k        |       8.6 |     41.6 |     79.3 |       20.2 |        — |     29.1 | Peregrine |
| elu_100k               |      60.0 |    132.5 |    121.8 |      131.5 |    862.0 |     77.3 | Peregrine |
| hard_tanh_100k         |       8.6 |     40.5 |     34.2 |       42.2 |        — |     35.4 | Peregrine |
| relu6_100k             |       8.6 |     40.8 |     45.9 |       50.2 |    728.5 |    112.2 | Peregrine |
| hardswish_100k         |      10.0 |     39.5 |     68.9 |      203.9 |        — |     28.0 | Peregrine |
| gelu_100k              |      95.8 |     72.9 |    135.7 |      241.7 |    848.6 |    213.9 | PyTorch |
| selu_100k              |      63.7 |    132.7 |     85.8 |      134.6 |    736.5 |     82.6 | Peregrine |
| softsign_100k          |      38.3 |    123.5 |     42.3 |       47.3 |        — |     58.3 | Peregrine |
| cross_entropy_64x10    |       2.6 |     37.4 |     22.2 |      615.8 |   3303.1 |     54.4 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.3 |     18.8 |       43.0 |   1102.8 |     12.2 | Peregrine |
| mse_loss_64x10         |       3.6 |      4.8 |     22.2 |       39.1 |    438.4 |     23.6 | Peregrine |
| huber_loss_64x10       |       5.0 |      5.0 |     32.8 |      238.0 |        — |     47.5 | Peregrine |
| smooth_l1_loss_64x10   |       4.8 |      5.0 |     32.8 |      235.7 |        — |     47.4 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.4 |     17.4 |      374.1 |        — |     60.4 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.1 |    113.1 |      237.3 |        — |     69.0 | Peregrine |
| rmsnorm_64x512         |      18.6 |     66.7 |     32.6 |      439.9 |        — |     73.3 | Peregrine |
| conv1d_1x32x128_k3     |      20.3 |     53.5 |     28.1 |      513.3 |        — |     73.5 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     43.8 |    261.7 |       62.9 |        — |     44.0 | Peregrine |
| groupnorm_4x64x16x16   |      22.8 |     54.0 |    222.5 |      770.4 |        — |    269.0 | Peregrine |
| rnn_seq32_128_256      |     197.8 |    267.9 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1049.8 |    805.9 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     797.4 |    776.4 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     795.0 |   1258.5 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     925.4 |   1128.0 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     900.1 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1280.0 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      60.2 |    257.5 |    480.8 |      124.0 |   2336.5 |    526.0 | Peregrine |
| rand_normal_100k       |     236.5 |    971.4 |    685.8 |      339.1 |   3226.1 |    620.6 | Peregrine |
| rand_bernoulli_100k    |     118.4 |    250.0 |    447.8 |      218.2 |        — |    536.8 | Peregrine |
| rand_uniform_1M        |     600.4 |   2564.7 |   4532.9 |      426.4 |   2359.5 |   2256.0 | TensorFlow |
| rand_normal_1M         |    2366.6 |   9702.0 |   6580.7 |     2065.2 |   3209.7 |   2904.1 | TensorFlow |
| rfft_1k                |       2.2 |      4.5 |     20.5 |       42.6 |        — |     60.2 | Peregrine |
| rfft_4k                |       7.6 |     14.8 |     30.3 |       54.2 |        — |     70.1 | Peregrine |
| rfft_16k               |      30.2 |     65.2 |     78.3 |      103.7 |        — |    117.7 | Peregrine |
| fft_1k                 |       3.3 |      6.6 |     22.5 |        8.8 |        — |     42.0 | Peregrine |
| fft_4k                 |      12.2 |     26.2 |     40.4 |       17.4 |        — |     55.3 | Peregrine |
| norm_l2_1k             |       1.1 |      1.2 |     18.7 |       69.0 |        — |      3.8 | Peregrine |
| solve_64x64            |      11.8 |     18.4 |    101.0 |       24.4 |        — |     32.1 | Peregrine |
| inv_64x64              |      36.8 |     26.2 |     51.5 |       32.5 |        — |     44.4 | PyTorch |
| cholesky_64x64         |       6.1 |     45.4 |     21.4 |       19.6 |        — |     19.8 | Peregrine |
| svd_64x64              |     275.2 |    279.0 |    288.5 |      494.1 |        — |    303.9 | Peregrine |
| qr_64x64               |      41.1 |     77.9 |     58.6 |       83.6 |        — |     65.2 | Peregrine |
| eigh_64x64             |     376.1 |    214.4 |    232.6 |      143.7 |        — |    238.8 | TensorFlow |
| det_64x64              |      19.0 |     19.9 |        — |       22.9 |        — |     28.6 | Peregrine |
| solve_128x128          |      49.9 |     45.0 |    190.2 |       76.6 |        — |     85.2 | PyTorch |
| inv_128x128            |      92.5 |     61.8 |     90.8 |      143.1 |        — |     83.2 | PyTorch |
| cholesky_128x128       |      35.4 |     51.5 |     26.2 |       62.1 |        — |     36.3 | MLX |
| svd_128x128            |     985.9 |    988.5 |    967.4 |     1845.0 |        — |   1010.1 | MLX |
| qr_128x128             |     189.0 |    224.2 |    196.3 |      325.7 |        — |    190.8 | Peregrine |
| eigh_128x128           |    1825.7 |    696.6 |    721.7 |      720.2 |        — |    746.1 | PyTorch |
| det_128x128            |      41.2 |     49.6 |        — |       81.8 |        — |     75.9 | Peregrine |
| solve_256x256          |     188.4 |    182.5 |    737.5 |      374.6 |        — |    263.8 | PyTorch |
| inv_256x256            |     458.8 |    294.7 |    251.4 |      843.6 |        — |    337.0 | MLX |
| cholesky_256x256       |     145.0 |     77.5 |     57.2 |      281.1 |        — |    118.8 | MLX |
| svd_256x256            |    6024.1 |   5599.9 |   5763.0 |     7889.4 |        — |   5795.0 | PyTorch |
| qr_256x256             |    1017.4 |    995.7 |   1012.0 |     1693.2 |        — |    967.8 | JAX |
| eigh_256x256           |    5915.0 |   3450.0 |   3358.8 |     4553.2 |        — |   3533.6 | MLX |
| det_256x256            |     140.6 |    205.3 |        — |      433.1 |        — |    206.2 | Peregrine |
| matmul_bias_gelu_196x768x3072 |    1786.3 |    871.1 |        — |     2359.0 |   1215.3 |   2119.8 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3242.8 |   1900.0 |        — |     3669.5 |   1223.4 |   3467.1 | tinygrad |
| add_layernorm_196x768  |     105.8 |    106.2 |        — |     1217.2 |   1176.2 |    228.7 | Peregrine |
| add_layernorm_196x1024 |     140.6 |    104.4 |        — |     1311.2 |   1139.2 |    280.3 | PyTorch |
| matmul_f32_196x768x3072 |     659.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14384.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1435.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   25949.3 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.54x** (Peregrine is faster)
- **Peregrine vs MLX: 0.40x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.29x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.37x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 100/141 ops
- PyTorch: 19/141 ops
- JAX: 8/141 ops
- TensorFlow: 7/141 ops
- MLX: 6/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
