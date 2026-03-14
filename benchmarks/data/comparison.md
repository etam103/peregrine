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
| matmul_128x128         |      13.3 |      6.2 |     20.5 |       95.0 |    420.0 |     78.8 | PyTorch |
| matmul_256x256         |      59.1 |     31.7 |     48.0 |      201.4 |    422.1 |    147.2 | PyTorch |
| matmul_512x512         |     208.2 |    142.3 |    167.4 |      686.0 |    425.7 |    521.6 | PyTorch |
| matmul_1024x1024       |    1005.6 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9517.4 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.5 |     38.3 |     28.4 |       46.2 |    186.4 |     35.0 | Peregrine |
| add_500k               |      61.7 |     56.3 |     80.2 |       83.7 |    187.7 |     65.6 | PyTorch |
| add_1M                 |     104.0 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     553.0 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     852.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.9 |     40.4 |     31.4 |       44.3 |    187.5 |     30.7 | Peregrine |
| mul_500k               |      61.5 |     57.5 |     81.7 |       71.8 |    189.9 |     59.6 | PyTorch |
| mul_1M                 |     125.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     505.5 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     961.7 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |      49.3 |     65.9 |     60.7 |       65.1 |    220.8 |     46.3 | JAX |
| exp_500k               |     162.2 |    137.9 |    223.0 |      104.1 |    221.1 |    119.0 | TensorFlow |
| exp_1M                 |     179.3 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |     460.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |     835.9 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.7 |     38.3 |     24.9 |       40.5 |    334.0 |     97.5 | Peregrine |
| relu_1M                |      82.0 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     30.9 |     16.1 |       11.2 |    614.6 |     30.3 | Peregrine |
| softmax_8x512          |       4.2 |     32.9 |     20.0 |       14.1 |    607.8 |     32.8 | Peregrine |
| mlp_fwd_64x784         |      34.4 |     27.8 |     51.2 |      243.7 |   1852.4 |    177.4 | PyTorch |
| mlp_fwd_256x784_wide   |     411.9 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     808.6 |   1272.2 |    846.5 |     8406.5 |  24134.7 |   5011.8 | Peregrine |
| train_step_256_wide    |    3333.1 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.7 |     37.8 |     24.5 |       47.9 |    164.2 |     34.1 | Peregrine |
| square_100k            |       8.7 |     39.4 |     23.4 |       14.9 |    179.2 |     30.0 | Peregrine |
| rsqrt_100k             |      21.5 |     44.3 |     31.3 |       50.2 |        — |     92.2 | Peregrine |
| floor_100k             |       8.7 |     38.8 |     23.4 |       15.8 |    416.2 |     28.5 | Peregrine |
| ceil_100k              |       8.7 |     39.6 |     23.0 |       15.8 |    353.4 |     27.8 | Peregrine |
| round_100k             |       8.7 |     41.7 |     23.3 |       46.6 |        — |     29.0 | Peregrine |
| sign_100k              |       8.7 |     41.6 |     27.3 |       44.8 |    783.3 |     36.1 | Peregrine |
| expm1_100k             |      63.1 |    110.4 |    103.3 |      150.6 |        — |     98.8 | Peregrine |
| log2_100k              |      55.6 |     90.5 |     97.7 |      158.5 |    161.6 |     57.2 | Peregrine |
| log10_100k             |      58.0 |     95.0 |    106.7 |      147.5 |        — |     55.9 | JAX |
| log1p_100k             |      75.4 |     82.7 |    132.6 |       97.1 |        — |    103.8 | Peregrine |
| erf_100k               |     100.7 |     57.9 |    104.5 |       55.1 |        — |     49.9 | JAX |
| sinh_100k              |      51.0 |    134.6 |     96.5 |      135.7 |    527.2 |    108.6 | Peregrine |
| cosh_100k              |      46.3 |    131.5 |     92.7 |      134.8 |    458.6 |     69.0 | Peregrine |
| arcsin_100k            |      52.1 |     77.5 |     97.5 |       55.3 |   2890.8 |    111.7 | Peregrine |
| arccos_100k            |      60.7 |     91.6 |    114.8 |       51.8 |        — |    196.2 | TensorFlow |
| arctan_100k            |      53.1 |     91.1 |     97.5 |       60.5 |   2970.8 |    215.0 | Peregrine |
| arcsinh_100k           |     204.8 |    158.4 |    331.8 |      139.7 |        — |    114.6 | JAX |
| maximum_100k           |      12.5 |     37.8 |     28.3 |       41.9 |    189.3 |     31.1 | Peregrine |
| minimum_100k           |      12.5 |     41.6 |     27.8 |       41.5 |    371.6 |     30.8 | Peregrine |
| power_100k             |     153.7 |    239.2 |    210.3 |      273.9 |        — |    140.9 | JAX |
| arctan2_100k           |      95.1 |    134.9 |    144.1 |       76.5 |        — |    313.0 | TensorFlow |
| logaddexp_100k         |     272.5 |    153.2 |    255.1 |      357.8 |        — |    141.5 | JAX |
| clip_100k              |       8.7 |     40.4 |     35.0 |       42.2 |    526.5 |     36.1 | Peregrine |
| where_100k             |      16.4 |     49.4 |     28.1 |       66.2 |    276.7 |     34.1 | Peregrine |
| greater_100k           |      12.5 |     48.0 |     25.9 |       51.1 |    188.4 |     26.5 | Peregrine |
| equal_100k             |      12.5 |     31.3 |     24.3 |       56.1 |    287.6 |     28.6 | Peregrine |
| sum_axis_256x512       |      18.8 |     41.3 |     23.6 |       56.1 |    207.8 |     49.8 | Peregrine |
| mean_axis_256x512      |      18.8 |     43.6 |     24.3 |       48.9 |    289.9 |     45.5 | Peregrine |
| max_axis_256x512       |      13.7 |     54.5 |     40.5 |       51.5 |    202.5 |     45.1 | Peregrine |
| min_axis_256x512       |      13.7 |     54.4 |     41.2 |       47.8 |    326.6 |     48.2 | Peregrine |
| var_256x512            |      45.7 |    279.9 |     61.8 |      223.2 |        — |     79.0 | Peregrine |
| prod_axis_256x512      |      24.1 |     41.6 |     25.7 |       50.7 |        — |     55.7 | Peregrine |
| logsumexp_256x512      |      95.5 |    216.2 |    110.3 |      335.3 |        — |    271.6 | Peregrine |
| cumsum_256x512         |     117.0 |     75.7 |    130.2 |      189.8 |    611.1 |    212.2 | PyTorch |
| argmax_axis_256x512    |      51.7 |     95.0 |    172.9 |       71.8 |   1323.6 |    174.3 | Peregrine |
| sum_axis_1024x1024     |     175.1 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |     442.7 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |       8.2 |     41.3 |     56.7 |       54.4 |   1785.7 |     37.8 | Peregrine |
| triu_256x256           |       7.7 |     39.7 |     56.2 |       55.1 |   1799.2 |     36.0 | Peregrine |
| repeat_64x128_2x3      |       7.9 |     47.4 |     30.6 |       75.1 |        — |     27.8 | Peregrine |
| pad_64x128             |       2.3 |      4.0 |     18.1 |       82.3 |     89.2 |     18.0 | Peregrine |
| stack_8x64x128         |       4.3 |      8.6 |     45.3 |       52.8 |    911.1 |    158.6 | Peregrine |
| diagonal_512x512       |       0.8 |      0.6 |     29.2 |       12.4 |        — |      9.4 | PyTorch |
| silu_100k              |      64.0 |     71.7 |     83.1 |      228.0 |    326.0 |     51.8 | JAX |
| softplus_100k          |     180.7 |    152.2 |    262.7 |      129.9 |    775.9 |    157.7 | TensorFlow |
| mish_100k              |     286.1 |    309.4 |    371.5 |      248.8 |   1145.0 |    235.4 | JAX |
| leaky_relu_100k        |       8.7 |     41.1 |     77.2 |       19.4 |        — |     30.2 | Peregrine |
| elu_100k               |      60.0 |    123.6 |    119.8 |      133.8 |    855.9 |     81.7 | Peregrine |
| hard_tanh_100k         |       8.7 |     40.9 |     34.4 |       42.0 |        — |     40.0 | Peregrine |
| relu6_100k             |       8.7 |     40.6 |     44.2 |       52.1 |    726.2 |    113.4 | Peregrine |
| hardswish_100k         |      10.0 |     39.1 |     68.1 |      201.0 |        — |     26.6 | Peregrine |
| gelu_100k              |      95.3 |     73.7 |    140.1 |      241.5 |    851.1 |    212.1 | PyTorch |
| selu_100k              |      63.7 |    122.7 |     88.4 |      130.7 |    744.6 |     81.8 | Peregrine |
| softsign_100k          |      38.1 |    117.1 |     42.2 |       46.8 |        — |     55.6 | Peregrine |
| cross_entropy_64x10    |       2.6 |     38.4 |     23.2 |      612.1 |   3277.5 |     55.0 | Peregrine |
| l1_loss_64x10          |       0.9 |      5.2 |     19.0 |       42.5 |   1107.3 |     11.8 | Peregrine |
| mse_loss_64x10         |       3.9 |      4.8 |     21.1 |       38.6 |    443.2 |     23.9 | Peregrine |
| huber_loss_64x10       |       5.0 |      4.8 |     32.8 |      236.6 |        — |     47.5 | PyTorch |
| smooth_l1_loss_64x10   |       4.9 |      5.0 |     34.8 |      234.9 |        — |     47.1 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      6.3 |     17.6 |      372.1 |        — |     60.9 | Peregrine |
| cosine_sim_loss_64x64  |       1.8 |     10.2 |    109.5 |      235.7 |        — |     66.8 | Peregrine |
| rmsnorm_64x512         |      18.8 |     65.9 |     32.8 |      438.4 |        — |     75.0 | Peregrine |
| conv1d_1x32x128_k3     |      21.0 |     55.9 |     27.9 |      508.1 |        — |     75.3 | Peregrine |
| avgpool2d_1x16x32x32   |      25.0 |     43.9 |    261.0 |       62.5 |        — |     43.1 | Peregrine |
| groupnorm_4x64x16x16   |      21.8 |     54.7 |    221.4 |      753.9 |        — |    264.3 | Peregrine |
| rnn_seq32_128_256      |     195.3 |    268.9 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1016.6 |    804.6 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     751.4 |    776.5 |        — |          — |        — |        — | Peregrine |
| optim_adam_64          |     793.1 |   1257.7 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     913.3 |   1116.0 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     901.3 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1269.8 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |      60.3 |    257.3 |    481.2 |      126.9 |   2362.7 |    537.2 | Peregrine |
| rand_normal_100k       |     236.5 |    970.8 |    685.0 |      338.4 |   3222.7 |    614.5 | Peregrine |
| rand_bernoulli_100k    |     118.9 |    250.0 |    447.0 |      222.3 |        — |    526.7 | Peregrine |
| rand_uniform_1M        |     601.8 |   2564.7 |   4533.1 |      422.8 |   2375.2 |   2268.5 | TensorFlow |
| rand_normal_1M         |    2368.1 |   9704.7 |   6570.0 |     2060.8 |   3201.1 |   2850.5 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     25.4 |       42.1 |        — |     46.6 | Peregrine |
| rfft_4k                |       6.5 |     14.8 |     34.9 |       53.8 |        — |     65.0 | Peregrine |
| rfft_16k               |      30.4 |     65.3 |     77.3 |      104.7 |        — |    116.7 | Peregrine |
| fft_1k                 |       3.2 |      6.6 |     25.9 |        8.6 |        — |     18.0 | Peregrine |
| fft_4k                 |      12.2 |     26.2 |     41.0 |       17.3 |        — |     57.4 | Peregrine |
| norm_l2_1k             |       1.1 |      1.2 |     20.8 |       68.8 |        — |      4.0 | Peregrine |
| solve_64x64            |      12.0 |     24.7 |     94.9 |       24.4 |        — |     32.2 | Peregrine |
| inv_64x64              |      37.3 |     25.9 |     47.1 |       32.4 |        — |     42.6 | PyTorch |
| cholesky_64x64         |       9.7 |     47.4 |     21.7 |       19.3 |        — |     19.5 | Peregrine |
| svd_64x64              |     276.0 |    279.0 |    291.2 |      483.0 |        — |    302.8 | Peregrine |
| qr_64x64               |      41.2 |     82.9 |     54.4 |       83.4 |        — |     63.7 | Peregrine |
| eigh_64x64             |     379.7 |    216.1 |    231.3 |      140.1 |        — |    238.3 | TensorFlow |
| det_64x64              |      23.2 |     19.8 |        — |       22.9 |        — |     28.8 | PyTorch |
| solve_128x128          |      50.2 |     44.9 |    184.3 |       77.3 |        — |     86.3 | PyTorch |
| inv_128x128            |      93.2 |     61.8 |     87.7 |      139.6 |        — |     82.9 | PyTorch |
| cholesky_128x128       |      50.5 |     47.5 |     26.8 |       58.6 |        — |     35.8 | MLX |
| svd_128x128            |     986.2 |    989.5 |    993.5 |     1822.2 |        — |   1011.6 | Peregrine |
| qr_128x128             |     188.1 |    220.2 |    191.7 |      327.3 |        — |    190.7 | Peregrine |
| eigh_128x128           |    1840.2 |    702.4 |    716.7 |      710.7 |        — |    740.8 | PyTorch |
| det_128x128            |      52.1 |     49.6 |        — |       82.1 |        — |     77.3 | PyTorch |
| solve_256x256          |     188.7 |    182.0 |    727.5 |      377.5 |        — |    265.4 | PyTorch |
| inv_256x256            |     459.7 |    291.2 |    246.1 |      847.8 |        — |    333.1 | MLX |
| cholesky_256x256       |     226.4 |     80.6 |     53.2 |      281.0 |        — |    120.3 | MLX |
| svd_256x256            |    5792.3 |   5706.8 |   5587.4 |     8023.6 |        — |   5835.3 | MLX |
| qr_256x256             |     989.0 |    994.1 |    999.8 |     1692.9 |        — |    984.7 | JAX |
| eigh_256x256           |    5989.3 |   3453.5 |   3417.0 |     4512.0 |        — |   3503.6 | MLX |
| det_256x256            |     212.6 |    203.4 |        — |      433.0 |        — |    207.1 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1789.7 |    936.2 |        — |     2424.4 |   1228.5 |   2182.2 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    3226.9 |   1898.3 |        — |     3701.9 |   1225.1 |   3357.8 | tinygrad |
| add_layernorm_196x768  |     105.8 |    102.1 |        — |     1204.0 |   1110.5 |    227.5 | PyTorch |
| add_layernorm_196x1024 |     140.6 |    104.4 |        — |     1281.2 |   1117.1 |    273.5 | PyTorch |
| matmul_f32_196x768x3072 |     521.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14456.2 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1462.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26017.1 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.55x** (Peregrine is faster)
- **Peregrine vs MLX: 0.41x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.30x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.05x** (Peregrine is faster)
- **Peregrine vs JAX: 0.39x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 97/141 ops
- PyTorch: 22/141 ops
- JAX: 9/141 ops
- TensorFlow: 7/141 ops
- MLX: 5/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
