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
| matmul_128x128         |       7.1 |      6.2 |     28.4 |       50.2 |    424.1 |     81.4 | PyTorch |
| matmul_256x256         |      37.4 |     31.8 |     81.4 |      152.4 |    425.9 |    152.5 | PyTorch |
| matmul_512x512         |     193.0 |    134.0 |    221.2 |      695.1 |    460.4 |    504.6 | PyTorch |
| matmul_1024x1024       |    1035.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9072.8 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.8 |     40.1 |     32.4 |       49.6 |    203.6 |     39.4 | Peregrine |
| add_500k               |     118.2 |     57.3 |     84.4 |       85.2 |    184.9 |     61.1 | PyTorch |
| add_1M                 |     125.2 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     534.6 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     877.2 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      12.4 |     40.7 |     33.0 |       43.5 |    185.7 |     30.0 | Peregrine |
| mul_500k               |      96.7 |     58.0 |     88.0 |       72.0 |    188.4 |     59.6 | PyTorch |
| mul_1M                 |     177.7 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     536.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     950.2 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |     100.0 |     62.3 |     73.6 |       65.4 |    224.4 |     46.8 | JAX |
| exp_500k               |     193.5 |    138.9 |    241.9 |      102.4 |    220.0 |    122.6 | TensorFlow |
| exp_1M                 |     286.1 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1145.0 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2184.6 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       8.8 |     40.2 |     27.0 |       38.5 |    345.4 |     99.2 | Peregrine |
| relu_1M                |     128.4 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     30.1 |     19.0 |       11.4 |    617.8 |     32.8 | Peregrine |
| softmax_8x512          |       4.3 |     33.8 |     20.1 |       14.2 |    617.5 |     34.2 | Peregrine |
| mlp_fwd_64x784         |      33.1 |     28.0 |     54.6 |      245.8 |   1820.1 |    179.9 | PyTorch |
| mlp_fwd_256x784_wide   |     422.8 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     817.6 |   1276.8 |    869.1 |     8796.6 |  24610.8 |   5544.6 | Peregrine |
| train_step_256_wide    |    3265.0 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.5 |     40.6 |     28.5 |       49.7 |    160.4 |     25.7 | Peregrine |
| square_100k            |       8.6 |     39.9 |     24.7 |       15.3 |    178.0 |     32.2 | Peregrine |
| rsqrt_100k             |      84.7 |     42.1 |     39.2 |       51.0 |        — |     82.8 | MLX |
| floor_100k             |      46.6 |     41.3 |     25.7 |       15.8 |    420.8 |     28.1 | TensorFlow |
| ceil_100k              |      46.6 |     39.2 |     30.8 |       15.8 |    352.0 |     30.4 | TensorFlow |
| round_100k             |      48.1 |     41.9 |     28.2 |       44.2 |        — |     27.5 | JAX |
| sign_100k              |      54.4 |     39.5 |     32.8 |       48.2 |    810.1 |     37.4 | MLX |
| expm1_100k             |     178.4 |    108.4 |    119.8 |      144.1 |        — |     93.2 | JAX |
| log2_100k              |     107.5 |     86.8 |    111.0 |      143.3 |    165.2 |    101.8 | PyTorch |
| log10_100k             |     115.4 |     88.4 |    122.9 |      147.2 |        — |     66.2 | JAX |
| log1p_100k             |     100.0 |     84.6 |    138.3 |       91.7 |        — |    114.6 | PyTorch |
| erf_100k               |     100.5 |     57.6 |    109.0 |       58.9 |        — |     59.4 | PyTorch |
| sinh_100k              |      51.0 |    132.8 |    105.1 |      126.2 |    527.1 |    128.2 | Peregrine |
| cosh_100k              |      46.3 |    130.4 |    102.8 |      124.4 |    483.9 |     78.7 | Peregrine |
| arcsin_100k            |      53.7 |     79.5 |    102.6 |       56.5 |   2939.8 |    121.2 | Peregrine |
| arccos_100k            |     109.0 |     88.3 |    119.4 |       52.2 |        — |    206.6 | TensorFlow |
| arctan_100k            |      53.1 |     94.9 |    101.9 |       57.6 |   3112.1 |    217.7 | Peregrine |
| arcsinh_100k           |     143.8 |    155.4 |    357.5 |      131.5 |        — |    131.5 | JAX |
| maximum_100k           |      12.4 |     38.8 |     26.6 |       43.6 |    191.8 |     38.1 | Peregrine |
| minimum_100k           |      12.5 |     41.2 |     30.8 |       43.4 |    380.5 |     51.4 | Peregrine |
| power_100k             |     391.3 |    240.1 |    238.0 |      323.8 |        — |    201.6 | JAX |
| arctan2_100k           |    1112.7 |    129.6 |    157.4 |       77.1 |        — |    357.6 | TensorFlow |
| logaddexp_100k         |     409.5 |    151.4 |    280.5 |      400.8 |        — |    218.2 | PyTorch |
| clip_100k              |       8.7 |     42.6 |     39.2 |       43.0 |    538.6 |     49.6 | Peregrine |
| where_100k             |      93.1 |     49.9 |     29.4 |       66.3 |    275.5 |     55.9 | MLX |
| greater_100k           |      70.0 |     47.6 |     21.5 |       61.6 |    191.7 |     45.9 | MLX |
| equal_100k             |      71.3 |     32.5 |     24.4 |       62.9 |    289.1 |     47.3 | MLX |
| sum_axis_256x512       |     113.0 |     40.1 |     20.8 |       49.6 |    207.5 |     63.5 | MLX |
| mean_axis_256x512      |     112.5 |     42.2 |     25.5 |       54.8 |    293.5 |     53.6 | MLX |
| max_axis_256x512       |     154.2 |     52.8 |     47.3 |       48.7 |    203.3 |     62.2 | MLX |
| min_axis_256x512       |     154.3 |     53.1 |     49.4 |       47.8 |    326.9 |     56.7 | TensorFlow |
| var_256x512            |     235.7 |    276.8 |     60.9 |      227.5 |        — |    101.9 | MLX |
| prod_axis_256x512      |     149.1 |     37.8 |     26.3 |       53.4 |        — |     56.9 | MLX |
| logsumexp_256x512      |     383.3 |    195.2 |    119.9 |      351.0 |        — |    329.6 | MLX |
| cumsum_256x512         |     125.7 |     73.1 |    144.5 |      199.8 |    626.8 |    237.6 | PyTorch |
| argmax_axis_256x512    |     154.7 |     97.8 |    184.3 |       74.3 |   1321.4 |    208.6 | TensorFlow |
| sum_axis_1024x1024     |     940.0 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    1936.1 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      35.8 |     35.9 |     63.6 |       56.6 |   1832.6 |     44.2 | Peregrine |
| triu_256x256           |      35.2 |     35.6 |     59.8 |       55.6 |   1808.2 |     44.3 | Peregrine |
| repeat_64x128_2x3      |     128.8 |     49.8 |     32.4 |       75.3 |        — |     30.3 | JAX |
| pad_64x128             |      16.8 |      4.3 |     21.8 |       84.1 |     89.1 |     19.1 | PyTorch |
| stack_8x64x128         |      17.2 |      8.8 |     47.1 |       61.8 |    936.4 |    196.0 | PyTorch |
| diagonal_512x512       |       0.8 |      0.6 |     28.4 |       12.5 |        — |      4.2 | PyTorch |
| silu_100k              |      66.3 |     69.8 |     89.8 |      211.5 |    329.6 |     59.8 | JAX |
| softplus_100k          |     300.6 |    153.1 |    286.7 |      134.3 |    786.0 |    201.8 | TensorFlow |
| mish_100k              |     471.0 |    312.4 |    408.3 |      244.5 |   1167.4 |    291.5 | TensorFlow |
| leaky_relu_100k        |       8.0 |     41.0 |     85.1 |       20.2 |        — |     48.8 | Peregrine |
| elu_100k               |     152.5 |    124.9 |    125.8 |      145.5 |    877.5 |    125.0 | PyTorch |
| hard_tanh_100k         |      50.5 |     40.3 |     37.7 |       42.1 |        — |     65.5 | MLX |
| relu6_100k             |      51.8 |     39.8 |     54.0 |       52.9 |    749.3 |    132.4 | PyTorch |
| hardswish_100k         |      84.0 |     40.2 |     75.5 |      197.8 |        — |     56.2 | PyTorch |
| gelu_100k              |      77.9 |     75.2 |    148.3 |      247.3 |    885.2 |    271.6 | PyTorch |
| selu_100k              |     161.3 |    132.0 |     88.5 |      137.6 |    748.5 |    124.3 | MLX |
| softsign_100k          |      35.0 |    122.0 |     50.9 |       46.0 |        — |     83.3 | Peregrine |
| cross_entropy_64x10    |       2.6 |     39.8 |     24.0 |      626.0 |   3431.6 |     60.9 | Peregrine |
| l1_loss_64x10          |       1.0 |      5.5 |     19.2 |       43.8 |   1127.8 |     13.0 | Peregrine |
| mse_loss_64x10         |       4.3 |      5.0 |     20.7 |       39.6 |    451.0 |     25.1 | Peregrine |
| huber_loss_64x10       |       5.8 |      4.9 |     39.6 |      237.7 |        — |     49.2 | PyTorch |
| smooth_l1_loss_64x10   |       5.6 |      5.2 |     40.5 |      243.8 |        — |     49.8 | PyTorch |
| kl_div_loss_64x10      |       2.5 |      6.3 |     23.5 |      379.1 |        — |     71.8 | Peregrine |
| cosine_sim_loss_64x64  |      13.6 |     10.3 |    125.2 |      237.5 |        — |     47.8 | PyTorch |
| rmsnorm_64x512         |      57.8 |     67.5 |     39.0 |      439.6 |        — |     83.7 | MLX |
| conv1d_1x32x128_k3     |      20.7 |     54.0 |     29.9 |      519.1 |        — |     71.3 | Peregrine |
| avgpool2d_1x16x32x32   |      25.1 |     45.2 |    283.6 |       63.8 |        — |     53.6 | Peregrine |
| groupnorm_4x64x16x16   |      72.6 |     53.9 |    235.1 |      770.2 |        — |    276.7 | PyTorch |
| rnn_seq32_128_256      |     195.8 |    267.5 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1149.8 |    807.4 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     810.3 |    782.3 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     809.9 |   1301.1 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     993.3 |   1122.0 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |    1004.0 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1283.8 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     106.8 |    257.3 |    520.5 |      128.2 |   2436.7 |    547.5 | Peregrine |
| rand_normal_100k       |     794.0 |    973.3 |    748.1 |      342.1 |   3275.7 |    639.7 | TensorFlow |
| rand_bernoulli_100k    |     319.2 |    250.0 |    490.7 |      208.9 |        — |    544.9 | TensorFlow |
| rand_uniform_1M        |    1069.0 |   2568.5 |   4873.0 |      434.2 |   2433.4 |   2346.8 | TensorFlow |
| rand_normal_1M         |    7661.4 |   9732.6 |   7038.8 |     2092.7 |   3336.2 |   3019.8 | TensorFlow |
| rfft_1k                |       2.2 |      4.4 |     20.6 |       43.5 |        — |     42.8 | Peregrine |
| rfft_4k                |       6.5 |     14.9 |     29.7 |       53.5 |        — |     66.3 | Peregrine |
| rfft_16k               |      30.3 |     65.2 |     83.8 |      122.2 |        — |    123.7 | Peregrine |
| fft_1k                 |       3.3 |      6.6 |     24.4 |        8.7 |        — |     17.5 | Peregrine |
| fft_4k                 |      12.2 |     26.2 |     44.4 |       17.2 |        — |     56.4 | Peregrine |
| norm_l2_1k             |       1.1 |      1.3 |     20.8 |       69.5 |        — |      4.0 | Peregrine |
| solve_64x64            |      11.9 |     18.1 |    100.3 |       24.4 |        — |     35.1 | Peregrine |
| inv_64x64              |      37.4 |     26.3 |     50.5 |       32.4 |        — |     37.8 | PyTorch |
| cholesky_64x64         |       9.5 |     41.6 |     22.0 |       19.3 |        — |     20.2 | Peregrine |
| svd_64x64              |     276.8 |    284.5 |    305.3 |      502.4 |        — |    304.4 | Peregrine |
| qr_64x64               |      41.5 |     82.6 |     62.4 |       83.6 |        — |     63.0 | Peregrine |
| eigh_64x64             |     381.3 |    213.6 |    236.4 |      144.3 |        — |    238.8 | TensorFlow |
| det_64x64              |      23.2 |     20.2 |        — |       22.8 |        — |     33.9 | PyTorch |
| solve_128x128          |      49.9 |     45.0 |    209.5 |       76.0 |        — |     85.6 | PyTorch |
| inv_128x128            |      92.2 |     62.1 |     97.1 |      139.0 |        — |     86.6 | PyTorch |
| cholesky_128x128       |      50.6 |     49.6 |     31.5 |       60.3 |        — |     37.4 | MLX |
| svd_128x128            |     992.2 |    998.4 |   1025.6 |     1825.0 |        — |   1020.1 | Peregrine |
| qr_128x128             |     188.8 |    223.4 |    205.6 |      327.4 |        — |    193.0 | Peregrine |
| eigh_128x128           |    1845.7 |    703.2 |    747.8 |      715.1 |        — |    751.7 | PyTorch |
| det_128x128            |      52.2 |     49.6 |        — |       81.9 |        — |     76.6 | PyTorch |
| solve_256x256          |     189.0 |    178.1 |    747.4 |      378.2 |        — |    261.3 | PyTorch |
| inv_256x256            |     466.9 |    301.3 |    250.6 |      851.1 |        — |    336.8 | MLX |
| cholesky_256x256       |     226.4 |     78.6 |     53.3 |      283.8 |        — |    117.3 | MLX |
| svd_256x256            |    5892.5 |   5781.4 |   6162.9 |     8113.8 |        — |   5996.2 | PyTorch |
| qr_256x256             |    1020.4 |   1003.9 |   1065.7 |     1705.2 |        — |    979.6 | JAX |
| eigh_256x256           |    6065.6 |   3454.5 |   3615.8 |     4627.9 |        — |   3576.7 | PyTorch |
| det_256x256            |     213.8 |    208.7 |        — |      434.1 |        — |    206.4 | JAX |
| matmul_bias_gelu_196x768x3072 |    1209.0 |    934.7 |        — |     2440.2 |   1259.2 |   2142.1 | PyTorch |
| matmul_bias_gelu_196x1024x4096 |    2184.3 |   2046.4 |        — |     3721.7 |   1276.0 |   3499.1 | tinygrad |
| add_layernorm_196x768  |     108.5 |    110.3 |        — |     1258.4 |   1143.5 |    231.7 | Peregrine |
| add_layernorm_196x1024 |     139.0 |    108.0 |        — |     1316.1 |   1148.5 |    287.2 | PyTorch |
| matmul_f32_196x768x3072 |     696.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14580.8 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1528.9 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26318.7 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 0.93x** (Peregrine is faster)
- **Peregrine vs MLX: 0.67x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.52x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.10x** (Peregrine is faster)
- **Peregrine vs JAX: 0.60x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 64/141 ops
- PyTorch: 35/141 ops
- MLX: 17/141 ops
- TensorFlow: 14/141 ops
- JAX: 10/141 ops
- tinygrad: 1/141 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
