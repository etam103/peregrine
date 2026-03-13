# Peregrine vs ML Frameworks — Wall-Clock Benchmark

All benchmarks run on CPU with `nice -n 10`. Times in microseconds (lower is better).

**Versions:**
- PyTorch: 2.10.0
- MLX: 0.30.6
- TensorFlow: 2.21.0
- tinygrad: ?
- JAX: 0.9.1

| Operation | Peregrine | PyTorch | MLX | TensorFlow | tinygrad | JAX | Best |
|-----------|----------: | --------: | --------: | ----------: | --------: | --------: | ----:|
| matmul_128x128         |       5.8 |      5.8 |     28.4 |       93.3 |    508.2 |    220.3 | Peregrine |
| matmul_256x256         |      59.0 |     30.1 |     81.4 |      189.9 |    513.6 |    211.9 | PyTorch |
| matmul_512x512         |     202.5 |    124.9 |    221.2 |      728.7 |    491.2 |    695.0 | PyTorch |
| matmul_1024x1024       |    1051.0 |        — |        — |          — |        — |        — | Peregrine |
| matmul_2048x2048       |    9603.6 |        — |        — |          — |        — |        — | Peregrine |
| add_100k               |      12.8 |     33.6 |     32.4 |       60.0 |    221.7 |     63.6 | Peregrine |
| add_500k               |     111.5 |     94.8 |     84.4 |      138.2 |    216.2 |    426.1 | MLX |
| add_1M                 |     158.3 |        — |        — |          — |        — |        — | Peregrine |
| add_5M                 |     553.4 |        — |        — |          — |        — |        — | Peregrine |
| add_10M                |     962.3 |        — |        — |          — |        — |        — | Peregrine |
| mul_100k               |      13.0 |     30.6 |     33.0 |       51.3 |    224.8 |     30.6 | Peregrine |
| mul_500k               |     132.4 |     97.6 |     88.0 |      139.6 |    214.8 |    189.7 | MLX |
| mul_1M                 |     133.6 |        — |        — |          — |        — |        — | Peregrine |
| mul_5M                 |     501.8 |        — |        — |          — |        — |        — | Peregrine |
| mul_10M                |     913.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_100k               |     112.4 |     56.1 |     73.6 |       67.9 |    270.4 |     48.5 | JAX |
| exp_500k               |     204.6 |    191.3 |    241.9 |      161.2 |    268.5 |    227.3 | TensorFlow |
| exp_1M                 |     307.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_5M                 |    1098.6 |        — |        — |          — |        — |        — | Peregrine |
| exp_10M                |    2139.7 |        — |        — |          — |        — |        — | Peregrine |
| relu_100k              |       9.0 |     28.8 |     27.0 |       42.0 |    400.2 |    108.0 | Peregrine |
| relu_1M                |     104.4 |        — |        — |          — |        — |        — | Peregrine |
| softmax_8x128          |       1.2 |     30.7 |     19.0 |       12.2 |    740.0 |     63.8 | Peregrine |
| softmax_8x512          |       4.2 |     33.2 |     20.1 |       14.3 |    749.4 |     48.0 | Peregrine |
| mlp_fwd_64x784         |      33.6 |     26.8 |     54.6 |      271.1 |   2411.7 |    218.9 | PyTorch |
| mlp_fwd_256x784_wide   |     420.8 |        — |        — |          — |        — |        — | Peregrine |
| train_step_64          |     816.6 |   1257.6 |    869.1 |     9763.3 |  29045.4 |   5925.0 | Peregrine |
| train_step_256_wide    |    3312.9 |        — |        — |          — |        — |        — | Peregrine |
| reciprocal_100k        |       8.5 |     46.0 |     28.5 |       50.0 |    195.5 |     44.5 | Peregrine |
| square_100k            |       8.6 |     29.0 |     24.7 |       15.2 |    204.2 |     51.4 | Peregrine |
| rsqrt_100k             |      76.1 |     30.4 |     39.2 |       53.8 |        — |     87.0 | PyTorch |
| floor_100k             |      46.6 |     27.1 |     25.7 |       16.7 |    491.0 |     49.5 | TensorFlow |
| ceil_100k              |      46.6 |     28.7 |     30.8 |       16.6 |    432.5 |     34.7 | TensorFlow |
| round_100k             |      46.6 |     30.7 |     28.2 |       49.8 |        — |     45.0 | MLX |
| sign_100k              |      54.4 |     29.3 |     32.8 |       52.5 |    927.3 |     36.9 | PyTorch |
| expm1_100k             |     142.0 |     67.8 |    119.8 |      107.3 |        — |     95.2 | PyTorch |
| log2_100k              |     108.4 |     53.4 |    111.0 |      156.8 |    197.1 |     58.0 | PyTorch |
| log10_100k             |     106.5 |     53.9 |    122.9 |      158.7 |        — |     87.0 | PyTorch |
| log1p_100k             |     126.6 |     50.9 |    138.3 |       83.0 |        — |    113.2 | PyTorch |
| erf_100k               |     121.7 |     39.2 |    109.0 |       63.5 |        — |     50.5 | PyTorch |
| sinh_100k              |      51.0 |    101.5 |    105.1 |      140.8 |    661.0 |    140.0 | Peregrine |
| cosh_100k              |      46.3 |     99.2 |    102.8 |      134.5 |    560.9 |     88.6 | Peregrine |
| arcsin_100k            |      52.1 |     54.9 |    102.6 |       57.3 |   3720.7 |    127.0 | Peregrine |
| arccos_100k            |     106.1 |     53.5 |    119.4 |       57.8 |        — |    208.3 | PyTorch |
| arctan_100k            |      53.2 |     73.3 |    101.9 |       62.1 |   3840.3 |    219.8 | Peregrine |
| arcsinh_100k           |     141.6 |    128.7 |    357.5 |      140.2 |        — |    141.4 | PyTorch |
| maximum_100k           |      12.5 |     33.9 |     26.6 |       47.9 |    227.2 |     55.4 | Peregrine |
| minimum_100k           |      12.5 |     30.4 |     30.8 |       45.6 |    437.3 |     58.9 | Peregrine |
| power_100k             |     393.2 |    210.9 |    238.0 |      276.8 |        — |    167.0 | JAX |
| arctan2_100k           |    1169.0 |    107.9 |    157.4 |       73.4 |        — |    303.4 | TensorFlow |
| logaddexp_100k         |     408.4 |    123.6 |    280.5 |      397.8 |        — |    186.5 | PyTorch |
| clip_100k              |       8.6 |     30.2 |     39.2 |       44.3 |    650.8 |     52.3 | Peregrine |
| where_100k             |      93.1 |     33.8 |     29.4 |       58.8 |    325.4 |     45.4 | MLX |
| greater_100k           |      81.0 |     36.5 |     21.5 |       56.6 |    219.2 |     46.8 | MLX |
| equal_100k             |      81.1 |     31.4 |     24.4 |       64.7 |    362.1 |     38.7 | MLX |
| sum_axis_256x512       |     112.8 |     34.7 |     20.8 |       58.2 |    249.4 |     16.7 | JAX |
| mean_axis_256x512      |     112.7 |     38.6 |     25.5 |       58.6 |    352.4 |     33.5 | MLX |
| max_axis_256x512       |     154.5 |     39.2 |     47.3 |       65.0 |    246.6 |     32.9 | JAX |
| min_axis_256x512       |     157.3 |     39.1 |     49.4 |       56.6 |    395.9 |     16.2 | JAX |
| var_256x512            |     235.7 |    225.0 |     60.9 |      260.6 |        — |     83.9 | MLX |
| prod_axis_256x512      |     149.2 |     28.4 |     26.3 |       54.5 |        — |     55.4 | MLX |
| logsumexp_256x512      |     381.1 |    144.3 |    119.9 |      373.6 |        — |    329.9 | MLX |
| cumsum_256x512         |     123.2 |     53.9 |    144.5 |      206.1 |    763.4 |    260.7 | PyTorch |
| argmax_axis_256x512    |     158.0 |     65.5 |    184.3 |       82.2 |   1592.1 |    198.2 | PyTorch |
| sum_axis_1024x1024     |     942.4 |        — |        — |          — |        — |        — | Peregrine |
| var_1024x1024          |    1940.2 |        — |        — |          — |        — |        — | Peregrine |
| tril_256x256           |      34.7 |     35.3 |     63.6 |       56.0 |   2285.7 |     34.5 | JAX |
| triu_256x256           |      34.5 |     35.7 |     59.8 |       54.6 |   2255.6 |     43.9 | Peregrine |
| repeat_64x128_2x3      |     125.0 |     36.8 |     32.4 |       80.3 |        — |     30.5 | JAX |
| pad_64x128             |      17.8 |      3.9 |     21.8 |       90.9 |    112.0 |     21.7 | PyTorch |
| stack_8x64x128         |      15.7 |      8.5 |     47.1 |       64.6 |   1173.2 |    212.2 | PyTorch |
| diagonal_512x512       |       0.8 |      0.7 |     28.4 |       13.1 |        — |     11.0 | PyTorch |
| silu_100k              |      64.0 |     48.1 |     89.8 |      274.5 |    397.5 |     90.5 | PyTorch |
| softplus_100k          |     298.4 |    122.6 |    286.7 |      127.7 |    976.5 |    157.3 | PyTorch |
| mish_100k              |     500.9 |    293.1 |    408.3 |      255.8 |   1496.2 |    291.2 | TensorFlow |
| leaky_relu_100k        |       8.0 |     40.9 |     85.1 |       19.2 |        — |     36.6 | Peregrine |
| elu_100k               |     142.2 |    102.3 |    125.8 |       69.0 |   1132.8 |     82.6 | TensorFlow |
| hard_tanh_100k         |      52.0 |     29.5 |     37.7 |       51.5 |        — |     52.0 | PyTorch |
| relu6_100k             |      50.5 |     37.8 |     54.0 |       55.0 |    902.4 |    150.7 | PyTorch |
| hardswish_100k         |      83.7 |     29.8 |     75.5 |      245.7 |        — |     32.8 | PyTorch |
| gelu_100k              |      77.4 |     45.2 |    148.3 |      292.2 |   1060.0 |    278.5 | PyTorch |
| selu_100k              |     158.8 |    105.1 |     88.5 |       80.7 |    897.2 |     84.9 | TensorFlow |
| softsign_100k          |      34.9 |     90.2 |     50.9 |       48.1 |        — |    102.7 | Peregrine |
| cross_entropy_64x10    |       2.6 |     37.6 |     24.0 |      680.1 |   4216.7 |     77.2 | Peregrine |
| l1_loss_64x10          |       1.0 |      6.9 |     19.2 |       49.2 |   1403.5 |     17.4 | Peregrine |
| mse_loss_64x10         |       3.7 |      6.3 |     20.7 |       42.7 |    528.7 |     31.9 | Peregrine |
| huber_loss_64x10       |       5.2 |      6.3 |     39.6 |      262.0 |        — |     63.7 | Peregrine |
| smooth_l1_loss_64x10   |       5.1 |      6.4 |     40.5 |      263.8 |        — |     65.3 | Peregrine |
| kl_div_loss_64x10      |       2.5 |      7.5 |     23.5 |      418.7 |        — |     96.7 | Peregrine |
| cosine_sim_loss_64x64  |      13.6 |     11.4 |    125.2 |      264.4 |        — |     94.8 | PyTorch |
| rmsnorm_64x512         |      57.6 |     52.2 |     39.0 |      451.9 |        — |     97.0 | MLX |
| conv1d_1x32x128_k3     |      20.0 |     45.5 |     29.9 |      537.2 |        — |     86.1 | Peregrine |
| avgpool2d_1x16x32x32   |      27.1 |     31.9 |    283.6 |       69.1 |        — |     59.4 | Peregrine |
| groupnorm_4x64x16x16   |     109.3 |     39.2 |    235.1 |      842.0 |        — |    173.2 | PyTorch |
| rnn_seq32_128_256      |     198.7 |    278.4 |        — |          — |        — |        — | Peregrine |
| lstm_seq32_128_256     |    1151.4 |    855.3 |        — |          — |        — |        — | PyTorch |
| gru_seq32_128_256      |     853.9 |    813.8 |        — |          — |        — |        — | PyTorch |
| optim_adam_64          |     814.6 |   1362.4 |        — |          — |        — |        — | Peregrine |
| optim_rmsprop_64       |     941.8 |   1081.8 |        — |          — |        — |        — | Peregrine |
| optim_lion_64          |     947.2 |        — |        — |          — |        — |        — | Peregrine |
| optim_adafactor_64     |    1336.4 |        — |        — |          — |        — |        — | Peregrine |
| rand_uniform_100k      |     109.6 |    258.3 |    520.5 |      139.1 |   2993.3 |    641.8 | Peregrine |
| rand_normal_100k       |     829.6 |    976.8 |    748.1 |      369.5 |   4107.5 |    728.2 | TensorFlow |
| rand_bernoulli_100k    |     312.2 |    251.7 |    490.7 |      232.9 |        — |    697.2 | TensorFlow |
| rand_uniform_1M        |    1067.3 |   2780.8 |   4873.0 |      598.5 |   2914.6 |   2493.8 | TensorFlow |
| rand_normal_1M         |    7669.8 |  10379.8 |   7038.8 |     2609.8 |   4127.3 |   3150.8 | TensorFlow |
| rfft_1k                |       2.2 |      4.8 |     20.6 |       46.8 |        — |     55.6 | Peregrine |
| rfft_4k                |       6.6 |     15.8 |     29.7 |       57.9 |        — |     80.8 | Peregrine |
| rfft_16k               |      30.3 |     78.6 |     83.8 |      100.4 |        — |    144.4 | Peregrine |
| fft_1k                 |       3.3 |      7.1 |     24.4 |        9.3 |        — |     54.5 | Peregrine |
| fft_4k                 |      12.2 |     26.5 |     44.4 |       15.8 |        — |     60.6 | Peregrine |
| norm_l2_1k             |       1.1 |      1.5 |     20.8 |       80.0 |        — |      6.7 | Peregrine |
| solve_64x64            |      12.0 |     20.5 |    100.3 |       26.0 |        — |     37.1 | Peregrine |
| inv_64x64              |      37.5 |     24.9 |     50.5 |       36.5 |        — |    129.4 | PyTorch |
| cholesky_64x64         |       9.7 |     43.1 |     22.0 |       19.8 |        — |     38.0 | Peregrine |
| svd_64x64              |     276.7 |    289.6 |    305.3 |      484.8 |        — |    872.1 | Peregrine |
| qr_64x64               |      41.4 |     76.6 |     62.4 |       85.0 |        — |     91.2 | Peregrine |
| eigh_64x64             |     381.4 |    218.8 |    236.4 |      154.5 |        — |    291.6 | TensorFlow |
| det_64x64              |      22.6 |     21.1 |        — |       22.3 |        — |     52.3 | PyTorch |
| solve_128x128          |      50.0 |     41.0 |    209.5 |       79.5 |        — |    124.4 | PyTorch |
| inv_128x128            |      95.9 |     56.6 |     97.1 |      156.9 |        — |    320.5 | PyTorch |
| cholesky_128x128       |      50.4 |     49.7 |     31.5 |       65.2 |        — |     75.9 | MLX |
| svd_128x128            |     989.2 |   1016.5 |   1025.6 |     1861.5 |        — |   5569.4 | Peregrine |
| qr_128x128             |     188.0 |    232.0 |    205.6 |      358.3 |        — |   3723.2 | Peregrine |
| eigh_128x128           |    1845.2 |    715.0 |    747.8 |      693.5 |        — |   1538.6 | TensorFlow |
| det_128x128            |      52.4 |     47.4 |        — |      110.6 |        — |    115.7 | PyTorch |
| solve_256x256          |     189.2 |    176.8 |    747.4 |      384.2 |        — |    556.1 | PyTorch |
| inv_256x256            |     474.4 |    311.1 |    250.6 |     1007.1 |        — |    975.7 | MLX |
| cholesky_256x256       |     226.5 |     91.3 |     53.3 |      300.7 |        — |   1877.0 | MLX |
| svd_256x256            |    5975.2 |   5443.8 |   6162.9 |     8502.3 |        — |  16842.4 | PyTorch |
| qr_256x256             |    1009.5 |   1034.5 |   1065.7 |     1829.3 |        — |   7128.3 | Peregrine |
| eigh_256x256           |    6044.7 |   3167.6 |   3615.8 |     4572.8 |        — |   8205.1 | PyTorch |
| det_256x256            |     212.8 |    209.5 |        — |      468.0 |        — |    389.3 | PyTorch |
| matmul_bias_gelu_196x768x3072 |    1111.1 |   1475.1 |        — |     3314.7 |   1532.9 |   2934.2 | Peregrine |
| matmul_bias_gelu_196x1024x4096 |    2175.8 |   2781.8 |        — |     4916.6 |   1555.2 |   4963.2 | tinygrad |
| add_layernorm_196x768  |     107.8 |    105.6 |        — |     1580.7 |   1385.1 |    292.3 | PyTorch |
| add_layernorm_196x1024 |     182.7 |    116.5 |        — |     1721.2 |   1383.7 |    319.6 | PyTorch |
| matmul_f32_196x768x3072 |     625.5 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x768x3072 |   14481.7 |        — |        — |          — |        — |        — | Peregrine |
| matmul_f32_196x1024x4096 |    1517.0 |        — |        — |          — |        — |        — | Peregrine |
| matmul_i8_196x1024x4096 |   26252.0 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_128x128     |       5.2 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_256x256     |       4.7 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_512x512     |       4.6 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_1024x1024   |       5.3 |        — |        — |          — |        — |        — | Peregrine |
| gpu_matmul_2048x2048   |       6.0 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_100k           |       4.9 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_500k           |       4.7 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_1M             |       5.1 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_5M             |       6.1 |        — |        — |          — |        — |        — | Peregrine |
| gpu_add_10M            |       8.1 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_100k           |       4.7 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_500k           |       4.7 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_1M             |       4.7 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_5M             |       5.6 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mul_10M            |       6.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_100k           |       4.5 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_500k           |       4.3 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_1M             |       4.6 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_5M             |       6.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_exp_10M            |       7.0 |        — |        — |          — |        — |        — | Peregrine |
| gpu_relu_100k          |       4.5 |        — |        — |          — |        — |        — | Peregrine |
| gpu_relu_1M            |       4.6 |        — |        — |          — |        — |        — | Peregrine |
| gpu_softmax_8x128      |       2.2 |        — |        — |          — |        — |        — | Peregrine |
| gpu_softmax_8x512      |       4.2 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mlp_fwd_64x784     |      30.8 |        — |        — |          — |        — |        — | Peregrine |
| gpu_mlp_fwd_256x784_wide |      31.1 |        — |        — |          — |        — |        — | Peregrine |
| gpu_train_step_64      |    1223.0 |        — |        — |          — |        — |        — | Peregrine |
| gpu_train_step_256_wide |    4177.2 |        — |        — |          — |        — |        — | Peregrine |
| gpu_train_fused_64     |    1277.6 |        — |        — |          — |        — |        — | Peregrine |
| gpu_train_fused_256_wide |    4666.3 |        — |        — |          — |        — |        — | Peregrine |
| het_sequential_gpu_gpu |    2878.6 |        — |        — |          — |        — |        — | Peregrine |
| het_pipelined_gpu_cpu  |    1496.3 |        — |        — |          — |        — |        — | Peregrine |

**Geometric mean ratio (Peregrine / Framework):**
- < 1.00 = Peregrine is faster
- \> 1.00 = Framework is faster

- **Peregrine vs PyTorch: 1.04x** (Peregrine is slower)
- **Peregrine vs MLX: 0.68x** (Peregrine is faster)
- **Peregrine vs TensorFlow: 0.48x** (Peregrine is faster)
- **Peregrine vs tinygrad: 0.08x** (Peregrine is faster)
- **Peregrine vs JAX: 0.46x** (Peregrine is faster)

**Wins by framework:**
- Peregrine: 99/173 ops
- PyTorch: 39/173 ops
- MLX: 14/173 ops
- TensorFlow: 13/173 ops
- JAX: 7/173 ops
- tinygrad: 1/173 ops

---
*Median of timed iterations (warmup excluded). Lower is better.*
