# 海光平台实现 Double Blocking QR 分解

## 编译
执行以下命令编译程序：
```bash
make
```

## 执行
运行程序时使用以下命令：
```bash
./mydoubleblocking_Iter_V6 m n nb b
```

### 参数说明
- `nb` 必须大于 `b`：`nb > b`
- 建议设置：`b = 32`

### 实验数据
- cuSOLVER只需要m和n，double blocking算法需要额外的nb和b，所以相同m，n执行时，cuSOLVER数据不变

#### 不生成Q的情况
- double blocking算法默认生成W，Y，R矩阵，其中W和Y矩阵可以用来生成Q
- 对于cuSolver，执行cusolverDnSgeqrf函数只生成Householder向量，而不显式生成
- 下面表格数据为不显式生成Q的性能对比
  
| `m`     | `n`     | `nb`   | `b`   | Double Blocking Time(s) | Double Blocking TFLOPS | cuSolver Time(s) | cuSolver TFLOPS |  
|---------|---------|--------|-------|--------------------------|------------------------|------------------|----------------|  
| 2048    | 2048    | 128    | 32    | 0.0855                   | 0.1340                 | 0.0882           | 0.1299         |  
| 4096    | 2048    | 128    | 32    | 0.1155                   | 0.2480                 | 0.1066           | 0.2686         |  
| 4096    | 4096    | 128    | 32    | 0.2078                   | 0.4410                 | 0.2265           | 0.4044         |  
| 8192    | 1024    | 128    | 32    | 0.0625                   | 0.2633                 | 0.0667           | 0.2470         |  
| 8192    | 2048    | 128    | 32    | 0.1269                   | 0.4962                 | 0.1446           | 0.4355         |  
| 8192    | 4096    | 128    | 32    | 0.2681                   | 0.8543                 | 0.3244           | 0.7061         |  
| 8192    | 8192    | 128    | 32    | 0.5420                   | 1.3524                 | 0.7516           | 0.9753         |  
| 8192    | 8192    | 1024   | 32    | 0.5216                   | 1.4054                 | 0.7516           | 0.9753         |  
| 8192    | 8192    | 2048   | 32    | 0.5363                   | 1.3669                 | 0.7516           | 0.9753         |  
| 16384   | 8192    | 128    | 32    | 0.7990                   | 2.2934                 | 1.4249           | 1.2861         |  
| 16384   | 8192    | 1024   | 32    | 0.7457                   | 2.4574                 | 1.4249           | 1.2861         |  
| 16384   | 128     | 64     | 32    | 0.0101                   | 0.0533                 | 0.0110           | 0.0488         |  
| 16384   | 256     | 64     | 32    | 0.0179                   | 0.1190                 | 0.0249           | 0.0857         |  
| 16384   | 512     | 64     | 32    | 0.0349                   | 0.2435                 | 0.0531           | 0.2983         |  
| 16384   | 1024    | 64     | 32    | 0.0696                   | 0.4836                 | 0.1128           | 0.2983         |  
| 16384   | 16384   | 128    | 32    | 1.9083                   | 3.0730                 | 1.4249           | 1.2861         |  
| 16384   | 16384   | 1024   | 32    | 1.6347                   | 3.5872                 | 3.7777           | 1.5523         |  
| 16384   | 16384   | 2048   | 32    | 1.6922                   | 3.4654                 | 3.7777           | 1.5523         |  


#### 生成Q的情况
- double blocking算法通过额外执行gemm生成Q矩阵
- cuSolver通过额外执行cusolverDnSorgqr函数生成Q矩阵
- 下面表格数据为显式生成Q的性能对比
  
| `m`     | `n`     | `nb`   | `b`   | Double Blocking Time(s) | Double Blocking TFLOPS       | cuSolver Time(s) | cuSolver TFLOPS      |  
|---------|---------|--------|-------|--------------------------|-----------------------------|------------------|----------------------|  
| 128     | 128     | 64     | 32    | 0.0057                   | 0.0010                      | 0.0049           | 0.0012               |  
| 256     | 256     | 64     | 32    | 0.0103                   | 0.0044                      | 0.0137           | 0.0032               |  
| 512     | 512     | 128    | 32    | 0.0192                   | 0.0186                      | 0.0322           | 0.0112               |  
| 1024    | 1024    | 128    | 32    | 0.0403                   | 0.0710                      | 0.0721           | 0.0396               |  
| 2048    | 2048    | 128    | 32    | 0.0878                   | 0.2610                      | 0.1694           | 0.1352               |  
| 4096    | 2048    | 128    | 32    | 0.1250                   | 0.4580                      | 0.2077           | 0.2756               |  
| 4096    | 4096    | 128    | 32    | 0.2220                   | 0.8254                      | 0.4432           | 0.4134               |  
| 8192    | 1024    | 128    | 32    | 0.0822                   | 0.4006                      | 0.1345           | 0.2448               |  
| 8192    | 2048    | 128    | 32    | 0.1557                   | 0.8094                      | 0.2919           | 0.4316               |  
| 8192    | 4096    | 128    | 32    | 0.3283                   | 1.3954                      | 0.6529           | 0.7016               |  
| 8192    | 8192    | 128    | 32    | 0.6519                   | 2.2490                      | 1.5025           | 0.9758               |  
| 8192    | 8192    | 1024   | 32    | 0.6411                   | 2.2866                      | 1.5025           | 0.9758               |  
| 8192    | 8192    | 2048   | 32    | 0.6454                   | 2.2714                      | 1.5025           | 0.9758               |  
| 16384   | 256     | 64     | 32    | 0.0377                   | 0.1132                      | 0.0488           | 0.0876               |  
| 16384   | 512     | 64     | 32    | 0.0705                   | 0.2410                      | 0.1049           | 0.1620               |  
| 16384   | 1024    | 64     | 32    | 0.1300                   | 0.5178                      | 0.2243           | 0.3000               |  
| 16384   | 8192    | 128    | 32    | 1.3191                   | 2.7784                      | 2.8832           | 1.2712               |  
| 16384   | 8192    | 1024   | 32    | 1.2643                   | 2.8988                      | 2.8832           | 1.2712               |  
| 16384   | 16384   | 128    | 32    | 3.0871                   | 3.7990                      | 7.5833           | 1.5466               |  
| 16384   | 16384   | 1024   | 32    | 2.8081                   | 4.1766                      | 7.5833           | 1.5466               |  
| 16384   | 16384   | 2048   | 32    | 2.8277                   | 4.1476                      | 7.5833           | 1.5466               |  
  

