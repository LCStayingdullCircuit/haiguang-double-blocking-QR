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
