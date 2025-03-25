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

| `m`     | `n`     | `nb`   | `b`   | Double Blocking Time(s) | Double Blocking TFLOPS | cuSolver Time(s) | cuSolver TFLOPS |  
|---------|---------|--------|-------|--------------------------|------------------------|------------------|----------------|  
| 512     | 256     | 128    | 32    | 0.0124                   | 0.0045                 | 0.0077           | 0.0073         |  
| 512     | 512     | 128    | 32    | 0.0197                   | 0.0091                 | 0.0170           | 0.0105         |  
| 1024    | 512     | 128    | 32    | 0.0230                   | 0.0194                 | 0.0184           | 0.0243         |  
| 1024    | 1024    | 128    | 32    | 0.0401                   | 0.0357                 | 0.0382           | 0.0375         |  
| 2048    | 1024    | 128    | 32    | 0.0463                   | 0.0773                 | 0.0422           | 0.0848         |  
| 2048    | 2048    | 128    | 32    | 0.0855                   | 0.1340                 | 0.0882           | 0.1299         |  
| 4096    | 2048    | 128    | 32    | 0.1155                   | 0.2480                 | 0.1066           | 0.2686         |  
| 4096    | 4096    | 128    | 32    | 0.2078                   | 0.4410                 | 0.2265           | 0.4044         |  
| 8192    | 4096    | 128    | 32    | 0.2681                   | 0.8543                 | 0.3244           | 0.7061         |  
| 8192    | 8192    | 128    | 32    | 0.5420                   | 1.3524                 | 0.7516           | 0.9753         |  
| 8192    | 8192    | 1024   | 32    | 0.5216                   | 1.4054                 | 0.7516           | 0.9753         |  
| 8192    | 8192    | 2048   | 32    | 0.5363                   | 1.3669                 | 0.7516           | 0.9753         |  
| 16384   | 8192    | 128    | 32    | 0.7990                   | 2.2934                 | 1.4249           | 1.2861         |  
| 16384   | 8192    | 1024   | 32    | 0.7457                   | 2.4574                 | 1.4249           | 1.2861         |  
| 16384   | 16384   | 128    | 32    | 1.9083                   | 3.0730                 | 1.4249           | 1.2861         |  
| 16384   | 16384   | 1024   | 32    | 1.6347                   | 3.5872                 | 3.7777           | 1.5523         |  
| 16384   | 16384   | 2048   | 32    | 1.6922                   | 3.4654                 | 3.7777           | 1.5523         |  
