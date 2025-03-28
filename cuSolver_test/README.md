
# cuSolver 性能对比程序

## 编译程序

```bash
make
```

---

## 执行说明

### 参数说明
- m > n
### 生成 Q矩阵



```bash
./cusolver_orgqr_WY_float m n
```

### 不生成 Q矩阵


```bash
./cusolver_geqrf_float m n
```


