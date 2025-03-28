
# cuSolver 性能对比程序

## 编译程序

执行以下命令编译程序：

```bash
make
```

---

## 执行说明

### `m > n` 时生成 Q矩阵

运行以下命令：

```bash
./cusolver_orgqr_WY_float m n
```

### 不生成 Q矩阵

运行以下命令：

```bash
./cusolver_geqrf_float m n
```

---

确保根据具体需求确定参数 `m` 和 `n` 的值，以满足程序对矩阵维度的要求。
