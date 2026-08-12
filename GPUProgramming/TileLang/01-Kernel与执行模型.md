# Kernel 与执行模型

## `T.Kernel`

`T.Kernel` 声明一次 kernel launch 的 grid 和每个 CTA 的线程数：

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...
```

对 CUDA target，可建立下面的近似对应：

```text
T.Kernel 的 grid 维度 -> blockIdx.x / y / z
threads               -> 每个 CTA 的线程数
kernel body           -> 每个 CTA 独立执行一次
```

`bx`、`by` 是 block index，不是 tile 内的 thread index。普通 TileLang 程序通常使用 `T.Parallel` 或 tile primitive 表达 CTA 内并行，不必直接读取 `threadIdx`。

## 从 grid 到 tile

以二维输出为例：

```python
with T.Kernel(
    T.ceildiv(N, BN),
    T.ceildiv(M, BM),
    threads=threads,
) as (bx, by):
    row_base = by * BM
    col_base = bx * BN
```

逻辑关系为：

```text
grid.x = ceildiv(N, BN)
grid.y = ceildiv(M, BM)

CTA(bx, by)
  -> output[by * BM : (by + 1) * BM,
            bx * BN : (bx + 1) * BN]
```

一个 CTA 负责哪个 tile，是 kernel 中的索引公式决定的，不是 `T.Kernel` 自动猜出来的。

## 三个不同的大小

不要混淆：

| 参数 | 含义 |
| --- | --- |
| 问题规模 | 完整张量的 $M,N,K$ |
| CTA tile | 一个 CTA 处理的 $B_M,B_N,B_K$ |
| thread tile | 一个线程最终持有或计算的局部元素集合 |

CTA tile 由程序员选择；thread tile 可能由显式 layout 指定，也可能由 layout inference 和 tile primitive 共同确定。

## Tile 大小的权衡

Tile 太小：

- global-memory 数据复用不足；
- 单 CTA 的工作量和 ILP 不足；
- block 数量、调度和边界开销相对增大。

Tile 太大：

- shared memory 占用上升；
- 每线程寄存器和 fragment 压力上升；
- 可驻留 CTA/warp 数下降；
- pipeline buffer 的容量成本被 stage 数进一步放大；
- 边界浪费可能增加。

因此 tile shape 不是越大越好：

```text
reuse / ILP
    vs
register / shared memory / occupancy
```

TileLang 暴露这些选择，但不替用户建立唯一正确的数学分块。

## CTA 内并行

一个 CTA 内通常同时存在三类工作：

```text
协作搬运 global -> shared
shared tile 上的整体计算
fragment 上的 elementwise / reduction
```

它们可以使用不同的线程映射。协作加载要求 global access 合并；计算 layout 则要匹配 MMA 或后续消费者。两者不应为了“看起来一致”而强行使用同一映射。

```text
copy layout     -> 优先考虑合并访存和向量化
compute layout  -> 优先匹配 primitive / fragment consumer
```

## 边界 tile

当问题规模不能整除 tile shape 时，最后一个 CTA 只处理部分有效元素。常见处理方式：

- 显式写 `if` predicate；
- 使用能被安全访问合法化处理的规则访问；
- 对输入越界位置填充计算的单位元，例如 reduction sum 填 0。

边界逻辑首先是正确性问题，其次才是性能问题。需要验证：

- load 不越界；
- store 不覆盖无效位置；
- padding 不改变 reduce、softmax 等算子语义。

## 普通与 Persistent Kernel

普通映射通常是：

```text
一个逻辑 tile -> 一个 CTA
```

Persistent 映射是：

```text
固定数量 CTA -> 每个 CTA 在生命周期内处理多个逻辑 tile
```

前者让 GPU scheduler 分发大量 CTA；后者把一部分任务领取顺序和跨任务状态交给程序。详细讨论见 [[06-Persistent与性能分析]]。

## 检查清单

- grid 的每一维分别枚举什么逻辑任务？
- 一个 CTA 负责的 tile 是否覆盖全部输出且不重复写？
- `threads` 是否满足 tile primitive 的线程组织要求？
- shared memory 和 register 使用量允许多少 CTA 驻留？
- copy mapping 与 compute mapping 是否各自适合其访问模式？
- 边界 tile 的 load、计算单位元和 store 是否正确？

