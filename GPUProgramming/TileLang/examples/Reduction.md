# TileLang Reduction

这个例子把二维矩阵的每一行求和：

$$
y_i=\sum_{j=0}^{N-1}x_{ij}.
$$

CUDA reduction 的完整性能分析见 [[../../CUDA Reduction 优化|CUDA Reduction 优化]]。

## Reducer 写法

下面展示单个 CTA 处理一个 `(BM, BN)` tile 时的核心结构：

```python
with T.Kernel(T.ceildiv(M, BM), threads=threads) as bx:
    x_shared = T.alloc_shared((BM, BN), "float16")
    x_frag = T.alloc_fragment((BM, BN), "float16")
    out = T.alloc_reducer(
        (BM,),
        "float32",
        op="sum",
        replication="all",
    )

    T.clear(out)

    for no in T.Pipelined(T.ceildiv(N, BN), num_stages=num_stages):
        T.copy(x[bx * BM, no * BN], x_shared)
        T.copy(x_shared, x_frag)

        for i, j in T.Parallel(BM, BN):
            out[i] += x_frag[i, j]

    T.finalize_reducer(out)
    T.copy(out, y[bx * BM])
```

这是概念骨架；边界 tile 需要 safe access 或显式 predicate。

## 1. 分块

```text
CTA bx
  -> BM 个输出行

每次 no
  -> 每行 BN 个输入元素
```

Reduction domain $N$ 被拆成多个 $B_N$ tile。`out[i]` 跨 `no` 循环保存第 `i` 行的累计结果。

## 2. 数据路径

```text
global x tile
  -> x_shared
  -> x_frag
  -> T.Parallel 中更新 reducer partial
  -> T.finalize_reducer
  -> global y
```

`x_shared` 支持 CTA 协作搬运；`x_frag` 让 elementwise/reduction 输入按 fragment layout 分布到线程寄存器。

## 3. Parallel 与 Reducer Layout

循环描述完整逻辑域：

```python
for i, j in T.Parallel(BM, BN):
    out[i] += x_frag[i, j]
```

编译器需要协调：

```text
(i, j) 的 loop owner
x_frag[i, j] 的 fragment owner
out[i] 的 partial-result owner
```

同一行的不同 `j` 可能由不同线程处理，所以 `out[i]` 在 finalize 前可以分散为多个 partial result。

## 4. Finalize

`T.finalize_reducer(out)` 完成跨线程归约。调用前后语义不同：

```text
before finalize: thread-local / group-local partials
after finalize : complete row sums according to result layout
```

任何需要完整 `out[i]` 的后处理都应发生在 finalize 之后。

## 5. `replication="all"`

这里选择 fully replicated result，表示相关线程都能本地看到完整的 `out[i]`。适合 finalize 后还要做逐行 normalization 或广播计算。

如果 finalize 后只需唯一线程写回 global，并且没有多线程后处理，fully replicated 可能不是必要的。应根据后续 consumer 决定结果 layout，而不是机械地总写 `"all"`。

它与 fragment replication 的区别：

```text
T.Fragment(..., replicate=R)
  -> 输入/中间 fragment 的通用 ownership

alloc_reducer(..., replication="all")
  -> reduction 最终结果的 ownership
```

## 6. Pipeline

外层 `no` 循环可以重叠：

```text
load next N tile
  with
reduce current N tile
```

但 reduction 常受以下因素影响：

- global bandwidth；
- 每线程在飞 load 数；
- fragment/loop layout；
- warp shuffle 和 barrier；
- reducer partial 的依赖链。

若单个 tile 内 reduction 延迟很长，增加 stage 不一定能消除累加依赖。

## 7. 边界单位元

当 `N` 不能整除 `BN` 时，越界输入必须等价于 sum 的单位元：

```text
invalid x -> 0
```

对 max/min 则不能继续填 0：

```text
max padding -> 足够小的值
min padding -> 足够大的值
```

边界 predicate、buffer 初始化和 reducer identity 必须共同保持数学语义。

## 阅读一个 Reduction Kernel

```text
T.Kernel         -> 每个 CTA 负责哪些输出
T.copy           -> 输入搬运和 layout 转换
T.Parallel       -> 完整逻辑 reduction domain
alloc_reducer    -> partial-result 状态
replication      -> 最终结果在哪些线程可见
finalize_reducer -> partial 到 complete result
T.Pipelined      -> 输入 tile 搬运与计算重叠
```

性能验证最后回答：global bandwidth 是否接近目标、shuffle/barrier 是否过重、结果 replication 是否必要，以及 stage 是否提供真实 overlap。

