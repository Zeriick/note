# Layout 与线程映射

Layout 回答的不是“tile 有多大”，而是：

> Tile 中每个逻辑元素由哪个线程处理，在线程或存储空间中的哪个位置。

## 三种映射

TileLang 中最常见的映射可分为：

```text
Fragment layout
logical index -> (thread id, thread-local index)

Parallel-loop layout
logical iteration -> (thread id, thread-local iteration)

Shared layout
logical coordinate -> shared-memory address / swizzle
```

它们的输入都可以是 `(i, j, ...)`，但输出空间不同，不能混为一谈。

## Fragment Layout

一个二维 fragment：

```python
frag = T.alloc_fragment((M, N), dtype)
```

逻辑上包含 $M\times N$ 个元素。Fragment layout 把每个 `(i,j)` 映射成：

```text
(owner thread, thread-local slot)
```

例如 8 个元素分给 4 个线程：

```text
logical index: 0  1  2  3  4  5  6  7
thread id:     0  1  2  3  0  1  2  3
local slot:    0  0  0  0  1  1  1  1
```

可以用显式 `T.Fragment` 表达类似映射：

```python
layout = T.Fragment(
    (8,),
    forward_fn=lambda i: (i % 4, i // 4),
)
```

`forward_fn` 的第一个结果是 thread mapping，第二个结果是 thread-local index。真实 MMA fragment 往往更复杂，并受指令约束，不应仅凭连续编号猜测。

## `T.Parallel`

`T.Parallel` 描述完整逻辑迭代域：

```python
for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]
```

代码表达的是“所有 `(i,j)` 都要执行”，不是“每个线程执行一次”。编译器还需要决定：

```text
(i, j)
  -> thread id
  -> 该线程第几次 local iteration
```

当 $M\times N$ 大于线程数时，一个线程通常处理多个逻辑迭代。local iteration 可能被展开、向量化或保持为线程内循环。

## Parallel Loop Layout

必要时可以显式给 `T.Parallel` 提供 layout：

```python
loop_layout = T.Fragment(
    (M, N),
    forward_fn=lambda i, j: (...thread..., ...local...),
)

for i, j in T.Parallel(M, N, loop_layout=loop_layout):
    ...
```

对一个 $k$ 维 `T.Parallel`，layout 的输入维数必须也是 $k$。嵌套 parallel loop 的 layout 作用于整个循环域，而不是分别给每层循环独立决定线程映射。

显式 loop layout 适用于：

- 需要固定 vector/coalescing 模式；
- producer 和 consumer 映射必须对齐；
- 特殊指令或跨线程通信要求特定 ownership；
- layout inference 的结果不符合性能目标。

## Layout Inference

省略显式 layout 时，layout inference 会结合以下信息补全映射：

- fragment 的 shape 和访问方式；
- `T.Parallel` 的逻辑迭代域；
- 输入、输出之间的索引关系；
- `T.copy`、`T.gemm`、reduction 等 tile operator 的 layout 约束；
- 已显式指定的 buffer 或 loop layout；
- target 的线程和指令要求。

结果可能包括：

```text
哪个 thread 执行哪个逻辑 iteration
该 thread 的哪个 local index 对应哪个 fragment 元素
不同 fragment 之间如何兼容
copy 是否具有可合并、可向量化的访问形状
```

Inference 解决的是约束传播和合法映射，不等于自动找到全局最优 schedule。Tile shape、线程数、stage 数和算法结构仍需调优。

## 一条完整的数据映射链

以下代码：

```python
for i, j in T.Parallel(BM, BN):
    C_local[i, j] += A_local[i, j]
```

至少涉及两类 layout：

```text
Parallel-loop layout
(i, j) -> 当前由哪个 thread 执行

Fragment layout
(i, j) -> C_local / A_local 位于哪个 thread 的哪个 slot
```

若执行该迭代的 thread 正好拥有两个 operand，访问可以保持本地。若不一致，则需要：

- 改变 loop layout；
- 改变 fragment layout；
- 使用能够完成跨线程交换的 primitive；
- 经 shared memory 重排。

这也是 layout inference 需要同时观察 loop 和 buffer 的原因。

## Shared Layout 与 Swizzle

Shared layout 映射：

```text
logical (i, j) -> physical shared-memory address
```

它主要影响：

- shared-memory bank conflict；
- vectorized load/store 的连续性；
- TMA tensor map；
- MMA/WGMMA 对 shared operand 的布局要求。

Swizzle 改变物理地址排列，但不改变 tile 的逻辑 shape 和数学语义：

```text
same logical tile
  -> different physical shared address
```

因此调试 shared 数据时，要分别检查逻辑坐标是否正确和物理布局是否高效。

## Replication

普通 Fragment layout：

```text
logical indices -> (thread id, local slot)
```

`T.Fragment(..., replicate=R)` 增加一个 thread-replica 维度：

```text
(logical indices, replica id)
  -> (thread id, local slot)
```

概念示例：

```python
layout = T.Fragment(
    (8,),
    forward_fn=lambda i, rep: (
        rep * 4 + i % 4,
        i // 4,
    ),
    replicate=2,
)
```

同一个逻辑元素出现两套 ownership：

```text
(index 5, replica 0) -> (thread 1, local slot 1)
(index 5, replica 1) -> (thread 5, local slot 1)
```

逻辑值没有增加一个业务维度；物理上有多个线程本地副本。

用途是让多个 warp/thread group 本地消费同一逻辑值，减少 shuffle、shared-memory 中转或 broadcast。代价是：

- 更多寄存器或计算；
- 更复杂的 ownership；
- 多副本写回时需要 guard，避免重复 store；
- 独立修改副本时必须保证语义一致。

## Fragment Replication 与 Reducer Replication

两者都产生“多份本地可见值”，但控制对象不同：

| 接口 | 含义 |
| --- | --- |
| `T.Fragment(..., replicate=R)` | 通用 layout 增加 replica dimension |
| `T.alloc_reducer(..., replication="all")` | reduction 完成后，结果在相关线程上 fully replicated |

一句话区分：

```text
Fragment replicate: 数据的 ownership 从一开始如何重复分布
Reducer replication: reduction 完成后结果如何分发
```

Reducer 细节见 [[04-计算原语与归约]]。

## Layout 可视化

已有显式 layout 时，可使用官方 `plot_layout` 查看映射：

```python
from tilelang.tools import plot_layout

plot_layout(fragment_layout, name="fragment", formats="png")
```

Fragment 图中通常用颜色区分 thread，并标注 thread-local index。也可以打开 layout inference visualization，查看编译器为 fragment 推断出的 shape、thread、local index 和 replication。

可视化用于回答：

- 相邻逻辑元素落在哪些线程？
- 每个线程持有多少 local element？
- 是否存在不期望的 replication？
- producer/consumer layout 是否直观兼容？

## 检查清单

- 现在讨论的是 fragment、parallel-loop 还是 shared layout？
- Layout 的输入是逻辑坐标，输出空间是什么？
- 每个逻辑元素有唯一 owner，还是存在 replica？
- 每个线程持有多少 local slot？
- 连续 lane 是否对应连续 global address？
- Shared layout 是否存在 bank conflict 或不符合 MMA/TMA 要求？
- 多副本 store 是否有唯一写入者或正确 predicate？

