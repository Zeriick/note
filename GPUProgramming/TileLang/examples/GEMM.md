# TileLang GEMM

这个例子只用于串联 TileLang 概念。GEMM 的完整优化路径见 [[../../GEMM|GEMM 优化]]。

## Kernel 骨架

```python
@T.prim_func
def gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(
        T.ceildiv(N, BN),
        T.ceildiv(M, BM),
        threads=threads,
    ) as (bx, by):
        A_shared = T.alloc_shared((BM, BK), "float16")
        B_shared = T.alloc_shared((BK, BN), "float16")
        C_local = T.alloc_fragment((BM, BN), "float32")

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=num_stages):
            T.copy(A[by * BM, ko * BK], A_shared)
            T.copy(B[ko * BK, bx * BN], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * BM, bx * BN])
```

## 1. 数学分块

CTA `(bx, by)` 负责：

```text
C tile: [by * BM, bx * BN], shape = (BM, BN)
```

每次 `ko` 处理：

```text
A tile: shape = (BM, BK)
B tile: shape = (BK, BN)
```

因此一个 CTA 内的数学结构是：

$$
C_{tile}mathrel{+}=A_{tile}^{(k)}B_{tile}^{(k)}.
$$

`BM/BN/BK` 是用户选择的调度参数，不由 `T.gemm` 自动决定。

## 2. 空间分层

```text
A, B global tensor
  -> A_shared, B_shared
  -> T.gemm
C_local fragment
  -> C global tensor
```

- Global 保存完整矩阵；
- Shared 保存当前 $K$ step 的输入 tile，供 CTA 内重复消费；
- Fragment 保存分散在各线程寄存器中的输出累加器。

`C_local` 的逻辑 shape 是 `(BM, BN)`，并不表示每个线程持有完整输出 tile。

## 3. Layout

至少存在以下映射：

```text
global -> shared copy mapping
shared-memory physical layout / swizzle
T.gemm operand layout
C_local fragment ownership
fragment -> global store mapping
```

它们可以由 layout inference、tile primitive 约束和显式 annotation 共同确定。

Copy 与 compute 的理想映射不同：

```text
global copy  -> 连续 lane 访问连续地址
MMA compute  -> 满足目标指令的 fragment/shared layout
```

Shared memory 是两者之间常用的重排边界。

## 4. Pipeline

每个 `ko` 有两个阶段：

```text
producer: load A/B tile
consumer: GEMM
```

多 stage 试图让：

```text
load(ko + d) overlap compute(ko)
```

增加 `num_stages` 会扩大预取距离，也会按 stage 复制 shared buffer 状态。必须同时检查 shared-memory 用量、occupancy 和实际 overlap。

## 5. 输出

在所有 $K$ tile 累加完成后：

```python
T.copy(C_local, C[by * BM, bx * BN])
```

Store lowering 需要把 fragment ownership 转换成 global-memory 写入。有效实现应同时满足：

- 每个输出元素只有正确写入者；
- global store 尽量 coalesced；
- dtype conversion 符合语义；
- 边界 CTA 不越界。

若直接 store 的 layout 不合适，也可能先经 shared memory 重排。

## 参数关系

| 参数 | 主要收益 | 主要代价 |
| --- | --- | --- |
| 增大 `BM/BN` | 输入复用、计算量 | 累加器和 shared tile 增大 |
| 增大 `BK` | 每 step 工作量、搬运粒度 | shared memory、边界浪费 |
| 增大 `threads` | 协作搬运和并行度 | 每线程工作减少、调度约束变化 |
| 增大 `num_stages` | 预取距离、latency hiding | 多 buffer、occupancy 下降 |

这些参数相互耦合，应搜索完整组合。

## 阅读一个 GEMM Kernel

```text
T.Kernel       -> grid 与 CTA tile
alloc_shared   -> 输入 tile 的片上缓存
alloc_fragment -> 输出的寄存器 ownership
T.copy         -> 数据搬运与重排
T.Pipelined    -> K-loop 时间重叠
T.gemm         -> 目标相关矩阵乘加
final T.copy   -> fragment 写回 global
```

最后再用 profiler 回答：tile 复用是否有效、Tensor Core 是否忙、copy 是否被隐藏，以及资源是否限制驻留。

