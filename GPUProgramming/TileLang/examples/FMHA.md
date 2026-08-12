# TileLang FMHA

本例用一个短序列 FMHA（Fused Multi-Head Attention）kernel 串联 TileLang 的 grid、shared memory、fragment、warp reduction 和 Tensor Core GEMM（General Matrix Multiplication，通用矩阵乘）。这里关注代码如何表达调度；通用数学与优化原则见 [[../../FMHA 优化|FMHA 优化]]。

Q、K、V 分别表示 Query、Key 和 Value；QK 表示 $QK^\mathsf T$，PV 表示 softmax probability 与 V 的矩阵乘。代码中的 BF16 是 bfloat16，FP16/FP32 分别表示 16/32-bit floating point。

下面以一组便于说明的静态 tile 为例：

```text
threads      = 128 = 4 warps
query tile   = 16
key tile     = 128
head dim     = D, 且满足 Tensor Core alignment
input/output = BF16 或 FP16
accumulation = FP32
```

这是特定 shape 下的调度选择，不是 FMHA 的通用定义。当 key tile 大于 128 或 head dimension 不同时，warp ownership、GEMM tile 和 online softmax 策略都可能需要改变。

## Kernel 分块

Grid 的三维分别是 batch、head 和 query block：

```python
QUERY_TILE = 16
query_blocks = (compiled_max_seqlen + QUERY_TILE - 1) // QUERY_TILE

with T.Kernel(
    batch_size,
    num_heads,
    query_blocks,
    threads=128,
) as (batch, head, query_block):
    query_start = query_block * QUERY_TILE
```

一个 CTA（Cooperative Thread Array，即 CUDA thread block）负责

```text
Q tile: [query_start : query_start + 16, 0 : D]
K tile: [0 : 128, 0 : D]
V tile: [0 : 128, 0 : D]
O tile: [query_start : query_start + 16, 0 : D]
```

与“一个 CTA 一个 query”相比，16 个 query 可复用同一份 K/V tile，grid 的 query 维也缩小为原来的约 $1/16$。

## Packed 序列寻址

输入可以按 token 紧凑存储 QKV：

```text
qkv.shape = [total_tokens, 3 * num_heads * D]
```

对当前 batch 和 head：

```python
hidden_size = num_heads * D
sequence_start = cu_seqlens[batch]
sequence_length = cu_seqlens[batch + 1] - sequence_start
head_column = head * D

Q[token, d] = qkv[sequence_start + token, head_column + d]
K[token, d] = qkv[sequence_start + token, hidden_size + head_column + d]
V[token, d] = qkv[sequence_start + token, 2 * hidden_size + head_column + d]
```

一个常见错误是用 `max_seqlen` 代替当前 `sequence_length` 做 load guard，从而读到 packed buffer 中下一个 sample。

## Buffer 组织

```python
queries_shared = T.alloc_shared((16, D), dtype)
keys_shared = T.alloc_shared((128, D), dtype)
values_shared = T.alloc_shared((128, D), dtype)

# FP32 score，供 stable softmax 使用
scores_shared = T.alloc_shared((16, 128), "float32")

# 低精度未归一化权重，作为 PV GEMM operand
pv_weights_shared = T.alloc_shared((16, 128), dtype)
denominators_shared = T.alloc_shared((16,), "float32")

scores_fragment = T.alloc_fragment((16, 128), "float32")
output_fragment = T.alloc_fragment((16, D), "float32")
score_values = T.alloc_local((4,), "float32")
```

三种 scope 的含义不同：

| Buffer | Scope | 作用 |
| --- | --- | --- |
| `queries/keys/values_shared` | Shared | CTA 协作加载，为 GEMM 提供可复用 tile |
| `scores_fragment` | Fragment | QK 的逻辑 FP32 输出，实际分散在各线程寄存器中 |
| `score_values` | Local | 每个 lane 私有的 4 个 score |
| `scores_shared` | Shared | 把 fragment 转换成 softmax 所需的 row/key 访存 |
| `pv_weights_shared` | Shared | 把 softmax numerator 转换成第二个 GEMM 的 operand |
| `output_fragment` | Fragment | PV 的 FP32 累加结果 |

Fragment 的逻辑 shape 不表示每个 thread 都保存完整矩阵。具体 ownership 由 GEMM policy、layout inference 和 lowering 决定，见 [[../03-Layout与线程映射|Layout 与线程映射]]。

## 协作加载与 mask

```python
for local_query, d in T.Parallel(16, D):
    if query_start + local_query < sequence_length:
        queries_shared[local_query, d] = Q[query_start + local_query, d]
    else:
        queries_shared[local_query, d] = T.cast(0, dtype)

for key, d in T.Parallel(128, D):
    if key < sequence_length:
        keys_shared[key, d] = K[key, d]
    else:
        keys_shared[key, d] = T.cast(0, dtype)

T.sync_threads()
```

Q/K padding 可以填 0，但还不能直接把无效 score 当作 0。在 softmax 中，0 仍对应权重 $e^0=1$。因此 QK 后必须显式将无效 `(query, key)` 改成 $-\infty$ 的有限近似。

## Tensor Core QK

```python
T.clear(scores_fragment)
T.gemm(
    queries_shared,
    keys_shared,
    scores_fragment,
    transpose_B=True,
    policy=T.GemmWarpPolicy.FullRow,
)

scale = 1.0 / D**0.5
for local_query, key in T.Parallel(16, 128):
    if query_start + local_query < sequence_length and key < sequence_length:
        scores_shared[local_query, key] = (
            scores_fragment[local_query, key] * scale
        )
    else:
        scores_shared[local_query, key] = -3.402823e38

T.sync_threads()
```

GEMM 的逻辑 shape 是

$$
(16\times D)(128\times D)^\mathsf{T}
\longrightarrow 16\times128.
$$

`T.gemm` 之前必须清零 output fragment，因为它的语义通常是累加而不是无条件覆盖。能否使用 Tensor Core 取决于 dtype、shape、alignment、shared layout 和 target，不能只因为写了 `T.gemm` 就假设已生成 MMA（Matrix Multiply-Accumulate，矩阵乘加）指令。

## Warp-row softmax

128 个 key 恰好可以分给 32 个 lane，每个 lane 持有 4 个 key。四个 warp 每次并行处理四行，共执行四个 row wave：

```text
wave 0: warp 0..3 -> row 0..3
wave 1: warp 0..3 -> row 4..7
wave 2: warp 0..3 -> row 8..11
wave 3: warp 0..3 -> row 12..15
```

对 `warp` 和 `lane`：

```python
lane = thread % 32
warp = thread // 32

for row_wave in T.serial(4):
    local_query = row_wave * 4 + warp

    # 该条件依赖 warp，但在同一 warp 的 32 个 lane 中一致。
    if query_start + local_query < sequence_length:
        row_max = -3.402823e38
        for key_group in T.unroll(4):
            key = key_group * 32 + lane
            score_values[key_group] = scores_shared[local_query, key]
            row_max = T.max(row_max, score_values[key_group])
        row_max = T.warp_reduce_max(row_max)

        row_sum = 0.0
        for key_group in T.unroll(4):
            key = key_group * 32 + lane
            probability = 0.0
            if key < sequence_length:
                probability = T.exp(score_values[key_group] - row_max)
            pv_weights_shared[local_query, key] = T.cast(probability, dtype)
            row_sum += probability
        row_sum = T.warp_reduce_sum(row_sum)

        if lane == 0:
            denominators_shared[local_query] = row_sum
```

每行的 max 和 sum 都只在一个 warp 内完成，不需要 shared partial 或 CTA barrier。`local_query` 依赖 `warp`，但在同一 warp 中一致，因此包围 warp reduction 的 valid-row 分支必须是 warp-uniform。

这段代码的正确性依赖两个整除关系：

$$
4\text{ warps}\times4\text{ waves}=16\text{ rows},
$$

$$
32\text{ lanes}\times4\text{ keys/lane}=128\text{ keys}.
$$

如果 tile shape 改变，必须重新证明 ownership 是完备且不重叠的。

## 加载 V 与发布数据

```python
for key, d in T.Parallel(128, D):
    if key < sequence_length:
        values_shared[key, d] = V[key, d]
    else:
        values_shared[key, d] = T.cast(0, dtype)

T.sync_threads()
```

这个 barrier 同时建立三类可见性：

```text
V cooperative load complete
all warps have written pv_weights_shared
lane 0 of each row has written denominator
```

后续 PV GEMM 会跨 warp 消费这些数据，因此不能只依赖 warp-local 执行顺序。将多个 publication 需求合并到一个必要 barrier，是减少同步的实用方法。

## Tensor Core PV 与延后归一化

```python
T.clear(output_fragment)
T.gemm(
    pv_weights_shared,
    values_shared,
    output_fragment,
    policy=T.GemmWarpPolicy.FullRow,
)

for local_query, d in T.Parallel(16, D):
    if query_start + local_query < sequence_length:
        O[query_start + local_query, d] = T.cast(
            output_fragment[local_query, d]
            / denominators_shared[local_query],
            dtype,
        )
```

PV GEMM 的 shape 是

$$
(16\times128)(128\times D)
\longrightarrow16\times D.
$$

GEMM 使用未归一化的

$$
\widetilde P_{ij}=e^{S_{ij}-m_i}.
$$

因为 denominator $\ell_i$ 对整个输出行相同，可以在 PV 结束后再做一次逐行除法：

$$
O_{i,:}=\frac{(\widetilde PV)_{i,:}}{\ell_i}.
$$

这避免在 PV 前多做一次 $16\times128$ normalization pass。

## Shared-memory 预算

对 `QUERY_TILE=16`、`KEY_TILE=128`、`D=64`、BF16/FP32，不假设 buffer alias 的逻辑用量为：

```text
Q shared:              16 * 64  * 2 B =  2,048 B
K shared:             128 * 64  * 2 B = 16,384 B
V shared:             128 * 64  * 2 B = 16,384 B
FP32 scores:           16 * 128 * 4 B =  8,192 B
BF16 PV weights:       16 * 128 * 2 B =  4,096 B
FP32 denominators:     16 * 4 B       =     64 B
logical total:                              47,168 B
```

`scores_fragment` 和 `output_fragment` 是分布在线程寄存器中的 fragment，不计入 shared-memory 总量。但它们会影响 register pressure 和 occupancy。

这个预算已接近传统 48 KiB 静态 shared-memory 界限。实际 lowering 还可能引入 alignment 或 swizzle 开销，也可能通过 liveness 复用 K/V 空间。因此不能只看源码数组大小，应检查 lowering 后资源报告。

## 与 scalar 基线的关系

在引入两个 GEMM 之前，可以先实现一条完全显式的基线：

```text
one thread owns one key
  -> scalar QK dot product
shared/warp max reduction
shared/warp sum reduction
one thread owns one or more output dimensions
  -> scalar PV accumulation
```

它的教学价值是所有 ownership、mask 和同步都直接可见。然后可以按以下顺序逐步替换：

```text
scalar QK + scalar PV
  -> Tensor Core QK + scalar PV
  -> Tensor Core QK + Tensor Core PV
  -> warp-row softmax
```

这个顺序能分别回答 QK、PV 和 softmax 是否是当前瓶颈。若一次同时替换 GEMM、layout、reduction 和 pipeline，遇到编译或正确性问题时很难定位原因。

## 可选的 K/V padding

当 scalar QK/PV 直接访问 shared K/V 时，逻辑 shape `(128, 64)` 的 BF16 row stride 为 128 Bytes，固定维度的跨行访存可能严重 bank conflict。可用两列 padding 改变物理 stride：

```python
KV_PADDING = 2
kv_shared = T.alloc_shared((128, D + KV_PADDING), dtype)
```

这个技巧不应盲目套用到 `T.gemm` operand。MMA lowering 对 shared layout 有自己的 alignment/swizzle 要求，一个适合 scalar access 的 padding 不一定适合 Tensor Core。应根据实际 consumer 分别设计 layout。

## 编译期专门化

Kernel 可以保留运行时 ABI（Application Binary Interface，应用二进制接口），同时用 Python 参数专门化 grid 和 buffer shape：

```python
@T.prim_func
def kernel(
    qkv: T.Tensor((total_tokens, packed_size), dtype),
    cu_seqlens: T.Tensor((compiled_batch_size + 1,), "int32"),
    max_s: T.int64,
    num_heads: T.int64,
    head_dim: T.int64,
    out: T.Tensor((total_tokens, hidden_size), dtype),
    batch_size: T.int64,
):
    with T.Kernel(
        compiled_batch_size,
        compiled_num_heads,
        query_blocks,
        threads=128,
    ):
        ...
```

这里 `compiled_*` 是静态 Python 整数，用于 lowering；运行时参数用于保持调用接口。缓存 compiled kernel 时，cache key 必须包含所有会改变代码生成的 shape、dtype 和 target，否则可能错用旧 kernel。

## 验证顺序

1. 用 CPU 模拟 ownership，证明 16 行和 128 个 key 恰好被覆盖一次。
2. 对短序列、warp 边界和 tile 边界长度与 reference attention 比较。
3. 检查 frontend IR（Intermediate Representation，中间表示）中的 GEMM shape、fragment shape、warp reduction 和 barrier 位置。
4. 检查 CUDA lowering 后的 MMA 指令、shared memory、register 和 spill。
5. 分开测量首次 JIT 和 steady-state kernel 时间。
6. 每次只改一项主要调度，保留可直接比较的基线。

## 阅读这个 Kernel

```text
T.Kernel             -> CTA 拥有 16 个 query rows
T.Parallel load      -> Q/K/V 协作搬运和边界填充
first T.gemm         -> Tensor Core QK
fragment -> shared   -> score layout 转换与 mask
warp-row reduction   -> FP32 stable softmax
shared publication   -> PV weights / denominator / V 可见
second T.gemm        -> Tensor Core PV
final division       -> 用 FP32 denominator 归一化并写回
```

最后再用 profiler 回答：Tensor Core 是否忙碌、fragment 物化是否过重、barrier 是否仍占主导、shared/register 是否限制 occupancy，以及 padding 计算在真实长度分布下占多大比例。
