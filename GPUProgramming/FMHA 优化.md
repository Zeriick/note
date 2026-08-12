# FMHA 优化

FMHA（Fused Multi-Head Attention）把 attention 的 QK、softmax 和 PV 融合到一个 kernel 中。关键不只是减少 kernel launch，而是让 score 和 probability 尽量停留在 register/shared memory，避免在 global memory 物化完整的 $L\times L$ 中间矩阵。

TileLang 代码实例见 [[TileLang/examples/FMHA|TileLang FMHA]]，GEMM（General Matrix Multiplication，通用矩阵乘）和 reduction 的基础见 [[GEMM]] 与 [[CUDA Reduction 优化]]。

## 数学结构

对一个 attention head，设

$$
Q\in\mathbb{R}^{L_q\times D},\qquad
K,V\in\mathbb{R}^{L_k\times D}.
$$

其中 Q、K、V 分别表示 Query、Key 和 Value，$D$ 是单个 head 的维度。

Scaled dot-product attention 为

$$
S=\frac{QK^\mathsf{T}}{\sqrt D}+B,\qquad
P=\operatorname{softmax}(S),\qquad
O=PV.
$$

$B$ 表示可选的 mask 或 attention bias。对不允许参与 attention 的位置令 $B_{ij}=-\infty$；无 mask 的位置可令 $B_{ij}=0$。因果 attention 通常屏蔽 $j>i$ 的位置，而本文后续的短序列例子只展示 padding mask；二者的数学处理相同，但有效 key 边界不同。

softmax 按 score 的 key 维度逐行计算：

$$
m_i=\max_j S_{ij},\qquad
\ell_i=\sum_j e^{S_{ij}-m_i},\qquad
O_{i,:}=\frac{\sum_j e^{S_{ij}-m_i}V_{j,:}}{\ell_i}.
$$

减去行最大值不改变 softmax 结果，但可防止 $e^{S_{ij}}$ 溢出。实现中通常让 Q/K/V 保持 BF16（bfloat16）或 FP16（16-bit floating point），而 score、最大值、分母和输出累加使用 FP32（32-bit floating point）。

## 为什么要融合

直接实现通常经过三个独立算子：

```text
QK^T -> global S
S    -> global P
PV   -> global O
```

除读写 Q/K/V/O 外，还会额外读写 $O(L_qL_k)$ 的 S 和 P。融合后的数据路径是：

```text
global Q/K
  -> shared tile
  -> register/fragment scores
  -> row-wise softmax
  -> shared/register weights
global V
  -> shared tile
  -> register/fragment output
  -> global O
```

对短序列或小 tile，可以在一个 CTA（Cooperative Thread Array，即 CUDA thread block）中保留整个 score tile。对长序列，必须沿 key 维分块，并在每次只保留当前 tile 的同时更新 online softmax 状态。两者属于同一融合思路，不要把“没有 global score 矩阵”误解为“从不保留 score tile”。

## Online softmax

长序列不能一次保存一整行 score 时，可以流式维护三个状态：

```text
m : 已处理 score 的最大值
l : 相对 m 的指数和
o : 相对 m 的未归一化输出
```

对新 score $s$ 及其 value $v$：

$$
\begin{aligned}
m' &= \max(m,s),\\
\alpha &= e^{m-m'},\\
\beta &= e^{s-m'},\\
l' &= \alpha l+\beta,\\
o' &= \alpha o+\beta v.
\end{aligned}
$$

最终输出为 $o/l$。当一次并入一个 key tile 时，把 $s$、$\beta$ 和 $\beta v$ 分别替换成 tile 的最大值、指数和与加权和，同样可以合并。

这个 recurrence 的重点是：当最大值变大时，旧的 $l$ 和 $o$ 必须同时乘 $e^{m-m'}$ 重标定。只更新 max 而不修正旧累加值会破坏数学等价性。

## Query tiling 与 K/V 复用

最简单的 ownership 是“一个 CTA 负责一个 query”。这容易实现，但同一 head 的每个 CTA 都要重新读取 K 和 V。

若一个 CTA 改为负责 $B_q$ 个 query，则可以让这些 query 共享同一份 K/V tile：

```text
one-query CTA:
  load Q[1, D]
  load K[Lk, D]
  load V[Lk, D]

Bq-query CTA:
  load Q[Bq, D]
  load K[Lk, D] once
  load V[Lk, D] once
```

忽略 cache 后，K/V 的 CTA 级重复读取次数约下降 $B_q$ 倍，block 数也由 $L_q$ 降为 $\lceil L_q/B_q\rceil$。对长序列，这个复用发生在每一个 key tile 内，不需要一次把完整 K/V 放入 shared memory。代价是：

- shared-memory 中需要保留更多 Q 行和 softmax 状态；
- 单个 CTA 的执行时间变长；
- register/shared-memory 增长可能降低 occupancy；
- 若各 query 仍串行处理，同步成本可能随 $B_q$ 增长。

因此 $B_q$ 是复用、资源和并行度之间的调度参数，不是越大越好。

## Softmax reduction 的 ownership

### Shared-memory tree

一种通用基线是每个 thread 持有一个 key score，然后在 shared memory 中做 max 和 sum 树形归约。它结构直观，但每一层都需要 CTA barrier。

### 两级 warp reduction

更常见的方法是：

```text
lane-local value
  -> warp shuffle reduction
warp lane 0
  -> shared partial[warp]
warp 0
  -> second warp reduction
```

对 $W$ 个 warp，shared scratch 只需 $W$ 个值。这会显著减少 shared load/store 和 block-wide barrier，但行结果仍然需要在 warp 之间传递。

### 一个 warp 独占一行

若 key tile 较短，可以让一个 warp 独立处理一行。例如 key tile 宽度为 128 时，lane $l$ 可以拥有

$$
l,\quad l+32,\quad l+64,\quad l+96
$$

四个 key。lane 先在 register 中对四个值做局部 max/sum，再做一次 warp reduction。这种 ownership 可以完全消除 softmax 内部的跨 warp 通信。

代价是每个 lane 需同时保留多个 score，寄存器压力可能上升。如果 key tile 很大，一个 warp 独占一行还会导致过长的 lane-local 循环。

## Shared-memory bank conflict

Shared memory 的逻辑矩阵是正确的，不代表物理访存高效。在典型的 32-bank、bank width 为 4 Bytes 的模型下，元素大小为 $E$ Bytes、行 stride 为 $S$ 个元素时，坐标 $(r,c)$ 的 bank 可估算为

$$
\operatorname{bank}(r,c)
=\left\lfloor\frac{(rS+c)E}{4}\right\rfloor\bmod 32.
$$

例如 $D=64$、BF16 元素 $E=2$，若 $S=64$，则固定维度 $d$ 下

$$
\operatorname{bank}(k,d)=\left\lfloor d/2\right\rfloor\bmod 32.
$$

一个 warp 的 32 个 lane 各访问一行 K，却全部落到同一 bank。在每行后 padding 两个 BF16 元素，$S=66$：

$$
\operatorname{bank}(k,d)
=\left(k+\left\lfloor d/2\right\rfloor\right)\bmod 32.
$$

现在相邻 key 行会依次落到不同 bank。这两列 padding 不改变逻辑 shape，只改变物理 stride。

该公式是静态诊断模型，实际还要考虑 broadcast、宽访存、swizzle、MMA layout 和具体 GPU 架构。修改 padding 前应先写出“一条 warp 指令中每个 lane 访问哪个坐标”，不能只看数组 shape。

## 每线程处理多个元素

对 PV 的 scalar 实现，可以让每个 thread 同时累加相邻的两个输出维度：

```python
for key in keys:
    p = weight[key]          # 只读一次
    out0 += p * V[key, 2*d]
    out1 += p * V[key, 2*d + 1]
```

这可以：

- 在两个 FMA 之间复用 softmax weight；
- 改善某些 BF16 shared access 的 bank 覆盖；
- 减少循环和地址计算的逻辑迭代数。

但每个 thread 需要更多 accumulator，store ownership 也发生变化。它是局部优化，不会改变 PV 总 FMA（Fused Multiply-Add，融合乘加）数。

类似地，把 QK dot-product 的一条累加链拆成多条 partial 可以增加 ILP（Instruction-Level Parallelism，指令级并行）：

```python
even = 0.0
odd = 0.0
for i in range(D // 2):
    even += Q[2*i]     * K[2*i]
    odd  += Q[2*i + 1] * K[2*i + 1]
score = even + odd
```

只有当单条 FMA 依赖链是实际瓶颈时才会受益；多一个 accumulator 也可能增加 register pressure。

## 把 QK 和 PV 映射到 Tensor Core

QK 和 PV 都是规则矩阵乘：

$$
S_{tile}=Q_{tile}K_{tile}^{\mathsf T},\qquad
N_{tile}=\widetilde P_{tile}V_{tile}.
$$

当 shape、dtype、alignment 和 layout 满足目标 MMA（Matrix Multiply-Accumulate，矩阵乘加）路径时，应优先把大比例的 scalar FMA 改为 Tensor Core GEMM。常见数据精度是：

```text
BF16/FP16 Q,K -> Tensor Core QK -> FP32 score fragment
FP32 stable softmax
BF16/FP16 unnormalized numerator
    -> Tensor Core PV -> FP32 output fragment
FP32 denominator division -> BF16/FP16 output
```

令

$$
\widetilde P_{ij}=e^{S_{ij}-m_i},\qquad
\ell_i=\sum_j\widetilde P_{ij}.
$$

则可以先计算

$$
N=\widetilde PV,
$$

再逐行做

$$
O_{i,:}=N_{i,:}/\ell_i.
$$

因此不必先物化完整的 normalized probability 矩阵，只需保留每行一个 FP32 denominator。把 $\widetilde P$ cast 到低精度会引入额外误差，必须以目标 dtype 的误差预算验证。

## Buffer 生命期复用

QK 完成后 K tile 已经死亡，PV 才需要 V tile。如果两者 shape 和 layout 兼容，可以复用同一块 shared allocation：

```text
phase 1: shared buffer stores K
barrier / QK complete
phase 2: same shared buffer stores V
```

对逻辑上分开的 buffer，编译器也可能根据 liveness 自动 alias。但静态资源估算应同时检查：

- 不假设 alias 的保守总量；
- lowering 后实际分配量；
- 额外 alignment、swizzle 或 pipeline stage 引入的开销。

复用前必须确保所有 consumer 已经完成，并在跨线程覆盖前建立正确的同步边。

## 变长 packed 序列

变长 batch 常把 token 紧凑存储，并用 cumulative sequence lengths 描述边界：

```text
cu_seqlens = [0, L0, L0+L1, ...]
sequence_start  = cu_seqlens[b]
sequence_length = cu_seqlens[b+1] - cu_seqlens[b]
```

对 batch $b$ 的 local token $t$，global token 下标是

$$
\operatorname{token}(b,t)=\operatorname{cu\_seqlens}[b]+t.
$$

实现必须同时保证：

- 无效 Q 行不写输出；
- 无效 K 的 score 等价于 $-\infty$；
- 无效 V 和 PV operand 填 0；
- cooperative load 不读到下一个 packed sample；
- 包围 barrier 的分支必须 CTA-uniform。

特别要区分两种控制流：

```text
if query_tile_is_valid:
    ... CTA barrier ...       # 条件必须整个 CTA 一致

if row_is_valid:
    ... warp reduction ...    # 若条件在 warp 内一致则可以安全
```

部分 thread 绕过 CTA barrier 而其他 thread 到达 barrier，会导致死锁或未定义行为。

## 优化顺序

一条可复用的实验路线是：

| 阶段 | 主要改变 | 想验证的瓶颈 |
| --- | --- | --- |
| 正确基线 | 一个 CTA 负责一行，FP32 累加 | mask、索引和 softmax 语义 |
| Query tiling | 一个 CTA 处理多行 | K/V 重复加载和 block 调度 |
| Warp reduction | shared tree 换成两级 warp 归约 | barrier 和 shared traffic |
| Shared padding | 只改物理 row stride | bank conflict |
| 多元素 ownership | 一个 thread 处理相邻输出 | weight 复用、bank 覆盖、循环开销 |
| Tensor Core QK | 只替换 QK | scalar dot-product 吞吐 |
| Tensor Core PV | 只替换 PV | scalar weighted sum 吞吐 |
| Warp-row softmax | 一个 warp 独占一行 | 跨 warp 同步 |

每次只改一个主要变量。否则即使新版更快，也无法判断是 layout、精度、Tensor Core 还是同步变化起作用。负结果同样有价值：它能排除不是当前主瓶颈的方向。

## 正确性检查

- 与 FP32 reference 比较，并根据 BF16/FP16 输入输出设置合理容差。
- 覆盖长度 `1/31/32/33/63/64/65/...`，特别检查 warp 和 tile 边界。
- 验证每个 `(query, key)` 和 `(query, output_dimension)` 恰好有一个 owner。
- 检查无效 score 的 max 单位元是 $-\infty$，sum 单位元是 0。
- 确认 denominator 在消费前已发布给所有线程。
- 确认输出不越过 packed sequence 边界。
- 对 online softmax，单独验证 max 变化时的 rescale recurrence。

## 性能检查

- Q/K/V 的 global load 是否 coalesced，是否有可删除的重复读取；
- QK/PV 的主要 FMA 是否已映射到 Tensor Core；
- warp 级 shared address 是否存在 bank conflict；
- CTA barrier 的动态次数，而不只是源码中的静态位置数；
- shared-memory 与 register 用量是否限制 resident blocks/warps；
- fragment 到 shared 的物化是否成为额外数据通路；
- 无效 padding 计算占比是否过高；
- 首次 JIT/lowering 时间和 steady-state kernel 时间是否被混在一起。
