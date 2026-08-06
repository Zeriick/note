# GEMM 优化

GEMM（General Matrix Multiplication）计算：

$$
C=AB,
\qquad
A\in\mathbb{R}^{M\times K},\;
B\in\mathbb{R}^{K\times N},\;
C\in\mathbb{R}^{M\times N}.
$$

单个输出元素为：

$$
C_{ij}=\sum_{k=0}^{K-1}A_{ik}B_{kj}.
$$

本文按下面的优化路径记录：

```text
Naïve GEMM
  -> Global-memory coalescing
  -> Shared-memory tiling
  -> 分析 scheduler 的 Active / Eligible / Issued
  -> Thread / register tiling
  -> Multi-stage pipeline
  -> cp.async / TMA
```

## Naïve GEMM

最直接的 GPU 映射是：一个线程计算一个输出元素 $C_{ij}$。

```cpp
__global__ void gemm_naive(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M,
    int N,
    int K)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= M || j >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
    }

    C[i * N + j] = sum;
}
```

这里假设三个矩阵都是 row-major。线程在 $K$ 方向完成 $K$ 次乘加，最后写出一个结果。

### 相邻线程的数据依赖

四个相邻线程计算一个 $2\times2$ 输出块：

```text
T00 -> C[i    , j    ]    T01 -> C[i    , j + 1]
T10 -> C[i + 1, j    ]    T11 -> C[i + 1, j + 1]
```

这四个输出只依赖：

- $A$ 的第 $i$、$i+1$ 两行；
- $B$ 的第 $j$、$j+1$ 两列。

因此数据存在复用关系：

- `T00` 和 `T01` 使用同一行 $A$；
- `T10` 和 `T11` 使用同一行 $A$；
- `T00` 和 `T10` 使用同一列 $B$；
- `T01` 和 `T11` 使用同一列 $B$。

Naïve kernel 没有使用 shared memory 或寄存器分块来显式组织这种跨线程复用。各线程从代码语义上独立加载自己的 $A$ 行和 $B$ 列；实际硬件可能通过 cache 合并或复用其中一部分访问，但这种复用受访问模式和 cache 容量限制。

### 单线程算术强度

一个线程执行 $K$ 次 FMA。按一次乘法加一次加法计算：

$$
\text{FLOPs}=2K.
$$

对 FP32 而言，忽略 cache 命中并按线程的独立访存计算：

- 读取 $K$ 个 $A$ 元素：$4K$ Bytes；
- 读取 $K$ 个 $B$ 元素：$4K$ Bytes；
- 写入一个 $C$ 元素：$4$ Bytes。

因此：

$$
AI_{\rm naive}
=\frac{2K}{4K+4K+4}
=\frac{2K}{8K+4}
\xrightarrow[K\to\infty]{}
0.25\ \text{FLOPs/Byte}.
$$

这个模型对应 $C=AB$。若计算 $C=\alpha AB+\beta C$ 且需要读取旧的 $C$，还要额外计入一次 $C$ 的读取，但当 $K$ 很大时不改变极限值 $0.25$。

### 为什么这是个问题

Roofline 模型：

$$
P\leq\min\left(P_{\rm peak},\;BW\times AI\right).
$$

Naïve GEMM 的有效算术强度低，每从 global memory 搬运 1 Byte 数据只做约 $0.25$ FLOPs，因此容易落在 Roofline 的 memory-bound 区域。瓶颈不是 FMA 数量不足，而是同一批输入数据没有在片上存储中得到充分复用。

作为对照，若理想地只从主存读取一次 $A$、$B$，并写出一次 $C$，整个 GEMM 的算术强度为：

$$
AI_{\rm ideal}
=\frac{2MNK}{4(MK+KN+MN)}.
$$

当 $M=N=K=n$ 时：

$$
AI_{\rm ideal}=\frac{n}{6}\ \text{FLOPs/Byte}.
$$

理想算术强度随问题规模线性增长，而 Naïve 模型约为常数 $0.25$。两者的差距就是 GEMM 的主要优化空间。

### Naïve 小结

Naïve GEMM 的特点：

- 一个线程只计算一个 $C$ 元素；
- 每个线程沿 $K$ 方向执行 $K$ 次 FMA；
- 没有显式的跨线程协作加载；
- 没有 shared-memory tiling；
- 没有 register tiling；
- 从线程独立访存模型看，算术强度约为 $0.25$ FLOPs/Byte；
- 下一步优化的核心是：把 $A$、$B$ 的 tile 搬到片上，并让一次加载服务于多个输出元素。

## Global-memory coalescing

CUDA 的 block 可以是二维或三维的，但硬件首先把线程坐标线性化：

```cpp
int linear_tid = threadIdx.x
               + blockDim.x * (
                     threadIdx.y
                     + blockDim.y * threadIdx.z);

int warp_id = linear_tid / 32;
int lane_id = linear_tid % 32;
```

因此 `threadIdx.x` 是变化最快的维度。若 `blockDim.x == 32`，一个 warp 恰好对应二维线程块的一行：

```text
ty 固定，tx = lane_id = 0..31
```

C/C++ 的二维矩阵通常为 row-major。对 $X\in\mathbb{R}^{R\times C}$：

$$
X[i,j]\longrightarrow X[i\cdot ld+j],
$$

其中无 padding 时 $ld=C$。最后一维连续：列加一时地址只增加一个元素，行加一时地址增加一整行。

所以基本约定是：

> 让连续的 lane，也就是连续的 `threadIdx.x`，对应 row-major 矩阵中连续的最后一维。

对 FP32，若一个 warp 访问：

```text
lane l -> X[row][col + l]
```

则 32 个 lane 覆盖连续的 $32\times4=128$ Bytes。典型分析中，这对应四个连续的 32-Byte sector。若改成：

```text
lane l -> X[row + l][col]
```

相邻 lane 的地址跨度变成 $ld\times4$ Bytes，可能覆盖很多离散 sector，浪费带宽。实际 transaction 数还取决于架构、对齐和 cache；重点是让一个 warp 的地址覆盖尽可能少的 sector。

在 tiled GEMM 中，应采用：

| 数据操作 | `tx` 对应的维度 |
| --- | --- |
| 加载 $A$ | $K$，即 `A[row][k0 + tx]` |
| 加载 $B$ | $N$，即 `B[k0 + ty][block_col + tx]` |
| 写回 $C$ | $N$，即 `C[row][block_col + tx]` |

还要区分两件事：

- Global-memory coalescing：让 warp 的 global 地址覆盖尽可能少的 sector。
- Shared-memory bank conflict：让 warp 的 shared 地址合理映射到 bank，或使用合法的 broadcast。

必要时可以在 global-to-shared 加载时改变 shared 布局，甚至转置或 padding；计算映射、global 加载映射和 shared 布局不必完全相同。

## Shared-memory tiling

一个 block 负责 $C$ 的一个 $B_M\times B_N$ tile。完整的 $K$ 维被分成长度为 $B_K$ 的小段，每轮协作加载：

$$
A_s\in\mathbb{R}^{B_M\times B_K},
\qquad
B_s\in\mathbb{R}^{B_K\times B_N},
$$

然后做：

$$
C_{\rm tile}\mathrel{+}=A_sB_s.
$$

最简单的版本令 $B_M=B_N=B_K=T$，仍然由一个线程计算一个输出：

```cpp
template<int T>
__global__ void gemm_shared_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* C,
    int M,
    int N,
    int K)
{
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;
    const int col = blockIdx.x * T + tx;

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] =
            row < M && k0 + tx < K
                ? A[row * K + k0 + tx]
                : 0.0f;

        Bs[ty][tx] =
            k0 + ty < K && col < N
                ? B[(k0 + ty) * N + col]
                : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < T; ++k) {
            acc = fmaf(As[ty][k], Bs[k][tx], acc);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
```

### 两次同步分别保护什么

第一次 `__syncthreads()` 保证整个 tile 写完后才能读取，因为一个线程只写 `As[ty][tx]`、`Bs[ty][tx]`，却会读取其他线程写入的整行或整列。

第二次 `__syncthreads()` 保证所有线程读完当前 tile 后，任何线程才能用下一轮数据覆盖 shared memory：

```text
协作写当前 tile
  -> __syncthreads(): 写完后才能读
使用当前 tile 计算
  -> __syncthreads(): 读完后才能覆盖
```

所有线程必须一致地到达 block barrier。因此边界线程不能在循环前提前 `return`；它们应加载 0、参与同步，最后再有条件地写回。

### 数据复用

一个 $A_s[i,k]$ 会被同一输出行中的 $B_N$ 个输出使用，一个 $B_s[k,j]$ 会被同一输出列中的 $B_M$ 个输出使用。global memory 中的输入只加载到 shared memory 一次，随后在 block 内反复读取。

当 $T=32$ 时：

- 每个 $A$ 元素最多服务 32 个输出；
- 每个 $B$ 元素最多服务 32 个输出；
- shared memory 用量为 $2\times32\times32\times4=8$ KiB；
- block 有 $32\times32=1024$ 个线程。

### Global-memory 算术强度

一个 $B_M\times B_N\times B_K$ 分块完成：

$$
2B_MB_NB_K\ \text{FLOPs}.
$$

每轮从 global memory 读取：

$$
4(B_MB_K+B_KB_N)\ \text{Bytes}.
$$

忽略最终一次 $C$ 写回，或认为它在较大的 $K$ 上被摊薄：

$$
AI_{\rm tile}
\approx
\frac{2B_MB_NB_K}{4(B_MB_K+B_KB_N)}
=\frac{B_MB_N}{2(B_M+B_N)}.
$$

若 $B_M=B_N=T$：

$$
AI_{\rm tile}\approx\frac{T}{4}.
$$

所以 $T=32$ 时：

$$
AI_{\rm tile}\approx8\ \text{FLOPs/Byte}.
$$

这相对 Naïve 的 $0.25$ 提高约 32 倍。

课件中若写成：

$$
\frac{2B_MB_NB_K}
{4(B_MB_N+B_MB_K+B_KB_N)}
=\frac{B_MB_N}{2(B_M+B_N)},
$$

这个等号并不严格，因为左边计入了 $C$ tile，右边却省略了它。正确理解是：完整 kernel 让 `acc` 跨所有 $K$-tile 留在寄存器中，只在最后写一次 $C$；当 $K$ 较大时，这次写回被摊薄，才得到右侧的渐近式。

### Shared 访问模式

当 `blockDim.x == 32` 时，在内层循环中：

```cpp
As[ty][k]
```

是同一 warp 读取同一地址，可由 shared broadcast 处理；

```cpp
Bs[k][tx]
```

是同一 warp 读取连续的 32 个 FP32，通常落到不同 bank。这个教学版本的 shared 访问比较规整。

### 当前局限

Shared tiling 只说明 global 数据复用提高了，不保证计算流水线持续繁忙。这个版本仍有：

- 一个线程只计算一个输出，寄存器复用不足；
- 只有一个 `acc`，形成长循环依赖链；
- 每轮两个 block-wide barrier；
- global-to-shared 加载与计算串行；
- $32\times32=1024$ 线程达到常见的 block 线程上限；
- 每次 FMA 仍要从 shared memory 取一个 $A$ 和一个 $B$。

## Scheduler：Active、Eligible、Issued

Nsight Compute 中三类 warp 的关系是：

```text
Active / resident warp
  -> 下一条指令已满足依赖
Eligible warp
  -> 被 scheduler 选中
Issued warp
```

即：

$$
\text{Issued}\subseteq\text{Eligible}\subseteq\text{Active}.
$$

- **Active**：warp 已驻留在 SM 上且尚未结束。等待 memory、scoreboard 或 barrier 时仍算 active。
- **Eligible**：下一条指令的操作数和依赖已就绪，本周期可以被发射。
- **Issued**：scheduler 从 eligible 集合中实际选择并发射的 warp。

课件示例中每个 scheduler 约有：

```text
Active Warps Per Scheduler   = 15.94
Eligible Warps Per Scheduler = 3.09
Issued Warp Per Scheduler    = 0.40
No Eligible                  = 59.57%
```

这表示 resident warps 几乎已经很多，但约 60% 的周期没有任何 warp 能发出下一条指令。高 occupancy 并没有转化为高 issue utilization：

$$
\text{high occupancy}\not\Rightarrow\text{high throughput}.
$$

大量 active warp 可能一起处在：

- global-memory long scoreboard；
- shared-memory short scoreboard；
- `__syncthreads()` barrier；
- 单一 `acc` 的 FMA 依赖链；
- 同一阶段的成批等待。

若 eligible warp 呈现“很多个同时就绪，然后很多周期全为 0”，scheduler 无法用繁忙周期补回空闲 issue slot。需要结合 Nsight 的 Warp State Statistics、Source Counters 和源代码关联继续定位：

| Stall | 常见含义 |
| --- | --- |
| Long Scoreboard | 等待 L2/DRAM 等长延迟数据 |
| Short Scoreboard | 等待 shared 等较短内存依赖 |
| Barrier | 等待 block 同步 |
| Wait / dependency | 等待前序计算结果 |
| MIO Throttle | shared / memory instruction pipeline 压力 |
| Not Selected | 已 eligible，但 scheduler 选择了其他 warp |

Shared-memory tiling 解决的是 global bytes/FLOP；下一步要用多个独立累加器提高 ILP，并减少 shared load/FMA。

## Thread / register tiling

令一个 block 仍计算 $B_M\times B_N=32\times32$ 输出，但每个线程计算 $T_M\times T_N=2\times2$：

```cpp
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 8;
constexpr int TM = 2;
constexpr int TN = 2;

// block = dim3(BN / TN, BM / TM) = dim3(16, 16)
int tc = threadIdx.x;
int tr = threadIdx.y;

int row0 = blockIdx.y * BM + tr * TM;
int col0 = blockIdx.x * BN + tc * TN;

float acc[TM][TN] = {};
```

线程 `(tr, tc)` 负责：

$$
\begin{bmatrix}
C[row_0,col_0] & C[row_0,col_0+1]\\
C[row_0+1,col_0] & C[row_0+1,col_0+1]
\end{bmatrix}.
$$

因此：

```text
CTA tile:    32 × 32
Thread tile:  2 × 2
Thread grid: 16 × 16 = 256 threads = 8 warps
```

同一个输出 tile 从 1024 个线程降为 256 个线程。每线程工作增加，但 block 调度更灵活，也不再触碰 1024-thread 上限。

### 每个 k 做一个小外积

对一个固定的 $k$，线程从 shared memory 读取：

$$
a_0,a_1\quad\text{和}\quad b_0,b_1,
$$

然后做：

$$
\begin{bmatrix}a_0\\a_1\end{bmatrix}
\begin{bmatrix}b_0&b_1\end{bmatrix}
=
\begin{bmatrix}
a_0b_0&a_0b_1\\
a_1b_0&a_1b_1
\end{bmatrix}.
$$

对应代码：

```cpp
#pragma unroll
for (int k = 0; k < BK; ++k) {
    float a0 = As[tr * TM + 0][k];
    float a1 = As[tr * TM + 1][k];
    float b0 = Bs[k][tc * TN + 0];
    float b1 = Bs[k][tc * TN + 1];

    acc[0][0] = fmaf(a0, b0, acc[0][0]);
    acc[0][1] = fmaf(a0, b1, acc[0][1]);
    acc[1][0] = fmaf(a1, b0, acc[1][0]);
    acc[1][1] = fmaf(a1, b1, acc[1][1]);
}
```

即：

```text
2 个 A 值 + 2 个 B 值 -> 4 次 FMA
```

旧的 $1\times1$ 映射计算 4 个输出需要 4 个线程、共 8 个 shared 元素读取；新的 $2\times2$ 映射只需要 4 个 shared 元素读取，shared load 数减半。

一般地，$T_M\times T_N$ thread tile 每个 $k$：

$$
T_M+T_N\ \text{个 shared 元素读取}
\longrightarrow
T_MT_N\ \text{次 FMA}.
$$

相对于 shared-memory 字节数的简化算术强度为：

$$
AI_{\rm shared}
=\frac{2T_MT_N}{4(T_M+T_N)}
=\frac{T_MT_N}{2(T_M+T_N)}.
$$

所以：

$$
AI_{\rm shared}(1\times1)=0.25,
\qquad
AI_{\rm shared}(2\times2)=0.5.
$$

### 多个累加器提高 ILP

一个输出只有一条依赖链：

```text
acc -> FMA -> acc -> FMA -> acc
```

$2\times2$ tile 有四条独立累加链：

```text
acc00 -> acc00 -> ...
acc01 -> acc01 -> ...
acc10 -> acc10 -> ...
acc11 -> acc11 -> ...
```

编译器可以交错安排四条 FMA 链，在等待一个累加器结果时发射另一个累加器的 FMA，从而提高 eligible 指令的连续性。

`float acc[TM][TN]` 只有在索引可静态展开、寄存器压力合适时才会被标量化到寄存器。应检查编译器的 register usage 和 local load/store，避免 spilling。

### 协作加载与计算映射应解耦

当前：

$$
B_MB_K=32\times8=256,
\qquad
B_KB_N=8\times32=256,
$$

正好等于 block 的 256 个线程。因此每轮可让每线程各加载一个 $A$ 元素和一个 $B$ 元素：

```cpp
int tid = threadIdx.y * blockDim.x + threadIdx.x;

int a_row = tid / BK;
int a_col = tid % BK;

int b_row = tid / BN;
int b_col = tid % BN;
```

其中 $B$ 的映射让一个 warp 读取一行 32 个连续元素，天然 coalesced；$A$ 的 `32×8` 朴素映射会让一个 warp 覆盖 4 行、每行 8 个连续元素，不一定是最佳 coalescing。高性能实现通常单独设计加载映射，可能使用向量化 load、转置或不同的 lane layout。

同理，`blockDim.x=16` 时一个 warp 跨两个 `ty` 行；每线程负责两个连续输出列也不代表每条标量 store 指令天然连续。可能需要 `float2`/`float4` 写回或更合适的 warp tile 映射。

### Thread tile 的代价

增大 $T_M,T_N$ 会提高复用和 ILP，但每线程至少需要 $T_MT_N$ 个累加器，寄存器需求大致随：

$$
T_MT_N+T_M+T_N
$$

增长。过大的 thread tile 可能降低 occupancy，甚至 spilling 到 local memory。因此优化目标是在 ILP、register pressure、occupancy、load/store 布局之间平衡，而不是让 tile 无限增大。

## Multi-stage pipeline

即使有 register tiling，基本循环仍然是：

```text
load tile 0 -> compute tile 0
load tile 1 -> compute tile 1
load tile 2 -> compute tile 2
```

global-to-shared copy 和 compute 串行。单个 shared buffer 不能重叠二者，因为加载下一 tile 会覆盖仍在使用的当前 tile。

### 双缓冲

使用两套 shared buffer：

```cpp
__shared__ float As[2][BM][BK];
__shared__ float Bs[2][BK][BN];
```

形成：

```text
预热：copy tile 0 -> B0，等待 B0 ready

稳态：compute tile 0 on B0 || async copy tile 1 -> B1
      compute tile 1 on B1 || async copy tile 2 -> B0
      compute tile 2 on B0 || async copy tile 3 -> B1

排空：等待并计算最后一个 tile
```

不重叠时每 tile 时间近似：

$$
T_{\rm tile}=T_{\rm copy}+T_{\rm compute}.
$$

理想重叠后接近：

$$
T_{\rm tile}\approx\max(T_{\rm copy},T_{\rm compute}).
$$

### 多 stage 与依赖距离

若有 $S$ 个 stage，异步请求可以比消费位置提前更多 tile 发出。可隐藏的时间粗略为：

$$
(S-1)T_{\rm compute\ per\ tile}.
$$

当它足以覆盖 copy latency 时，等待可以大幅减少。但每个 stage 都要保存一份 $A/B$ tile：

$$
S_{\rm shared}
=4S(B_MB_K+B_KB_N)\ \text{Bytes}
$$

对 $B_M=B_N=32,B_K=8$，每 stage 为 2 KiB，2/3/4 stages 分别需要 4/6/8 KiB 的 tile 空间。更深 pipeline 会增加 shared-memory、barrier、地址计算和状态管理成本，可能降低 occupancy；stage 不是越多越好。

### Buffer 的正确状态

每个 stage 必须满足：

```text
EMPTY
  -> async copy
FILLING
  -> copy complete
READY
  -> compute / consume
IN USE
  -> all consumers finished
EMPTY
```

必须同时保护：

1. copy 完成后，消费者才能读取；
2. 所有消费者读完后，生产者才能覆盖。

异步搬运不是取消同步，而是把 wait 尽量推迟到真正消费数据之前。

## `cp.async` 与 TMA

### `cp.async`

传统 global-to-shared 代码概念上经过：

```text
global memory -> register -> shared memory
```

`cp.async` 提供异步 global-to-shared copy：warp 发出若干 copy，将它们提交为 group，继续执行与该 tile 无关的计算，在消费前再等待对应 group 完成。

```text
issue async copy
  -> commit group
  -> execute independent compute
  -> wait group
  -> consume shared tile
```

它有助于：

- 把下一 tile 的数据搬运与当前 tile 的计算重叠；
- 减少显式 load-to-register、store-to-shared 的依赖链；
- 增加在飞请求；
- 构建双缓冲或多-stage software pipeline。

但它仍消耗 global/L2 带宽、shared 写入带宽和指令发射资源，也仍需要正确的 commit、wait 与协作同步。

### Tensor Memory Accelerator

TMA 面向大块、多维 tensor tile 的异步搬运。相比每个线程描述自己的小片段，TMA 使用 tensor descriptor 描述：

- global 基地址；
- 维度与 stride；
- tile 形状；
- 元素类型；
- 边界和布局信息。

随后由少量线程发起整个 tile 的异步传输，硬件负责大部分多维地址生成和数据移动。可以粗略对比：

| 机制 | 搬运粒度 | 描述方式 |
| --- | --- | --- |
| 普通 load + store | 单元素或向量 | 每线程显式执行 |
| `cp.async` | 每线程的小片段 | warp 内线程协作 |
| TMA | 多维 tensor tile | 少量线程提交 descriptor |

TMA 很适合 producer-consumer 或 warp specialization：producer warp/group 负责推进异步 copy，consumer warp/group 负责读取 shared tile 并执行 MMA/FMA。双方通过 shared-memory barrier 管理 buffer 的 ready/empty 状态。

TMA 也不是无限带宽的独立 DMA：它仍受 L2/DRAM、shared-memory 容量和同步正确性约束。

## 优化层次总结

| 优化 | 主要问题 | 主要收益 |
| --- | --- | --- |
| Naïve | 每线程重复从 global 取完整行/列 | 正确性基线 |
| Global coalescing | warp 地址散落在太多 sector | 提高有效 global 带宽 |
| Shared-memory tiling | 跨线程数据复用依赖 cache | 降低 global bytes/FLOP |
| Register tiling | shared load 多、单累加器依赖链 | 提高 shared 复用和 ILP |
| Multi-stage pipeline | copy 与 compute 串行 | 隐藏 global-to-shared latency |
| `cp.async` / TMA | 搬运依赖链、地址与发射开销 | 异步、流水化 tensor copy |

数据流可概括为：

```text
Global memory
  -> coalesced async copy / TMA
Shared-memory multi-stage buffers
  -> thread / register tiling
Registers with multiple accumulators
  -> FMA or Tensor Core MMA
C epilogue and coalesced store
```

Profile 时应分别回答：

1. global load/store 是否合并，DRAM/L2 带宽是否有效；
2. shared tiling 是否真的减少 global 流量；
3. shared access 是否有 bank conflict 或 pipeline 压力；
4. active warp 中有多少能成为 eligible；
5. stall 是 memory、barrier 还是依赖链造成；
6. register tiling 是否提高 ILP，是否发生 spilling；
7. async pipeline 是否真正重叠 copy 与 compute；
8. 更多 stages 是否因 shared/register 压力反而降低并发度。
