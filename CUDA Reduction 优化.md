# CUDA Reduction 优化

Reduction 是典型的 memory-bound 算子：每个 FP32 元素只贡献一次加法，却至少需要从 global memory 读取 4 Bytes。优化目标不是增加无用 FLOP，而是减少访存和同步开销，并制造足够的在飞请求。

## 性能模型

Roofline：

$$P \le \min(P_{\rm peak},\; BW \times I), \qquad I=\frac{\rm FLOP}{\rm Bytes}.$$

Little's Law：

$$\text{在飞字节} \approx \text{带宽} \times \text{延迟}.$$

每个 SM 能制造的在飞字节可近似理解为：

```text
resident warps × 32 lanes × bytes/load/lane × MLP
```

- Occupancy：有多少 resident warps。
- 向量化：一条 load 每个 lane 搬多少 Bytes。
- MLP（Memory-Level Parallelism）：每个 lane/warp 有多少条独立 load 同时在飞。
- 三者乘积足够覆盖 bandwidth-delay product 后，再继续增加通常不会提高带宽。

## 优化演进

| 实现 | 核心设计 | 主要收益 |
| --- | --- | --- |
| Shared-memory 树形归约 | grid-stride loop；线程内 register 累加；shared-memory 树形归约；每 block 一次 atomic | warp 内 global load 连续；控制 block/atomic 数量；建立正确的分层归约基线 |
| Warp-shuffle 分层归约 | 每个 warp 用 shuffle 在 register 中归约；每 warp 只写一个 shared partial；第一个 warp 完成 block 归约 | shared 数组从 `BLOCK` 降到最多 32 个 float；多次 block barrier 降为一次；大幅减少 shared load/store |
| `float4` 向量化读取 | 输入改为 `float4*`，每 lane 每次读取 16 Bytes | load、地址计算和循环次数约降为 1/4；每条 warp load 覆盖连续 512 Bytes；更容易打满带宽 |

### Grid-stride + shared-memory tree

```cpp
constexpr int BLOCK = 256;

__global__ void reduce_shared_tree(
    const float* __restrict__ x,
    float* out,
    int n)
{
    __shared__ float s[BLOCK];

    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = blockIdx.x * blockDim.x + tid;
         i < n;
         i += gridDim.x * blockDim.x) {
        sum += x[i];
    }

    s[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, s[0]);
    }
}
```

约束：启动时应满足 `blockDim.x == BLOCK`，并通常令 `BLOCK` 为 2 的幂。`out` 在 kernel 执行前初始化为 0。

```text
global x
  -> 每线程 grid-stride 读取多个元素
register sum
  -> s[tid]
shared-memory tree reduction
  -> block sum
atomicAdd(out)
```

固定一次迭代时，相邻 lane 的下标差为 1：

```text
i(lane) = common_base + lane
```

所以一个 warp 读取 32 个连续 FP32，形成连续的 128B 区域。线程下一轮虽然跳过 `gridDim.x * blockDim.x`，但 warp 内仍然连续。

`s[tid]` 对 FP32 shared memory 的典型 bank 映射为：

```text
bank = tid mod 32
```

因此 lane 0..31 分别访问 bank 0..31。树形归约使用连续活跃线程：

```text
s[tid] += s[tid + stride]
```

左右两次 shared load 各自在单条 warp 指令内一 lane 一 bank，基本无 bank conflict。后半段的主要问题不是 bank conflict，而是 active lanes 逐渐减少，并且每一级都要 `__syncthreads()`。

### 两级 warp shuffle

```cpp
__inline__ __device__ float warp_reduce_sum(float value)
{
    constexpr unsigned mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

__global__ void reduce_warp_shuffle(
    const float* __restrict__ x,
    float* out,
    int n)
{
    float sum = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        sum += x[i];
    }

    sum = warp_reduce_sum(sum);

    __shared__ float warp_sum[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    if (lane == 0) {
        warp_sum[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        const int warp_count = blockDim.x / 32;
        sum = lane < warp_count ? warp_sum[lane] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            atomicAdd(out, sum);
        }
    }
}
```

约束：这个简化版本要求 `blockDim.x` 为 32 的倍数，因此所有参与第一次 shuffle 的 warp 都是完整 warp。若要支持 partial warp，需要用 `__activemask()` 或 `__ballot_sync()` 构造正确 mask。

```text
每线程 register sum
  -> warp_reduce_sum
每 warp lane 0 写 warp_sum[warp_id]
  -> 一次 __syncthreads()
第一个 warp 读取连续 warp sums
  -> warp_reduce_sum
lane 0 atomicAdd
```

对 256-thread block：

```text
8 个 warp -> 只写 warp_sum[0..7]
```

第一个 warp 的 lane 0..7 连续读取 `warp_sum[0..7]`，对应 shared bank 0..7，无冲突。warp 内数据通过 shuffle 网络在 register 间交换，不再反复经过 shared memory。

### `float4` 向量化读取

```cpp
__global__ void reduce_float4(
    const float4* __restrict__ x4,
    float* out,
    int n4)
{
    float sum = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n4;
         i += gridDim.x * blockDim.x) {
        const float4 value = x4[i];
        sum += value.x + value.y + value.z + value.w;
    }

    sum = warp_reduce_sum(sum);

    __shared__ float warp_sum[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    if (lane == 0) {
        warp_sum[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        const int warp_count = blockDim.x / 32;
        sum = lane < warp_count ? warp_sum[lane] : 0.0f;
        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            atomicAdd(out, sum);
        }
    }
}
```

调用时通常传入：

```cpp
const int n4 = n / 4;
reduce_float4<<<grid, block>>>(
    reinterpret_cast<const float4*>(x), out, n4);
```

单 lane：

```text
float  load = 4 Bytes
float4 load = 16 Bytes
```

单 warp：

```text
32 lanes × 16 Bytes = 连续 512 Bytes
```

硬件仍会把 512B 拆成若干 cache sector / memory transaction；`float4` 的意义是每个 lane 用一条宽指令取回四个连续元素。它改善的是指令宽度和在飞字节，不改变算法的 FLOP/Byte，也不代表使用了 CPU 风格的四路向量 ALU。

注意：

- `float4*` 起始地址必须满足 16B 对齐。
- `n4=n/4` 只覆盖 4 的整数倍部分；`n%4` 的 tail 需要单独处理或安全 padding。
- 标量 warp load 本来就可能完全 coalesced；`float4` 主要减少 load/循环/地址指令并增加每次发射的 Bytes。
- 宽 load 可能增加寄存器压力；最终检查 occupancy、spill 和 SASS，而不是只看源码类型。

## 判断优化是否有效

最终目标是固定正确工作量的 wall-clock，而不是单独追求 GB/s 或 TFLOP/s。重点检查：

1. warp 地址是否连续、对齐、无 overfetch；
2. shared access 是否一 lane 一 bank；
3. 是否有足够的 resident warps、向量宽度和独立 outstanding loads；
4. 展开/向量化是否因寄存器压力降低 occupancy；
5. 输入 global load、block reduction、atomic 各占多少时间；
6. reduction 加法顺序变化会带来 FP32 舍入差异，测试应使用误差容忍。

一句话总结：先用 grid-stride loop 建立连续 global load 和分层归约，再用 warp shuffle 消除大部分 shared traffic 与 barrier，最后用 `float4` 宽 load 降低内存指令开销并把有效带宽推近硬件上限。
