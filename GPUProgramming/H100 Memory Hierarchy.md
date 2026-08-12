# NVIDIA H100 存储层次与跨 SM 通信

本文整理 NVIDIA H100（GH100/Hopper）的计算与存储层次、跨 SM 通信、L1/Shared Memory 统一存储、分布式 L2、HBM 和芯片外互联。文中的带宽与延迟数字主要用于建立数量级直觉；它们会随 H100 型号、时钟、数据布局、缓存命中、L2 分区距离、并发度和测量方法变化，并不是每条指令的固定硬件承诺。

整体数据路径可以概括为：

```text
Thread
  -> Register
  -> 当前 SM 的 Shared Memory / L1
  -> 同一 Thread Block Cluster 内的 DSMEM
  -> 全 GPU 共享的分布式 L2
  -> HBM / Global Memory
  -> PCIe（CPU/DPU）或 NVLink（其他 GPU/Grace CPU）
```

越靠近计算单元，容量通常越小、延迟越低、作用范围越局部；越往外容量越大，但延迟、通信和同步成本越高。

## H100 产品与芯片组织

不同 H100 产品的主要差异为：

| 项目 | 完整 GH100 die | H100 SXM5 | H100 PCIe |
| --- | ---: | ---: | ---: |
| SM 数量 | 最多 144 | 132 | 114 |
| L2 容量 | 60 MiB | 约 50 MiB | 约 50 MiB |
| 功耗量级 | — | 约 700 W | 约 350 W |
| HBM 带宽量级 | — | 约 3 TB/s | 约 2 TB/s |

完整 GH100 芯片由 8 个 GPC（Graphics Processing Cluster）组成，每个 GPC 最多 18 个 SM：

```text
8 GPC × 18 SM/GPC = 144 SM
```

144 是完整 die 的设计上限。具体产品会因为良率、功耗和产品分档关闭部分 SM，因此 SXM5 和 PCIe 型号启用的 SM 数不同。

GPC 是物理硬件组织，不等于 CUDA 的 Thread Block Cluster。一个 GPC 有 18 个 SM，不表示一个 cluster 默认包含 18 个 SM 或 18 个 block。

## SM 与 quadrant

SM（Streaming Multiprocessor）是执行 CUDA thread block 的主要计算单元。每个 SM 可以理解为包含 4 个相似的 quadrant，每个 quadrant 大致包含：

- Warp scheduler：选择准备好的 warp 指令并发射；
- Tensor Core：执行矩阵乘加等张量运算；
- Other compute，也就是常说的 CUDA Core 及其他执行单元：执行普通 FP、INT 等运算；
- LD/ST：处理 load/store；
- 一份 `16K × 32-bit` register file。

四个 quadrant 合起来，一个 SM 有：

$$
4\times 16\text{K}\times 32\text{ bit}
=64\text{K}\times 32\text{ bit}
=256\text{ KiB}.
$$

也就是每个 SM 有 65,536 个 32-bit 寄存器。Register Memory 有时简称 RMEM，不要与 CUDA 的 local memory 混为一谈。

GPU 通常以 warp 为调度单位：

```text
1 warp = 32 threads
```

普通算术指令的依赖延迟可处于约 `4～15 cycles` 的量级。具体数字取决于指令类型、流水线、数据依赖和调度；吞吐量与单条依赖链的延迟也不是同一个指标。

### Tensor Core 与 WGMMA

WGMMA 是 Hopper 面向 warpgroup 的矩阵乘加指令族，可用于充分利用 Tensor Core。它让多个 warp 协作发起较大的矩阵乘加，是高性能 Hopper GEMM、FlashAttention 等 kernel 的核心构件之一。

Tensor Core 和 TMA 不应混淆：

- Tensor Core / WGMMA 负责计算；
- TMA 负责异步搬运 tensor tile；
- 高性能 kernel 常让二者形成流水线。

## Register：最快的线程私有状态

寄存器保存线程的临时变量、地址、循环状态和累加器。它们最靠近执行单元，通常具有最低访问成本，但容量有限。

寄存器使用量会影响 occupancy：

```text
每线程寄存器增加
  -> 每 block 消耗的寄存器增加
  -> 每个 SM 可同时驻留的 block/warp 可能减少
  -> 隐藏延迟的能力可能下降
```

如果编译器无法把线程私有数据全部留在寄存器中，会发生 register spilling。严格地说，spill 的目标是线程私有的 local-memory 地址空间，而不是直接把 L1 当作额外寄存器文件：

```text
Register spill
  -> local-memory load/store
  -> 可能命中当前 SM 的 L1
  -> L1 miss 后访问 L2
  -> L2 miss 后访问 HBM
```

L1 可以缓存 spill 产生的 local-memory 访问，但 L1 本身不能被编译器直接扩展成寄存器文件。即使命中 L1，spill 也会增加指令数、地址计算、缓存压力和依赖延迟。

## Unified Shared Memory / L1 / Texture Cache

H100 每个 SM 有 256 KiB 的统一片上存储，供 Shared Memory、L1 和 Texture Cache 使用。可以概括为：

```text
┌──────────────────────────────────┐
│ Shared Memory | software dial | L1│
└──────────────────────────────────┘
                 256 KiB
```

这里的 `software dial` 表示 shared-memory carveout 可以由软件表达偏好。H100 支持的 Shared Memory 档位包括：

```text
0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KiB/SM
```

可通过 `cudaFuncSetAttribute()` 和 `cudaFuncAttributePreferredSharedMemoryCarveout` 表达 carveout 偏好。它不是从 0 到 256 KiB 任意逐字节划分，也不是绝对命令；运行时会结合 kernel 需求和硬件支持的档位选择配置。

### 为什么 256 KiB 中最多只有 228 KiB Shared Memory

H100 每 SM 的 Shared Memory 容量上限是 228 KiB。CUDA 为每个 thread block 保留 1 KiB，因此单个 block 最多能寻址约 227 KiB：

```text
228 KiB/SM - 1 KiB system reservation/block
  -> 单 block 最多约 227 KiB
```

多 block 驻留时可以用下面的简化关系建立直觉：

```text
有效空间约为 228 KiB - resident_blocks × 1 KiB
```

这只是突出每 block 都有保留开销。真实 occupancy 还同时受寄存器、线程数、warp 数、barrier、cluster size 和其他硬件资源限制。静态 Shared Memory 仍有传统的 48 KiB 限制；使用更大的动态 Shared Memory 需要显式 opt-in。

### Shared Memory 与 L1 的语义差异

| 属性 | Shared Memory | L1 Cache |
| --- | --- | --- |
| 作用范围 | 当前 thread block；cluster 模式下可被同 cluster 远程访问 | 当前 SM 私有 |
| 管理方式 | 程序显式放置和寻址 | 硬件自动缓存 |
| 数据复用 | 程序明确组织 tile | 依赖地址局部性和替换策略 |
| 同步 | 跨线程复用时通常需要 barrier | cache 机制本身自动处理 |
| 常见风险 | bank conflict、同步错误、容量限制 | miss、thrashing、不可控替换 |

Shared Memory 显式寻址，不需要 cache tag comparison 和 hit/miss 判断，因此行为通常比 L1 更可预测。不过，SMEM 仍可能因为 bank conflict 而序列化；糟糕的 SMEM 布局不一定比良好的 L1 命中快。

### SMEM/L1 带宽和延迟

一种常见的近似量级是：

```text
Bandwidth ≈ 128 B/cycle
Latency   ≈ 20～30 cycles（约十几纳秒）
```

这些是建立直觉的近似值。实际结果取决于访问类型、bank conflict、L1 命中、指令依赖、时钟和争用。需要区分：

- Latency：一个依赖访问多久后可以使用结果；
- Throughput/Bandwidth：大量 warp 连续发出访问时，每周期最多服务多少数据。

### L1 cache line 与 sector

H100 的 L1 可按下面的结构理解：

```text
L1 cache line = 128 B
              = 4 × 32 B sector
```

一个 warp 的 32 个线程各读取一个连续 FP32 时：

$$
32\text{ threads}\times 4\text{ B}=128\text{ B}.
$$

这正好覆盖一个连续 128 B 区域。Sector 化允许硬件以 32 B 粒度追踪和获取 cache line 的部分数据；如果线程访问稀疏地址，则可能触碰许多不同 sector，产生 over-fetch 和更多事务。

## TMA：Tensor Memory Accelerator

TMA 是 Hopper 的异步张量搬运单元，不是 Tensor Core。它主要负责：

```text
GMEM <-> SMEM 的 load/store transfer
支持多维地址生成和 swizzling
```

TMA 可以搬运 1D 到 5D tensor，主要路径包括：

```text
Global Memory -> Shared Memory
Shared Memory -> Global Memory
同一 cluster 中，一个 SM 的 Shared Memory -> 另一个 SM 的 Shared Memory
```

它的优点包括：

- 一个线程即可发起较大的异步搬运；
- 地址生成由专用硬件处理；
- 搬运时不需要让整个 warp 执行大量普通 load/store；
- 减少搬运过程中使用寄存器作为中转；
- 支持 swizzling，以改善 Shared Memory 的 bank 映射；
- 可与 barrier/mbarrier 和计算流水线配合；
- 适合 warp specialization：一部分 warp 管数据搬运，另一部分 warp 做计算。

典型流水线：

```text
TMA 搬运 tile N+1 到 SMEM
             与此同时
WGMMA/Tensor Core 计算 tile N
```

小请求使用 TMA 可能比普通 async copy 延迟更高，因为描述符解析和地址生成有固定开销。TMA 更适合较大、多维、可流水化的数据块，而不是替代所有细粒度 load。

## Thread Block Cluster 与 `across cluster`

Hopper 在 grid 和 thread block 之间加入了可选的 Thread Block Cluster：

```text
Grid
  -> Thread Block Cluster
       -> Thread Block
            -> Warp
                 -> Thread
```

同一 cluster 的 blocks 保证并发驻留，从而可以执行 cluster 级同步，并访问彼此的 Shared Memory。

TMA 可以在同一 cluster 的不同 SMEM 区域之间异步搬运，也可用于把一份 global-memory tile multicast 到 cluster 中多个 block 的 Shared Memory。这类路径可以统称为 across-cluster transfer。

它不表示：

- 任意两个 SM 都能互访；
- 整个 GPU 的 SMEM 自动组成一块内存；
- 一个 SM 可以直接访问另一个 SM 的 L1；
- 不需要 cluster 生命周期和同步规则。

H100 的可移植最大 cluster size 是 8 blocks；H100 还可显式选择非可移植的 16-block cluster。Cluster size 增大可能降低同时活跃的 clusters/blocks 数量，因此应使用 cluster occupancy API 评估。

## Distributed Shared Memory（DSMEM）

Cluster 中每个 block 仍有自己的本地 Shared Memory，但其他同 cluster block 可以读取、写入或执行原子操作。所有这些分散的 SMEM 区域构成逻辑上的 Distributed Shared Memory：

```text
Cluster DSMEM
  = Block 0 的本地 SMEM
  + Block 1 的本地 SMEM
  + ...
```

DSMEM 不是物理上集中、等延迟的一整块 SRAM：

```text
SM A / Block A                      SM B / Block B
┌──────────────┐  cluster fabric  ┌──────────────┐
│ local SMEM A │ <--------------> │ local SMEM B │
└──────────────┘                   └──────────────┘
```

### DSMEM 与 TMA across cluster 的区别

| 机制 | 含义 | 典型粒度 | 典型用途 |
| --- | --- | --- | --- |
| DSMEM load/store/atomic | 线程远程寻址另一个 block 的 SMEM | 细粒度 | 计数器、查表、跨 block 交换、atomic |
| TMA cluster copy | 专用硬件异步批量搬运到远程 SMEM | 大块、多维 | tensor tile 搬运与流水化 |
| TMA multicast | 一份源 tile 写入 cluster 中多个 SMEM | 大块 | 多个 block 复用相同输入 |

DSMEM 相比本地 Shared Memory 带宽更低、延迟更高，但通常仍比绕到 L2 更局部。这是因为远程 SMEM 需要经过 cluster fabric，而本地 SMEM 不需要；L2 又位于更远的全芯片存储层级。

这仍不是绝对的性能保证。DSMEM 应尽量采用合并访问、对齐到 32 B segment，并避免非单位 stride、热点和严重原子竞争。

局部性顺序可概括为：

```text
本地 SMEM
  -> 同 cluster 的远程 SMEM / DSMEM
  -> L2
  -> HBM
```

## 全 GPU 共享的分布式 L2

从 CUDA 编程语义看，L2 是所有 SM 共享的缓存；从物理实现看，它由分布在芯片不同位置的多个 partition/slice 组成，并通过片上互联和 partition crossbar 连接：

```text
多个 SM/GPC
   -> on-chip network / partition crossbar
      -> L2 partition 0
      -> L2 partition 1
      -> ...
      -> 对应的 memory controller / HBM partition
```

某个 global address 会根据硬件地址映射落到特定 L2/memory partition。因而同一个请求 SM 到不同 L2 partition 的物理距离可能不同。

### Near partition 与 far partition

相对请求 SM/GPC 而言，L2 slice 可以区分为物理路径较近的 `near partition` 和较远的 `far partition`：

- Near partition：片上网络跳数更少，可能获得更低延迟和更高带宽；
- Far partition：需要经过更多互联/crossbar，可能受到更多争用，带宽显著降低；
- 所有 partition 在软件上仍形成统一 L2，不是显式暴露的 CUDA NUMA node。

普通 CUDA 程序通常不能直接指定“把这个地址分配到离 SM 7 最近的 L2 partition”。程序只能通过数据地址、对齐、stride、padding、block 到数据的映射和访问分布间接影响 partition 使用。

特定 microbenchmark 可能观察到以下 L2 hit 量级：

```text
Near-partition aggregate read bandwidth ≈ 12～14 TB/s
Far partition significantly slower
Latency ≈ 200 cycles
```

`12～14 TB/s` 应视为特定条件或 microbenchmark 下的片上聚合读取量级，不是每个 kernel 的保证值。它和 HBM 的 `2～3 TB/s` 不矛盾：前者是 L2 命中后的片上传输，后者是 L2 miss 后访问外部 HBM 的带宽。

### Partition crossbar 与 partition camping

Partition crossbar 让任意 SM 能通过统一 global address 访问任意 L2 partition，同时也意味着远端请求会消耗片上网络资源。

如果某种地址/stride 模式让大量请求集中到少数 partition，会出现 partition camping：

```text
许多 SM -> 同一个 L2/HBM partition -> 拥堵
其他 partition                         -> 空闲
```

可通过 padding、改变 leading dimension、重新布局 tile 或改变 block-to-data 映射，使请求更均匀地分布。目标不是把所有请求都塞进一个“near” partition，而是在局部性和分区负载均衡之间取得平衡。

### L2 cache line 与 fetch granularity

L2 可按下面的 cache-line 结构理解：

```text
L2 cache line = 128 B
              = 4 × 32 B sector
```

和 L1 一样，L2 使用 128 B line、32 B sector 的理解模型。

可以用 `cudaDeviceSetLimit` 为最大 L2 fetch granularity 提供 32、64 或 128 B 的性能提示。对应的限制项是：

```cpp
cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, bytes);
```

它是性能提示，不会改变物理 cache-line 大小，硬件可能钳制或忽略：

- 较大 fetch：空间局部性好时可减少后续请求，但稀疏访问可能 over-fetch；
- 较小 fetch：稀疏访问浪费较少，但连续访问可能需要更多请求。

### L2 容量

完整 die 与实际产品启用的容量不同：

```text
60 MiB on the full die
50 MiB on H100 SXM/PCIe products
```

这里同样区分了完整 GH100 设计与实际 H100 产品启用的容量。

### L2 residency control / persisting cache

CUDA 可以设置一部分 L2 作为 persisting-data set-aside，并为某个 stream 定义 access-policy window，把一段 global-memory 地址标记为更倾向保留：

```text
GMEM 中频繁复用的地址窗口
  -> persisting access hint
  -> 在 L2 中具有更高保留倾向
```

它适合反复使用的小权重、lookup table 和跨 kernel 复用的数据。它不是把 cache line 永久锁住，也不是把 L2 变成软件管理的 Shared Memory；硬件仍可能限制、替换或忽略提示。

### L2 的压缩与 global atomics

L2/内存分区附近还包含数据压缩电路，并参与处理 global atomics：

- Inline compression 可减少某些可压缩数据占用的 L2/HBM 传输带宽；收益依赖数据模式；
- Global atomic 请求会被送往目标地址所属的 L2/memory partition；大量线程竞争同一地址仍会严重序列化。

高性能归约通常逐层聚合：

```text
线程寄存器累加
  -> warp 内归约
  -> block 内 SMEM 归约
  -> 可选的 cluster DSMEM 归约
  -> 少量 global atomic
```

### 内存流量与 SM 功耗预算

在总功耗上限固定时，更好的 tiling、复用和更少的 L2/HBM 流量可能降低内存系统的动态功耗，使动态频率/功耗管理为 SM 计算留下更多预算。这是一种 workload 与功耗管理之间的间接效果，不是普通 CUDA memory API；不存在通用的 `movePowerFromL2ToSM()` 编程接口。

## HBM / Global Memory / Device Memory

板载 HBM 在 CUDA 语境中常称为：

```text
VRAM / device memory / GMEM
80 GB（常见 H100 配置）
```

CUDA 中的 Global Memory/Device Memory 通常由 HBM 提供。H100 HBM 路径的近似量级为：

```text
H100 SXM  ≈ 3 TB/s
H100 PCIe ≈ 2 TB/s
Latency   ≈ 200～1000 cycles（约 100～500 ns）
```

这里 PCIe 型号的 `2 TB/s` 是该卡本地 HBM 的带宽，不是 PCIe 链路带宽。

HBM 单次访问延迟很高，但聚合带宽很大。GPU 通过大量 resident warps 和 memory-level parallelism 隐藏延迟，因此高性能访问需要：

- Warp 内地址连续并合并访问；
- 足够多的并发请求；
- 利用 L2/L1 命中；
- 把反复使用的数据分块搬到 SMEM/寄存器；
- 让数据搬运与计算重叠。

### 32/64/128 B memory transaction granularity

内存系统会以 `32/64/128 B` 等粒度组织请求。Warp 的线程访问会被硬件合并为若干内存事务，而不是每条 C++ 的 4 B load 都独立发出事务。

一个 warp 连续读取 FP32：

```text
32 threads × 4 B = 128 B contiguous range
```

是理想访问模式之一。若 32 个线程访问相距很远的地址，就可能触碰许多 32 B sector，实际搬运量远大于有效数据量。

### 80 GB 与 141 GB

常见 H100 配置提供 80 GB HBM，相关产品 H200 可提供 141 GB。容量、HBM 类型和带宽必须按具体 SKU 确认，不能只用“Hopper”或“H100/H200”名称推断。

## Constant Memory

Constant Memory 是约 64 KiB 的 CUDA 只读地址空间，并有相应缓存/广播路径：

- Host 写入，kernel 通常只读；
- 一个 warp 的线程读取同一个地址时适合广播；
- 同一 warp 读取许多不同 constant 地址时可能序列化；
- 适合小型 lookup table、共同参数、shape/stride 等常量。

Constant Memory 描述的是 CUDA 地址空间和访问路径，不应理解为 HBM 旁边物理独立的一块 64 KiB DRAM。

## Local Memory

Local Memory 的名字容易误导：它是每线程私有的逻辑地址空间，容量上限可达约 `512 KiB/thread`，但后备存储通常在 device memory/HBM，并可以被 L1/L2 缓存。

它常用于：

- Register spill；
- 无法放入寄存器的线程私有数组；
- 编译器难以静态索引、因此不能寄存器化的数据。

这里的 `512 KiB/thread` 是地址空间/实现限制量级，不表示 GPU 真的为每个线程预留 512 KiB 的片上 SRAM。

## PCIe Gen 5：GPU 与 CPU/DPU

PCIe Gen 5 x16 的理论带宽量级是：

```text
单方向约 64 GB/s
同时双向合计约 128 GB/s
```

实际有效带宽会因为协议、事务、拓扑和软件栈低于理论值。和本地 HBM 相比：

```text
本地 HBM：约 2～3 TB/s
PCIe：    约几十 GB/s/方向
```

因此 CPU-GPU 传输通常需要：

- 减少 Host/Device 往返；
- 尽量让数据长期驻留 GPU；
- 使用 pinned memory；
- 使用异步 memcpy；
- 把传输和 kernel 执行重叠；
- 合并细碎传输。

## NVLink 4：GPU 与 GPU/Grace CPU

H100 SXM 的 NVLink 4 带宽量级为：

```text
18 links × 25 GB/s/方向
单方向合计约 450 GB/s
同时双向合计约 900 GB/s
```

它可连接其他 GPU，也可在相应系统中连接 Grace CPU。NVLink 适合 NCCL collective、AllReduce、Tensor Parallelism、Pipeline Parallelism 和 peer access。

NVLink 显著快于 PCIe，但仍低于访问 GPU 自己的本地 HBM，更远低于本地 SMEM/L1 的片上带宽。因此多 GPU 程序仍需要优先保持数据局部性，并减少跨 GPU 通信量。

## 层级、可见性与性能总结

| 层级 | 主要可见范围 | 管理方式 | 容量/带宽量级 | 相对特点 |
| --- | --- | --- | --- | --- |
| Register | 单线程 | 编译器/硬件 | 64K × 32-bit/SM | 最快；过多会降低 occupancy |
| Local SMEM | Thread block | 软件显式管理 | 最多 228 KiB/SM；227 KiB/block | 低延迟；需同步；可能 bank conflict |
| L1 | 当前 SM | 硬件管理 | 与 SMEM/Texture 共用 256 KiB | 自动缓存；当前 SM 私有 |
| DSMEM | 同 cluster blocks | 软件寻址 + cluster fabric | 各 block SMEM 的逻辑总和 | 比本地 SMEM 慢，通常比 L2 局部 |
| L2 | 全 GPU | 硬件管理，可给 residency hint | 产品约 50 MiB | 分布式 partition；存在 near/far |
| HBM/GMEM | 全 GPU | 程序分配、缓存自动 | 常见配置 80 GB；约 2～3 TB/s | 大容量、高延迟、高聚合带宽 |
| NVLink peer | 其他 GPU/Grace | 通信或 peer access | 约 450 GB/s/方向 | 快于 PCIe，慢于本地 HBM |
| PCIe host | CPU/DPU | 显式或统一内存机制 | 约 64 GB/s/方向理论值 | 通常最远、最慢 |

需要始终区分 latency 和 bandwidth：HBM 单次访问延迟很高，但在足够并发、访问合并良好时可以提供 TB/s 级聚合带宽；Shared Memory 延迟低、行为可控，但容量很小。

## 用 GEMM 串起整个层级

以 $C=AB$ 为例，高性能 Hopper kernel 的典型数据流是：

```text
HBM 中的 A/B
  -> L2
  -> TMA 异步加载 tile 到 SMEM
  -> WGMMA/Tensor Core 读取 SMEM 中的 tile
  -> 寄存器中保存操作数片段和 accumulator
  -> 必要时用 cluster/DSMEM 进行跨 block 协作
  -> 结果经 SMEM/寄存器写回 HBM
  -> 多 GPU 场景通过 NVLink 通信
```

流水线通常是：

```text
TMA 搬运 tile N+1
        ||
Tensor Core 计算 tile N
        ||
前一个 tile 的结果写回
```

性能优化的核心不是让 HBM 单次访问变得低延迟，而是：

1. 一次从 HBM 取入一个 tile；
2. 在 SMEM 和寄存器中多次复用；
3. 用足够并发隐藏不可避免的等待；
4. 用 TMA/pipeline 把搬运和计算重叠；
5. 让本地访问多、DSMEM/L2 访问少、HBM 访问更少；
6. 在多 GPU 场景进一步减少 PCIe/NVLink 通信。

更完整的 GEMM 优化过程见 [GEMM](GEMM.md)。

## 知识点速记

1. GH100 完整设计最多 144 SM，即 8 GPC × 18 SM/GPC；H100 SXM5/PCIe 分别启用较少 SM。
2. 每个 SM 有 4 个 quadrant；每个 quadrant 有 scheduler、Tensor/普通计算、LD/ST 和 `16K × 32-bit` register file。
3. 每 SM 共 64K 个 32-bit 寄存器；寄存器压力会降低 occupancy，并可能产生 local-memory spill。
4. Tensor Core 负责计算，WGMMA 用于 warpgroup 矩阵乘加；TMA 负责数据搬运。
5. H100 每 SM 的 L1/Texture/SMEM 统一存储为 256 KiB，SMEM carveout 最大 228 KiB。
6. CUDA 每 block 保留 1 KiB SMEM，因此单 block 最多约 227 KiB。
7. SMEM 软件显式管理、行为可预测；L1 硬件管理、依赖 cache hit/miss。
8. L1 line 为 128 B，并以 4 个 32 B sector 组织；连续 32-thread FP32 warp 正好覆盖 128 B。
9. TMA 支持 GMEM/SMEM 双向搬运、多维地址生成、swizzling、cluster 内 SMEM 搬运和 multicast。
10. 小 TMA 请求可能因为地址生成等固定开销而不如普通 async copy。
11. Thread Block Cluster 是 CUDA 协作层级，不等于 GPC；同 cluster blocks 保证并发驻留。
12. DSMEM 允许同 cluster block 读取、写入和 atomic 操作彼此的 SMEM。
13. 本地 SMEM 通常快于 DSMEM，DSMEM 通常比绕到 L2 更局部，但性能依赖访问模式。
14. L1 是每 SM 私有缓存，不能被另一 SM 直接寻址；register spill 只是可能经 L1 缓存。
15. L2 对软件统一、物理上分区；SM 到 L2 slice 存在 near/far 拓扑差异。
16. Near/far 不是普通 CUDA 显式暴露的 NUMA 分配接口；地址映射和布局会间接影响它。
17. L2 line 同样为 128 B、4 × 32 B sectors；最大 L2 fetch granularity 可以给出 32/64/128 B 性能提示。
18. 完整 die 有 60 MiB L2，H100 SXM/PCIe 产品约 50 MiB。
19. L2 支持 persisting-data residency hint、inline compression 和 global atomic 路径。
20. 特定测试中的近端 L2 读取约 12～14 TB/s 是片上命中量级；HBM 约 2～3 TB/s 是芯片外内存路径。
21. HBM 延迟可处于约 200～1000 cycles 的量级，必须靠并发、合并访问、缓存和数据复用隐藏。
22. Constant Memory 约 64 KiB，适合同 warp 同地址广播；它是地址空间/访问路径，不是独立 HBM。
23. Local Memory 是线程私有但由 device memory 后备；512 KiB/thread 的限制不是片上实际预留容量。
24. 常见 H100 配置提供 80 GB HBM；H200 等相关产品可提供 141 GB，容量与带宽应按 SKU 确认。
25. PCIe Gen5 x16 理论约 64 GB/s/方向、128 GB/s 双向合计。
26. NVLink 4 为 18 × 25 GB/s/方向，约 450 GB/s/方向、900 GB/s 双向合计。
27. 局部性和复用的最终目标是让大多数计算使用寄存器/SMEM，让 DSMEM/L2/HBM/芯片外通信逐层减少。

## 参考资料

- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API: Device Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)
