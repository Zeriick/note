# Persistent 与性能分析

Persistent kernel 把空间上的大量逻辑任务，转换成固定数量 CTA 在时间上的连续处理。

## 普通与 Persistent 映射

普通 kernel：

```text
logical tile 0 -> CTA 0
logical tile 1 -> CTA 1
...
logical tile N -> CTA N
```

Persistent kernel：

```text
少量 resident CTA
  -> tile 0
  -> tile P
  -> tile 2P
  -> ...
```

其中 $P$ 通常与目标驻留 worker 数有关。CTA 完成一个 tile 后不退出，而是继续领取下一个 tile。

## `T.Persistent`

TileLang 提供高级 persistent loop：

```python
with T.Kernel(worker_count, threads=threads) as block_id:
    for bx, by in T.Persistent(
        [m_blocks, n_blocks],
        worker_count,
        block_id,
    ):
        ...
```

参数可以建立下面的直觉：

```text
domain      : 逻辑任务空间
wave_size   : 一波有多少 worker
index       : 当前 worker id
group_size  : 任务分组/遍历粒度
```

具体 tile 顺序由 lowering 和参数共同决定，不应假设它必然是简单 row-major。

## 为什么使用 Persistent

可能收益：

- 控制 resident worker 数，减少过多 CTA 的 launch/scheduling 开销；
- 在 CTA 生命周期内复用 shared/register 状态；
- 自定义 tile 领取顺序，改善 cache locality；
- 在 MegaKernel 或多阶段融合中保持中间状态；
- 对不规则任务实现更直接的负载均衡。

它提供的是更强 schedule control，不是免费性能。

## 主要代价

### 尾部效应

若不同 tile 工作量差异大，最后少数 CTA 可能拖住整个 kernel：

```text
多数 workers 已完成
少数 workers 处理长尾任务
GPU 利用率快速下降
```

### 长期 Live State

跨 task 保留的 fragment、shared buffer 和队列状态会长期占用资源：

```text
state lifetime 增长
  -> register/shared pressure 持续存在
  -> resident workers 可能减少
```

### 同步与确定性

动态领取和跨任务状态复用会增加：

- 队列和 atomic 开销；
- phase/barrier 管理；
- 多个 worker 的写入顺序差异；
- 浮点 reduction 的非确定性；
- 错误状态泄漏到下一 task 的风险。

每个新 task 开始前，应明确哪些状态清零、覆盖、继承或同步。

## Persistent 不一定适合

下列情况普通 grid 往往更简单：

- 每个 tile 工作量规则且足够大；
- GPU scheduler 已能良好分配 CTA；
- 没有跨 tile 状态可复用；
- 逻辑 tile 数不多；
- persistent 使 occupancy 或尾部行为变差。

判断依据必须是端到端 benchmark，而不是“CTA 数更少”。

## 性能分析顺序

建议固定使用以下顺序：

```text
1. correctness
2. end-to-end latency
3. throughput / roofline 位置
4. 资源与 occupancy
5. memory access
6. scheduler stalls / pipeline overlap
7. PTX/SASS
```

不要先看某个漂亮的底层指标，再反推 kernel 一定更快。

## 1. Correctness

先覆盖：

- 随机输入；
- 全零、极值、非有限值等特殊输入；
- tile 整除和不整除的 shape；
- 很小和很大的 reduction domain；
- 多种 dtype；
- persistent 下多 wave 和尾部任务。

对浮点 reduction，应根据累加顺序和 dtype 设定合理的 `rtol/atol`，不要直接用 bitwise equality。

## 2. 基准测试

测量时区分：

- 编译时间；
- warmup；
- 单 kernel latency；
- 包含框架调度和数据准备的端到端 latency。

记录完整配置：

```text
shape / dtype / target
tile shape / threads / num_stages
是否 persistent / warp specialized
软件版本 / GPU clock policy
```

只记录“最快 0.1 ms”而不保存配置，结果无法复现。

## 3. 性能模型

先判断主要瓶颈：

```text
memory-bound
compute-bound
latency / dependency-bound
occupancy / resource-bound
launch / synchronization-bound
```

GEMM 使用算术强度和 Tensor Core 利用率；reduction 重点观察带宽、在飞请求和同步。对应模型见：

- [[../GEMM|GEMM 优化]]
- [[../CUDA Reduction 优化|CUDA Reduction 优化]]

## 4. 资源与 Occupancy

重点记录：

- registers per thread；
- static/dynamic shared memory per CTA；
- resident blocks/warps；
- register spill；
- persistent worker 数。

Occupancy 不是越高越好。只要有足够 warps/ILP 隐藏延迟，降低 occupancy 换取更多 register tiling 可能更快。真正危险的是资源增长后，latency hiding 已不足或发生 spill。

## 5. Memory Access

检查：

- global load/store 是否 coalesced；
- 向量化宽度和 alignment 是否匹配；
- 实际 DRAM/L2 流量是否接近模型；
- shared-memory bank conflict；
- cache hit rate 是否由可复现的复用产生；
- persistent tile order 是否改善或破坏 locality。

## 6. Pipeline 与 Stall

重点区分：

```text
没有足够 producer 请求
copy 延迟没有被 compute 隐藏
consumer 等待数据
barrier 等待
Tensor Core / ALU dependency
资源导致可运行 warp 不足
```

增加 stage 只可能解决其中一部分。若瓶颈是低效 layout、bank conflict 或计算依赖链，增加 buffer 数不会直接修复。

## 工具分工

| 工具 | 主要用途 |
| --- | --- |
| TileLang profiler / benchmark | 正确性、延迟、参数搜索 |
| Nsight Compute（NCU） | 单 kernel 指标、memory、scheduler、occupancy |
| Nsight Systems（NSYS） | 端到端 timeline、kernel/CPU overlap |
| Layout visualization | 查看 thread/local slot/replication |
| Lowered source、PTX、SASS | 确认最终指令与访存 |
| IKET | TileLang CUDA kernel 内的命名 marker/range 与 Perfetto trace |

IKET 是实验性 instrumentation 工具，不属于 `tilelang.language`：

```python
from tilelang.tools.cuda import iket

with iket.range("compute"):
    ...
```

它适合观察 kernel 内 phase；NCU 更适合硬件计数器，NSYS 更适合 host/device 全局时间线。

## 调优参数

常见搜索空间：

```text
BM, BN, BK
threads
num_stages
vector width
explicit layout / swizzle
persistent worker count / group size
warp specialization strategy
```

参数之间强耦合：改变 tile shape 会同时改变复用、fragment layout、shared memory、寄存器和 pipeline 深度预算。调优结果应作为完整配置保存，而不是分别选每个参数的局部最优值。

## 最终检查清单

- 与 reference 的数值结果是否覆盖边界 shape？
- Benchmark 是否排除了编译并有稳定 warmup？
- 当前瓶颈分类有指标支持吗？
- Tile shape 的收益是否被 register/shared pressure 抵消？
- Layout 是否产生合并访存、合理 local slot 和必要而不过度的 replication？
- Pipeline 是否形成真实 overlap？
- Persistent 是否改善 locality/调度，而不是只减少 CTA 数？
- PTX/SASS 是否与预期 primitive lowering 一致？

