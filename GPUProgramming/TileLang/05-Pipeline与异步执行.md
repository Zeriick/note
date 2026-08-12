# Pipeline 与异步执行

`T.Pipelined` 是用户可见的软件流水控制流：它描述循环迭代之间的 producer-consumer 重叠，而不是单纯换一种 `for` 写法。

## 基本形式

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_shared)
    T.copy(B[ko * BK, bx * BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

单次逻辑迭代包含：

```text
producer: copy A/B tile
consumer: gemm 当前 tile
```

普通串行执行：

```text
copy(k) -> compute(k) -> copy(k+1) -> compute(k+1)
```

流水执行希望形成：

```text
copy(k+1) 与 compute(k) 重叠
```

编译器根据依赖重写出 prologue、steady state 和 epilogue。

## Stage 的含义

`num_stages=N` 表示允许 producer-consumer 在多个逻辑 stage/buffer slot 上滚动。它的主要作用不是“把计算切成 N 段”，而是拉开生产和消费之间的依赖距离。

直观上：

```text
stage slot 0: 正被 consumer 使用
stage slot 1: 已完成、等待消费
stage slot 2: producer 正在填充
```

实际状态和 lowering 取决于 target、primitive 和依赖分析，但必须满足同一条原则：

> Producer 不能覆盖 consumer 尚未使用完的 buffer slot。

## Buffer 状态

每个流水 buffer slot 可以抽象为：

```text
FREE -> WRITING -> READY -> READING -> FREE
```

正确性要求：

- producer 只能写 `FREE` slot；
- consumer 只能读 `READY` slot；
- 异步 copy 完成后才能从 `WRITING` 进入 `READY`；
- consumer 完成后才能把 slot 重新交给 producer。

这对应两类同步：

```text
producer completion: 数据已经可读
consumer release   : 旧数据已经不再被读
```

只等待 copy 完成，并不能证明 buffer 可以被下一轮覆盖。

## 为什么能重叠

流水重叠依赖硬件具备独立或可并发推进的工作路径，例如：

- global-memory load 与 Tensor Core compute；
- `cp.async` 与普通计算；
- TMA copy engine 与 WGMMA；
- warp specialization 下不同 warp group 的 producer/consumer。

`T.Pipelined` 表达高层关系，具体是否使用 `cp.async`、TMA 或普通 load/store，由 target、shape、alignment、pass 配置和 primitive lowering 决定。

因此不能把：

```python
num_stages=3
```

直接等同于“必然生成三重 TMA 流水”。最终必须检查 lowered code 和 profiler。

## 自动与手动 Pipeline

规则 GEMM-like loop 优先使用：

```python
T.Pipelined(iters, num_stages=N)
```

编译器从 buffer 读写关系推断 producer/consumer。

对于特殊语句顺序，可以显式给出 `stage` 和 `order`：

```python
for ko in T.Pipelined(
    num_tiles,
    stage=[0, 0, 1],
    order=[0, 1, 2],
):
    T.copy(A[ko * BK], A_shared)
    T.copy(B[ko * BK], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

数组按可执行语句的源码顺序对应：

```text
copy A -> stage 0, order 0
copy B -> stage 0, order 1
gemm   -> stage 1, order 2
```

普通标量别名不是独立 pipeline operation：

```python
base: T.int32 = ko * BK
```

它通常由编译器在使用处重放，不应占用 `stage/order` 条目。手动 schedule 时应只标注 copy、GEMM、reduce、store、wait、commit 等实际操作。

## `T.copy` 与 `T.async_copy`

常规代码优先使用 `T.copy`，让 pipeline 和 lowering 选择合适实现。

显式 `T.async_copy` 用于手动控制异步 global-to-shared copy。它不会自动插入所有 wait；使用者必须确保：

- 对应 async group 已提交；
- consumer 前已经 wait；
- 跨线程消费前满足 shared-memory 可见性；
- buffer slot 不会过早复用。

只在需要控制特殊 ordering、grouping 或无法由常规 pipeline 表达时使用手动异步接口。

## Stage 数的代价

增加 stage 可能提高 latency hiding，也会增加：

- shared-memory buffer 容量；
- barrier/mbarrier 状态；
- 地址和循环状态；
- 某些情况下的 register live range；
- prologue/epilogue 相对开销。

典型权衡：

```text
更多 stages
  -> 更长的预取距离
  -> 可能有更多在飞 copy
  -> 但每 CTA 资源增加
  -> occupancy 可能下降
```

当 2 stages 已足以隐藏延迟时，继续增加通常没有收益。

## 尾部与短循环

若循环迭代数很少，steady state 占比低：

```text
total = prologue + short steady state + epilogue
```

此时 pipeline setup 和额外 buffer 可能得不偿失。需要按真实问题规模评估，不能只在很长的 $K$ 循环上验证。

## 与 GEMM 优化的关系

Pipeline 解决的是时间重叠；tiling 解决的是数据复用：

```text
shared-memory tiling -> 减少 global traffic
register tiling      -> 增加 ILP 和片上复用
multi-stage pipeline -> 隐藏剩余搬运延迟
```

前两层没有建立好时，单独增加 pipeline stage 通常不能挽救低效 kernel。硬件背景见 [[../GEMM#Multi-stage pipeline|GEMM：Multi-stage pipeline]]。

## 检查清单

- 循环中的 producer 和 consumer 分别是什么？
- 哪个 buffer 跨 iteration 复用？
- Producer 完成和 consumer release 分别由什么保证？
- `num_stages` 是否使 shared/register 资源超出预算？
- 循环是否足够长，能进入有意义的 steady state？
- 最终 lowering 使用普通 copy、`cp.async` 还是 TMA？
- Profiler 中 copy 与 compute 是否真的重叠？

