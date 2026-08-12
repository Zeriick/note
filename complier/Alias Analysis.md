# Alias Analysis：内存别名分析

Alias analysis（别名分析）判断两个指针或内存访问在某个程序点是否可能指向重叠的内存位置。它是 load/store 调度、死存储删除、循环向量化、内存依赖分析和调用副作用建模的基础。

“别名”是语义问题；具体分析可以使用 points-to sets、类型/基于对象的规则、调用摘要，或 [[PAG|Pointer Assignment Graph（PAG）]] 上的约束求解与合法路径可达性来回答。

## 查询结果

对内存区域 $(p,s_p)$ 和 $(q,s_q)$，常见查询结果包括：

- `NoAlias`：能够证明两个区域不重叠。
- `MayAlias`：两者可能重叠，或分析无法安全排除重叠。
- `MustAlias`：在查询所定义的程序点和语义下，两者必然表示同一区域。
- `PartialAlias`：两个有大小的区域部分重叠，但起始位置或范围不完全相同。

具体 IR 和分析 API 的结果集合可能不同。对 sound may-analysis 而言，`NoAlias` 需要证明；超时、库函数建模不足或信息丢失时，安全退化通常是 `MayAlias`/`Unknown`，而不是 `NoAlias`。

## Points-to 与 alias 的关系

Points-to analysis 尝试为指针 $p$ 计算可能指向的抽象对象集合：

$$
pts(p)=\{o_1,o_2,\ldots\}.
$$

若

$$
pts(p)\cap pts(q)=\varnothing,
$$

则可以据此证明 `NoAlias`；若交集非空，通常只说明 `MayAlias`，因为抽象对象可能合并多个 runtime objects 或互斥路径上的状态。证明 `MustAlias` 通常需要更强的唯一性、偏移、大小和程序点信息。

Alias query 不一定要先完整物化所有 points-to sets。Demand-driven analysis 可以从查询对出发，寻找一条证明共同对象或合法 alias path 的 witness。

## PAG 是表示，alias analysis 是问题

PAG 常把 address/copy/load/store/field/call/return 等指针约束表示成带标签的图。求解器可以在图上传播 points-to sets，合并等价类，或搜索满足 CFL/Dyck 等路径语言的 witness。因此 PAG 可以支撑 alias analysis，但不能把两个概念互换：

- 同一张 PAG 可以由不同精度和预算的求解器使用。
- alias analysis 也可以组合基于类型、基对象、偏移、调用副作用或动态检查的其他技术，不必显式构建 PAG。
- 无标签的普通图连通不等于 alias；load/store、field offset 与 call/return context 需要符合分析语义。

## 精度维度

- **Flow sensitivity**：是否区分 [[complier/cfg/dom|CFG]] 不同程序点的 points-to state。
- **Path sensitivity**：是否区分不同分支条件，避免合并互斥路径。
- **Context sensitivity**：是否区分不同 call sites 或 calling contexts。
- **Field sensitivity**：是否区分 aggregate object 的字段和偏移。
- **Heap abstraction**：多个 runtime allocations 何时被合并为同一 abstract object。

更高精度通常能证明更多 `NoAlias`，但会增加状态数量、求解时间和内存占用。

## 与优化 pass 的关系

Alias analysis 通常是优化的正确性门卫，而不是变换本身。例如，若要把 load 移过 store，必须证明该 store 不会改写 load 读取的区域。`MayAlias` 并不证明运行时一定发生别名；它只是阻止缺乏进一步证明的不安全变换。

[[IPSCCP]] 和 [[CVP]] 的核心分别是 SSA 格值/可执行边传播与支配条件推理，并不依赖 PAG 作为自己的传播图。Alias/mod-ref 信息更常被相邻的内存化简、调用副作用分析或具体实现使用，用来证明 load 不会被 store/call clobber，从而间接暴露更多常量或路径化简机会。这是 pipeline 层面的配合，不代表每个 CVP/IPSCCP 实现都直接查询 alias analysis。

## 正确性边界

分析的 soundness 不只取决于求解器是否搜索完成，还取决于程序建模是否覆盖 external calls、inline assembly、integer-pointer conversions、unions、memory intrinsics、varargs、allocators 和语言本身的未定义行为规则。漏掉可能的指针流会把本应 `MayAlias` 的结果错误地收窄为 `NoAlias`，进而导致错误优化。
