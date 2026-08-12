# Jump Threading

Jump threading（跳转穿线）是一种控制流优化：当编译器能根据“从哪条入边到达当前块”确定该块末尾的分支结果时，让这条入边直接连接到必然选择的后继。

它与 [[结构化并发|结构化并发]] 不是同一概念，但二者共享一种设计直觉：具有明确入口、出口和嵌套边界的区域更容易局部推理；任意跨边界的边会让树状结构退化为一般图。

## 基本例子

```text
P ──► C: if (x == 0) ──true──► T
                       └false─► F
```

如果编译器知道在 `P -> C` 这条边上 `x == 0` 恒成立，就可以把它改成：

```text
P ───────────────────────────► T

      C: if (x == 0) ──true──► T
                       └false─► F
```

这里被优化的不是“`C` 的条件在所有路径上恒真”，而是“从 `P` 进入 `C` 时恒真”。`C` 仍可能被其他 predecessor 到达，并在那些路径上选择不同的后继。

“threading”可以理解为把 `P` 这条控制流路径像一根线一样穿过 `C`，接到已知目标 `T`。

## 典型信息来源

- 前面已有相同或可蕴含当前条件的判断。
- φ：从不同 predecessor 进入时，φ 选择不同的值。
- 常量传播、SCCP 的可执行边和值格信息。
- [[CVP]]、范围分析、关系分析提供的路径 predicate，例如 `x > 10` 蕴含 `x > 0`。
- `switch` case 边携带的取值约束。
- 已有 `assume`、非空、no-wrap 等语义事实。

最典型的 SSA 例子是：

```text
P0 ── value 0 ──┐
                ▼
              x = phi [0, P0], [1, P1]
P1 ── value 1 ──┘
              branch (x == 0), T, F
```

在 `P0 -> C` 上条件恒真，在 `P1 -> C` 上条件恒假，因此理论上可以得到：

```text
P0 ──► T
P1 ──► F
```

## 和相邻优化的区别

### Branch folding / branch pruning

如果条件在整个块中恒真：

```text
C: branch true, T, F
```

直接把 `C` 的 terminator 改成 `jump T` 即可。这种变换只需要块级事实，不需要区分它是从哪个 predecessor 到达的。

Jump threading 更细：条件可能只对 `C` 的某个 predecessor 确定，因此它改的是特定入边，而不是直接删除 `C` 的条件分支。

### Jump forwarding / 空块消除

```text
P -> B -> T
```

如果 `B` 只有无条件跳转，改成 `P -> T` 不需要路径推理，通常属于 CFG simplification 或 trivial block elimination。Jump threading 的核心是利用某条入边上的条件事实。

### Tail duplication

如果中间块包含不能直接跳过的计算，可以复制一小段指令或基本块，再把复制后的路径专门化。这是更激进的 jump threading，也可以视为与 tail duplication 的结合。它会增加代码尺寸，并显著增加 SSA 修复难度。

## 收益链条

Jump threading 的直接收益是减少动态条件跳转；更大的收益常来自它暴露的后续优化：

```text
路径穿线
  -> 删除不可达边
  -> 简化 φ
  -> 常量传播 / DCE
  -> 删除空块、合并基本块
```

可能收益包括：

- 减少分支与分支预测压力。
- 消除不可执行边。
- 让 φ 退化为单值。
- 暴露常量和不可达块。
- 给后续循环、冗余消除和代码布局提供更简单的 CFG。

代价主要是：

- 块复制引起代码膨胀。
- CFG 与 SSA 修复复杂。
- 需要使 dominance、loop、range、MemorySSA 等缓存分析失效或更新。
- 不受约束的改边可能制造 irreducible control flow。

## Reducible 与 irreducible 风险

普通自然循环具有唯一 header：

```text
outside -> H -> B
           ^    |
           |____|
```

循环 SCC `{H, B}` 的所有外部入口边都指向 `H`，而且 `H` 支配 `B`。这是 reducible 的循环。

假设有一条路径 `P -> H`，且从 `P` 到达时能证明 `H` 的分支必然去 `B`。若 threading 直接产生：

```text
outside -> H -> B
           ^    ^
           |____|
P --------------|
```

现在循环 SCC 同时有外部入口 `outside -> H` 和 `P -> B`。`H` 不再支配 `B`，SCC 找不到支配全部成员的唯一 header，CFG 变成 irreducible。

要注意：header 有多个 predecessor 不等于多入口循环。只要所有循环外的边都进入同一个 header，仍然是单入口；危险的是外部边进入同一循环 SCC 的不同节点。

很多自然循环分析、循环规范化和循环优化都默认这种单入口结构，因此 jump threading 不能只验证语义等价，还要验证它是否保留了后续 pass 所依赖的 CFG 形状。若编译器没有完整支持 irreducible CFG，宁可保守拒绝跨循环入口的候选。

## 通用实现经验

### 把事实绑定到边，而不是块

Jump threading 使用的是 `P -> C` 上成立的事实，而不是 `C` 上对所有路径成立的事实。实现中最好显式表示 edge context，避免把某个 predecessor 的约束写进块级缓存，随后错误地用于其他 predecessor。

对一个值做边上转译时，通常遵循这些规则：

- 常量和支配 predecessor 的定义可以直接复用。
- `C` 中的 φ 结果替换为该 predecessor 对应的 incoming value。
- 小型纯表达式可以在代入后继续折叠，但要限制递归深度和表达式规模。
- 遇到无法安全解释的 load、call、异常语义或 poison/undef 情况就停止推导。

证明分支方向与物化 CFG 改写最好分离。先得到一个完整、不可变的改写计划，确认目标、值映射、循环合法性和成本都满足要求，再修改 IR；否则很容易在改掉旧边之后才发现新边的 φ 输入无法构造。

### 改边的难点是 SSA，而不是 terminator

把 `P -> C` 改成 `P -> T` 本身很简单，真正困难的是为新边构造正确的数据流：

- `C` 的 φ 在 `P -> C` 上选择了哪些值？
- `C -> T` 原本给 `T` 的各个 φ 传递什么值？
- 把前一组选择代入后一组表达式后，能否得到 `P -> T` 的 incoming values？
- 删除旧边后，旧 φ 输入是否同步删除？

例如，`C -> T` 传递的值若依赖 `C` 中的某个 φ，就要先把该 φ 替换成 `P` 对应的 incoming value。若它还依赖 `C` 中的普通指令，则只能在该值已于 `P` 可用、能够安全重算，或者允许复制相关指令时继续。

因此适合提供一个统一的 SSA-aware edge rewrite 抽象，由它原子地完成：

- 收集并转译新边的 incoming values；
- 验证目标 φ 的完整性；
- 更新 terminator；
- 删除旧边的 φ 输入并加入新边输入。

不要让不同 CFG pass 各自手写这套逻辑。还要明确 IR 如何表示平行边：如果同一 terminator 有多个 operand 指向同一 successor，φ 输入究竟按 predecessor 还是按 edge occurrence 区分，会直接影响改写正确性。

### 不复制版本的合法性边界

无需复制的 threading 只能绕过“对该路径透明”的块。无副作用并不自动等于透明：一个纯计算如果产生了后继仍需使用的值，跳过它同样会破坏语义。

比较稳妥的条件是：

- 分支方向能在入边上静态确定。
- 被绕过的指令没有必须发生的副作用、异常或 trap。
- 目标所需的所有值都已在 predecessor 可用，或能通过 φ 代入得到。
- 特殊控制流和无法精确定义的数据流边直接拒绝。

这类限制会漏掉机会，但建立了清晰的正确性基线。更激进的优化应作为“路径复制”单独处理，而不是悄悄放宽“可跳过”的定义。

### 用单入口不变量保护 reducibility

一个实用的快速过滤规则是：如果目标位于循环内、源位于循环外，那么新边只能指向该循环 header。

```text
source outside L && target inside L && target != header(L)
    => reject
```

它对嵌套循环和不完整 LoopInfo 可能过于保守，因此更强的兜底方式是检查改写后的 SCC：每个有环 SCC 应只有一个接收外部边的入口节点，并由该节点支配 SCC 内所有节点。

局部规则适合快速筛选，SCC/dominance 检查适合调试、验证或处理复杂候选。重要的是不要只检查“目标是不是一个 loop block”，因为改写本身可能改变原有 loop membership。

### 复制需要同时考虑安全性和收益

如果条件或后继输入依赖中间计算，可以复制相关的纯指令切片或建立一个专用块。复制时需要维护旧值到新值的映射，并把 φ 输入替换成当前入边上的实际值。

可复制不等于值得复制。通常需要限制：

- 新增指令和基本块数量；
- 单个候选及整个函数的代码增长；
- 重复处理同一结构的次数；
- 冷热路径上的收益；
- 后续是否真的能删除分支或不可达代码。

还要单独审查可能 trap 的算术、内存访问、异常、原子/volatile 操作、别名影响，以及 poison、undef、no-wrap 等语义。最好把“允许复制”和“允许投机执行”视为两个不同问题：复制到只在原路径执行的专用块，通常比把指令提前塞进 predecessor 更安全。

### 分析失效与迭代

改写 CFG 后，旧的可达性、dominance、post-dominance、loop、路径事实、频率和依赖 CFG 的内存分析通常都不再可靠。除非实现了严格的增量维护，否则应保守失效并重算。

一次 threading 常会暴露新的 φ 折叠、常量分支和不可达块，因此适合用有预算的 worklist 与 CFG cleanup 配合。预算不仅控制编译时间，也防止路径复制与新机会互相触发，造成代码尺寸失控。

## 容易遗漏的测试

除基本的 true/false 穿线外，重点覆盖：

- 只穿部分 predecessor，原条件块仍然可达。
- 目标含多个 φ，且 outgoing value 依赖中间 φ 的边上转译。
- 同一 successor 的重复边或特殊 edge identity。
- 被绕过块含副作用、trap、异常边或不可投机指令时正确拒绝。
- 循环外到非 header 的候选不会产生多入口 SCC。
- 复制达到预算后稳定停止，不会反复复制等价结构。
- 改写后重建 CFG、dominance 和 loop 信息，并由 IR verifier 检查 φ、use-def 与 terminator 一致性。
- 改写前后的解释执行或端到端结果一致。

## 一句话总结

Jump threading 利用特定入边上的已知事实，把 predecessor 直接接到条件块必然选择的 successor。真正困难的不是改一个 block operand，而是同时保证路径语义、SSA/φ 完整、分析失效正确、代码增长受控，并保持后续循环基础设施依赖的 reducible CFG。
