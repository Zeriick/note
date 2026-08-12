# PAG：Pointer Assignment Graph 与图上的可达性

PAG（Pointer Assignment Graph，指针赋值图）把程序中的指针传播关系表示成一张带标签的图。它通常用来回答：

> 一个 pointer value 能否沿合法的赋值、访存、字段访问和跨函数传递关系影响另一个 pointer？

不同论文和工具对 PAG 的节点、边和内存抽象会有差异；本文记录的是一套通用理解框架，而不是某个具体实现的接口。

## 1. 基本定义

PAG 通常可以写成：

$$
G=(V,E,\lambda).
$$

- $V$ 是节点集合。
- $E\subseteq V\times V$ 是有向边集合。
- $\lambda:E\rightarrow\Sigma$ 给每条边附上操作标签。

### 1.1 节点

常见节点包括：

- pointer-typed variables 或 SSA values；
- 函数的 pointer parameters 和 return values；
- load、`getelementptr` 等指令产生的中间 pointer values；
- abstract memory objects，例如 allocation sites、stack objects、globals；
- field-sensitive 实现中的 object-field nodes。

源代码变量与 PAG 节点通常不是一一对应的。在 SSA IR 中，同一个源代码变量的不同定义会成为不同 SSA values；一次字段寻址、load 或类型转换也可能产生独立节点。

### 1.2 常见边

下面是一种常见的边方向约定：

| 程序操作 | PAG 边 | 含义 |
|---|---|---|
| `p = q` | $q\xrightarrow{Assign}p$ | pointer value 从 `q` 复制到 `p` |
| `p = *q` | $q\xrightarrow{Load}p$ | 从 `q` 定位的内存中读出 pointer |
| `*p = q` | $q\xrightarrow{Store}p$ | 把 `q` 的 pointer value 写到 `p` 定位的内存 |
| `p = &q->f` | $q\xrightarrow{Field_o}p$ | 从 base `q` 取得 offset 为 $o$ 的字段地址 |
| actual-to-formal | $a\xrightarrow{Call}f$ | 实参 pointer 传给形参 |
| return-to-receiver | $r\xrightarrow{Ret}p$ | 返回的 pointer 传给调用点结果 |

具体工具可能拆分出 `Addr`、`Copy`、`Gep`、`Load`、`Store` 等更多边，也可能使用相反的边方向。阅读实现时必须先确认其约定，不能只凭标签名判断。

## 2. 从约束到图

经典 inclusion-based pointer analysis 会把程序转换成 points-to constraints。例如：

```c
p = &x;
q = p;
r = *s;
*t = q;
```

可以产生近似约束：

$$
x\in pts(p),
$$

$$
pts(p)\subseteq pts(q),
$$

$$
o\in pts(s)\Rightarrow pts(o)\subseteq pts(r),
$$

$$
o\in pts(t)\Rightarrow pts(q)\subseteq pts(o).
$$

PAG 把这些约束显式编码成节点和边。求解器可以在图上做集合传播、节点合并或按需可达性查询。

## 3. 为什么需要反向边

某些 CFL-reachability formulation 会为每条正向边：

$$
u\xrightarrow{\sigma}v
$$

加入一条分析用的反向边：

$$
v\xrightarrow{\bar\sigma}u.
$$

例如：

$$
q\xrightarrow{Assign}p
$$

对应：

$$
p\xrightarrow{\overline{Assign}}q.
$$

反向边不表示程序执行了反向赋值。它允许查询先逆着 value flow 找到共同来源，再顺着另一条 value-flow path 到达目标。

例如：

```c
p = q;
r = q;
```

图中有：

```text
q --Assign--> p
q --Assign--> r
```

查询 `p` 和 `r` 时，可以考察：

```text
p --reverse(Assign)--> q --Assign--> r
```

它表达“`p` 和 `r` 具有共同的 pointer-value 来源”。

## 4. 普通图可达不等于 alias

如果忽略边标签，只要存在：

$$
p\leadsto q
$$

就判定 alias，会产生大量伪关系。Load、Store、Field、Call 和 Ret 不能任意拼接：

- 一次通过 Store 进入内存的传播，必须与语义相容的 Load 配合；
- 不同 field offsets 不能随意合并；
- 从一个 call site 进入函数后，不能从另一个不相关的 call site 返回；
- 逆向和正向 value flow 必须组成合法结构。

因此，许多 PAG formulation 使用带标签的路径语言。

对路径：

$$
\pi=(v_0,v_1,\ldots,v_m),
$$

定义标签串：

$$
w(\pi)=\lambda(v_0,v_1)\lambda(v_1,v_2)\cdots\lambda(v_{m-1},v_m).
$$

只有当 $w(\pi)$ 属于分析定义的合法语言时，$v_m$ 才是从 $v_0$ **语义可达**的，而不只是拓扑可达。

## 5. CFL reachability

CFL reachability（Context-Free-Language Reachability）把上下文无关文法与图路径结合起来：

> 是否存在一条从 $u$ 到 $v$ 的路径，使其边标签串能够由指定 grammar 推导？

不同分析会采用不同 grammar。一种常见的抽象结构是：

```text
Flow  -> ordinary pointer-value propagation
Alias -> reverse(Flow) Alias Flow
Alias -> matched Load/Store relation
Alias -> matched field-offset relation
Alias -> epsilon
```

它表达三类核心关系：

1. 沿 ordinary assignments 和跨函数传递进行 value flow；
2. 通过成对的 Store/Load 穿过内存；
3. 从共同来源分叉到两个 pointers。

### 5.1 Store/Load 匹配

考虑：

```c
*r = q;
p = *r;
```

`q` 的 pointer value 先写入 `r` 指向的 memory cell，再从该位置读入 `p`。因此 `q` 可以通过一对匹配的 Store/Load 与 `p` 建立 value-flow 关系。

这里不能只因为图上先出现 Store、后出现 Load 就认为二者匹配；承载它们的 memory pointers 也必须可能指向同一抽象内存位置。

### 5.2 Field-offset 匹配

如果两个 base pointers 可能 alias，并且都取 offset $o$ 的字段，那么相应 field pointers 也可能 alias：

$$
\overline{Field_o}\;Alias\;Field_o.
$$

offset 必须匹配。`&x->left` 与 `&x->right` 即使共享同一个 base object，只要字段 offset 不同，field-sensitive analysis 就不应仅因为它们属于同一个 struct 而合并。

## 6. Context-sensitive reachability

如果把 `Call` 和 `Ret` 当成普通复制边，图上可能出现 unrealizable path：

```text
从 call site A 进入函数
经过函数体
从 call site B 返回
```

真实执行中的调用和返回必须正确嵌套。常见做法是给每个 call site 编号 $i$，把进入和返回映射为成对括号：

```text
进入 call site i：⟦i
返回 call site i：⟧i
```

合法 context 类似 Dyck language：

```text
⟦1 ⟦2 ⟧2 ⟧1    合法
⟦1 ⟧2            不合法
⟦1 ⟦2 ⟧1 ⟧2     不合法
```

context-sensitive、field-sensitive reachability 通常要求同一条路径同时满足：

1. value/memory/field 标签约束；
2. call/return nesting 约束。

因此，节点在图上连通仍不够；路径还必须是可由真实调用栈实现的路径。

## 7. Points-to 与 alias query

pointer analysis 通常计算：

$$
pts(p)=\{o_1,o_2,\ldots\},
$$

其中每个 $o_i$ 是 abstract memory object。

两个 pointers 的 may-alias 判断通常是：

$$
pts(p)\cap pts(q)\neq\varnothing.
$$

但 alias query 不一定必须先完整物化两个 points-to sets。Demand-driven solver 可以直接搜索一条能够证明共同对象或合法 alias path 的 witness；找到 witness 后就提前结束。

需要区分：

- `MayAlias`：存在 alias 的可能性，或者分析无法排除；
- `NoAlias`：分析能够证明不 alias；
- `MustAlias`：在所定义的语义和程序点上必然 alias；
- `Unknown`：资源或建模不足，不能安全回答。

对 sound may-analysis 来说，搜索超时或预算耗尽时，安全结果通常是 `MayAlias` 或 `Unknown`，而不是 `NoAlias`。

## 8. 常见求解风格

### 8.1 Andersen-style inclusion solving

用集合包含关系传播 points-to facts：

$$
pts(q)\subseteq pts(p).
$$

通常比较精确，但在复杂程序上可能昂贵。

### 8.2 Steensgaard-style unification

把相关节点或 points-to sets 合并为等价类：

$$
pts(q)=pts(p).
$$

可以用 union-find 高效实现，但合并会传播额外的伪 alias，通常比 inclusion-based analysis 粗糙。

### 8.3 Demand-driven CFL reachability

不预先求出所有 pointer pairs，而是在查询 $(p,q)$ 时搜索合法路径。它适合少量 on-demand queries，但单个困难 query 仍可能遍历很大的图。

### 8.4 Summary/index-based solving

预先为函数、图区域或常见路径构造 summary/index，减少重复遍历。它在 upfront cost、memory 和 per-query latency 之间做权衡。

## 9. 精度维度

PAG 本身并不自动决定分析精度；精度取决于节点抽象、边构造和求解规则。

### Field sensitivity

区分 aggregate object 的不同字段；field-insensitive analysis 可能把所有字段折叠到同一位置。

### Context sensitivity

区分同一个函数在不同 call sites 或不同 calling contexts 下的 facts。

### Flow sensitivity

区分不同程序点上的 pointer state；flow-insensitive analysis 忽略语句顺序。

### Heap abstraction

决定多个 runtime objects 是否被同一个 abstract object 表示，例如 allocation-site abstraction 会把同一分配点产生的多个对象合并。

### Path sensitivity

区分不同控制流条件下的状态，避免把互斥路径上的 facts 无条件合并。

更高 sensitivity 往往减少伪 alias，但也增加状态数量、路径组合和求解成本。

## 10. 为什么可达性可能很贵

成本不仅来自图的节点和边很多，还来自路径状态：

- 当前存在多少未匹配的 Load/Store；
- call stack/context 有多深；
- Field offsets 是否相容；
- 正向和反向 flow 是否组成合法结构；
- 多套语言约束是否需要同时满足；
- indirect calls 和 external functions 是否需要额外建模。

常见的可扩展性策略包括：

- 限制 context depth；
- 限制 access-path 或 indirection depth；
- 设置 per-query time/edge/path budget；
- 合并 contexts 或 fields；
- 预计算 summaries/indexes；
- 只对部分 queries 使用高精度分析；
- 预算耗尽时返回 conservative result 或 `Unknown`。

必须区分两个概念：

- **搜索完成**：相对于当前 PAG，所有规定范围内的路径都已处理；
- **PAG soundness**：PAG 本身覆盖了所有需要建模的 concrete pointer flows。

即使搜索完整结束，如果 PAG 因 external calls、inline assembly、integer-pointer conversions、union、memory intrinsics、varargs 或 allocator modeling 等问题漏边，分析结果仍可能不 sound。

## 11. 与其他 IR 图和分析的关系

### 11.1 PAG 与 SSA

[[complier/ssa/构造|SSA]] 保证每个 SSA value 只有一个静态定义，并用 def-use 链和 φ 节点表示标量值传播。PAG 可以把 pointer-typed SSA values 当作节点，从 copy、φ、call 和 return 中生成指针传播边。

但 SSA def-use 图不能单独回答别名问题。对 `p = *q` 或 `*p = q` 来说，必须知道 `p`/`q` 可能指向哪些 abstract memory objects，才能把 load/store 与内存内容联系起来。PAG 正是通过 object nodes 和 address/copy/load/store/field 等约束补上这一层。MemorySSA 虽然也建模内存定义与使用，但它主要组织内存版本和 clobber 关系，仍不等同于 PAG。

### 11.2 PAG 与 CFG

| [[complier/cfg/dom|CFG]] | PAG |
|---|---|
| 节点通常是 basic blocks 或 instructions | 节点通常是 pointer SSA values 和 abstract objects |
| 边表示可能的控制转移 | 边表示 pointer value 或 memory flow |
| 可达表示代码可能按某条控制路径执行 | 可达表示 pointer 信息可能传播 |
| 常用 DFS、dominator、loop 等分析 | 常用 constraint solving、CFL/Dyck reachability |
| 关注 branch、exception、call/return 控制结构 | 关注 Assign、Load、Store、Field、Call、Ret 等标签 |

一条 CFG path 说明语句可能按某种顺序执行，但不自动说明某个 pointer value 会沿这条路径传播；一条 PAG path 抽象了 value/memory flow，但如果分析不是 path-sensitive，它也可能包含控制上不可同时实现的传播组合。

Flow-sensitive pointer analysis 会在 CFG 程序点之间传递 points-to state；path-sensitive analysis 还会区分分支条件。相反，flow-insensitive PAG 中的指针约束通常不按执行顺序分版本，因此不能把它的可达性直接解读为某个具体程序点的路径事实。

### 11.3 PAG 与 alias analysis

[[Alias Analysis|Alias analysis]] 是要回答的语义问题：两个 pointers 或内存访问是否可能指向重叠区域。PAG 是承载指针约束的一种分析表示。求解器可以在 PAG 上计算 points-to sets 或搜索合法 alias paths，再对查询返回 `NoAlias`、`MayAlias`、`MustAlias` 或工具定义的其他结果。

因此“构建了 PAG”并不等于“已经得到 alias 结果”：还需要明确节点抽象、路径语言、求解策略和保守失败语义。反过来，alias analysis 也可以组合类型、基对象、偏移和调用副作用等其他技术，不一定显式构建 PAG。

### 11.4 PAG 与 CVP/IPSCCP

[[IPSCCP]] 和 [[CVP]] 也在传播程序事实，但不应与 PAG-based analysis 混为一谈：

- IPSCCP 主要沿 SSA def-use 链和调用关系传播格值，同时维护 CFG 边的可执行性。
- CVP 主要利用支配当前程序点的 predicate/range 条件化简指令。
- PAG 主要组织指针值和抽象内存对象之间的约束；普通 PAG 可达性本身不能证明一个值是常量，也不能证明某个 predicate 在当前路径成立。

PAG-based alias analysis 可以在优化 pipeline 中间接帮助这两类 pass：例如让相邻的内存化简或调用副作用分析证明某个 store/call 不会 clobber 一次 load，从而暴露可供常量或路径传播消费的事实。具体 CVP/IPSCCP 实现未必直接查询 alias analysis，更不会用 PAG 取代 SSA 或 CFG 传播。

## 12. 阅读 PAG 实现时的检查清单

1. 节点代表 SSA values、variables，还是 abstract objects？
2. allocation、copy、load、store、GEP 分别使用什么边？
3. 边方向是什么？是否显式存储 reverse edges？
4. struct fields 和 array indices 如何抽象？
5. heap objects 按 allocation site、type 还是 context 区分？
6. indirect calls、external functions 和 allocator wrappers 如何建模？
7. call/return 是否进行 context matching？
8. solver 是 global fixed point 还是 demand-driven query？
9. timeout/budget exhaustion 返回 `MayAlias`、`NoAlias` 还是 `Unknown`？
10. 声称的 soundness 针对哪种语言子集和哪些实现假设？

## 13. 一句话总结

PAG 上的可达性不是简单地问：

> 图中 $p$ 能不能走到 $q$？

而是问：

> 是否存在一条从 $p$ 到 $q$ 的路径，使它的 pointer-value、memory、field 和 call/return 标签满足分析定义的合法传播规则？

普通 reachability 只关心 connectivity；PAG-based alias analysis 还必须关心 path language、memory abstraction、calling context，以及图本身是否 sound。
