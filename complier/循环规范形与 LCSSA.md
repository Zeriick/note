# 循环规范形与 LCSSA

循环优化通常不会直接支持任意形状的控制流图，而是先把自然循环规整成一组明确的结构约定。循环规范形（loop canonical form）降低了后续变换的组合复杂度；Loop-Closed SSA（LCSSA）则把循环内部值的流出集中到出口边界。

不同编译器对“canonical loop”的精确定义并不完全相同。本文记录通用组件、语义风险和常见约定，不把某一种实现的判定函数当成普遍定义。

## 1. 自然循环与 Reducible CFG

若 CFG 边：

```text
latch -> header
```

的目标 `header` 支配源 `latch`，这条边是 backedge 候选。由 `header`、`latch` 以及所有能反向到达 `latch` 而不穿过 `header` 的节点，可构成一个 natural loop。

自然循环具有单入口性质：所有循环外进入循环区域的边都先到 `header`。这使循环可以用树状 nesting relation 表示。

如果一个强连通分量有多个外部入口，CFG 是 irreducible 的。很多基于 natural loop 的优化无法直接处理这种结构，或者必须先做 node splitting 等规整化。

支配与回边详见 [[dom|支配、后支配与支配边界]]。

## 2. 循环术语

### Header

循环入口，也是 backedge 的目标。所有循环外路径通常应先进入 header。

### Latch

拥有指向 header 的回边的块。一个循环可能有多个 latches。

### Preheader

循环外、紧邻 header 的专用前驱块：

```text
outside -> preheader -> header
```

preheader 通常要求只有 header 一个 successor，且 header 的所有循环外入边都先汇入 preheader。它为循环不变量、初始值和边界计算提供唯一放置点。

### Exiting Block

位于循环内、拥有至少一条离开循环的 successor edge 的块。

### Exit Block

位于循环外、接收循环退出边的块。

### Dedicated Exit

如果 exit block 的所有 predecessors 都来自该循环，则它是该循环的 dedicated exit。这样循环变换可以改写出口而不干扰无关路径。

### Parent Loop 与 Subloop

自然循环可以嵌套。一个 block 可能同时属于内层和外层循环；通常把它归入最内层 loop，并通过 parent relation 表示外层包含关系。

## 3. 常见 Canonical Form

一种常见的规范循环要求：

- 存在 preheader；
- 有唯一 latch；
- loop exits 是 dedicated exits；
- header Phi 清楚区分 preheader incoming 与 latch incoming；
- 循环保持单入口和 reducible；
- 必要的 critical edges 已拆分。

概念结构：

```text
preheader
    |
    v
  header <------ latch
    |              ^
    v              |
   body -----------+
    |
    v
dedicated exit -> continuation
```

规范形不是审美要求，而是接口约束。若每个循环变换都必须独立处理多入口、多 latch、共享 exit 和 critical edge，正确性条件会迅速膨胀。

## 4. Canonicalization 做什么

### 建立 Preheader

若 header 有多个循环外 predecessors，新建 preheader，让所有循环外入边先进入它，再由 preheader 跳到 header。

Header Phi 也要同步重写：原来多个外部 incoming 可能先在 preheader 合并，再作为一个 incoming 进入 header。

### 合并多个 Latches

可新建统一 latch，让原回边先进入统一 latch，再跳向 header。Header Phi 的多个循环内 incoming 也可能需要在新 latch 合并。

### 建立 Dedicated Exits

若一个 exit block 同时有循环内、循环外 predecessors，可以拆出专用 exit，再连接到原 continuation。边界 Phi 必须按照新边重新路由。

### Split Critical Edges

在 edge 上需要插入 Phi copy、状态保存或其他 edge-specific 指令时，拆出单独 block。详见 [[构造|SSA 构造与维护]]。

这些步骤都不仅是 CFG 变形，还必须同时维护 SSA incoming values。

## 5. Loop Rotation

源级 `while` 常先形成 header-tested loop：

```text
preheader -> header(test)
                 | true
                 v
                body -> latch -> header
                 |
               false/exit
```

Loop rotation 把主要条件移到 latch，形成带 guard 的 do-while 形态：

```text
preheader -> guard(test)
              | false -> exit
              v true
             body
              |
             latch(test) --continue--> body
              |
             exit
```

旋转后的重要性质是：一旦通过 guard 进入循环，body 的第一次执行已经确定。这会简化 must-execute 推理、循环内存提升和某些归纳变量变换。

## 6. 零次迭代语义

Rotation 不能消灭原循环的零次路径。Guard 的职责正是保留：

```c
while (condition) {
    body;
}
```

在 `condition` 初始为 false 时不执行 `body`。

因此，把 body 中的操作移到 guard 之前可能改变语义：

- 原本零次循环时不会执行的 load 现在执行，可能访问非法地址；
- 除法可能在原本不执行的路径上除零；
- call、store 或原子操作可能产生新的可观察效果；
- `undef` 或 poison 的传播范围可能扩大。

所有 loop-hoisting、promotion、peeling 和 unrolling 都必须明确处理 `trip count = 0`。

## 7. Must-Execute 与 Safe-to-Speculate

这两个概念不能混为一谈。

### Must-Execute

在给定控制流条件下，操作是否在每次相关循环执行中一定发生。例如 rotated loop 的某些 body block 一旦进入循环就可能 guaranteed execute。

### Safe-to-Speculate

即使原路径不执行该操作，提前执行是否仍不改变可观察行为。纯整数加法通常比 load、除法和 call 更容易安全投机，但仍受 overflow、poison 等 IR 语义影响。

一个操作不是 must-execute，并不代表一定不能移动；若它 safe-to-speculate，可能仍可提前。反过来，一个操作在某路径 must-execute，也不表示可以跨越内存 clobber 或移动到循环外。

## 8. LCSSA

普通 SSA 只要求 value 唯一定义。LCSSA 额外要求：循环内定义、循环外使用的 value 必须先通过 exit Phi 离开循环。

普通 SSA：

```text
loop:
    v = ...
    ...
exit:
    use(v)
```

LCSSA：

```text
loop:
    v = ...
    ...
exiting-edge:
    v.out = phi [v, loop]
exit/after:
    use(v.out)
```

更准确地说，Phi 位于循环 exit block，incoming use 位于相应 exiting edge。

## 9. 为什么 LCSSA 有用

### 集中枚举 Live-Out Values

只需查看 exit Phis，就能知道哪些循环内 values 逃逸到循环外。

### 简化循环复制

Unroll、peel、split、versioning 等变换只需在循环边界重建出口定义，而不必搜索函数中所有远处 uses。

### 简化循环删除

删除整个循环时，可以集中替换 exit Phi，而不是逐一处理任意外部 use。

### 改善嵌套循环所有权

内层循环的 value 先通过内层 exit，外层再决定如何使用，使 loop nesting 上的数据边界更清楚。

## 10. 构造 LCSSA

对循环内定义 `v`：

1. 收集所有循环外 uses；
2. 找到这些 uses 可由哪些 loop exits 到达；
3. 在相关 dedicated exits 放置 Phi；
4. 从每条 exiting edge 传入 `v`；
5. 如果多个 exits 的值在更外部继续汇合，按 dominance frontier 放置 merge Phi；
6. 沿支配树 rename 外部 uses。

LCSSA 构造本质上是一个限制在循环外区域的 SSA 修复问题。

## 11. Nested Loops 与 LCSSA

对嵌套循环：

```text
Outer
  └─ Inner
```

若 `Inner` 中定义的 value 在 `Outer` 中、但 `Inner` 外使用，它首先需要通过 `Inner` 的 exit Phi。若随后还在 `Outer` 外使用，则还要经过 `Outer` 的 exit Phi。

这形成逐层封闭的边界，而不是从最内层定义直接连到函数远处。

## 12. 常见循环变换的结构前提

### LICM

- operands 对循环 invariant；
- 插入 preheader 后仍可用；
- 操作无副作用或内存依赖已证明；
- 若不 guaranteed execute，则需 safe-to-speculate。

### Peeling

复制头部或尾部少量迭代，需要更新 header Phi 的 start/recurrence incoming、退出条件和 live-out values。

### Unrolling

不仅复制 body，还要串接各副本间 recurrence、调整 latch condition、处理 remainder 和 LCSSA exits。

### Loop Split / Versioning

把一个循环分为多个 phase 或 fast/slow path，需要证明迭代域覆盖完整、不重叠或按语义正确重叠，并重建边界 SSA。

### Interchange / Fusion / Fission / Tiling

除了结构规范形，还需要 [[循环依赖分析]] 证明执行顺序变化不会违反 memory dependence。

## 13. 常见错误

- 把“存在 header”当成已经 canonicalized。
- 未检查多 latch 或共享 exit。
- 新增从循环外到非 header 的边，制造多入口 SCC。
- rotation 后把 guard 前移操作，破坏零次迭代语义。
- 把 must-execute 与 safe-to-speculate 混为一谈。
- 循环复制只更新 CFG，没有重建 header Phi recurrence。
- 循环内 value 直接在循环外使用，破坏 LCSSA。
- 只处理单个 exit，遗漏其他退出路径。
- 内层循环变换错误地把外层 invariant 当成全函数 invariant。

## 14. 关系速记

```text
[[dom|Dominance]]
  -> natural loop / backedge
  -> preheader / latch / dedicated exits
  -> canonical loop
  -> rotation + zero-trip guard
  -> [[SCEV]] / memory / dependence proofs
  -> loop transforms

[[构造|SSA]]
  -> LCSSA
  -> 集中管理 loop live-out values
```
