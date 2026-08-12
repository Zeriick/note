# 支配、后支配与支配边界

CFG（Control-Flow Graph，控制流图）描述程序的控制可能如何在 basic blocks 之间转移。支配（dominance）则把“所有到达某点的执行都必须经过哪里”变成可查询的结构，是 [[构造|SSA 构造]]、代码移动、循环识别、控制依赖和部分冗余消除的共同基础。

## CFG 的节点与边

一个 basic block 是一段单入口、内部没有控制分叉的指令序列，末尾的 terminator 决定后继。CFG 中：

- 节点通常是 basic blocks；
- 边 $A\rightarrow B$ 表示执行可能从 $A$ 的 terminator 转移到 $B$；
- 从 entry 可达的图路径是可能的控制顺序的过度近似，但不保证路径条件在数据语义上可同时满足。

## Dominance

对从 entry 可达的节点 $A$ 和 $B$，如果每条从 entry 到 $B$ 的路径都经过 $A$，则称 $A$ 支配（dominate）$B$。若 $A$ 支配 $B$ 且 $A\neq B$，则称 $A$ 严格支配 $B$。$B$ 的 immediate dominator 是最靠近 $B$ 的严格支配者；所有 immediate-dominator 关系组成 dominator tree。

支配关系让编译器可以安全地说：如果 $A$ 中的条件在到达 $B$ 前必然成立，那么该事实可能用于化简 $B$。但仍需根据分支方向和指令语义判断具体是哪个 predicate 成立。

例如：

```text
       entry
       /   \
      L     R
       \   /
        merge
          |
         exit
```

- `entry` 支配所有从入口可达的节点。
- `L` 不支配 `merge`，因为可以从 `R` 到达 `merge`。
- `merge` 支配 `exit`。

支配是自反关系：任意可达节点都支配自己。若只讨论不相等节点之间的关系，应明确使用“严格支配”。

## Immediate Dominator 与支配树

对入口以外的可达节点 `B`，它的 immediate dominator（直接支配者，记作 `idom(B)`）是严格支配 `B`、且距离 `B` 最近的节点。

把每个节点连到它的 `idom`，得到 dominator tree。它满足：

- `A` 支配 `B`，当且仅当 `A` 是支配树中 `B` 的祖先；
- `B` 的全部严格支配者，就是从 `idom(B)` 沿父指针一直到根的节点。

可以对支配树做深度优先遍历，为每个节点记录进入和离开编号。此时：

```text
pre[A] <= pre[B] && post[B] <= post[A]
```

成立，当且仅当 `A` 是 `B` 的祖先，也就是 `A` 支配 `B`。

### 指令级支配

若 definition 和 use 位于不同基本块，定义所在块通常必须支配使用所在块。若二者位于同一块，还需要 definition 出现在 use 之前。

Phi 的 incoming use 位于对应 predecessor edge 上，而不是简单位于 Phi 所在块开头，所以应按边判断 incoming value 是否可用。

## 支配关系的计算

### 数据流方程

最直接的定义式是：

$$
Dom(entry)=\{entry\},
$$

$$
Dom(B)=\{B\}\cup\bigcap_{P\in pred(B)}Dom(P).
$$

反复迭代到不动点即可得到支配集合。这个算法概念清晰，适合解释定义和处理小图。

### Immediate-dominator 算法

工程实现通常直接求 `idom`：

- Cooper–Harvey–Kennedy 算法按 reverse postorder 迭代，简单且在许多现实 CFG 上很快；
- Lengauer–Tarjan 算法使用 DFS tree、semi-dominator、link/eval 和路径压缩，在大图上具有更好的复杂度。

理解 Lengauer–Tarjan 时要区分 DFS parent、semi-dominator 和 immediate dominator。DFS tree 只是搜索结构，不能当成 dominator tree；semi-dominator 也是计算 `idom` 的中间量，不是最终父节点。

## Dominance frontier

节点 $A$ 的 dominance frontier 是这样一组节点 $B$：$A$ 支配 $B$ 的某个 predecessor，但 $A$ 不严格支配 $B$。直观上，它是“由 $A$ 控制的路径开始与其他路径合流”的边界，因此是 [[构造|SSA 构造]] 中放置 Phi 节点的核心工具。

在前面的 diamond 中，`merge` 同时属于 `DF(L)` 和 `DF(R)`。

一种常见计算方法是：对 predecessor 数量至少为 2 的块 `B`，遍历每个 predecessor `P`：

```text
runner = P
while runner != idom(B):
    DF(runner).add(B)
    runner = idom(runner)
```

### Iterated Dominance Frontier

若变量在多个块中定义，Phi 不一定只放在这些定义块的一阶 dominance frontier。新放置的 Phi 本身也是新定义，它的 frontier 可能继续需要 Phi，因此使用 iterated dominance frontier（IDF）：

```text
worklist = 变量的定义块
while worklist 非空:
    D = pop(worklist)
    for Y in DF(D):
        if Y 尚未放置 Phi:
            在 Y 放置 Phi
            push(Y)
```

Pruned SSA 还会加入 live-in 条件，避免在变量已不活跃的 frontier 上放置无用 Phi。

## Post-dominance

支配从入口看路径；后支配（post-dominance）从出口看路径。

若从节点 `A` 到任意正常出口的每条路径都经过节点 `B`，则称 `B` 后支配 `A`。直觉上，执行一旦到达 `A`，在正常终止前就不可避免地到达 `B`。

函数可能有多个 `return`。一种统一求解方式是：

1. 反转 CFG 边；
2. 创建 virtual exit；
3. 让 virtual exit 连接所有真实出口；
4. 从 virtual exit 计算普通 dominance。

得到的 immediate dominator 在原图语义下就是 immediate post-dominator。

必须明确“出口”的语义。无限循环、异常退出和不可返回调用是否算作出口，会改变后支配结果。

## 控制依赖

控制依赖描述某个分支结果是否决定另一个节点会不会执行，它与 post-dominance 密切相关。

粗略地说，如果 `B` 后支配 `A` 的某个 successor，但不后支配 `A`，那么 `B` 的执行依赖于 `A` 选择了哪条边。控制依赖常用于程序切片、if-conversion、谓词化和代码放置。

## 典型用途

- [[构造|SSA 构造]]：IDF 放置 Phi，dominator tree 驱动 rename。
- 自然循环：若 `header` 支配回边源 `latch`，`latch -> header` 是自然循环回边候选。
- LICM：移动后的定义仍需支配所有 uses；是否可安全投机还需额外证明。
- [[GVN、FRE 与 PRE]]：等价表达式只有在可用定义支配当前位置时才能直接复用。
- [[Jump Threading]]：新增边可能改变支配关系，甚至把单入口循环变成 irreducible CFG。
- Shrink wrapping：保存点通常要支配资源使用，恢复位置要覆盖所有离开区域的路径。

## CFG 与 PAG

[[PAG|Pointer Assignment Graph（PAG）]] 和 CFG 回答不同问题：

| CFG | PAG |
|---|---|
| 节点通常是 basic blocks | 节点通常是 pointer values 和 abstract objects |
| 边表示可能的控制转移 | 边表示指针值或内存约束 |
| 可达性与顺序、分支、循环有关 | 合法路径与 copy/load/store/field/call/return 语义有关 |

CFG path 本身不说明 pointer value 沿该路径传播，PAG path 也不必然对应一条可实现的控制路径。flow-sensitive pointer analysis 会把 points-to state 与 CFG 程序点关联，并沿 CFG 边执行 transfer/merge；path-sensitive analysis 还会区分分支条件。flow-insensitive PAG 则通常忽略指令顺序，可能合并来自互斥路径的 facts。

## 边界与常见误区

- 基本块在源码、数组或汇编中的排列顺序不表示支配关系。
- “大多数路径经过”不是支配；必须是所有相关路径。
- 从入口不可达的块不能直接套用可达区域的支配结论。
- DFS tree、loop nesting tree 和 dominator tree 是不同结构。
- `A` 支配 `B` 不代表 `B` 后支配 `A`。
- 块级支配不足以回答同一块内的指令顺序。
- Phi operand 是 edge use，应按相应 predecessor edge 判断可用性。
- dominance 只证明路径覆盖，不证明操作无副作用、不会 trap 或值得移动。
- CFG 改变后，原有支配关系不再自动成立。

## 关系速记

```text
CFG
 ├─ dominance
 │   ├─ dominator tree
 │   ├─ dominance frontier
 │   ├─ SSA Phi placement
 │   ├─ natural-loop backedge
 │   └─ value availability
 └─ reversed CFG + virtual exit
     └─ post-dominance
         ├─ control dependence
         └─ exit/resource placement
```
