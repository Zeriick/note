# GVN、FRE 与 PRE

Global Value Numbering（GVN，全局值编号）、Full Redundancy Elimination（FRE，完全冗余消除）和 Partial Redundancy Elimination（PRE，部分冗余消除）都在处理重复计算，但回答的问题不同：

```text
GVN：两个 expressions 是否等价？
Availability：某个等价 value 在当前位置是否可用？
FRE：当前计算是否在所有到达路径上都已有可用结果？
PRE：当前计算是否只在部分路径上已有，能否通过插入变成完全冗余？
```

把这些问题混为一谈，最常见的错误就是“表达式等价，所以直接复用”，却忽略已有 definition 不支配当前 use。

## 1. Value Numbering

Value numbering 为语义等价的 expressions 分配同一编号或同一等价类。

```text
a = x + y
b = x + y
```

若两次计算语义相同，则：

```text
VN(a) = VN(b)
```

后续优化可以选择用一个 value 替换另一个，但还需要判断 dominance 和 availability。

## 2. Local Value Numbering

Local Value Numbering（LVN）只在一个 basic block 内工作。按顺序扫描 instructions，维护 expression key → value number：

```text
key(add, VN(x), VN(y)) -> VN(a)
```

同一 block 中前一 instruction 天然位于后一 instruction 之前，因此 availability 比全局情况简单。

LVN 常处理：

- common subexpression；
- copy propagation；
- constant folding；
- algebraic identities；
- commutative operand canonicalization。

## 3. Global Value Numbering

GVN 跨 basic blocks 建立值等价关系。它需要同时考虑：

- CFG 和 [[dom|支配]]；
- Phi 的 congruence；
- expression canonicalization；
- memory state；
- call effects；
- loop recurrence；
- unreachable paths。

GVN 可以基于 dominator tree 遍历、SSA congruence closure、partition refinement 或其他等价类算法实现。

## 4. Expression Key

一个纯 expression 的 key 通常包括：

```text
opcode
operand value numbers
result type
semantic flags
payload / layout
```

### Commutative Operation

对 `add`、`mul`、`and` 等可交换操作，应规范化 operands：

```text
key(add, min(VN(x),VN(y)), max(...))
```

否则 `x+y` 与 `y+x` 会得到不同 key。

### Flags 也是语义

`nsw add`、普通 wrapping add、saturating add 并不一定等价。Fast-math flags、rounding mode、exception behavior、alignment 和 address space 也可能属于 key。

### Type 不能省略

相同 bit pattern 在不同整数宽度、浮点类型或 pointer 语义下可能不是同一个 expression。

## 5. Phi 与 Congruence

```text
x = phi [a, P1], [b, P2]
y = phi [c, P1], [d, P2]
```

如果在每条对应入边上：

```text
VN(a) = VN(c)
VN(b) = VN(d)
```

那么 `x` 与 `y` 可能属于同一 congruence class。

但 incoming 必须按相同 CFG edges 对齐。只比较 Phi operands 的无序集合是不够的。

循环 Phi 还可能表示 recurrence。两个初值和 step 等价的归纳变量，可借助 [[SCEV]] 证明等价；普通 GVN 未必能直接识别所有递推等式。

## 6. Memory Expressions

Load 不能只用 pointer value number 作为 key：

```c
x = *p;
*q = 1;
y = *p;
```

即使 `p` 没变，只要 `q` 可能 alias `p`，第二次 load 就可能读到不同值。

更可靠的 key/查询需要包括：

- memory location；
- access type/size；
- reaching memory version；
- alias result；
- call ModRef effects；
- volatile/atomic 语义。

详见 [[内存优化证明：Alias Analysis 与 MemorySSA]]。

Call 只有在具有足够强的纯函数、只读或 effect summary 时，才可像普通 expression 一样编号。未知 call 通常不能仅按 callee 和 arguments 合并。

## 7. Equivalence 不等于 Availability

考虑：

```text
        entry
        /   \
       L     R
       |     |
   a=x+y     |
       \     /
        merge
          |
      b=x+y
```

GVN 可以证明 `a` 与 `b` expression 等价，但 `a` 不支配 `merge`，因为从 `R` 路径到达时 `a` 不存在。因此不能直接把 `b` 替换为 `a`。

Availability 需要回答：

- 是否存在等价 definition；
- 它是否支配当前位置；
- 沿所有路径是否保持有效；
- 对 memory expression，中间是否有 clobber；
- 对可能 trap 的 expression，复用是否改变执行路径。

## 8. Full Redundancy Elimination

如果当前 expression 在所有到达路径上都已有一个可用等价值，它是 fully redundant。

最简单情形：

```text
a = x + y
...
b = x + y
```

`a` 支配 `b`，且 `x/y` 对应 values 未变，则可以：

```text
replace uses(b, a)
erase b
```

若不同 predecessor 各有一个等价 definition，也可以在 merge 处用 Phi 合并它们，再消除当前 expression：

```text
L: a = x + y
R: c = x + y
merge:
    v = phi [a,L], [c,R]
    b = x + y   ; 可替换为 v
```

## 9. Partial Redundancy

若 expression 只在部分路径上已有，它是 partially redundant：

```text
        entry
        /   \
       L     R
       |     |
   a=x+y     |
       \     /
        merge
          |
      b=x+y
```

在 `L` 路径上 `b` 冗余，在 `R` 路径上不冗余。PRE 可以在 `R` 插入：

```text
R:
    c = x + y
```

然后在 merge 建 Phi：

```text
v = phi [a,L], [c,R]
```

最终用 `v` 替换 `b`。

PRE 的目标不是盲目减少静态 expressions，而是在不改变语义且值得的情况下，把 partial redundancy 转成 full redundancy。

## 10. Anticipatability 与 Availability

经典 PRE 同时考虑：

### Available

从入口到当前位置的每条路径上，expression 是否都已计算且未被 kill。

### Anticipatable

从当前位置到出口的每条相关路径上，expression 是否都会在 operands 被改变前计算。

可用性是前向性质，anticipatability 是后向性质。Lazy Code Motion（LCM）利用二者把计算放到足够早以消除冗余、但又尽可能晚以避免无用投机的位置。

## 11. SSA-PRE

在 SSA 上，PRE 可以把 expression occurrence 视作一种需要构造 SSA 的“虚拟变量”：

1. 收集 real occurrences；
2. 在 iterated dominance frontier 放置 Phi-like merge；
3. 沿支配树 rename occurrences；
4. 对每条入边做 Phi translation；
5. 判断哪些 operands 缺失；
6. 在必要 edges/blocks 插入 computations；
7. 用合并 value 替换 redundant occurrences。

### Phi Translation

若 merge block 中 expression 使用 Phi result：

```text
x = phi [a,P1], [b,P2]
y = x + 1
```

把 `x+1` 翻译到 `P1` edge 得 `a+1`，翻译到 `P2` edge 得 `b+1`。这使 PRE 能判断各入边是否已有等价 expression，或需要插入什么。

Phi translation 必须尊重 edge identity 和 value availability。

## 12. Critical Edge

若 computation 必须只在某条 `P -> S` edge 执行，而 `P` 有多个 successors、`S` 有多个 predecessors，则需要 split critical edge，再在 split block 中插入。

否则：

- 放在 `P` 会让其他 outgoing paths 也执行；
- 放在 `S` 会让其他 incoming paths 也执行。

Edge splitting 与 Phi 维护详见 [[构造|SSA 构造与维护]]。

## 13. Safe Speculation

PRE 会把 expression 插入原本没有计算它的路径，因此必须判断是否 safe-to-speculate。

风险包括：

- integer division by zero；
- `INT_MIN / -1` 等异常/overflow 边界；
- invalid load；
- volatile/atomic access；
- call effect；
- floating exception/rounding；
- poison、undef 和 no-wrap 语义；
- 增加原本不存在的非终止计算。

“没有 side effect”不自动等于“安全投机”。可能 trap 的纯计算仍会改变程序行为。

## 14. Profitability

PRE 可能减少动态重复计算，也可能：

- 增加静态 code size；
- 在冷路径插入不值得的计算；
- 延长 live range；
- 增加 register pressure 和 spill；
- 破坏 rematerialization；
- 使 address mode 变差。

成本应结合：

- block/edge frequency；
- expression cost；
- 插入次数与消除次数；
- live-range extension；
- target instruction forms；
- code-size budget。

详见 [[寄存器分配]] 和 [[指令选择与合法化]]。

## 15. 与相邻优化的区别

### Common Subexpression Elimination（CSE）

CSE 通常指发现并消除重复 expression；GVN 是建立等价关系的一种全局方法，FRE/PRE 是利用这些关系进行改写的策略。

### Copy Propagation

处理 `b = copy a`，不一定需要 expression equivalence。

### LICM

把 loop-invariant computation 移到循环外，关注循环执行次数与不变量；PRE 关注路径上的部分冗余。二者可能产生相似代码移动，但证明和 placement 不同。

### [[CVP]] / [[整数范围分析]]

它们用路径 predicate 或取值范围证明 expression/branch 可简化，不只依赖 syntactic expression key。

### [[IPSCCP]]

它利用 SSA lattice 与 executable edges 传播常量和不可达性；GVN 则更广泛地建立非恒定 expressions 之间的等价。

## 16. 常见误区

- Value number 相同就直接替换，忽略 dominance。
- 只按 opcode 和 operands 构 key，遗漏 type、flags、layout。
- 对 load 忽略 memory version 和 clobber。
- 把 unknown call 当成 pure expression。
- Phi congruence 忽略 incoming edges 的对应关系。
- PRE 在新路径插入可能 trap 的 expression。
- 在 critical edge 上把 computation 放到错误 block。
- 只计算减少的 expressions，不看新增计算、live range 和 spill。
- 把静态等价与某程序点的 value availability 混为一谈。

## 17. 关系速记

```text
Expression canonicalization
  -> value number / congruence class
  -> [[dom|dominance]] + availability
      ├─ all paths available -> FRE
      └─ some paths available -> PRE
           -> edge insertion + Phi translation

Memory expression
  -> [[内存优化证明：Alias Analysis 与 MemorySSA|MemorySSA / clobber proof]]

Profitability
  -> code size + frequency + [[寄存器分配|register pressure]]
```
