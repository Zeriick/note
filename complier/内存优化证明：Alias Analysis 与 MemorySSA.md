# 内存优化证明：Alias Analysis 与 MemorySSA

标量 SSA 能直接回答“这个 value 由哪条指令定义”，却不能直接回答“这次 load 读到哪次 store 的结果”。原因是多个 pointer values 可能指向同一内存位置，call 也可能通过不可见的别名读写内存。

内存优化通常需要一条分层证明链：

```text
Pointer Value
-> Memory Object / Base
-> Object-relative Location / Region
-> Alias result
-> Call ModRef effect
-> Memory partition
-> MemorySSA version
-> Clobber query
-> Transformation
```

[[Alias Analysis]] 单独整理 alias query、points-to 与精度维度；本文保留必要定义，重点解释这些结论如何与 call effects、MemorySSA 和 clobber query 组成完整内存证明。

## 1. 为什么普通 SSA 不够

```c
*p = 7;
x = *q;
```

即使 `p` 和 `q` 是不同 SSA values，也不能断言 load 与 store 无关。它们可能：

- 指向同一对象的同一位置；
- 指向同一对象的重叠区间；
- 指向不同对象；
- 因信息不足而无法判断。

因此，内存依赖不能只比较 pointer identity。

## 2. Memory Object、Location 与 Region

### Memory Object

Memory object 表示一块抽象存储来源，例如：

- global object；
- stack allocation；
- heap allocation site；
- pointer parameter 所代表的外部对象；
- unknown object。

对象身份回答“可能属于哪块分配”，但还没有回答对象内的具体位置。

### Memory Location

一次具体访问至少需要：

```text
(base object, byte offset, access size, alignment, address space, qualifiers)
```

例如 `int32 a[10]` 中的 `a[3]` 可描述为：

```text
object = a
offset = 12 bytes
size   = 4 bytes
```

### Memory Region

当 offset 或 size 不是精确常量时，可使用区间或符号区域：

```text
[offset, offset + size)
```

Region 适合表示数组切片、循环 footprint 和函数参数相对访问范围。

## 3. 地址规范化

同一地址可能有许多等价写法：

```text
base + (i * 4) + 8
(base + 8) + (i << 2)
gep(base, i + 2)
```

若每个优化各自匹配语法形状，会漏掉大量等价地址。常见做法是先规范化为：

```text
base object
+ invariant byte offset
+ Σ(coefficient_k * symbolic_index_k)
```

规范化应保留：

- pointer provenance 或对象来源；
- element/layout 与 byte offset 的关系；
- access size；
- 整数扩展、截断和 overflow 语义。

[[SCEV]] 常用于表达循环相关的地址递推。

## 4. Alias Analysis 的回答

不同系统使用的枚举略有差异，常见结果包括：

- `NoAlias`：两次访问一定不重叠；
- `MayAlias`：可能重叠，或信息不足；
- `MustAlias`：确定指向同一位置；
- `PartialAlias`：确定存在部分重叠，但不是完全相同位置。

这些结果不是简单的“投票”。组合多个 alias providers 时，只能在有证明的情况下增强结论。无法证明 `NoAlias` 必须保留为 `MayAlias`，不能把 unknown 当成独立对象。

### 常见证明来源

- 不同且不逃逸的 allocation objects；
- 对象相对 offset 和 access size 不重叠；
- 地址表达式差值可证明非零或区间分离；
- 类型或 layout 规则提供的补充约束；
- 调用点证明两个形式参数来自不同实际对象；
- 语言显式提供的 `restrict`、noalias 等契约。

类型信息通常只能作为证据之一。类型相同不表示 alias，类型不同也不总能安全推出 no-alias，具体取决于语言和 IR 的别名规则。

## 5. 参数别名

```c
void f(int *p, int *q) {
    *p = 1;
    x = *q;
}
```

不同形式参数 `p`、`q` 默认可能接收同一个实际参数：

```c
f(a, a);
```

所以“参数名字不同”或“ParamObject 不同”不能自动推出 `NoAlias`。只有调用约定、语言属性或调用点分析提供额外事实时，才能增强结论。

## 6. Call Effect 与 ModRef

对某个 location，call 的效果常抽象为 ModRef：

- `NoModRef`：既不读也不写；
- `Ref`：可能读，不写；
- `Mod`：可能写，不读；
- `ModRef`：可能读也可能写。

考虑：

```c
*p = 7;
f(q);
x = *p;
```

要把 `x` 替换为 7，至少要证明 `f(q)` 对 `p` 所在 location 是 `NoMod`。这通常需要：

1. callee summary 描述它读写哪些 globals 和 parameters；
2. 在 callsite 把 formal objects 映射到 actual objects；
3. 用 alias analysis 判断 actual regions 是否可能与 `p` 重叠；
4. 确认 callee 不写 unknown memory。

未知外部调用通常必须保守地视为可能读写可达内存，除非 intrinsic 或函数属性提供更强契约。

## 7. Memory Partition

给整个函数建立单一 memory state 最保守，但每次 store 都可能阻断所有 loads；为每个字节建立独立状态又不可行。

Memory partition 在两者之间折中：把可能交互的 objects/regions 放在同一 partition，把能证明独立的访问分开。

```text
Partition A: local object x
Partition B: global g + may-alias parameter p
Partition C: unrelated local array y
```

Partition 过粗会降低优化精度；过细若缺乏 alias 证明则会不安全。构建时常使用 union-find 合并可能相互影响的 objects。

## 8. MemorySSA 的节点

MemorySSA 为每个 partition 建立类似 SSA 的内存版本链。常见节点包括：

### LiveOnEntry

函数入口时未知的初始内存状态。

### MemoryDef

写入或可能 clobber 内存的操作，产生新 memory version：

```text
M1 = MemoryDef(store p, defining=M0)
```

### MemoryUse

只读取内存的操作，引用它观察到的 reaching memory version：

```text
MemoryUse(load p, defining=M1)
```

### MemoryPhi

多条 CFG 路径的 memory versions 在合流点合并。

### MemoryExit

表示函数出口对当前 memory state 的可观察需求。它有助于判断某些写是否能在函数结束前删除。

MemorySSA 通常是分析图，而不是普通指令列表中的真实 opcodes。

## 9. MemorySSA 的构造

构造过程与标量 SSA 类似：

1. 为每条内存指令确定影响的 partitions；
2. 将 store、可能写内存的 call 等视为 definitions；
3. 根据 definitions 的 dominance frontier 放置 MemoryPhi；
4. 沿 dominator tree rename 当前 memory version；
5. load/只读 call 记录 reaching version；
6. store/写 call 产生新 version；
7. 在 CFG edges 上连接 MemoryPhi incoming；
8. 为可观察出口连接 MemoryExit。

MemorySSA 版本只说明“候选 reaching state”，并不自动证明某个 `MemoryDef` 精确覆盖目标 location。Partition 内仍可能包含多个 may-alias regions，因此查询通常还要结合 alias 和 location identity。

## 10. Diamond 示例

```c
if (c)
    *p = 1;
else
    *p = 2;
x = *p;
```

概念 MemorySSA：

```text
LiveOnEntry M0

then: M1 = MemoryDef(store p, 1, defining=M0)
else: M2 = MemoryDef(store p, 2, defining=M0)

merge: M3 = MemoryPhi(M1, M2)
       MemoryUse(load p, defining=M3)
```

这个 load 不能直接替换成某一条 store 的 value，因为两条路径写入不同值。若要消除 load，需要额外构造标量 Phi：

```text
v = phi [1, then], [2, else]
```

如果 else 不写 `p`，MemoryPhi 会合并 `M1` 和 `M0`，更不能假设所有路径都看到 1。

## 11. Clobber Query

Clobber query 从某次 load 或 store 的 defining memory version 向前追踪，寻找最近的相关写入或边界。

查询要区分：

- 精确同址；
- 部分重叠；
- may-alias；
- 普通 CFG merge；
- loop-carried merge；
- cycle；
- LiveOnEntry；
- unknown call/barrier。

在 MemoryPhi 处，必须合并所有相关 incoming paths。找到“一条前驱路径上的 store”不表示这条 store 支配并覆盖所有路径。

循环 header 的 MemoryPhi 还会连接前一迭代的 state。若查询跨过 loop-carried edge，必须避免把“本次迭代的精确同址”错误提升为跨迭代的精确结论。

## 12. 常见内存优化及证明义务

### Store-to-Load Forwarding

```c
*p = v;
x = *p;
```

用 `v` 替换 load 至少需要：

- store 与 load 精确同址且类型/layout 兼容；
- 中间没有可能覆盖该位置的 clobber；
- store 的 value 在 load 位置可用；
- 所有到达 load 的路径都满足同一结论。

### Redundant Load Elimination

复用前一个 load 需要证明两次 load 之间没有可能修改目标位置的 store/call，并且前一 load 的 value 支配当前位置。

### Redundant Store Elimination

如果同一 memory state 下对同一位置重复写入同一值，后一次或前一次可能冗余；具体删除哪一次取决于可观察顺序和中间 reads。

### Dead Store Elimination

删除旧 store 需要证明所有未来可观察路径在读取旧值前都会被完整覆盖，或对象在退出时不可观察。只在一条路径找到覆盖 store 不够。

### Loop Memory Promotion

把循环内固定 location 提升为 loop-carried scalar SSA，通常需要：

- 地址对循环 invariant；
- 其他访问不 alias；
- call 不 clobber；
- preheader seed load 不破坏零次语义；
- header Phi 携带每轮状态；
- 所有 exits 写回最终值；
- write-only 情况下能证明首次读取前必写。

## 13. Footprint 与循环区域

循环访问通常不是单一 location，而是迭代域上的 footprint：

```text
object
+ affine byte offset(iteration variables)
+ access size
+ iteration domain
```

例如：

```c
for (i = 0; i < n; ++i)
    a[i] = 0;
```

footprint 大致为 `a` 中 `[0, 4n)` 的字节区域。要证明一次写覆盖另一组访问，需要同时证明对象一致、索引域覆盖和单次访问宽度正确。

这与 [[循环依赖分析]]、状态覆盖、循环 DSE 和跨函数 region summary 相关。

## 14. Volatile、Atomic 与并发

普通 MemorySSA/alias 优化不能自动跨越：

- volatile access；
- atomic ordering；
- fence；
- signal/interrupt 可见状态；
- 数据竞争下由语言内存模型定义的行为。

Atomic 是否可以相互重排、合并或消除，取决于具体 memory order 和语言规范。把它们当作普通 load/store 是不安全的。

## 15. Unknown 必须保持保守

下列情况通常应返回 unknown 或 `MayAlias`：

- base object 无法识别；
- 参数别名无法排除；
- pointer arithmetic 无法规范化；
- access size/layout 不完整；
- call effect 未知；
- 路径合流无法得到统一 reaching clobber；
- affine/region solver 超出能力或预算；
- 可能发生整数溢出或越界。

证明失败只表示当前方法无法推出结论，不表示程序中确实存在 alias 或 dependence。优化可以因此保守拒绝，但不能把未知解释为安全。

## 16. 与 PAG 的关系

[[PAG|Pointer Assignment Graph（PAG）]] 关注 pointer values 和 abstract objects 之间的传播、points-to 与 alias 约束；MemorySSA 关注 CFG 上内存状态的版本与 clobber 链。

二者可以互相提供信息，但不是同一种图：

```text
PAG / points-to
  -> 哪些 objects 可能被 pointer 指向
  -> 帮助 Alias Analysis

Alias + CFG + effects
  -> Memory partitions / MemorySSA
  -> reaching clobber
  -> memory transformations
```

## 17. 常见误区

- pointer identity 不同就判 `NoAlias`。
- 两个形式参数不同就假设独立。
- 同一 base object 就认为精确同址。
- 忽略 access size，错过 partial overlap。
- 找到一条路径上的 store 就做 forwarding。
- call 没有显式 store operand，就认为不写内存。
- MemorySSA 的 defining access 就一定是精确覆盖目标的 store。
- `MayAlias` 被当成 `NoAlias`。
- 循环内提升 seed load 时忽略 zero-trip path。
- 区域 solver 返回 unknown，却把它当成“不重叠”。
