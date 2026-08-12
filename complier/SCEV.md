# SCEV: Scalar Evolution

Scalar Evolution（标量演化，通常简称 SCEV）是一类以 LLVM SCEV 最具代表性的循环分析方法，用来回答：

> 一个标量值在循环中是如何变化的？

它最常用来分析循环里的变量、数组下标、指针偏移和循环次数。很多循环优化不会各自从头推导这些性质，而是共享一套递推表达式和边界推理。

## 核心表示

SCEV 会把 IR 里的标量表达式抽象成一套数学表达式。最重要的是 `AddRec`，也就是 add recurrence：

```text
{start,+,step}<L>
```

它表示在循环 `L` 中：

- 第 0 次迭代是 `start`。
- 每次迭代增加 `step`。
- 第 `k` 次迭代在相应整数语义允许时为 `start + k * step`。

例如：

```cpp
for (int i = 0; i < n; ++i) {
    a[i] = i;
}
```

在循环里，`i` 可以表示成：

```text
{0,+,1}<L>
```

如果 `int` 是 4 字节，那么 `&a[i]` 的地址偏移可以近似理解成：

```text
base(a) + {0,+,4}<L>
```

这类线性形式对循环优化非常关键。

## 常见 SCEV 形式

常见表达式包括：

- `SCEVConstant`：常量。
- `SCEVUnknown`：SCEV 无法继续理解的值，比如某些复杂 load、call 结果。
- `SCEVAddExpr` / `SCEVMulExpr`：普通加法、乘法表达式。
- `SCEVAddRecExpr`：循环递推表达式，是分析 induction variable 的核心。
- `SCEVMinMaxExpr`：某些 min/max 形式，用于表达边界。
- `SCEVZeroExtendExpr` / `SCEVSignExtendExpr` / `SCEVTruncateExpr`：整数位宽转换。

不同实现支持的节点集合不同。例如轻量实现可能只提供 Constant、Add、Mul、AddRec、UDiv 和 Unknown，而不直接表示全部 min/max 与 cast。消费者必须以实际表达能力为准。

## 表达式规范化与唯一化

如果 `a + b`、`b + a` 和 `(a + 0) + b` 被保留为不同表达式，SCEV 很难稳定比较它们。常见实现会做：

- 常量折叠；
- 加法、乘法扁平化；
- 可交换 operands 排序；
- 消除单位元；
- 合并相同项；
- recurrence 规范化；
- expression interning/uniquing。

Uniquing 让结构等价表达式共享 canonical node，便于快速比较和缓存。但它只表示在 SCEV 语义中的等价，不自动说明某个现成 SSA definition 在当前程序点可用；代码复用还需要 [[dom|支配]] 和 availability。

`Unknown(Value)` 通常按底层 SSA value 的 identity 区分。两个 unknown 即使打印名称相同，也不能在没有额外证明时合并。

## 从 SSA Phi 识别归纳变量

典型 basic induction variable：

```text
preheader:
    i.start = 0

header:
    i = phi [i.start, preheader], [i.next, latch]

latch:
    i.next = i + step
```

识别时需要：

- Phi 位于 loop header；
- 一个 incoming 来自循环外初值；
- 一个 incoming 来自 latch recurrence；
- recurrence value 能规约为 `i + step`；
- `step` 对当前循环 invariant；
- 类型和 overflow 语义允许相应推导。

得到：

```text
i = {i.start,+,step}<L>
```

循环规范形和 header/latch 术语详见 [[循环规范形与 LCSSA]]。

## 高阶与嵌套递推

二项 AddRec：

```text
{a,+,b}<L>
```

表示线性递推。三项形式：

```text
{a,+,b,+,c}<L>
```

可以表达二阶差分为常量的二次演化。直观上，value 的一阶 step 本身也在递推。

嵌套循环中，表达式可以包含相对不同 loops 的 AddRec。必须明确每个 recurrence 属于哪一层循环；内层计算中 loop-invariant 的 value，可能仍随外层循环变化。

## 能回答的问题

SCEV 常用来回答：

- 这个值是不是 induction variable？
- 每次循环迭代它增加多少？
- 某个数组下标是不是线性增长？
- 两个指针访问之间的距离是否固定？
- 循环的 backedge taken count 或 trip count 能不能推出来？
- 某个加法、乘法是否可以证明不会溢出？

这些答案会被很多循环优化使用，例如：

- `IndVarSimplify`：归一化/简化归纳变量。
- `LoopStrengthReduce`：把乘法或复杂地址计算转成递增形式。
- loop unroll/vectorize：估计循环次数、判断访问模式。
- bounds check elimination：证明某些访问一定在范围内。
- dependence analysis：判断不同迭代之间是否可能访问同一内存位置。

## Loop Disposition

相对某个循环 `L`，SCEV expression 常被分类为：

- loop invariant：在 `L` 的所有迭代中不变；
- loop computable：可由 `L` 的 recurrence 表达；
- loop variant/unknown：无法用当前模型安全描述；
- available at entry：不仅 invariant，而且组成 expression 的 values 在循环入口可用。

“Loop invariant”不等于“可以放到 preheader”。若底层 definition 不支配 preheader，或物化会引入新的 trap/effect，就不能直接外提。

## trip count 和 backedge taken count

SCEV 里经常会区分两个概念：

- `trip count`：循环 body 实际执行多少次。
- `backedge taken count`：循环回边被执行多少次。

如果循环至少执行一次，通常有：

```text
trip count = backedge taken count + 1
```

但如果循环可能 0 次执行，就必须把“是否进入循环”的条件也考虑进去。很多 off-by-one 的理解错误都来自这里。

### Exit count

还应区分某个 exiting block 对应的 exit count。多出口循环可能为每个 exiting edge 推导不同退出次数，而整个循环的 trip count 取决于最早实际退出路径。

### 继续条件的方向

需要先把 branch 解释成“何时继续循环”，而不是只看 compare opcode。例如：

```text
if (i < n) goto body else exit
```

和：

```text
if (i >= n) goto exit else body
```

语义相同，但 true/false edge 相反。

### 不同 Predicate

`<`、`<=`、`>`、`>=`、`!=` 的终止条件不同。以正 step 为例：

- `< bound` 与 `<= bound` 有一位边界差；
- `!= bound` 还要证明 recurrence 能精确到达 bound；
- step 为负时不等式方向相反；
- step 为 0 时循环可能不终止；
- 机器整数 wrap 可能让数学上单调的序列重新绕回。

因此不能只凭源码看起来像 `for (i=0; i<n; ++i)` 就套用公式。应从当前 [[构造|SSA]] recurrence、continue edge 和整数语义推导。

## 在某次迭代求值

对 affine AddRec：

```text
R = {start,+,step}<L>
```

第 `k` 次迭代的数学形式为：

$$
R(k)=start+k\cdot step.
$$

高阶 AddRec 可用有限差分和二项式系数求值。求值仍必须遵守位宽、符号扩展与 overflow 语义，不能无条件用无限精度整数替代机器整数。

## no-wrap 信息

SCEV 很依赖整数溢出语义。一个递推式如果可以证明 `nuw`、`nsw` 或更一般的 no-wrap 性质，那么优化空间会大很多。

例如：

```cpp
for (unsigned i = 0; i < n; ++i)
```

如果能证明 `i + 1` 在循环内不会绕回 0，那么 SCEV 可以更放心地推导循环次数和范围。反过来，如果可能溢出，很多推导就必须保守。

常见 no-wrap 信息：

- `nuw`：unsigned no-wrap；
- `nsw`：signed no-wrap；
- recurrence 在实际迭代域内不会 wrap；
- pointer offset 保持在合法对象或地址计算模型内。

必须区分“IR 指令显式携带 no-wrap flag”和“分析根据循环边界证明实际执行范围内不 wrap”。二者都可支持推理，但证明来源和可复用范围不同。

## SCEVExpander：把表达式物化回 IR

SCEV expression 是分析图，不是可执行指令。需要真正改写程序时，SCEVExpander 把表达式在指定位置物化为 IR。

物化需要考虑：

- expression operands 是否在插入点可用；
- 新 definitions 是否支配全部 uses；
- pointer 与 integer 运算的类型和 layout；
- 是否可以复用已有等价 value；
- 是否会把可能 trap 的操作放到原本不执行的路径；
- 新增计算的成本与 live range。

例如，把 `base + 4*i` 物化到 loop preheader，只在 `i` 对该位置可用且 expression 对循环 invariant 时合理。若 `i` 是当前循环的 AddRec，通常需要构造新的派生归纳变量，而不是在 preheader 直接求值。

## Loop Strength Reduction

Loop Strength Reduction（LSR，循环强度削弱）利用 SCEV 把每轮重复的乘法或复杂地址计算转为递增 recurrence。

```c
for (i = 0; i < n; ++i)
    use(base + i * 4);
```

可转为概念上的：

```text
p0 = base
for (...):
    use(p)
    p.next = p + 4
```

代价权衡包括：

- 减少乘法和地址重算；
- 新增 loop-carried Phi；
- 延长 pointer recurrence 的 live range；
- 增加寄存器压力；
- 可能影响寻址模式选择。

因此“算术更便宜”不自动表示最终代码更快。

## 指针与仿射内存地址

SCEV 常把地址表示为：

```text
base object
+ invariant offset
+ Σ(AddRec stride for each loop)
```

例如二维 row-major 数组 `A[i][j]`，在元素大小为 `E`、行长度为 `N` 时，byte offset 可表示为：

$$
i\cdot (N E)+j\cdot E.
$$

这为 [[循环依赖分析]] 提供每一层 loop 的 stride，也为 [[内存优化证明：Alias Analysis 与 MemorySSA|Alias/Region 分析]] 提供对象内 offset。

Pointer arithmetic 还涉及 provenance、对象边界、address space 和 layout，不能把所有 pointers 无条件视为普通整数。

## 表达式的有效边界

SCEV expression 通常引用具体 SSA values 和 loop identities。若 definitions、Phi recurrence、CFG exits 或 loop structure 被改写，原表达式可能不再描述当前程序。

一些实现可以只遗忘受影响的 value 或 loop，另一些需要重新构造整组表达式。核心原则是：递推表达式只在其结构与语义前提仍成立时有效。

## 局限与保守边界

SCEV 是分析，不是证明器。遇到下面情况时，它可能退回 `SCEVUnknown` 或给出很保守的结果：

- 循环控制流太复杂。
- induction variable 被非线性方式更新。
- 值来自难以分析的内存 load 或函数调用。
- 整数溢出、指针越界、`poison`/`undef` 语义无法安全排除。
- 多个循环嵌套且表达式跨层依赖复杂。

还要注意：

- `Unknown` 表示当前模型无法继续推导，不表示 value 随机或一定非线性；
- 两个 SCEV expressions 相等，不自动证明现有 SSA definition 在某点可复用；
- 数学闭式成立，不自动证明机器整数执行中无 wrap；
- exact trip count 可能只对某个 exit 或某组前置条件成立；
- pointer recurrence 的数值等价不自动保留 pointer provenance；
- 多出口、early break 和异常控制流可能让单一 trip-count 模型不完整。

所以看 SCEV 结果时，重点不是“它一定能算出来”，而是理解它在什么语义和结构前提下，把循环值表示成怎样的递推形式。

## 关系速记

```text
[[循环规范形与 LCSSA|Canonical loop]] + [[构造|Header Phi]]
  -> basic induction variable
  -> AddRec
  -> exit/backedge/trip count
  -> SCEVExpander / derived IV
  -> LSR / unroll / bounds reasoning
  -> affine byte address
  -> [[循环依赖分析]]
```
