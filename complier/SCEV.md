# SCEV: Scalar Evolution

**Scalar Evolution**，通常指 LLVM 里的一个分析框架，用来回答：

> 一个标量值在循环中是如何变化的？

它最常用来分析循环里的变量、数组下标、指针偏移、循环次数等。很多循环优化不会自己从头推导这些性质，而是先问 SCEV。

## 核心表示

SCEV 会把 IR 里的标量表达式抽象成一套数学表达式。最重要的是 `AddRec`，也就是 add recurrence：

```text
{start,+,step}<L>
```

它表示在循环 `L` 中：

- 第 0 次迭代是 `start`。
- 每次迭代增加 `step`。
- 第 `k` 次迭代大致是 `start + k * step`。

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

## trip count 和 backedge taken count

SCEV 里经常会区分两个概念：

- `trip count`：循环 body 实际执行多少次。
- `backedge taken count`：循环回边被执行多少次。

如果循环至少执行一次，通常有：

```text
trip count = backedge taken count + 1
```

但如果循环可能 0 次执行，就必须把“是否进入循环”的条件也考虑进去。很多 off-by-one 的理解错误都来自这里。

## no-wrap 信息

SCEV 很依赖整数溢出语义。一个递推式如果可以证明 `nuw`、`nsw` 或更一般的 no-wrap 性质，那么优化空间会大很多。

例如：

```cpp
for (unsigned i = 0; i < n; ++i)
```

如果能证明 `i + 1` 在循环内不会绕回 0，那么 SCEV 可以更放心地推导循环次数和范围。反过来，如果可能溢出，很多推导就必须保守。

## 局限

SCEV 是分析，不是证明器。遇到下面情况时，它可能退回 `SCEVUnknown` 或给出很保守的结果：

- 循环控制流太复杂。
- induction variable 被非线性方式更新。
- 值来自难以分析的内存 load 或函数调用。
- 整数溢出、指针越界、`poison`/`undef` 语义无法安全排除。
- 多个循环嵌套且表达式跨层依赖复杂。

所以看 SCEV 结果时，重点不是“它一定能算出来”，而是理解它在能证明时会把循环值表示成怎样的递推形式。
