# IPSCCP: InterProcedural Sparse Conditional Constant Propagation

`IPSCCP` 全称是 **InterProcedural Sparse Conditional Constant Propagation**，即“跨过程稀疏条件常量传播”。

可以把它拆成四个词理解：

- `InterProcedural`：不只看单个函数，也会利用调用关系传播函数参数、返回值等信息。
- `Sparse`：基于 SSA def-use 链传播，不需要像传统数据流分析那样反复扫描所有语句。
- `Conditional`：同时判断 CFG 边是否可执行，不会沿着已经证明不可达的分支继续传播。
- `Constant Propagation`：把能证明为常量的值替换成常量，并触发后续简化。

## SCCP 的基本模型

SCCP 通常同时维护两类状态：

1. value 的格状态。
2. basic block / CFG edge 的可达性。

value 的格可以粗略理解成：

- `unknown`：还不知道这个值是什么。
- `constant(c)`：已经证明这个值是常量 `c`。
- `overdefined`：已经证明它不是单一常量，或者目前无法安全当成常量。

CFG 边的状态则表示这条边是否可能被执行。比如：

```cpp
if (flag) {
    return 1;
} else {
    return 2;
}
```

如果 `flag` 被证明恒为 true，那么 false 分支就可以标记为不可执行；false 分支里的计算也不需要继续参与传播。

## 稀疏传播

在 SSA IR 中，一个值的使用者可以通过 def-use 链直接找到。因此当某个 SSA value 的格状态发生变化时，只需要把依赖它的用户放进 worklist，而不是扫描整个函数。

这就是 `Sparse` 的意义：传播沿着真实依赖关系走，通常比密集数据流分析更省。

## 跨过程传播

普通 SCCP 只在函数内部工作。IPSCCP 会进一步利用调用关系：

```cpp
static int f(int mode) {
    if (mode == 0) {
        return 10;
    }
    return 20;
}

int g() {
    return f(0);
}
```

如果所有可见调用点都传入同一个常量，或者某个函数返回值可以被证明为常量，IPSCCP 就可以把这种信息跨函数传播。这样函数内部的分支、调用点的返回值使用，都可能继续被折叠。

## 典型优化结果

IPSCCP 本身常见的结果包括：

- 函数参数在可见调用范围内变成常量。
- 函数返回值被证明为常量。
- 条件分支被证明只会走其中一边。
- 不可达 basic block 被标记出来，后续 pass 可以删除。
- 常量传入后暴露更多 `InstCombine`、`DCE`、`SimplifyCFG` 的机会。

## 和其他 pass 的区别

- 和普通常量传播相比，IPSCCP 会同时追踪 CFG 可执行性，所以不会被不可达路径上的值过早污染。
- 和 [[CVP]] 相比，IPSCCP 更关注 SSA value 是否能稳定变成常量；CVP 更关注某条路径上的条件关系能否简化后续判断。
- 和内联不同，IPSCCP 不一定需要把函数体复制到调用点，也能通过调用图信息进行一部分跨过程推理。

## 保守边界

跨过程分析必须保守。下面这些情况通常会降低它能证明的内容：

- 函数有外部可见调用点，编译器看不到所有 caller。
- 间接调用、虚调用、函数指针让目标不明确。
- 参数或返回值受内存、全局变量、volatile/atomic、异常路径等影响。
- 函数地址被取走后，调用集合难以完整枚举。

所以 IPSCCP 的结果通常是“能证明就替换，证明不了就保持 overdefined”，不会为了激进而牺牲正确性。
