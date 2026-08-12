# CVP: Correlated Value Propagation

`CVP` 全称是 **Correlated Value Propagation**，可以理解成“带路径条件的值传播”。

它在控制流路径上收集已经成立的 predicate/range 信息，然后用这些信息简化后续的比较、分支、算术和内存安全检查。

## 解决的问题

普通的常量传播通常只关心一个 SSA value 是否能被证明为常量。但很多优化机会不是来自“这个值全局恒等于某个常量”，而是来自“走到这条路径时，这个值一定满足某个条件”。

例如：

```cpp
if (i >= 0 && i < n) {
    if (i < n) {
        use(a[i]);
    }
}
```

进入外层 `if` 的 body 之后，`i < n` 已经成立，所以内层判断可以被化简为 true。这个事实不是 `i` 的全局事实，而是当前控制流路径上的事实。

## 核心思路

CVP 会利用支配当前 basic block 的条件信息：

- 来自前面分支的 `icmp` 条件，例如 `x < 10`、`p != null`。
- 来自 range 推理的信息，例如某个整数值落在 `[0, n)`。
- 来自已有 IR 语义的信息，例如某些操作隐含的非空、非负、无溢出条件。

拿到这些信息后，它会尝试调用通用的 simplification/value tracking 逻辑，把后续指令折叠掉。

常见结果包括：

- 比较指令变成常量 true/false。
- 条件分支变成无条件分支。
- `select`、`min/max`、边界检查被简化。
- 后续的 `SimplifyCFG`、`DCE` 可以删除不可达块和无用指令。

## 为什么叫 correlated

“correlated” 指的是某些事实只在特定路径上同时成立。比如：

```cpp
if (x == y) {
    // 在这里，x 和 y 是相等的
}
```

在 `if` 内部，`x == y` 可以帮助优化后续表达式；但出了这个分支以后，如果有多个路径汇合，就不能直接把这个事实当成全局事实。

所以 CVP 的关键不是单独知道 `x` 是什么，而是知道“在这个控制流上下文里，`x` 和其他值之间有什么关系”。

## 和 IPSCCP 的区别

- [[IPSCCP]] 更偏全局/跨过程的常量传播：它维护值的格状态和 CFG 边的可执行性。
- CVP 更偏路径敏感的局部化简：它利用支配条件推导当前点一定成立的事实。

简单说，IPSCCP 问的是“这个值能不能传播成常量”，CVP 问的是“在这条路径上，前面的判断能不能让后面的判断变得多余”。

## 与 PAG 和 alias analysis 的关系

CVP 的核心是 [[complier/cfg/dom|CFG]] 上的 dominance 与路径条件，不是 [[PAG]] 上的指针可达性。一条 flow-insensitive PAG path 可能合并互斥控制路径，因此不能直接当作“当前块中必然成立”的 predicate。

当机会来自 load、store 或 call 时，[[Alias Analysis|alias analysis]] 可以让相邻的内存化简或调用副作用分析证明内存不会被 clobber，PAG 则可以是 alias analysis 的底层表示。这会间接暴露更多供 CVP 消费的值和路径事实；具体 CVP 实现不一定直接查询 PAG 或 alias analysis。无法证明内存效果时，相关 pass 必须保守地保留可能的 clobber。

## 注意点

- CVP 只能使用能证明的事实，不能把某条路径上的条件错误推广到所有路径。
- 合流点要谨慎：如果两个 predecessor 带来的约束不同，合流后通常只能保留共同成立的部分。
- 整数溢出、`poison`、`undef`、指针别名等语义会让推理变保守。
- CVP 通常不是单独完成所有清理，而是和 `InstCombine`、`SimplifyCFG`、`DCE` 等 pass 配合，把简化后的 IR 继续收拾干净。
