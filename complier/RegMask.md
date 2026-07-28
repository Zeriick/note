# RegMask

`RegMask` 是 LLVM 后端机器指令层常见的一种寄存器位图，用来描述一条机器指令对物理寄存器的保留/破坏关系。它最常出现在 call、inline asm、patchpoint/statepoint 一类可能 clobber 寄存器的指令上。

一个容易记错的点是：很多场景里的 regmask 表示的是“哪些寄存器被保留”，而不是“哪些寄存器被破坏”。不在 mask 里的寄存器，通常就要按可能被 clobber 来处理。

## 和寄存器分配的关系

寄存器分配（RA, Register Allocation）要解决的问题是：把 virtual register 分配到具体 physical register。

如果某个 virtual register 在一次 call 前定义，并且在 call 后还要继续使用，那么它就是 live across call。此时 RA 必须知道这条 call 会破坏哪些物理寄存器：

- 如果把这个 live-through-call 的值分到 call-clobbered register，call 之后值可能就没了。
- 如果分到 call-preserved register，就可能不需要额外 spill/reload。
- 如果没有合适的 preserved register，就需要在 call 附近插入 spill/reload，或者选择其他代价更低的方案。

所以 RA 前/RA 中的 regmask，本质上是一份“用于分配决策的约束信息”。

## RA 前/RA 中的 callee regmask

在 RA 前或 RA 过程中，机器指令里通常已经带有目标相关的 call-preserved mask。它来自 ABI、calling convention、函数属性或目标后端的 lowering 规则。

这份 mask 会告诉 allocator：

- 哪些物理寄存器跨过这条 call 之后仍然可靠。
- 哪些物理寄存器如果承载 live value，就需要额外保护。
- 某个 live interval 是否和这条 regmask 产生冲突。

因此它主要回答的问题是：

> 当前还没完成最终分配时，哪些 physical register 可以安全地承载跨 call 活跃的值？

## RA 后的 post-RA regmask

RA 后，virtual register 已经被 rewrite 成 physical register。此时再看 regmask，关注点会从“指导怎么分配”转成“描述最终机器代码里的真实物理寄存器效果”。

post-RA 阶段更关心：

- 最终哪些 physical register 在某条指令处 live。
- 某条 call/特殊指令实际会影响哪些物理寄存器。
- 后续 post-RA pass、liveness、verifier、调度或栈图相关逻辑如何理解这些 clobber 信息。

所以可以把两份信息区分成：

- RA 前/RA 中：算一份“用于分配决策的 callee regmask”。
- RA 后：再算一份“基于最终物理寄存器结果的 post-RA regmask”。

## caller-saved 和 callee-saved

理解 regmask 时，经常会和 caller-saved/callee-saved 混在一起：

- caller-saved：调用者如果还想保留这些寄存器里的值，需要自己在 call 前后保存和恢复。
- callee-saved：被调用者按 ABI 负责保存和恢复，所以对调用者来说，跨 call 更稳定。

RegMask 通常把这种 ABI 约定编码成机器指令可查询的位图。RA 看到 live across call 的值时，就会倾向于让它避开 caller-saved clobber，或者为它安排 spill。

## 注意点

- 物理寄存器存在 alias、super-register、sub-register，判断一个寄存器是否被 clobber 时不能只看单个位。
- inline asm 或特殊 calling convention 可能让 regmask 和普通 call 不同。
- pre-RA mask 服务于“分配选择”，post-RA mask 服务于“最终机器指令语义和物理寄存器活跃性”。
- 如果 regmask 过于保守，可能导致不必要的 spill；如果过于激进，则可能把 live value 放进会被破坏的寄存器，造成错误代码。
