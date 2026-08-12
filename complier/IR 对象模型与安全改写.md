# IR 对象模型与安全改写

Intermediate Representation（IR，中间表示）不仅是一串指令。一个可优化的 IR 通常同时包含：

- 容器层级：module、function、basic block、instruction；
- 控制流图：basic blocks 之间的 edges；
- 数据依赖图：definition 与 uses；
- 类型、内存 layout、调用目标和其他语义信息；
- 为快速查询维护的反向索引。

因此，安全改写 IR 的本质是维护多种互相重叠的图不变量，而不是修改一个数组元素。

## 1. 容器层级与依赖图

常见的容器结构是：

```text
Module
├── Globals
└── Functions
    ├── Parameters
    └── Blocks
        └── Instructions
            ├── Results
            ├── Operands
            └── Payload / Attributes
```

容器关系回答“对象属于哪里”：

- instruction 属于哪个 block；
- block 属于哪个 function；
- function 属于哪个 module。

依赖图回答“对象指向谁、被谁使用”：

- instruction result 被哪些 instructions 使用；
- branch 指向哪些 successor blocks；
- call 指向哪个 function；
- Phi result 从哪些入边接收值。

容器树和依赖图不是同一结构。从 block 的 instruction list 删除一条指令，不会天然保证其他对象不再引用它。

## 2. Node、Value、Instruction 与 Opcode

### Node

Node 是可被 IR 引用的实体，例如 value、block、function。不同 IR 对 Node 的划分不同，核心目的是统一表达图中的端点。

### Value

Value 是可以作为数据 operand 的节点，通常包括：

- SSA temporary；
- parameter；
- constant；
- global address；
- physical register view；
- `undef`、`poison` 等特殊值。

### Instruction

Instruction 通常包含：

- opcode；
- 零个或多个 results；
- 零个或多个 operands；
- 可选的 flags、layout、memory semantics 或其他 payload；
- 所属 basic block。

### Opcode descriptor

实现可以为 opcode 定义 descriptor，声明：

- result 数量和类型；
- operand 数量、种类和类型；
- 是否为 terminator；
- 是否可交换；
- 是否支持变长 operands；
- 是否读写内存或具有其他 effects。

Factory 负责按 descriptor 构造指令，Viewer/Accessor 用语义化名称访问 operand。相比在各处散落 `operand(2)`，这种方式更不易因 operand 顺序变化而出错。

## 3. Def-Use 与 Use-Def

SSA value 的核心关系是：

```text
definition -> value -> uses
```

从 use 找 definition 叫 use-def；从 definition/value 找全部 uses 叫 def-use。优化经常需要双向查询：

- 常量传播从定义沿 uses 传播；
- Dead Code Elimination（DCE，死代码删除）检查 result 是否仍有 use；
- value replacement 把旧 value 的 uses 重定向到新 value；
- liveness 从 uses 和 definitions 推导活跃范围。

### 显式 Ref 对象

一种实现方式是把依赖边建模为独立对象：

```text
Ref<Source, Target>
```

不同边可以具有不同语义：

- DefRef：instruction → result value；
- UseRef：instruction → operand value；
- BlockRef：terminator → successor block；
- FunctionRef：call → callee；
- PhiRef：edge value → Phi result。

当 Ref retarget 时，应原子地：

1. 从旧 target 的 incoming index 中 detach；
2. 更新 target；
3. attach 到新 target 的 incoming index。

只改正向指针而不维护反向索引，会留下幽灵 use 或幽灵 predecessor。

## 4. Identity 与 Structural Equality

IR 中经常同时存在两种相等性。

### 对象身份相等

两个 SSA temporary 即使类型、名字和定义形状都相同，只要不是同一个定义，就仍是不同 values：

```text
a = x + y
b = x + y
```

`a` 和 `b` 在 identity 上不同。它们是否语义等价，应由 value numbering、等价类或其他证明回答，不能通过普通对象字段相等来假设。

以下映射经常必须使用 identity key：

- old value → cloned value；
- block → analysis result；
- instruction → rewrite plan；
- loop header → loop object。

### 结构或值相等

常量、立即数、类型描述和某些物理寄存器视图可以采用值语义。例如两个 `i32 0` 通常可视为相等。

IR 规范必须明确哪些对象使用 identity、哪些使用 structural/value equality。混用可能导致：

- 两个独立定义被错误合并；
- map 查询不到实际同一对象；
- clone 映射错位；
- live range 或 Phi incoming 被串接。

## 5. 所有权与生命周期

Instruction 常见三种状态。

### Detached

指令已经构造，但尚未插入 block。它可以拥有 operands 和 results，但没有合法执行位置。

### Attached

指令位于某个 block 的 instruction list，且 parent 指针与容器一致。

### Disposed / Erased

指令已经从 IR 移除，outgoing refs 已解除，parent 已清空。实现可以保留只读调试对象，但它不应继续参与合法 IR。

状态转换应明确：

```text
construct -> detached
insert    -> attached
erase     -> disposed
```

同一 instruction 同时出现在两个 blocks，或 list 已删除但 parent 仍指向旧 block，都会破坏遍历和语义查询。

## 6. Basic Block 与 CFG 不变量

典型基本块约束包括：

- block 属于唯一 function；
- block 非空；
- 恰有一条 terminator；
- terminator 是最后一条 instruction；
- Phi 构成 block 前缀；
- successor 由 terminator 定义；
- predecessor 与所有入边一致；
- function entry 是 function 内的合法 block。

不同 IR 可能允许空块、隐式 fallthrough 或 region terminator，所以这些是常见设计，不是所有 IR 的绝对规定。

## 7. 安全的局部编辑

### 插入

插入前检查：

- 新指令处于 detached 状态；
- 插入点属于正确 function；
- operands 在插入点可用；
- 不破坏 Phi prefix；
- 不在 terminator 后插入普通指令；
- 新操作在该路径上执行是安全的。

最后一项不是结构问题。例如把除法或 load 提前到条件分支之前，IR 结构可能完全合法，却引入原程序不会发生的 trap。

### 替换 value

常见模式：

```text
new = build replacement
insert new at legal position
redirect old-result uses to new-result
erase old instruction
```

要区分：

- 替换 instruction；
- 替换 instruction result；
- 只替换普通 UseRef；
- 是否还要更新 Phi、debug、metadata 或特殊引用。

多结果指令不能凭空推断旧 results 与新 results 的对应关系，映射必须显式给出。

### 删除

删除 instruction 前，要回答：

- 所有 results 是否无 use，或 uses 已重定向？
- instruction 是否有 observable effect？
- outgoing operand refs 是否会解除？
- parent/list 是否同时更新？
- 删除 terminator 后 block 如何保持合法？

“结果没有被使用”不代表指令可删除。store、call、volatile load、atomic、trap 和控制流都可能有可观察效果。

### 移动

移动 instruction 时要检查：

- operands 在新位置支配可用；
- 新定义仍支配所有 uses；
- 内存操作没有跨越可能 alias 的 clobber；
- 操作是否可能 trap；
- 控制依赖是否允许该操作在更多路径执行；
- 对循环而言，移动是否改变执行次数。

## 8. CFG Edge 编辑

### Redirect Edge

把 `P -> Old` 改成 `P -> New`，至少涉及：

- terminator target；
- `Old` 的 predecessor 信息；
- `New` 的 predecessor 信息；
- `Old` 中 Phi incoming 的删除；
- `New` 中 Phi incoming 的构造。

最后一步经常最难：新 incoming value 必须在 `P -> New` 上可用。

### Split Edge

```text
P -> S
```

变成：

```text
P -> M -> S
```

若 Phi incoming 按 predecessor block 表示，原来属于 `P -> S` 的 incoming 现在应属于 `M -> S`。边上数据必须随 edge 一起迁移。

### Erase Region

删除一组 blocks 时，通常要求：

- 不删除仍作为入口的块；
- 区域不存在未处理的外部入边；
- 区域内定义没有未处理的区域外 use；
- 先解除全部 refs，再从 function 容器移除；
- 所有边界 Phi 同步修复。

## 9. Clone 与 Region Mapping

复制 instruction 或 CFG region 需要显式映射：

```text
old block       -> new block
old value       -> new value
old instruction -> new instruction
```

常见顺序：

1. 先创建新 blocks；
2. 创建新 instructions 和 results；
3. 建立 value mapping；
4. 重写普通 operands；
5. 重写 block targets；
6. 修复区域入口/出口 Phi；
7. 处理只应保留一次的 effects；
8. 更新外部 uses。

克隆 payload、layout、flags 和特殊语义同样重要。只复制 opcode 与普通 operands 可能悄悄丢失 access size、alignment、no-wrap 或 calling-convention 信息。

## 10. 内存语义不能只依赖 Pointer Type

采用 opaque pointer 的 IR 可能只有统一 pointer type。此时内存访问含义还需要：

- base object；
- object-relative offset；
- pointee/layout；
- access size；
- alignment；
- address space；
- volatile/atomic 属性。

两个 pointer values 的 identity 不同，可能指向同一位置；同一个 pointer 以不同 access size 访问，也不一定代表同一 memory location。详见 [[内存优化证明：Alias Analysis 与 MemorySSA]]。

## 11. Plan-Then-Commit

复杂 IR 改写适合分成两个阶段。

### Plan

只读地完成：

- discover candidate；
- 检查结构；
- 证明语义安全；
- 计算所有新 values、edges 和 mappings；
- 估计收益；
- 得到不可变 plan。

### Commit

按照预先确定的顺序：

1. 创建 detached objects；
2. 插入新 definitions；
3. 重定向 uses；
4. 更新 CFG edges 和 Phi；
5. 删除旧对象；
6. 完成清理。

Commit 阶段不应再进行可能失败的复杂证明，否则失败可能留下半完成的 IR。

## 12. 结构正确与语义正确

结构合法不等于变换正确。结构不变量包括：

- parent/list 一致；
- terminator 位置正确；
- operand/result 类型匹配；
- SSA value 唯一定义；
- Phi incoming 完整。

但下面的错误可能保持结构完全合法：

- 把 load 移到原本不会执行的路径；
- 把 MAY_ALIAS 当成 NO_ALIAS；
- 删除仍可被调用观察到的 store；
- 有符号溢出推理错误；
- 修改循环后丢失零次迭代语义；
- 延长 live range，导致大量 spill 和性能回退。

安全改写必须同时维护结构、语言/IR 语义和目标约束。

## 13. 常见失败模式

| 症状 | 常见原因 |
|---|---|
| 已删除 value 仍显示有 use | outgoing/incoming ref 未解除 |
| predecessor 集合含不存在的边 | terminator 与反向 BlockRef 不一致 |
| 一个 SSA value 出现多个 definitions | 错误复用 result identity |
| instruction parent 错误 | 直接操作容器 list，没有维护 parent |
| Phi 缺少 incoming | 新增或重定向 edge 后未补数据流 |
| Phi 有重复 incoming | edge identity 与 predecessor identity 混淆 |
| clone 后 operand 仍引用旧区域 | value mapping 不完整 |
| 内存优化错编译 | 丢失 layout、access size、alias 或 effect 信息 |
| 合法 IR 运行结果错误 | 路径、trap、零次执行或 overflow 语义改变 |

## 14. 关系速记

```text
IR container tree
  └─ 决定 ownership 与顺序

CFG edges
  └─ 决定可达性、[[dom|支配]]、循环和 Phi predecessor

Def-use graph
  └─ 支撑 [[构造|SSA]]、传播、DCE、liveness

Memory semantics
  └─ 支撑 Alias Analysis、MemorySSA 与内存变换

Safe rewrite
  = 同时维护上述所有结构与语义
```
