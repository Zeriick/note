# TileLang

TileLang 是面向高性能算子的 tile-based DSL。用户显式描述数学分块、存储位置、数据搬运和主要流水结构；编译器补全线程级 layout，并把 tile primitive 降低到目标相关实现。

它不是自动调优器，也不替用户选择数学分块：

```text
问题规模与算子语义
  -> tile / block 划分
  -> CTA 与线程组织
  -> global / shared / fragment
  -> layout 与线程映射
  -> copy / gemm / reduce 等 tile primitive
  -> pipeline 与同步
  -> CUDA / HIP / CPU 等目标代码
```

## 三个层次

TileLang kernel 中常混用三个抽象层次：

| 层次 | 关注点 | 典型接口 |
| --- | --- | --- |
| 执行与存储 | grid、CTA、tile、memory scope | `T.Kernel`、`T.alloc_shared`、`T.alloc_fragment` |
| tile primitive | 搬运和计算的整体语义 | `T.copy`、`T.gemm`、`T.reduce_*` |
| 线程级控制 | 逻辑迭代如何映射到线程 | `T.Parallel`、`T.Fragment`、显式 layout |

同一个 kernel 可以在规则部分使用 tile primitive，在特殊边界或融合部分使用线程级表达。

## 用户与编译器的边界

用户主要决定：

- 一个 CTA 处理多大的逻辑 tile；
- 使用多少线程；
- 数据放在 global、shared、fragment 还是其他 scope；
- 哪些循环串行、并行或流水执行；
- 何处使用 GEMM、reduce 等整体原语。

编译器主要完成：

- 推断 fragment 和 `T.Parallel` 的线程 layout；
- 根据数据依赖插入或降低必要同步；
- 根据 shape、dtype、scope、alignment 和 target 选择 primitive 实现；
- 生成目标相关代码。

边界不是绝对的：需要精细控制时，可以显式提供 layout 或使用更低层的原语。

## 最小骨架

```python
@T.prim_func
def gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_shared = T.alloc_shared((BM, BK), "float16")
        B_shared = T.alloc_shared((BK, BN), "float16")
        C_local = T.alloc_fragment((BM, BN), "float32")

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_shared)
            T.copy(B[ko * BK, bx * BN], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * BM, bx * BN])
```

这段代码同时表达了：

```text
grid tile       : (bx, by)
global -> shared: A、B tile 搬运
shared -> compute: T.gemm
register state  : C_local fragment
时间重叠         : T.Pipelined
```

它仍没有回答 tile shape、线程数和 stage 数是否最优；这些是调度参数，需要结合硬件和 profiling 验证。

## 阅读顺序

1. [[01-Kernel与执行模型]]：CTA 如何领取 tile。
2. [[02-数据容器与存储层级]]：tile 在不同 memory scope 中如何表示。
3. [[03-Layout与线程映射]]：逻辑坐标如何落到线程、寄存器槽位和 shared 地址。
4. [[04-计算原语与归约]]：copy、GEMM、reduction 如何降低。
5. [[05-Pipeline与异步执行]]：如何重叠数据搬运与计算。
6. [[06-Persistent与性能分析]]：persistent schedule、调优和诊断。

贯穿示例：

- [[examples/GEMM|TileLang GEMM]]
- [[examples/Reduction|TileLang Reduction]]
- [[examples/FMHA|TileLang FMHA]]

相关基础：

- [[../GEMM|GEMM 优化]]
- [[../CUDA Reduction 优化|CUDA Reduction 优化]]
- [[../FMHA 优化|FMHA 优化]]
- [[../H100 Memory Hierarchy|H100 存储层次]]
- [[../TVM FFI|TVM FFI]]

## 版本说明

TileLang 仍在快速演进。本文优先记录稳定心智模型，接口以 2026-08-07 的官方仓库主线为准。遇到旧示例时应优先检查当前语言参考和源码。

官方资料：

- [TileLang 官方文档](https://tilelang.com/)
- [TileLang GitHub](https://github.com/tile-ai/tilelang)
