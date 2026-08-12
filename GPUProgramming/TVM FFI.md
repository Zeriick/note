# TVM FFI

## 一句话理解

TVM FFI 是一套面向机器学习系统的跨语言运行时 ABI。它用稳定的 C ABI、统一的 packed calling convention、带类型标签的 `Any`、引用计数对象系统，以及基于 DLPack 的 Tensor 互操作，把 Python、C++、Rust、动态库和编译器生成代码连接起来。

它不只是“让 Python 调用 C++”的绑定库，更接近机器学习运行时使用的一套通用二进制协议。

## TVM FFI 解决什么问题

FFI（Foreign Function Interface）定义不同语言和运行时如何互相调用。真正需要统一的不只是源码层的函数签名，还包括：

- 参数和返回值如何表示；
- 数据在寄存器、栈和内存中的布局；
- 函数符号如何导出和查找；
- 对象生命周期和内存所有权；
- 错误如何跨语言传播；
- Tensor 如何共享底层 storage 和 device pointer。

TVM FFI 不是单纯的“Python 调 C++”绑定库，更接近 ML 系统使用的一套通用二进制协议。核心 ABI 使用 C 而不是直接暴露 C++ ABI，以降低编译器、标准库和动态库版本差异带来的兼容风险。

## 五个核心抽象

### `Any` 与 `AnyView`

`TVMFFIAny` 是带类型标签的联合体（tagged union），逻辑上包含：

```text
type_index  -> 当前值的运行时类型
payload     -> 整数、浮点数、指针或对象句柄
```

它可以表示标量、字符串、`DataType`、`Device`、`Shape`、`Tensor`、`Function`、`Module` 和用户对象等。

- `AnyView`：非 owning，只在当前调用期间借用参数，避免不必要的复制和引用计数操作。
- `Any`：owning，用于安全持有值或承接返回值。

两者共享兼容的 ABI 表示，但所有权语义不同。

### `Function` 与 packed calling convention

不同的强类型函数：

```cpp
float foo(int x, Tensor y, std::string name);
```

跨 FFI 边界时统一表示为：

```text
args[0] = Int(x)
args[1] = Tensor(y)
args[2] = String(name)
```

边界只需要一种调用形态：

```text
(args pointer, number of args, result pointer)
```

接收端 wrapper 检查参数数量和类型，解包后再调用真正的强类型函数。这样做带来几项收益：

- Python、C++、Rust 和编译器生成代码复用同一 ABI；
- 可以在运行时按名字查找和调用函数；
- 函数自身可以作为跨语言对象和回调参数；
- 不需要为每个导出函数手写完整的语言绑定；
- AOT/JIT 生成代码可以直接构造参数数组并调用稳定入口。

### `Object` 与 `ObjectRef`

TVM FFI 不直接把任意 C++ 类暴露到 ABI 边界，而是使用自己的对象系统：

- `Object`/`ObjectObj` 保存真正的数据；
- `ObjectRef` 是持有引用的类型安全句柄；
- 公共对象头维护运行时类型索引、引用计数和 deleter。

这套机制避免依赖 C++ RTTI、虚表和 STL 内部布局，并统一解决跨语言对象的类型识别与生命周期管理。

### `Tensor`

Tensor 互操作基于 DLPack 风格的描述符。跨边界传递的重点不是复制 Tensor 数据，而是共享或借用：

```text
data pointer
shape
strides
dtype
device
byte offset
ownership/lifetime information
```

理想路径中，PyTorch、JAX、NumPy/CuPy 等框架的 Tensor 可直接向 native 代码暴露兼容描述符和底层 storage，不必复制整块 CPU/GPU 数据。

“零拷贝”不是无条件保证：不兼容的 dtype、device、布局、生命周期或 stream 语义仍可能要求额外适配，应用也可能主动生成 contiguous copy。

### `Module` 与 Registry

- `Function` 表示一个可调用对象；
- `Module` 是一组命名函数的容器；
- Global Registry 提供“字符串名称 → Function”的动态查找。

整体关系可以记成：

```text
Module / Global Registry
          |
          | 根据名称找到
          v
       Function
          |
          | args: AnyView[]
          | result: Any
          v
 Python / C++ / Rust / generated code
          |
          +-- Tensor、Object 等作为 Any 传递
```

## 一次 Python 到 C++ 的调用

以 `add_two(40)` 为例：

```text
Python 调用 add_two(40)
  -> binding 将 40 打包为 TVMFFIAny
  -> args[0] = { type = Int, payload = 40 }
  -> 调用统一的 C ABI 入口
  -> native wrapper 校验参数数量和类型
  -> 将 AnyView 解包成 int
  -> 调用真正的 C++ 函数
  -> 将 42 包装为 owning Any
  -> Python binding 转回 Python int
```

错误不能让 C++ exception 直接穿过不兼容的 ABI。安全调用路径会捕获异常，将其转为 FFI 错误状态，再由 Python 等调用方恢复成对应异常。

## TVM FFI 与 pybind11 的区别

| 方面 | pybind11 | TVM FFI |
| --- | --- | --- |
| 核心目标 | 将 C++ API 暴露给 Python | 统一 ML 系统的跨语言、跨框架 ABI |
| 调用形式 | 每个函数通常有专用 wrapper | 所有函数使用统一 packed ABI |
| 语言范围 | 主要是 C++ 与 Python | C、C++、Python、Rust、生成代码 |
| 动态注册 | 不是核心模型 | Registry 是核心能力 |
| Tensor 互操作 | 常结合框架专用方案 | 原生围绕 Tensor/DLPack 设计 |
| 编译器生成代码 | 不是主要目标 | 是重要使用场景 |
| C++ 类 | 适合自然暴露 C++ API | 倾向使用自有对象系统和稳定 C ABI |

普通 Python C++ 扩展通常用 pybind11 更直接；编译器 runtime、Kernel DSL、多框架算子库、动态加载模块和多语言基础设施更适合 TVM FFI 的抽象。

## 典型使用场景

TVM FFI 适用于：

- 编译器和模型 runtime；
- Kernel DSL 与算子库；
- Python、C++、Rust 共存的基础设施；
- PyTorch、JAX 等框架之间的 Tensor 互操作；
- 动态加载的模型或 kernel module；
- AOT/JIT 编译器生成的 host launcher；
- 需要函数注册、查找和跨语言回调的系统。

例如 Kernel DSL 可以在编译时同时生成 device kernel 和 native host launcher。Python 侧只做一次薄 FFI 调用，native launcher 直接读取 Tensor descriptor，并完成：

```text
dtype / rank / shape / layout validation
  -> argument marshaling
  -> grid 或 launch 参数计算
  -> device kernel launch
```

这样可以把高频 Python wrapper 中的 metadata 访问、动态检查和参数整理移到专门化 native code。TVM FFI 在这里提供紧凑的调用入口和直接 Tensor 互操作；具体的代码生成与优化仍由 Kernel DSL 或编译器负责。

这种方式对数微秒级的小 kernel 尤其有价值，因为 Python host overhead 此时可能与 kernel 执行时间相当，甚至成为主要瓶颈。

## 性能边界

Packed call 相比直接 C++ 调用仍有参数打包、运行时类型检查、间接调用和返回值转换等成本。对于启动较大算子或 device kernel 的调用，这部分开销通常容易摊薄；对于极细粒度标量操作，则应先批量化工作，而不是让每个元素单独经过一次 FFI。

```text
不合适：调用 FFI 一百万次，每次处理一个元素
合适：  调用 FFI 一次，由 kernel 处理一百万个元素
```

## 心智模型

```text
Any / AnyView       -> 跨边界传递的任意值
Object / ObjectRef  -> 有类型、引用计数的跨语言对象
Function            -> 使用统一 packed ABI 的函数
Tensor              -> 可通过 DLPack 互操作的张量
Module / Registry   -> 函数的组织、查找和动态加载
```

最核心的一句话是：

> TVM FFI 用稳定的 C ABI、带类型标签的通用值和统一的 Packed Function 调用约定，把不同语言、动态库、编译器生成代码和 Tensor 框架连接起来。

## 延伸阅读

- [Apache TVM FFI](https://tvm.apache.org/ffi/)
- [ABI Overview](https://tvm.apache.org/ffi/concepts/abi_overview.html)
- [Any and AnyView](https://tvm.apache.org/ffi/concepts/any.html)
- [Function and Module](https://tvm.apache.org/ffi/concepts/func_module.html)
- [Tensor and DLPack](https://tvm.apache.org/ffi/concepts/tensor.html)
- [Object and Class](https://tvm.apache.org/ffi/concepts/object_and_class.html)
