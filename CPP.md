### 指针和引用

1. \* 和 \& 可以作为声明的一部分， 也可以作为解引用/取地址符

```
int i = 42;
int &r = i;
int *p; 
p = &i;             \\ &取地址
*p = i;             \\ *解引用
int &r2 = *pi       \\ *解引用
```

### const
默认const对象仅本文件内有效，多文件共用一个的话加extern
用顶层表示指针本身是个常量，底层表示指针所指向的对象是个常量。
```
int *const p1 = &i;      顶层
const int *p2 = &ci;     底层
```
在执行对象的拷贝操作时，**底层 const 的限制却不能被忽视**。当执行对象的拷贝操作时，拷入和拷出的对象必须具有相同的底层 const 资格，或者两个对象的数据类型必须能够转换，一般来说，非常量可以转化为常量，反之不行。

### decltype与auto
有些表达式会让decltype返回一个引用类型，这就意味着表达式可以作为赋值语句的左值
decltype加括号则返回对对象的引用
```
int i = 42, *p=&i;
decltype(*p) c // 错误，c是int& 必须初始化
decltype((i)) d // 错误，d是int& 必须初始化
```

## 虚函数

运行时多态 虚表 
## 关键字
### explicit
禁止隐式转换

### static

+ magic static

### volatile

### inline 
inline（内联）函数的好处太多了：它没有宏的那些缺点，见[Item 2：避免使用define](https://harttle.land/2015/07/20/effective-cpp-2.html)；而且不需要付出函数调用的代价。 同时也方便了编译器基于上下文的优化。但inline函数也并非免费的午餐：

它会使得目标代码膨胀，运行时会占用更多的内存，甚至引起缓存页的失效和指令缓存的Miss，这些都会造成运行时性能的下降。 但是另一方面，如果inline函数足够小以至于生成的目标代码比函数调用还小，那么inline函数会产生更小的目标代码以及更高的指令缓存命中率。

## 模板

### CRTP
https://zhuanlan.zhihu.com/p/408668787
CRTP是Curiously Recurring Template Pattern的缩写
CRTP的应用很广泛，特别多的开源项目都会用到这种技术，经常被用在下面三种场景中：
1. 静态多态
2. 代码复用
3. 实例化多套基类静态变量和方法

## Cast

### Static cast
### const cast
用来实现底层const的添加和消除。请注意不能让const对象变为非const，如果这样会导致UB。
### reinterpret cast
最强的const
### dynamic cast
1) 如果表达式是指向或引用多态类型 Base 的指针或引用，而目标类型是指向或引用类型 Derived 的指针或引用，则会执行运行时检查：
   a) 检查表达式所指向/标识的最派生对象。如果在该对象中，表达式指向/引用 Derived 的公共基类，并且只有一个 Derived 类型的对象派生自表达式所指向/标识的子对象，则转换结果指向/引用该 Derived 对象。（这被称为"向下转换"。）
   b) 否则，如果表达式指向/引用最派生对象的公共基类，并且同时最派生对象具有 Derived 类型的明确公共基类，则转换结果指向/引用该 Derived 对象。（这被称为"侧向转换"。）
   c) 否则，运行时检查失败。如果在指针上使用 `dynamic_cast`，则返回目标类型的空指针值。如果在引用上使用，则抛出 `std::bad_cast` 异常。

2) 当在构造函数或析构函数中（直接或间接）使用 `dynamic_cast`，并且表达式引用当前正在构造/析构的对象时，该对象被视为最派生的对象。如果目标类型不是指向或引用构造函数/析构函数自身的类或其基类之一，则行为未定义。

## any，variant

```cpp
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>; std::variant<double, bool, std::string> var;
std::visit(overloaded { 
[](auto arg) { std::cout << arg << ' '; }, 
[](double arg) { std::cout << std::fixed << arg << ' '; }, 
[](const std::string& arg) { std::cout << std::quoted(arg) << ' '; }, 
	}, var);
```



- **公有继承**：当你希望派生类对象能够像基类对象一样使用基类的公有接口时。
- **保护继承**：当你希望派生类能够访问基类的成员，但不希望这些成员暴露给派生类的用户时。
- **私有继承**：当你希望派生类能够使用基类的实现，但不希望基类的接口暴露给派生类的用户时。

虚继承是为了解决 ​**菱形继承（Diamond Inheritance）​**​ 问题，确保在多重继承中，**公共基类（Common Base Class）​**​ 的实例在派生类中**仅存在一份**。例如：

cpp

```cpp
class A { int data; };
class B : virtual public A {}; // 虚继承
class C : virtual public A {}; // 虚继承
class D : public B, public C {}; // D 中仅有一份 A 的实例
```
- ​**内存开销**：每个虚继承的派生类增加一个 `vbptr`（通常 8 字节）。
- ​**访问性能**：访问虚基类成员需要通过 `vbptr` 查表，多一次间接寻址。