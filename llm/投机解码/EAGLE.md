_EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty_ 

[[llm/投机解码/README|投机解码总览]] · 前序：[[Medusa#局限|Medusa 的独立预测头]] · 后续：[[DSpark#设计|DSpark 的半自回归 block draft]]

## 洞察

1. **特征级自回归比 token 级更简单**：token 是离散的自然语言符号，而 second-to-top-layer 的特征是连续向量，规律性更强。在特征层做自回归预测比直接预测 token 更准确。
2. **采样引入的不确定性约束了性能**：token 采样的随机性导致后续特征序列存在歧义。例如 "I" 后面可能是 "am" 也可能是 "always"，对应的特征序列完全不同。Medusa 在预测 t+2 位置时，不确定 t+1 位置的真实目标是什么。
## 设计

对比medusa，EAGLE 的 drafting 本身是**自回归的**

**在特征层做自回归预测，并用"提前一个时间步的 token"消除不确定性。**  
- **Draft Model 结构**：Embedding 层 + 单层 Transformer Decoder + LM Head      
- Embedding 和 LM Head **直接复用**目标模型的参数（不训练）
- 只训练一个轻量的 Autoregression Head（对 70B 模型仅 0.99B 参数）

$f$ is **second-to-top-layer feature**：因为作者把 LM Head 看成最顶层，所以 LM Head 前面的 Transformer hidden state 就是“倒数第二层特征

EAGLE 希望用一个很轻的 draft head 近似这一步：

$$(fi,Emb⁡(xi+1))⟶f^i+1​$$

然后复用 target 的 LM Head：

$$q(x_{i+2}) = \operatorname{softmax} \left(W_{\mathrm{LM}}\hat f_{i+1}\right)$$

![[Pasted image 20260712004018.png]]

为什么一定要加入**当前 feature $f_i$​ 实际选出来的下一个 token $x_{{i+1}}$​**  (shifted token)

如果只用 $f_t$ 回归 $f_{t+1}$​，相同输入可能对应多个完全不同的 target，回归模型容易学出类似“平均 feature”的东西。

一旦把已经采样出的 $x_{t+1}$​ 也输入：

$$(f_t,x_{t+1}) \Rightarrow f_{t+1}$$​

歧义就大幅减少。

feature $f_t$​ 配上它产生的实际 token $x_{t+1}$，共同预测后续 feature。论文认为这解决了 feature-level autoregression 中由采样造成的 uncertainty

EAGLE drafter虽然是自回归的，但每一步不必只保留一个 token，可以保留多个候选
不过 EAGLE-1 的问题是：

> **树的形状是事先固定的。**

例如无论当前分布是：A: 0.99 B: 0.01 还是：A: 0.51 B: 0.49
都可能使用相同的宽度和深度。这就引出了 EAGLE-2。

## EAGLE2

EAGLE-2 **几乎没有改变 EAGLE drafter 本身**，也不改变 target verification。它只改变一件事：

> 给定有限的候选节点预算，应该把节点放在树的什么地方？

EAGLE-2 观察到，EAGLE drafter 的 confidence 与 target 最终接受该 token 的概率高度相关。

EAGLE-2 为每个节点定义近似 global acceptance value：

$$V(v)= \prod_{u\in\operatorname{path}(\mathrm{root},v)} q_{\rm draft}(u\mid\operatorname{parent}(u))$$

它并不要求这个值是精确的 acceptance probability，只用它来决定计算预算分配。最终正确性仍由 target model 的严格验证保证。

EAGLE-2 有两个阶段。
### Expansion

在当前最深一层里，选择 $V(v)$ 最大的若干节点，送进 drafter 继续展开：

这让高潜力路径变得更深。
### Reranking

展开结束以后，不能直接把所有被展开的节点交给 target model。

因为某个较浅但没被继续展开的节点，可能比某个很深的节点更有价值。因此 EAGLE-2 把所有生成过的节点按 $V(v)$ 重新排序，在固定 verification token budget 下选择 top-NNN。

由于：$V(child)≤V(parent)$ 选中一个子节点时，它的祖先通常也会排在它前面，因此最终仍然形成一棵连通树。之后再展平 token，并生成只允许看祖先的 tree mask。

## EAGLE3

drafter 的目标是预测 token，但 EAGLE-1 额外要求它：
> 还必须准确复现 target LLM 的高维 hidden representation。

在 EAGLE-1 架构上扩大训练数据，收益会很快饱和；feature regression constraint 限制了 drafter 的表达能力。

EAGLE-3 的轻量 decoder 直接输出一个向量：
$$a_i\in\mathbb R^{d_{\text{model}}}$$​
论文把它称为 **unconstrained vector**。它和  $\hat f_i​$ 形状类似，但语义不同：$a_i\not\approx f_i$​
EAGLE-3 不再要求这个向量逼近 target model 的真实 hidden feature。它只要求：
$W_{\mathrm{LM}}a_i$ 能够产生准确的 draft token 分布。

去掉 feature loss 后，就必须用另一种办法解决多步 rollout 的 train-test mismatch。
EAGLE-3 的解决方法叫 **training-time test**。
在训练期间真实模拟测试时的多步自回归过程。

EAGLE-3 仍要在 draft 阶段逐 token forward；这份串行依赖是 [[DSpark#设计|DSpark]] 转向并行 block 生成的主要动机。两者的共同目标不是让 draft 单独生成最终答案，而是提高 [[性能模型与指标#从接受率到期望提交长度|target 一次验证后的期望提交长度]]；严格的接受 / 残差验证基线见 [[Speculative Sampling]]。
