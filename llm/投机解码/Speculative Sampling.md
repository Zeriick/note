# Speculative Sampling：严格正确性的基线

[[llm/投机解码/README|投机解码总览]] · 对照：[[Medusa#Typical Acceptance|Medusa 的 Typical Acceptance]] · 如何评估收益：[[性能模型与指标]]

标准 speculative sampling（也称 speculative decoding）的关键承诺是：**输出的随机分布与只运行 target model 时完全一致**。它不是“draft 猜对就沿用”的启发式，而是用接受/拒绝与残差采样校正 draft 分布和 target 分布之间的差异。

原始算法见 [Leviathan et al., 2023](https://arxiv.org/abs/2211.17192) 与 [Chen et al., 2023](https://arxiv.org/abs/2302.01318)。

## 记号与前提

给定已提交的前缀 $x_{<t}$：

- $p_i(\cdot)=p(\cdot\mid x_{<t},y_{<i})$：第 $i$ 个位置的 **target** 条件分布，其中 $y_{<i}$ 是此前 draft 出来的候选；它也是最终希望采样的分布。
- $q_i(\cdot)$：drafter 在同一前缀下实际使用的 proposal 分布。
- $m$：一次 draft 的 token 数。

这里的 $p_i$、$q_i$ 必须是**真正参与采样的分布**。例如 temperature、top-k、top-p、min-p 或 logits processor 若改变了 target 的输出分布，就应把变换后的 target 分布当作 $p_i$；分母中的 $q_i$ 则必须是 drafter 实际采样时的分布，而不只是其原始 softmax logits。

## 单轮算法

1. drafter 自回归采样 $m$ 个候选：$y_1,\ldots,y_m$，其中 $y_i\sim q_i$。
2. 把整个候选前缀送入 target，一次 forward 并行得到 $p_1,\ldots,p_{m+1}$。在尚未发生拒绝的前提下，这些就是对应逻辑前缀的 target 条件分布。
3. 从左到右检查候选 $y_i$。以

   $$\alpha_i(y_i)=\min\left(1,\frac{p_i(y_i)}{q_i(y_i)}\right)$$

   的概率接受它；一旦拒绝，停止检查后面的候选。
4. 若第 $i$ 个候选被拒绝，从残差分布采样一个 target token：

   $$r_i(z)=\frac{[p_i(z)-q_i(z)]_+}{\sum_v[p_i(v)-q_i(v)]_+},\qquad [a]_+=\max(a,0).$$

   提交此前已接受的候选加上该 token，并开始下一轮。
5. 若 $m$ 个候选全部接受，则额外从 $p_{m+1}$ 采样一个 **bonus token**，再开始下一轮。

候选来自 $q_i$，因此被检查到的 $q_i(y_i)$ 总是正的；实现时无需为未被 proposal 采到的 token 计算比值。

## 为什么它仍然服从 target 分布

先只看一个位置，令 proposal 为 $y\sim q$。

- 被接受并输出为 $z$ 的概率是

  $$q(z)\min\left(1,\frac{p(z)}{q(z)}\right)=\min(p(z),q(z)).$$

- 拒绝总概率为 $1-\sum_v\min(p(v),q(v))$，这恰好等于 $\sum_v[p(v)-q(v)]_+$。
- 因此，拒绝后由 $r$ 输出 $z$ 的概率为 $[p(z)-q(z)]_+$。

两部分相加为

$$\min(p(z),q(z))+[p(z)-q(z)]_+=p(z).$$

也就是说，这一位置的最终输出恰好来自 $p$。多 token 情况只是在每个已接受前缀上重复同一个论证；第一次拒绝后补出的 token 恢复了 target 的自回归轨迹。因此 algorithm 的联合输出分布与 target-only sampling 相同（忽略有限精度实现的数值误差）。

## 不要和相近规则混淆

| 场景                            | 接受规则                                                      | 是否严格保持 target sampling 分布 |     |
| ----------------------------- | --------------------------------------------------------- | ------------------------- | --- |
| 标准 speculative sampling       | 随机接受 $\min(1,p/q)$；拒绝后从残差 $r$ 采样                          | 是                         |     |
| greedy decoding               | draft token 与 target 的 argmax 一致才接受；首个不一致处取 target argmax | 对确定性的 greedy target 是     |     |
| [[Medusa#Typical Acceptance]] | 按 target 概率或典型性阈值直接接受候选                                   | 通常不是；它是质量/吞吐折中，需单独说明采用的规则 |     |

“target 一次 forward 验证多个 token”本身不保证无损；保证来自上面的概率校正。阅读 [[EAGLE]]、[[DSpark]] 等方法时，需要分别问两个问题：**它如何提出候选？它采用哪一种验证/接受规则？**

## 实现检查清单

- target 与 draft 的 tokenization / vocabulary 要能在同一 token 空间中对齐；否则需要额外的映射或专门算法。
- 对 target 和 draft 使用的采样处理应记录清楚；不能用已截断或重标定的 $q$ 去代替真正的 proposal 概率。
- EOS、stop token、grammar / structured-output mask 等约束也属于条件分布的一部分；验证时要应用同样的约束。
- 记录每轮 draft 长度、接受位置和 bonus token，见 [[性能模型与指标#最小观测集|最小观测集]]；这样才能区分“理论无损”与“实际变快”。
