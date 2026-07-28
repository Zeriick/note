# 投机解码

投机解码（speculative decoding）的目标是：让一个**便宜的 drafter** 先提出多个候选 token，再让昂贵的 **target model** 一次并行验证，从而用更少的 target-model 解码轮次生成相同的输出。

## 一次解码循环

1. drafter 从当前前缀提出一条或多条候选续写；候选可以组织为树。
2. target model 通过 causal mask（树形候选时为 [[Medusa#Tree Attention|Tree Attention]]）并行计算候选位置的 logits。
3. 按验证规则接受最长的可接受前缀；遇到不接受的位置时，以 target 分布补一个 token，再开始下一轮。

因此，端到端加速不只取决于 drafter 有多小，还取决于：每轮能接受多少 token、候选树是否把预算放在高概率路径上，以及 target 验证是否能高效地批量执行。

## 这组笔记的脉络

| 方案 | draft 的基本单位 | 核心取舍 | 与下一步的关系 |
| --- | --- | --- | --- |
| [[Medusa]] | 多个独立 future-token head | 一次提出多步候选，但未来位置彼此不条件化 | [[EAGLE]] 改为特征级自回归，解决未来 token 的不确定性 |
| [[EAGLE]] / [[EAGLE#EAGLE2|EAGLE-2]] / [[EAGLE#EAGLE3|EAGLE-3]] | hidden feature / unconstrained vector | 草稿质量更高、树预算更自适应，但逐 token rollout 仍有串行成本 | [[DSpark]] 探索块级并行 draft，再用很轻的自回归校正 |
| [[DSpark]] | 并行 token block + Markov/RNN head | 降低 draft 阶段的串行轮次 | 把候选长度也作为随置信度和负载变化的调度决策 |

## 阅读入口

- [[Medusa]]：多头预测与候选树验证的起点。
- [[EAGLE]]：为什么 feature-level draft 能够提高接受率，以及 EAGLE-2 / 3 如何继续演进。
- [[DSpark]]：为什么“少量自回归校正”足以让并行 block draft 可用。
- [[Speculative Sampling]]：标准接受 / 残差采样如何严格保持 target 的输出分布。
- [[性能模型与指标]]：如何用接受长度、逐位置接受率和 wall-clock 判断方案是否真的更快。

## 相关概念

- **正确性（losslessness）**：见 [[Speculative Sampling]]。若要严格复现 target model 的采样分布，接受/拒绝规则与拒绝后的残差采样必须满足其校正条件。[[Medusa#Typical Acceptance|Typical Acceptance]] 则是偏向质量与吞吐折中的另一类规则，不能和严格的 rejection sampling 混为一谈。
- **接受长度**：一轮 target 验证实际提交的 token 数。它和草稿质量、树形候选覆盖率、temperature / top-p 等采样设置共同决定收益；具体口径和性能模型见 [[性能模型与指标]]。
- **端到端延迟**：除了目标模型 forward，还受 KV cache、tree mask / position id 构造、候选 gather，以及动态 batch 调度影响。
