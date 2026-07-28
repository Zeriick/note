_DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation_

[[llm/投机解码/README|投机解码总览]] · 对比：[[EAGLE#EAGLE3|EAGLE-3]] 的逐 token 自回归 drafter、[[Medusa#架构|Medusa]] 的独立多头预测

**Eagle-3** 是目前最成熟的方案之一。但它仍遵循自回归生成逻辑，也就是说每预测一个 Draft Token，就需要执行一次 Draft Model 的前向传播。生成 7 个 Token 就需要 7 次 Forward
DeepSeek 提出了 **DFlash**（Block Diffusion for Flash Speculative Decoding）。它放弃了逐 Token 的自回归方式，改为**块级并行预测**：在目标模型输出 Hidden States 后，Draft Model 利用非因果注意力一次性生成整个 Token Block

**扩散模型天然支持并行生成**：可以在单次前向传播中同时去噪所有位置。

## 设计 

先用一个较强的并行 drafter 一次性算出整段 draft 的“基础 logits”，再用极轻的串行 head，根据已经采样出的前一个 token 修正后续 logits；最后根据每个位置被接受的概率和当前服务负载，决定到底送多少 token 给 target 验证。

DSpark 的思路是：

> 重计算完整 drafter 太贵，但只根据前一个实际 token 修正 logits，可能已经足够消除大量模式碰撞。

DSpark 默认使用最简单的 **一阶 Markov head**：

$$B_k=B(x_{k-1},\cdot)$$

也就是只根据紧邻的前一个 token 修正当前 logits，不读取整个 block prefix。

理论上，可以为每个 token pair 保存一个转移矩阵：

$$B\in\mathbb R^{V\times V}$$

其中：$B[a,b]$B 表示前一个 token 是 $a$ 时，对当前 token $b$ 增加多少 bias。

但词表可能约有十万个 token，$V^2$ 不现实，所以 DSpark 使用低秩分解：$B=W_1W_2$​

其中：

$$W_1\in\mathbb R^{V\times r}, \qquad W_2\in\mathbb R^{r\times V}$$

DSpark 也研究了 RNN head。RNN 保持一个 block 内的 recurrent state，可以读取整个已生成 prefix，而不仅是前一个 token；但实验中它相对 Markov head 的额外收益较小，主要出现在较长 block 上，所以生产设计默认选择更简单的 Markov 版本。

这就是论文强调的：

> **A little autoregression goes a long way。**

从 [[Medusa#架构|Medusa]] 到 [[EAGLE#设计|EAGLE]] 再到 DSpark，draft 的依赖结构经历了“完全独立 → 完全自回归 → 块级并行加少量自回归校正”的连续取舍。DSpark 还把送往 target 的 token 数视作调度问题：置信度高或系统负载低时可以尝试更长 block；反之缩短 block，避免低接受率带来的验证浪费。参见 [[Speculative Sampling|正确性基线]] 与 [[性能模型与指标#最小观测集|性能观测集]]。
