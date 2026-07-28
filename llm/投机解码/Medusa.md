_Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads_

[[llm/投机解码/README|投机解码总览]] · 后续思路：[[EAGLE#洞察|特征级自回归 draft]]、[[DSpark#设计|半自回归 block draft]]

## 架构

**不引入独立 draft model，而是在原始 LLM 顶部"长"出多个预测头。**  

- **Medusa Heads**：在 LLM 最后一层隐藏状态上叠加 K 个轻量解码头（单层 FFN + 残差连接），第 k 个头负责预测第 (t+k+1) 个 token。 预测头轻量级
- **Tree Attention**：每个 head 取 top-s 预测，通过笛卡尔积构建候选树，用改造的 attention mask 并行验证所有候选路径。 并行验证多条链路 。 Medusa tree 是由多个未来位置的 top-k token 组合而成的临时候选前缀树；Tree Attention 把整棵树展平后，用“只能看祖先”的 attention mask，在一次 target-model forward 中验证所有根到叶路径，最终提交最长的可接受前缀。self-attention + tree-structured causal mask
- **Typical Acceptance**：放弃 rejection sampling，改用原始模型自身的概率阈值筛选"合理"候选，无条件接受第一个 token（第一个 token 来自**原始 target model 自己的 LM head**，采用 greedy decoding 或指定采样方式生成，所以当然可以直接接受），后续 token 根据阈值决定是否接受。

严格无损的接受 / 残差校正见 [[Speculative Sampling]]；Typical Acceptance 是不同的折中规则，不能把两者的“接受率”直接等同。

![[Pasted image 20260711144244.png]]

## 局限  
- Medusa Heads 的预测精度有限（约 0.6），因为每个 head 独立预测，忽略了位置间的依赖关系
- 随着预测距离增加，精度急剧下降

这正是 [[EAGLE#洞察|EAGLE 的出发点]]：不再让第 $t+2$ 个位置孤立地猜测，而是把已经选定的中间 token 显式作为条件，进行特征级自回归 rollout。
