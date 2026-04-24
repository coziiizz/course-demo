# 推测解码的实现与动态 Draft 深度分析

## 一、任务说明

### 1.1 背景

自回归（Autoregressive, AR）解码是当前大语言模型最常见的生成方式。  
在 AR 解码中，模型每次只生成 1 个 token，因此当模型参数规模较大时，推理延迟较高、吞吐量受限。

[Speculative Decoding（推测解码）](https://dl.acm.org/doi/10.5555/3618408.3619203)是一种用于加速大语言模型推理的方法。它的基本思想是：  
使用一个较小、较快的 draft model 先生成若干候选 token，再由较大的 target model 对这些 token 进行验证。  
如果 draft model 提出的 token 与 target model 的分布足够一致，则可以一次接受多个 token，从而减少 target model 的调用轮数，提高整体生成速度。

训练型 draft model 是这一方向的进一步发展。[EAGLE-3](https://arxiv.org/abs/2503.01840)不再沿用早期方法中的特征预测约束，而是改为直接 token prediction，并引入 [multi-layer feature fusion 与 training-time test](https://www.zhihu.com/question/591646269/answer/2005360155936707357)，以提升训练出来的 draft model 与 target model 的匹配程度。

### 1.2 推测解码

Speculative Decoding 的一个典型流程如下：

1. 给定输入前缀，target model 先基于当前前缀得到下一位置的分布；
2. draft model 基于当前前缀继续提出若干个候选 token；
3. target model 对这些候选 token 进行验证；
4. 按照接受/拒绝规则，接受一部分 token，若发生拒绝则进行修正采样；
5. 重复以上过程，直到生成结束。

在该过程中，推测深度（即每轮 draft 的 token 数量）会影响：
- 接受率（acceptance rate）
- 接受长度（acceptance length）
- 整体吞吐量（tokens/s）

如果 draft 太短，则每轮带来的加速有限；  
如果 draft 太长，则被 target model 拒绝的概率可能升高，反而降低收益。  

---

## 二、方法设计规范

### 2.1 Speculative Decoding 基础实现（必做）

你需要基于 PyTorch 与 Transformers，自行实现一个可运行的 speculative decoding 推理流程，要求至少包含以下部分：

- target model 与 draft model 的加载
- AR baseline 解码
- speculative decoding 解码
- KV cache 的使用
- 接受/拒绝逻辑
- 统计实验指标

建议你按照如下思路实现：

1. 对输入前缀进行 prefill；
2. draft model 每轮提出 K 个候选 token；
3. target model 对这 K 个 token 进行验证；
4. 根据接受规则决定接受多少个 token；
5. 若中途发生拒绝，则进行修正采样；
6. 重复直到生成结束。

你需要完成以下实验：

- 固定 draft-target 模型：
  - draft：Qwen3-0.6B
  - target：Qwen3-1.7B
- 比较不同固定推测深度：
  - 例如 `K = 1, 2, 4, 8`
  - acceptance rate
  - acceptance length
- 与普通 AR 解码比较以下指标：
  - tokens/s
  - speedup

### 2.2 动态 Draft 深度分析（必做）

固定 draft-target 模型后，继续探索动态 draft 深度策略。

你需要设计一个动态策略，使得每一轮 draft model 不再固定提出 K 个 token，而是根据当前状态动态决定是否提前停止。

可选思路包括但不限于：

- 当 draft model 当前 token 的最大概率低于某个阈值时，提前停止
- 当 top-1 与 top-2 的概率差距低于某个阈值时，提前停止
- 当分布熵（entropy）高于某个阈值时，提前停止

你需要比较：

- 固定 K 策略
- 动态 K 策略


### 2.3 选做内容（可选）

在完成必做部分的基础上，你可以进一步探索 **训练专用的 draft model** 。

本题中，固定：

- target model：**Qwen3-1.7B**

选做要求为：

- 使用 **EAGLE-3** 方法，为 Qwen3-1.7B 训练一个专用的 draft model
- 将训练得到的 draft model 用于 speculative decoding 推理
- 与必做部分中直接使用的同系列小模型 **Qwen3-0.6B** 作为 draft model 的结果进行比较

并回答以下问题：

- 专门为 target model 训练得到的 draft model，是否比直接使用同系列小模型更适合作为 draft？

如果你完成了该部分，请在报告中额外说明：

- EAGLE-3 的训练配置
- 使用的数据集或样本来源
- 训练资源与训练时长
- 训练后 draft model 的推理接入方式
- 与 Qwen3-0.6B 对比时的实验设置是否保持一致

## 三、实验要求

### 3.1 必做实验一：基础实现与固定 K 比较

你需要完成以下内容：

- 实现 AR baseline
- 实现 speculative decoding
- 在不同 K 下分别运行实验
- 对比 AR 与 speculative decoding 的表现

至少统计以下指标：

- **接受率（acceptance rate）**  
  被成功验证并接受的 draft token 数量 / 总 draft token 数量

- **接受长度（acceptance length）**  
  每轮中被成功接受的 draft token 数量  
  报告中建议同时给出：
  - 总接受 token 数
  - 平均每轮接受长度

- **吞吐量（tokens/s）**  
  生成 token 总数 / 总生成时间

- **加速比（speedup）**  
  speculative decoding 吞吐量 / AR 吞吐量

### 3.2 必做实验二：动态 K 策略比较

你需要至少实现一种动态 draft 深度策略，并与固定 K 策略比较。

### 3.3 选做实验：训练型 Draft Model 比较

如果完成选做部分，你需要补充展示：

- EAGLE-3 训练得到的 draft model 与 Qwen3-0.6B 的对比结果
- 对实验现象的分析

### 3.4 实验设置要求

为保证结果可比，请统一：

- 相同的 prompt 集
- 相同的 `max_new_tokens`
- 相同的采样设置（如 greedy / temperature / top-p）
- 相同的硬件环境
- 相同的计时方式

如你使用了额外的工程优化，请在报告中明确说明。

---

## 四、提交内容

### 4.1 代码

- 代码结构不做要求
- 要给出能够运行的python环境（requirement.txt）
- 要给出能成功运行代码的脚本

### 4.2 报告

报告至少应包含以下内容：

- 详细描述你的实现方案。
- 展示不同配置下的实验结果。建议使用表格和图形展示结果。
- 对实验现象进行分析。
- 如参考了论文、博客、开源实现或官方文档，请列出参考文献。

---

## 五、注意事项

截止日期： 2026年6月 日 23:59（UTC+8）。

如有疑问，请联系 fuliangliu@smail.nju.edu.cn。

