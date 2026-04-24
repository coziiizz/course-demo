# 基于连续潜在空间的语言模型推理（Latent Reasoning）

## 一、任务说明

### 1.1 背景介绍
近年来，大语言模型（LLM）在数学、逻辑和常识推理任务上表现出较强能力，其中一种常见做法是通过 **Chain-of-Thought（CoT）** 让模型显式生成逐步推理过程，再输出最终答案。然而，显式 CoT 通常需要生成较长的自然语言中间步骤，带来较高的推理开销，同时这些中间 token 中并非每一个都对真正的推理计算同等重要。

为此，近期研究开始探索 **latent reasoning**，即让模型在连续潜在空间中完成中间推理，而不是把每一步都显式解码成自然语言。Coconut 提出了一种代表性的连续潜在推理框架：在 latent reasoning 阶段，不再将上一步 hidden state 映射为离散 token，而是直接将最后 hidden state 作为下一步输入 embedding，从而使模型可以在语言空间之外进行中间推理。

在此基础上，后续工作进一步从训练方式和压缩方式上扩展 latent reasoning。例如，CODI 使用 teacher–student self-distillation，将显式 CoT 的推理能力蒸馏到连续空间；CoLaR 则进一步探讨了将多个 reasoning token 压缩到单个 latent 表征中，并研究压缩率与效率之间的平衡。

本次课程大作业将围绕这一方向展开，要求同学们围绕 latent reasoning 方法完成一个 **可分析、可比较、可讨论** 的课程项目。

### 1.2 任务目标
本项目要求同学们基于 **GPT-2** 模型，在 **GSM8K** 数据集上实现并分析连续潜在推理方法，重点关注以下问题：

1. 连续潜在推理是否能够在减少显式推理 token 的同时完成数学推理任务；
2. latent reasoning 与 Direct Answer、Explicit CoT 相比，在准确率上有何差异；
3. latent step 数量对模型性能有何影响；
4. latent reasoning 是否能够在显式输出长度上体现一定效率优势；
5. 在基础方法之上，是否可以设计简单改进进一步提升性能或效率。

---

## 二、数据集说明

### 2.1 数据集
本次大作业使用 **GSM8K** 数学推理数据集。
GSM8K 是一个面向小学数学文字题求解的推理数据集，广泛用于评估语言模型的逐步推理能力，也是多种 latent reasoning 工作常用的实验基准之一。Coconut 的实验中也使用了 GSM8K 来验证 latent reasoning 在数学推理任务上的可行性。

### 2.2 数据划分
原则上使用 GSM8K 的标准训练集与测试集。
如需验证集，可从训练集中划分一部分样本作为验证集，并在报告中明确说明划分方式与随机种子。若课程统一提供验证划分，则以课程提供的划分为准。

### 2.3 CoT 监督来源
本项目中的显式 CoT 监督可直接使用 GSM8K 原始数据中的解答过程文本。
同学们可根据实现需要对原始解答进行适度清洗，例如：

- 去除多余格式符号；
- 统一答案标记；
- 提取最终数值答案。

若对 CoT 文本做了额外处理，需要在报告中说明。

### 2.4 输出形式与评测方式
模型输出为一段文本，其中应包含最终答案。
评测时只提取 **最终数值答案** 进行比较，并以数值匹配作为 Accuracy 的统计标准。

建议统一使用类似如下格式：

`The answer is 42.`

这样可以减少不同输出风格带来的评测差异。

---

## 三、方法设计规范

### 3.1 基础要求
请在统一设置下实现以下三种推理方式，并进行比较：

#### （1）Direct Answer
只输入题目，不生成中间推理过程，直接输出最终答案。

#### （2）Explicit CoT
输入题目后，让模型显式生成自然语言推理过程，再输出最终答案。

#### （3）Latent Reasoning（Coconut-style）
本项目的 latent reasoning 基础版统一参考 **Coconut** 的核心思路进行实现：

- 在问题输入后插入若干 latent steps；
- 每个 latent step 不对应可读 token，而是使用上一步的最后 hidden state 作为下一步输入 embedding；
- latent reasoning 结束后，再进入语言生成阶段输出最终答案。

也就是说，本项目的 latent reasoning 基线需要明确体现以下思想：

> 使用连续 hidden representation 替代显式中间推理 token，在潜在空间中完成若干步中间推理后，再解码答案。

### 3.2 训练建议
基础 latent reasoning 的训练建议参考 Coconut 的多阶段训练思路：

- **初始阶段**：使用标准 CoT 数据训练模型；
- **后续阶段**：逐步用 latent steps 替代部分显式 CoT 步骤；
- **最终阶段**：尽可能让模型依赖 latent reasoning 完成答案预测。

训练时不要求同学们完整复现 Coconut 的全部实验细节，但至少应体现“逐步从显式 CoT 过渡到 latent reasoning”的基本思路。

### 3.3 可参考的进阶方向
完成基础版后，可阅读 CODI、CoLaR 等相关工作，了解 latent reasoning 在监督信号、表示压缩和推理效率方面的不同设计思路：

- **CODI**：通过 teacher–student self-distillation，将显式 CoT 的 hidden activation 对齐到 latent reasoning。
- **CoLaR**：研究多个 reasoning token 的压缩、动态压缩率以及 latent reasoning 的速度控制。

这些工作主要作为理解 latent reasoning 研究思路的参考。本项目不要求同学们复现现有论文，更鼓励在基础方法之上提出自己的简化改进或分析方案。

### 3.4 鼓励的改进方向
在完成基础 Coconut-style latent reasoning 后，鼓励同学们围绕“如何让 latent reasoning 更有效或更高效”进行开放探索。可以尝试但不限于以下方向：

- **更好的监督信号**：例如设计 auxiliary loss、hidden-state alignment、答案一致性约束，或利用显式 CoT 为 latent states 提供额外监督；
- **更好的停止方法**：例如根据 hidden state 变化、预测置信度、entropy、相邻 latent step 的相似度等信号，动态决定是否继续 latent reasoning；
- **更好的 latent state transition**：例如改进 latent state 的更新方式，加入门控、残差连接、归一化或轻量变换模块；
- **更灵活的 latent length / compression 策略**：例如让不同样本使用不同数量的 latent steps，或尝试将多个 reasoning steps 压缩到更少的 latent states；
- **更合理的训练目标或 curriculum**：例如比较不同阶段训练策略、不同 CoT 替换比例，或不同答案监督方式对结果的影响；
- **更充分的分析**：例如分析 latent step 数量、生成长度、准确率之间的关系，或比较哪些类型的问题更适合 latent reasoning。

---

## 四、实验要求

### 4.1 必做实验

#### （1）基线对比实验
比较以下三种方法在测试集上的性能：

- Direct Answer
- Explicit CoT
- Latent Reasoning（Coconut-style）

至少报告以下指标：

- **Accuracy**
- **Average Generation Length**

其中：

- **Accuracy** 作为主指标；
- **Average Generation Length** 用于衡量模型显式输出长度与推理开销。

Coconut 在实验中也分析了新生成 token 数，以衡量 reasoning efficiency。

#### （2）Latent Step 数分析
请至少比较三组不同 latent step 数量，例如：

- `T = 1`
- `T = 2`
- `T = 4`

分析 latent step 数变化对准确率和显式生成长度的影响。

### 4.2 鼓励完成的进阶实验
在基础 latent reasoning 方法之上，鼓励同学们设计一种简单改进方法，并通过实验验证其有效性。

进阶实验可以从以下两个目标中任选其一：

1. 在准确率上优于基础 latent reasoning；
2. 在保持相近准确率的前提下，进一步减少 Average Generation Length。

---

## 五、实现与资源建议

### 5.1 模型规模
本项目统一使用 **GPT-2** 作为基础模型。
选择 GPT-2 的主要原因在于：

- 模型规模较小，便于课程项目复现；
- Coconut 与 CODI 等工作都在 GPT-2 尺度上进行了验证；
- 相比更大模型，更适合在有限 GPU 资源下完成实验。

### 5.2 运行资源
由于 latent reasoning 训练通常需要多次 forward 或多阶段训练，其训练时间可能长于普通 Direct Answer / CoT 微调。
请同学们根据自己的硬件资源合理设置：

- batch size
- epoch 数
- latent step 数
- 是否使用 gradient accumulation

若训练资源有限，可在报告中说明你所采用的折中方案。

### 5.3 复现原则
本项目更看重：

- 方法实现是否完整；
- 对比是否公平；
- 分析是否清楚；

而不是追求特别高的绝对分数。
如果由于资源限制未能进行大规模训练，也可以通过合理实验与分析展示你对 latent reasoning 的理解。

---

## 六、提交内容

### 6.1 实验报告
本次大作业仅需提交 **实验报告**，不要求提交完整代码。报告建议控制在合理篇幅内，重点说明方法、实验和分析结果。

报告需包含以下内容：

1. **任务与方法概述**
   简要说明 latent reasoning 的任务目标，并介绍 Direct Answer、Explicit CoT 和 Coconut-style Latent Reasoning 的基本实现思路。

2. **实验设置**
   说明所使用的数据集、模型、训练设置、latent step 数量以及答案抽取方式。若由于硬件资源限制对训练规模、epoch 数或数据量进行了简化，需要在此说明。

3. **实验结果**
   至少报告 Direct Answer、Explicit CoT 和 Latent Reasoning 的对比结果，并包含 Accuracy 和 Average Generation Length 两个指标。

4. **Latent Step 分析与讨论**
   比较不同 latent step 数量对性能和生成长度的影响，并结合实验结果讨论 latent reasoning 相比显式 CoT 的优点与不足。

5. **总结与参考资料**
   简要总结实验发现。若参考了论文、代码库或开源实现，需要列出对应资料。

### 6.2 报告要求
- 报告应条理清晰，重点突出实验结果与分析；
- 图表需有标题和必要说明；
- 不要求追求特别高的绝对分数，更看重实验是否公平、分析是否合理；
- 若使用了外部代码、论文或开源实现，需要在报告中注明；
- 若因资源限制简化实验，需要如实说明。

---

## 七、评分标准

| 评分项 | 分值 | 说明 |
|---|---:|---|
| 方法理解与实现说明 | 20% | 是否清楚说明 Direct Answer、Explicit CoT 与 Latent Reasoning 的实现思路 |
| 实验设计合理性 | 25% | 是否进行了公平对比，是否包含 latent step 分析 |
| 结果分析深度 | 30% | 是否对实验结果进行了充分讨论，而非仅给出表格 |
| 报告规范性 | 15% | 结构清晰，表达准确，图表规范，引用完整 |
| 创新与扩展 | 10% | 是否尝试并说明合理的进阶改进方法 |

---

## 八、参考论文与代码

请至少阅读并参考以下工作：

1. **Training Large Language Models to Reason in a Continuous Latent Space（Coconut）**  
   提出连续潜在推理的基础框架，在 GPT-2 与 GSM8K 等任务上验证 latent reasoning 的可行性。  
   参考代码：<https://github.com/facebookresearch/coconut>

2. **CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation**  
   通过 self-distillation 将显式 CoT 压缩到连续空间中，是 implicit CoT 的重要改进方向。  
   参考代码：<https://github.com/zhenyi4/codi>

3. **Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains（CoLaR）**  
   探索多个 reasoning token 的动态压缩、latent reasoning 的速度控制，以及压缩率与性能之间的平衡。  
   参考代码：<https://github.com/xiaomi-research/colar>

---

## 九、补充说明

1. 本项目的基础 latent reasoning 统一以 **Coconut-style** 为准，不要求完整复现所有论文细节，但必须体现连续 latent step 的核心思想。
2. 进阶部分为 **鼓励完成**，不强制要求。
3. 若同学们对训练资源、验证集划分或答案抽取方式进行了特殊处理，需要在报告中说明。
4. 若参考了其他文献、代码库或开源实现，也请在报告中列出对应参考资料。

祝大家探索顺利。
