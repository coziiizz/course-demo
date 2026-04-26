
# 测试时自适应（TTA）

## 任务背景

近年来，深度学习模型在标准数据集上取得了优异表现，然而，在实际部署中，由于测试数据与训练数据之间存在分布偏移（distribution shift），模型性能往往显著下降。
测试时自适应（Test-Time Adaptation, TTA）作为一种新兴技术，旨在**在不使用训练数据的情况下，仅利用测试数据对模型进行动态调整**，从而提升模型在未知环境中的泛化能力。

本任务旨在让同学们在掌握深度学习模型基础的前提下，深入理解并实践主流TTA方法，并通过实验对比它们在分布偏移场景中的性能、稳定性与效率。

---

## 任务目标

* 阅读并理解主流TTA方法的核心原理与实现机制
* 选择至少三种TTA方法进行实现
* 在分布偏移任务上进行测试与评估
* 比较它们的性能（准确率等）、推理开销等指标
* 撰写实验报告，分析各方法的优劣和适用场景

---

## 任务内容

### 1. 选题与准备

* 可选任务（任选其一）：

  * **图像任务**

    * 使用 CIFAR-10 / CIFAR-100 / ImageNet 数据集（至少三个)，可选择结合 CIFAR-10-C（含多种分布偏移）以及多标签图像数据集MS-COCO / VOC / NUS-WIDE
    * 基于CLIP / BLIP等预训练模型 (至少两个，推荐选择CLIP-VIT / CLIP RESNET50)

  * **时间序列任务**

    * 使用ETTh / EXCHANGE / Weather 等时序基准数据集（至少三个)
    * 基于Transformer / Linear 为架构的时间序列模型（至少两个)
      

---


### 2. 方法实现（至少选择三种，需覆盖不同策略类别）：

#### **提示调优类**

例如
* TPT（Test-time Prompt Tuning）
* DiffTPT（Diverse Data Augmentation with Diffusions）

#### **统计调整类**

例如
* DOTA（Distributional Test-Time Adaptation）
* ADAPT（Probabilistic Gaussian Alignment）

#### **参数更新类**

例如
* SHOT（Source Hypothesis Transfer）
* CoTTA（Continual Test-Time Adaptation）

可以使用PyTorch、开源实现或自行复现。

---

### 3. 实验与评估

* 在相同任务和模型初始化下，分别应用各TTA方法

* 控制变量（模型结构、数据、训练方式一致）

* 评估并记录以下指标：

  * 测试集性能（Accuracy / F1-score 等）
  * 不同分布偏移强度下的鲁棒性
  * 推理时间开销（是否影响实时性）
  * 是否需要额外存储（缓存/历史数据）
  * 是否存在性能退化或不稳定现象

---


## 进阶方向

### 1. 复杂分布偏移场景探索 (参考方法ADAPT)
探索更具挑战性的分布偏移类型，包括：
- 雾化（Fog / Haze）
- 噪声干扰（Gaussian Noise / Shot Noise）
- 模糊（Motion Blur / Defocus Blur）

### 2. 多标签任务下的 TTA 研究 (参考方法BEM)
将测试时自适应方法扩展至多标签任务，重点分析：
- 不同标签间相关性对 TTA 效果的影响
- 预测熵最小化在多标签场景下的适用性
- 各方法在多标签任务中的性能变化

### 3. 设计一种全新的 TTA 方法
可从以下视角出发：
- 引入权重正则化（Weight Regularization）
- 使用历史样本缓存（Memory Buffer）
- 动态调整学习率或更新频率



---

## 评分标准

| **评分项**           | **分值** |
| ----------------- | ------ |
| 实现完整性（≥3种方法）      | 30%    |
| 实验设计科学性（控制变量、复现性） | 20%    |
| 结果分析与对比深度         | 20%    |
| 报告规范性与表达清晰度       | 20%    |
| 创新点（改进方法、扩展实验等）   | 10%    |

---

## 参考资源

* TPT: [https://arxiv.org/abs/2209.07511](https://https://arxiv.org/abs/2209.07511)
* DiffTPT: [https://arxiv.org/abs//2308.06038](https://arxiv.org/abs/2308.06038)
* DOTA: [https://arxiv.org/abs/2409.19375](https://arxiv.org/abs/2409.19375)
* ADAPT: [https://arxiv.org/abs/2508.15568](https://arxiv.org/abs/2508.15568)
* CoTTA: [https://arxiv.org/abs/2203.13591](https://arxiv.org/abs/2203.13591)
* SHOT: [https://arxiv.org/abs/2002.08546](https://arxiv.org/abs/2002.08546)
* BEM:[https://arxiv.org/abs/2502.03777](https://arxiv.org/abs/2502.03777)

#### 如有疑问，请联系助教:za@smail.nju.edu.cn


