# 基于扩散模型的反演与图像编辑
## 一、任务说明

## 1.1 背景要求

近年来，扩散模型（Diffusion Models）在图像生成与编辑任务中表现出强大的能力。相比“从零生成”图像，许多实际应用更关注对一张**已有图像**进行修改，例如：

- 改变图像中的物体属性
- 修改场景风格

为了实现这类任务，一个关键步骤是将真实图像映射回扩散模型的隐空间或噪声轨迹中，即 **Diffusion Inversion（扩散反演）**。

扩散反演的核心思想是：

> 给定一张真实图像，寻找一个合适的 latent noise trajectory，使得扩散模型能够尽可能重建该图像，并支持后续基于 prompt 的可控编辑。

本次实验要求你基于 Hugging Face 或其他公开实现的预训练扩散模型，探索扩散反演方法在图像重建与图像编辑中的作用，重点比较不同方法在以下方面的表现：

- 重建保真度
- 编辑可控性
- 文本一致性

---

## 1.2 核心任务

你需要完成以下核心目标：

1. **实现真实图像到扩散模型噪声空间的反演**
2. **基于反演结果实现文本驱动图像编辑**
3. **比较不同 inversion 方法在重建与编辑任务中的差异**
4. **分析 inversion 质量与编辑质量之间的权衡关系**



---

## 二、方法设计要求

### 2.1 核心组件
| 模块     | 基础功能要求                            |
| -------- | --------------------------------------- |
| 反演模块 | 将输入图像映射到扩散模型隐空间/噪声轨迹 |
| 重建模块 | 根据反演结果重建原图                    |
| 编辑模块 | 根据 target prompt 生成编辑结果         |
| 评估模块 | 计算重建与编辑指标                      |

### 2.2 必选实现要求
本次作业要求**必须实现以下两种方法**：

1. **DDIM Inversion**
2. **Null-text Inversion**



---

## 三、实验实施要求

### 3.1 必做实验

#### 1. 基础功能验证
对输入图像分别进行：

- DDIM Inversion + Reconstruction
- Null-text Inversion + Reconstruction
- 基于 target prompt 的编辑生成

验证点：

- 重建图像是否接近原图
- 编辑结果是否符合目标提示词
- 人物身份或主体结构是否尽量保持

#### 2. 方法对比实验
对比 DDIM Inversion 与 Null-text Inversion：

- 重建质量差异
- 编辑结果差异
- 在不同图像/不同编辑任务下的表现差异

#### 3. 参数对比实验
至少比较 3 组参数设置，例如：

- `guidance_scale`
- `ddim_steps / inversion_steps`
- `optimization_steps`
- `learning_rate`

#### 4. 失败案例分析
至少展示若干失败案例，并分析原因，例如：

- 重建较好但编辑失败
- 编辑成功但身份漂移
- 结构破坏明显
- 反演不稳定导致结果崩坏

### 3.2 可选优化方向
可自行设计并实现改进方法，并与 DDIM Inversion、Null-text Inversion 进行比较，例如改进 Inversion 方式以提升重建质量和改善后续编辑效果，或结合 mask 等局部控制方式提高编辑区域的可控性并减少对非编辑区域的破坏。

---

## 四、提交要求

### 4.1 简化提交内容
```bash
project/
├── inversion/                  # 反演模块代码
│   ├── ddim_inversion.py
│   ├── null_text_inversion.py
│   └── custom_inversion.py     # 可选：自定义改进方法
├── editing/                    # 编辑与评估相关代码
│   ├── edit_pipeline.py
│   ├── evaluator.py
│   └── utils.py
├── data/                       # 实验所用数据或样例数据说明
├── results/                    # 结果展示
│   ├── reconstructions/        # 部分重建结果示例图
│   ├── edits/                  # 部分编辑结果示例图
│   └── metrics.csv             # 主要实验指标汇总表
└── report.pdf                  # 实验报告
```

### **4.2 提交内容**

- **完整代码**：包含反演、重建、编辑与评估流程
- **实验报告（PDF）**：包含方法说明、实验设置、结果分析
- **结果文件**：部分重建图像、部分编辑图像、指标结果表



## **五、性能指标**



| **指标**    | **说明**                         |
| ----------- | -------------------------------- |
| PSNR / SSIM | 衡量重建图像与原图的接近程度     |
| LPIPS       | 衡量感知重建误差                 |
| CLIP Score  | 衡量编辑结果与目标提示词的一致性 |



## **六、评分标准**

| 维度       | 细则说明                                       | 占比 |
| ---------- | ---------------------------------------------- | ---- |
| 功能完整性 | 完成反演、重建、编辑等核心功能                 | 30%  |
| 方法实现   | 正确实现 DDIM Inversion 与 Null-text Inversion | 25%  |
| 实验完成度 | 完成方法对比、参数实验与失败案例分析           | 15%  |
| 算法创新   | 实现可选优化方向，并对改进效果进行分析         | 10%  |
| 报告质量   | 结构清晰、结果分析合理、图表规范               | 20%  |



## **七、注意事项**

- 不得直接调用商用图像 API 完成任务
- 必须体现 **反演 + 重建 + 编辑** 全流程
- 必须实现 **DDIM Inversion** 和 **Null-text Inversion**
- 必须有定量评估与对比实验





## **八、实现支持**

### **推荐模型**

- Stable Diffusion v1.5
- Stable Diffusion 2.1
- SDXL
- 其他公开扩散模型

### **推荐数据集**

- PIE-Bench
- 其他公开图像编辑数据
- 自建小规模测试集（20–100 张图像）

### **参考资源**

- DDIM: *Denoising Diffusion Implicit Models* 
  arXiv: https://arxiv.org/abs/2010.02502

- Null-text Inversion: *Null-text Inversion for Editing Real Images using Guided Diffusion Models*  
  arXiv: https://arxiv.org/abs/2211.09794

- Prompt-to-Prompt: *Prompt-to-Prompt Image Editing with Cross-Attention Control*  
  arXiv: https://arxiv.org/abs/2208.01626

- PIE-Bench（数据集介绍见下述论文）: *Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code*  
  arXiv: https://arxiv.org/abs/2310.01506



