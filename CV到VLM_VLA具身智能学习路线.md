# 从计算机视觉到 VLM / VLA / 具身智能学习路线

> 目标：在已有计算机视觉基础上，逐步进入 **VLM（Vision-Language Model）→ VLA（Vision-Language-Action）→ 具身智能**，形成既懂视觉感知、又懂多模态理解和机器人动作建模的能力体系。

---

# 1. 先明确：具身智能到底需要什么能力？

具身智能不是单纯的计算机视觉，也不是单纯的大模型。

可以理解为：

```text
具身智能
│
├─ 视觉感知 Vision
│   ├─ 图像理解
│   ├─ 目标检测
│   ├─ 图像分割
│   └─ 视觉特征提取
│
├─ 语言理解 Language
│   ├─ Transformer
│   ├─ LLM
│   └─ 多模态对齐
│
├─ 动作 Action
│   ├─ Robot State
│   ├─ Action Space
│   ├─ Trajectory
│   └─ Policy
│
└─ 机器人 Robotics
    ├─ 坐标系
    ├─ 运动学
    ├─ 控制
    └─ 规划
```

因此：

\[
\text{Embodied AI}
=
\text{Vision}
+
\text{Language}
+
\text{Action}
+
\text{Robotics}
\]

---

# 2. CV、VLM、VLA 的关系

整体路线：

```text
Computer Vision
│
├─ CNN
├─ Detection
├─ Segmentation
└─ Visual Feature
        ↓
Vision Transformer
        ↓
CLIP
        ↓
VLM
Vision + Language
        ↓
理解图像、视频和语言
        ↓
VLA
Vision + Language + Action
        ↓
预测机器人动作
        ↓
具身智能
```

---

# 3. 是否必须先学完 CV 才能学 VLM / VLA？

答案：

> 不需要把传统 CV 全部学完，但必须具备核心视觉基础。

真正需要优先掌握的是：

```text
Python
↓
PyTorch
↓
CNN
↓
Detection / Segmentation 基础
↓
Transformer
↓
ViT
↓
CLIP
↓
VLM
↓
Robotics
↓
Imitation Learning
↓
VLA
```

不建议：

```text
传统视觉所有算法全部学完
↓
再学 VLM
```

也不建议：

```text
没有 CNN / Transformer / ViT 基础
↓
直接调 VLM / VLA
```

最佳路线是：

> CV 学到“能够理解视觉特征和视觉任务”为止，然后尽快进入 ViT / CLIP / VLM。

---

# 4. 当前基础

目前已有：

- [x] Python 基础
- [x] Linux 基础
- [x] Git 基础
- [x] PyTorch 使用经验
- [x] DnCNN
- [x] U-Net
- [x] NAFNet
- [x] Wiener Filter
- [x] 图像恢复基础
- [x] CNN 初步理解
- [x] 图像分割初步基础

因此并不是从零开始。

当前最关键的是：

```text
补齐 CV 基础
↓
迅速进入 Transformer / ViT
↓
从 CLIP 跨到 VLM
↓
再补 Robotics + Action
```

---

# 5. 总学习路线

```text
阶段 1：PyTorch + CNN 基础
        ↓
阶段 2：OpenCV 基础
        ↓
阶段 3：Detection + Segmentation
        ↓
阶段 4：Transformer
        ↓
阶段 5：Vision Transformer
        ↓
阶段 6：CLIP
        ↓
阶段 7：DINO / SAM / Grounding DINO
        ↓
阶段 8：VLM
        ↓
阶段 9：机器人学基础
        ↓
阶段 10：Imitation Learning
        ↓
阶段 11：VLA
        ↓
阶段 12：具身智能完整项目
```

---

# 阶段 1：PyTorch + CNN

## 1.1 PyTorch

必须掌握：

- [ ] Tensor
- [ ] Dataset
- [ ] DataLoader
- [ ] nn.Module
- [ ] forward
- [ ] Loss
- [ ] Optimizer
- [ ] backward
- [ ] train
- [ ] eval
- [ ] checkpoint
- [ ] GPU

训练流程必须能独立写：

```python
optimizer.zero_grad()

output = model(x)

loss = criterion(output, y)

loss.backward()

optimizer.step()
```

必须理解：

```text
Forward
↓
Loss
↓
Backward
↓
Gradient
↓
Optimizer
↓
Parameter Update
```

---

## 1.2 CNN

重点理解：

- [ ] Convolution
- [ ] Kernel
- [ ] Stride
- [ ] Padding
- [ ] Channel
- [ ] Feature Map
- [ ] Receptive Field
- [ ] Pooling
- [ ] Residual Connection

理解视觉模型最关键的概念：

```text
Image
↓
Feature Extraction
↓
Feature Map
↓
High-level Feature
```

### 完成标准

能够解释：

- CNN 为什么能提取图像特征？
- 浅层特征和深层特征有什么区别？
- Feature Map 是什么？
- Backbone 是什么？

---

# 阶段 2：OpenCV 基础

这一阶段不用学得特别深。

重点掌握：

- [ ] 图像读取
- [ ] RGB / BGR
- [ ] Gray
- [ ] Resize
- [ ] Crop
- [ ] Rotate
- [ ] Gaussian Filter
- [ ] Median Filter
- [ ] Canny
- [ ] Threshold
- [ ] Morphology

目标：

> 理解图像处理基本流程，而不是成为传统 CV 专家。

---

# 阶段 3：Detection + Segmentation

这是具身智能视觉感知的基础。

---

## 3.1 目标检测

重点：

- [ ] Bounding Box
- [ ] Confidence
- [ ] IoU
- [ ] NMS
- [ ] Precision
- [ ] Recall
- [ ] AP
- [ ] mAP

学习模型：

```text
YOLO
↓
DETR
↓
Grounding DINO
```

其中：

```text
YOLO
```

用于理解传统 Detection。

```text
DETR
```

用于理解 Transformer 如何做 Detection。

---

## 3.2 图像分割

重点：

- [ ] Semantic Segmentation
- [ ] Instance Segmentation
- [ ] Mask
- [ ] IoU
- [ ] Dice

模型：

```text
U-Net
↓
Mask R-CNN
↓
SAM
```

你已经有 U-Net 基础，因此重点是把它真正用于 segmentation。

---

## 为什么具身智能需要 Detection / Segmentation？

机器人需要回答：

```text
目标是什么？
在哪里？
轮廓是什么？
应该抓哪里？
```

例如：

```text
桌面
│
├─ 红杯子
├─ 苹果
└─ 手机
```

Detection：

```text
找到红杯子的位置
```

Segmentation：

```text
找到红杯子的精确轮廓
```

---

# 阶段 4：Transformer

这是从传统 CV 转向 VLM 的关键节点。

必须重点学习。

---

## 4.1 Self-Attention

理解：

\[
Q=XW_Q
\]

\[
K=XW_K
\]

\[
V=XW_V
\]

Attention：

\[
Attention(Q,K,V)
=
Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

必须理解：

```text
Q
K
V
Attention Score
Softmax
Context
```

---

## 4.2 Multi-Head Attention

理解：

```text
不同 Attention Head
↓
学习不同关系
↓
Concat
↓
Linear
```

---

## 4.3 Transformer Block

理解：

```text
Input
↓
Multi-Head Attention
↓
Residual
↓
LayerNorm
↓
FFN
↓
Residual
↓
LayerNorm
```

---

## 完成标准

必须可以独立解释：

- Self-Attention 是什么？
- 为什么除以 \(\sqrt{d_k}\)？
- Q/K/V 分别是什么？
- Multi-Head 有什么作用？
- Transformer 和 CNN 最大区别是什么？

---

# 阶段 5：Vision Transformer

这一阶段极其重要。

---

## 5.1 ViT 基本结构

```text
Image
↓
Patch
↓
Patch Embedding
↓
Position Embedding
↓
Transformer Encoder
↓
Feature
↓
Prediction
```

---

## 5.2 Patch

一张图片：

\[
224\times224
\]

例如切成：

\[
16\times16
\]

的 Patch。

Patch 数量：

\[
14\times14=196
\]

每个 Patch 变成一个 token。

---

## 5.3 ViT 必须掌握

- [ ] Patch
- [ ] Patch Embedding
- [ ] Position Embedding
- [ ] CLS Token
- [ ] Transformer Encoder
- [ ] Visual Token

---

## 为什么 ViT 很重要？

现代 VLM 常见结构：

```text
Image
↓
Vision Encoder
↓
Visual Tokens
↓
Projector
↓
LLM
↓
Answer
```

Vision Encoder 很多时候就是：

```text
ViT
```

或 ViT 的变体。

---

# 阶段 6：CLIP

CLIP 是从 CV 进入 VLM 的桥梁。

---

## 6.1 基本结构

```text
Image
↓
Image Encoder
↓
Image Embedding


Text
↓
Text Encoder
↓
Text Embedding
```

训练目标：

```text
正确图文对
↓
Embedding 靠近

错误图文对
↓
Embedding 远离
```

---

## 6.2 理解 Contrastive Learning

需要理解：

- [ ] Positive Pair
- [ ] Negative Pair
- [ ] Similarity
- [ ] Cosine Similarity
- [ ] Contrastive Loss

---

## 6.3 CLIP 能做什么？

```text
Zero-shot Classification
Image Retrieval
Text-to-Image Retrieval
Open Vocabulary Recognition
```

---

## 完成标准

能够回答：

- CLIP 为什么可以零样本分类？
- Image Encoder 和 Text Encoder 怎么对齐？
- 为什么 Embedding 很重要？
- Contrastive Learning 是什么？

---

# 阶段 7：DINO / SAM / Grounding DINO

这一阶段进入现代视觉基础模型。

---

# 7.1 DINO

重点理解：

```text
Image
↓
Vision Encoder
↓
Visual Feature
```

理解：

- Self-Supervised Learning
- Representation Learning
- Visual Embedding

---

# 7.2 SAM

SAM：

```text
Image
↓
Image Encoder
       +
Prompt
↓
Prompt Encoder
       ↓
Mask Decoder
       ↓
Segmentation Mask
```

掌握：

- [ ] Image Encoder
- [ ] Prompt Encoder
- [ ] Mask Decoder
- [ ] Point Prompt
- [ ] Box Prompt
- [ ] Mask Prompt

---

# 7.3 Grounding DINO

核心：

```text
Image
+
Text
↓
Grounding DINO
↓
Bounding Box
```

例如：

```text
"red cup"
↓
找到图片中的红杯子
```

---

# 7.4 Grounding DINO + SAM

这是非常值得做的小项目：

```text
Image
+
Text
↓
Grounding DINO
↓
Object Detection
↓
SAM
↓
Precise Mask
```

例如：

```text
"red cup"
↓
找到红杯子
↓
分割红杯子
```

这已经非常接近机器人视觉感知模块。

---

# 阶段 8：VLM

现在正式进入：

```text
Vision + Language
```

---

## 8.1 VLM 基础结构

最简单理解：

```text
Image
↓
Vision Encoder
↓
Visual Feature
↓
Projector
↓
LLM
↓
Text Output
```

---

## 8.2 核心模块

理解：

- [ ] Vision Encoder
- [ ] Visual Token
- [ ] Projector
- [ ] LLM
- [ ] Multimodal Alignment
- [ ] Instruction Tuning

---

## 8.3 重点模型

建议按顺序了解：

```text
CLIP
↓
BLIP / BLIP-2
↓
LLaVA
↓
Qwen-VL
↓
InternVL
```

不要求全部复现。

重点是理解架构演化。

---

## 8.4 LLaVA

非常适合作为第一个完整 VLM 学习对象。

结构：

```text
Image
↓
Vision Encoder
↓
Visual Feature
↓
Projector
↓
LLM
↓
Answer
```

需要理解：

- 为什么要 Projector？
- Visual Token 怎么进入 LLM？
- Vision Encoder 是否冻结？
- Instruction Tuning 是什么？

---

# 阶段 9：机器人学基础

到 VLA 之后必须开始补机器人。

---

## 9.1 坐标系

需要理解：

```text
World Frame
Robot Base Frame
Camera Frame
End-Effector Frame
Object Frame
```

---

## 9.2 Transformation Matrix

齐次变换矩阵：

\[
T=
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
\]

其中：

```text
R = Rotation
t = Translation
```

---

## 9.3 Rotation

掌握：

- [ ] Rotation Matrix
- [ ] Euler Angle
- [ ] Quaternion

---

## 9.4 Forward Kinematics

输入：

\[
q_1,q_2,\dots,q_n
\]

输出：

\[
x,y,z,R
\]

即：

```text
Joint Angle
↓
End-Effector Pose
```

---

## 9.5 Inverse Kinematics

反过来：

```text
Target Pose
↓
Joint Angle
```

---

# 阶段 10：Action 与 Imitation Learning

这是从 VLM 到 VLA 的关键。

---

## 10.1 Observation

机器人输入：

\[
o_t
\]

可能包括：

```text
Camera Image
Robot State
Language Instruction
```

---

## 10.2 Action

输出：

\[
a_t
\]

可能是：

```text
dx
dy
dz
roll
pitch
yaw
gripper
```

或者：

```text
Joint Position
Joint Velocity
```

---

# 10.3 Policy

VLA 最核心形式：

\[
\pi(a_t|o_t)
\]

表示：

> 给定当前 Observation，预测 Action。

---

# 10.4 Behavior Cloning

数据：

```text
Observation
+
Expert Action
```

模型学习：

```text
Observation
↓
Policy
↓
Predicted Action
```

本质上：

> 模仿人类或专家的操作。

---

## 10.5 Imitation Learning

需要掌握：

- [ ] Demonstration
- [ ] Behavior Cloning
- [ ] Policy
- [ ] Distribution Shift
- [ ] Offline Dataset

---

# 阶段 11：VLA

现在进入真正的：

```text
Vision
+
Language
+
Action
```

---

## 11.1 VLA 基本结构

```text
Image / Video
+
Language Instruction
+
Robot State
↓
Multimodal Model
↓
Action Tokens / Continuous Action
↓
Robot
```

---

## 11.2 和 VLM 的区别

VLM：

```text
Image + Text
↓
Text
```

VLA：

```text
Image + Text + State
↓
Action
```

因此：

\[
VLM:
P(text|image,text)
\]

而：

\[
VLA:
P(action|image,text,state)
\]

---

## 11.3 Action Representation

需要理解两种方式：

### Continuous Action

直接输出：

\[
[x,y,z,r,p,y,g]
\]

### Action Token

把动作离散化：

```text
Action
↓
Token
↓
Transformer Prediction
```

---

## 11.4 Action Chunking

不是只预测下一步：

\[
a_t
\]

而是一次预测：

\[
a_t,a_{t+1},...,a_{t+k}
\]

称为：

```text
Action Chunk
```

这在 VLA 中非常重要。

---

# 阶段 12：强化学习基础

VLA 初期可以不深入 RL，但后续需要认识。

学习：

- [ ] State
- [ ] Action
- [ ] Reward
- [ ] Policy
- [ ] Value
- [ ] Q Function
- [ ] MDP

之后再学习：

```text
PPO
SAC
Offline RL
RLHF / RL for Robotics
```

注意：

> 第一阶段进入具身智能，Imitation Learning 优先级高于 Reinforcement Learning。

---

# 13. 最终技能树

最终目标：

```text
Python
+
PyTorch
+
Computer Vision
+
CNN
+
Detection
+
Segmentation
+
Transformer
+
ViT
+
CLIP
+
SAM / DINO / Grounding DINO
+
VLM
+
Robotics
+
Imitation Learning
+
VLA
```

---

# 14. 如果未来走工业 CV

路线：

```text
OpenCV
↓
CNN
↓
YOLO
↓
U-Net
↓
PatchCore
↓
ONNX
↓
TensorRT
↓
C++
↓
CUDA
↓
Industrial Vision
```

---

# 15. 如果未来走 VLM / 具身智能

路线：

```text
PyTorch
↓
CNN
↓
Detection / Segmentation
↓
Transformer
↓
ViT
↓
CLIP
↓
SAM / DINO / Grounding DINO
↓
VLM
↓
Robotics
↓
Imitation Learning
↓
VLA
```

---

# 16. 两条路线的共同基础

前半段其实高度重合：

```text
                 ┌──────── Industrial CV
                 │
Python            │
PyTorch           │
CNN               │
Detection ────────┤
Segmentation      │
Transformer       │
                 │
                 └──────── Embodied AI
                           ↓
                          ViT
                           ↓
                          CLIP
                           ↓
                          VLM
                           ↓
                      Robotics
                           ↓
                         VLA
```

因此：

> 当前学习 CV 并不会浪费。

---

# 17. 优先级调整

如果目标逐渐偏向具身智能：

## 高优先级

```text
PyTorch           ★★★★★
CNN               ★★★★★
Transformer       ★★★★★
ViT               ★★★★★
CLIP              ★★★★★
VLM               ★★★★★
Detection         ★★★★☆
Segmentation      ★★★★☆
Robotics          ★★★★☆
Imitation Learning★★★★★
VLA               ★★★★★
```

---

## 中等优先级

```text
OpenCV            ★★★☆☆
SAM               ★★★★☆
DINO              ★★★★☆
Grounding DINO    ★★★★☆
C++               ★★★☆☆
CUDA              ★★★☆☆
```

---

## 可稍后再学

如果不走工业视觉：

```text
PatchCore         ★★☆☆☆
工业相机          ★★☆☆☆
TensorRT 深度优化 ★★☆☆☆
传统滤波深入      ★★☆☆☆
```

---

# 18. 最推荐的实际学习顺序

不要一下子学 VLA。

按下面顺序：

```text
Step 1
PyTorch + CNN
        ↓
Step 2
OpenCV 基础
        ↓
Step 3
YOLO + Detection
        ↓
Step 4
U-Net + Segmentation
        ↓
Step 5
Transformer
        ↓
Step 6
ViT
        ↓
Step 7
CLIP
        ↓
Step 8
Grounding DINO + SAM
        ↓
Step 9
LLaVA / Qwen-VL
        ↓
Step 10
Robotics Basics
        ↓
Step 11
Behavior Cloning
        ↓
Step 12
VLA
```

---

# 19. 推荐项目路线

最终建议形成四个项目。

---

# 项目一：图像恢复

已有基础：

```text
Wiener
DnCNN
U-Net
NAFNet
```

目标：

> 打牢 CNN / PyTorch / 图像特征基础。

---

# 项目二：Detection + Segmentation

```text
YOLO
+
U-Net
```

实现：

```text
Image
↓
YOLO
↓
Bounding Box
↓
U-Net / SAM
↓
Mask
```

---

# 项目三：Open-Vocabulary Vision

推荐：

```text
Grounding DINO
+
SAM
```

输入：

```text
Image
+
"red cup"
```

输出：

```text
Bounding Box
+
Mask
```

这个项目非常适合从 CV 过渡到 VLM / 具身智能。

---

# 项目四：简化版 VLA

最终项目：

```text
Image
+
Language
+
Robot State
↓
Model
↓
Action
```

例如：

```text
"pick up the red cube"
```

输出：

```text
Robot Action Sequence
```

可以先在：

```text
Simulation
```

环境中做。

---

# 20. 面试需要能回答的问题

## CV

- CNN 为什么可以提取图像特征？
- Feature Map 是什么？
- Backbone 是什么？
- YOLO 怎么做目标检测？
- IoU 是什么？
- NMS 是什么？
- U-Net 为什么需要 Skip Connection？

---

## Transformer

- Self-Attention 是什么？
- Q/K/V 是什么？
- Multi-Head Attention 有什么作用？
- Transformer 和 CNN 有什么区别？

---

## ViT

- 图像为什么可以变成 Token？
- Patch Embedding 是什么？
- Position Embedding 是什么？
- ViT 和 CNN 的区别？

---

## CLIP

- CLIP 怎么进行图文对齐？
- Contrastive Learning 是什么？
- 为什么 CLIP 可以 Zero-shot？

---

## VLM

- VLM 的基本架构是什么？
- Vision Encoder 在做什么？
- Projector 为什么存在？
- Visual Token 怎么进入 LLM？
- Multimodal Alignment 是什么？

---

## Robotics

- 什么是机器人坐标系？
- Transformation Matrix 是什么？
- Forward Kinematics 是什么？
- Inverse Kinematics 是什么？
- Quaternion 是什么？

---

## VLA

- VLM 和 VLA 有什么区别？
- Observation 是什么？
- Action Space 是什么？
- Policy 是什么？
- Behavior Cloning 是什么？
- Action Token 是什么？
- Action Chunking 是什么？

---

# 21. 当前最适合你的路线

不要立刻跳到：

```text
VLA
```

当前顺序：

```text
你现在
│
├─ DnCNN
├─ U-Net
├─ NAFNet
└─ PyTorch
        ↓
补 OpenCV
        ↓
YOLO
        ↓
Segmentation
        ↓
Transformer
        ↓
ViT
        ↓
CLIP
        ↓
Grounding DINO + SAM
        ↓
VLM
        ↓
Robotics
        ↓
Imitation Learning
        ↓
VLA
```

---

# 22. 当前打卡清单

先只关注前六步。

## Step 1：PyTorch / CNN

- [ ] Tensor
- [ ] Dataset
- [ ] DataLoader
- [ ] nn.Module
- [ ] Conv2d
- [ ] Feature Map
- [ ] Residual
- [ ] Loss
- [ ] Optimizer
- [ ] Backpropagation

---

## Step 2：OpenCV

- [ ] Image Read
- [ ] RGB / Gray
- [ ] Resize
- [ ] Filter
- [ ] Edge
- [ ] Threshold
- [ ] Morphology

---

## Step 3：Detection

- [ ] Bounding Box
- [ ] IoU
- [ ] NMS
- [ ] Precision
- [ ] Recall
- [ ] mAP
- [ ] YOLO

---

## Step 4：Segmentation

- [ ] Mask
- [ ] U-Net
- [ ] IoU
- [ ] Dice
- [ ] Semantic Segmentation
- [ ] Instance Segmentation

---

## Step 5：Transformer

- [ ] Attention
- [ ] Q/K/V
- [ ] Multi-Head Attention
- [ ] FFN
- [ ] Residual
- [ ] LayerNorm

---

## Step 6：ViT

- [ ] Patch
- [ ] Patch Embedding
- [ ] Position Embedding
- [ ] CLS Token
- [ ] Visual Token

完成这些以后，再进入：

```text
CLIP
↓
VLM
```

---

# 23. 学习原则

## 原则 1

不要一开始追最新 VLA。

先把：

```text
CV
+
Transformer
+
ViT
+
CLIP
```

搞懂。

---

## 原则 2

每个阶段都必须有代码。

```text
Theory
↓
Code
↓
Experiment
↓
Result
↓
Explanation
```

---

## 原则 3

不要只会调 API。

目标是可以解释：

```text
Input
↓
Encoder
↓
Feature
↓
Fusion
↓
Output
```

---

## 原则 4

VLA 最重要的新东西不是“大模型”，而是：

```text
Action
```

一定要重点理解：

```text
Observation
Policy
Action
Trajectory
Behavior Cloning
```

---

# 24. 最终目标

最终形成下面这一条完整能力链：

\[
\boxed{
Vision
\rightarrow
Representation
\rightarrow
Vision\text{-}Language
\rightarrow
Action
\rightarrow
Robot
}
\]

对应：

```text
CNN
↓
ViT
↓
CLIP
↓
VLM
↓
VLA
↓
Embodied AI
```

这就是后续进入具身智能最推荐的主路线。
