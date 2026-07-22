# KWS 模型训练报告

## 概况

| 项目 | 值 |
|------|-----|
| 网络架构 | BC-ResNet |
| 类别数 | 13 |
| 参数量 | 5,793（22.6 KB float32） |
| 训练平台 | PyTorch 2.5.1 + CUDA 12.1 |
| GPU | NVIDIA GeForce RTX 4070 Laptop |
| 数据集 | Google Speech Commands v2 |
| 训练轮数 | 50 epochs |
| 最佳验证准确率 | **86.64%**（epoch 33） |
| 测试集准确率 | **86.44%** |

---

## 类别定义（13 类）

```
['_silence_', '_unknown_', 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'zero', 'stop', 'up', 'yes']
```

> 注：在原始 12 类基础上增加 `zero`，替换了不可用的 `start`（Google Speech Commands v2 数据集中无此词）。

---

## 模型架构

```
Input: Log-Mel Spectrogram [1, 40, 101]
  │
  ├─ conv1: 1→16, 3×3, stride=(2,1) → BN → ReLU     [16, 20, 101]
  ├─ BC-ResBlock 1: 16→8,  stride=1                   [8,  20, 101]
  ├─ BC-ResBlock 2: 8→12,  stride=(2,1)               [12, 10, 101]
  ├─ BC-ResBlock 3: 12→16, stride=(2,1)               [16,  5, 101]
  ├─ DWConv: 16→16, 3×3 → BN → ReLU                   [16,  5, 101]
  ├─ PWConv: 16→20, 1×1 → BN → ReLU                   [20,  5, 101]
  ├─ conv2: 20→20, kernel=(5,1) → BN → ReLU           [20,  1, 101]
  ├─ expand: 20→32, 1×1 → BN → ReLU                   [32,  1, 101]
  ├─ GlobalAvgPool (time dim)                          [32,  1,   1]
  └─ FC: 32→13                                        [13]
```

每个 BC-ResBlock = expand(1×1) → DWConv(3×3) → project(1×1) + shortcut

---

## 训练配置

```bash
python train.py --epochs 50 --batch-size 64 --lr 0.001
```

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 初始学习率 | 0.001（分类器）/ 0.0001（骨干网络） |
| 学习率调度 | CosineAnnealingLR |
| 权重衰减 | 1e-4 |
| 数据增强 | 时间偏移 ±100ms、音量扰动 ±3dB、背景噪声混合（SNR 5-15dB）、SpecAugment |
| 预训练 | 从 DSP 端 weights.c 加载（fc 层随机初始化） |

---

## 训练历程

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | 备注 |
|-------|-----------|-----------|----------|----------|------|
| 1 | 1.1482 | 74.7% | 0.9199 | 66.7% | 初始 |
| 5 | 0.4586 | 84.9% | 0.5585 | 79.9% | |
| 10 | 0.4167 | 86.3% | 0.5136 | 81.9% | 第1次中断（编码bug） |
| 15 | 0.4012 | 86.7% | 0.5151 | 81.9% | |
| 20 | 0.3854 | 87.2% | 0.5019 | 82.4% | |
| 22 | 0.3819 | 87.4% | 0.4202 | 85.4% | 突破85% |
| 25 | 0.3729 | 87.5% | 0.4910 | 82.7% | |
| 30 | 0.3659 | 87.9% | 0.4380 | 84.8% | |
| 33 | 0.3626 | 88.0% | 0.3866 | **86.6%** | 最佳验证 |
| 35 | 0.3574 | 88.1% | 0.4906 | 82.9% | |
| 40 | 0.3528 | 88.2% | 0.4323 | 84.9% | |
| 44 | 0.3531 | 88.3% | 0.4263 | 85.3% | |
| 50 | 0.3461 | 88.6% | 0.4382 | 84.9% | 训练结束 |

---

## 测试集结果

**Test Accuracy: 86.44%**（11005 样本）

### 逐类准确率

| 类别 | 准确率 | 样本数 |
|------|--------|--------|
| stop | 98.8% | 411 |
| left | 97.1% | 412 |
| yes | 96.9% | 419 |
| zero | 96.7% | 418 |
| no | 96.0% | 405 |
| right | 95.7% | 396 |
| up | 95.1% | 425 |
| down | 94.6% | 406 |
| go | 94.3% | 402 |
| off | 92.0% | 402 |
| on | 91.4% | 396 |
| _unknown_ | 80.3% | 6513 |
| _silence_ | — | 0 |

### 混淆矩阵

```
           _sile  _unkn   down     go   left     no    off     on  right   zero   stop     up    yes
_silence_      0      0      0      0      0      0      0      0      0      0      0      0      0
_unknown_      0   5230    137    171    122    160     40    111    169    184    123     44     22
   down       0      4    384      7      1     10      0      0      0      0      0      0      0
     go       0      3      9    379      1      6      2      0      1      0      1      0      0
   left       0      1      0      1    400      2      0      0      1      1      0      1      5
     no       0      3      3      5      2    389      0      0      0      0      1      0      2
    off       0      2      0      9      2      0    370      2      0      0      1     16      0
     on       0      9      2      5      0      0     11    362      0      0      1      6      0
  right       0     12      0      1      1      0      1      0    379      0      1      1      0
   zero       0     10      0      1      0      0      1      0      1    404      1      0      0
   stop       0      2      0      1      1      0      0      0      0      1    406      0      0
     up       0      4      2      0      1      0      4      6      1      0      3    404      0
    yes       0      8      0      0      3      0      0      0      0      2      0      0    406
```

---

## 权重部署

训练好的权重已导出至 `training/checkpoints/`：

```
final_weights.c  (122 KB)
final_weights.h  (  5 KB)
```

**部署步骤：**

```bash
cp training/checkpoints/final_weights.c  Code/User/weights.c
cp training/checkpoints/final_weights.h  Code/User/weights.h
```

然后在 CCS 中重新编译工程即可。

---

## 文件索引

| 文件 | 用途 |
|------|------|
| `training/model.py` | BC-ResNet PyTorch 模型定义 |
| `training/train.py` | 训练脚本（支持 --resume 续训） |
| `training/data_loader.py` | 数据加载 + Mel 频谱预计算 |
| `training/load_weights.py` | 从 DSP weights.c 加载预训练权重 |
| `training/export_weights.py` | 导出训练权重为 weights.c/h |
| `training/verify.py` | PC-DSP 推理一致性验证 |
| `training/eval_model.py` | 独立模型评估脚本 |
| `training/deploy_weights.py` | 一键部署权重到 DSP 工程 |
| `training/checkpoints/best_model.pt` | 最佳模型 checkpoint |

---

## 复现训练

```bash
cd training
python train.py --epochs 50 --batch-size 64 --lr 0.001
```

首次运行会自动预计算 Mel 频谱缓存（约 30 分钟），后续运行直接加载缓存。

从 checkpoint 续训：

```bash
python train.py --epochs 50 --resume checkpoints/best_model.pt
```
