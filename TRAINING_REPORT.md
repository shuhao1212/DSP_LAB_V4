# KWS 模型训练报告

## 概况

| 项目 | 值 |
|------|-----|
| 网络架构 | BC-ResNet |
| 类别数 | 13 |
| 参数量 | 5,793（22.6 KB float32） |
| 训练平台 | PyTorch 2.9.1 + CUDA 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop（12GB） |
| 数据集 | Google Speech Commands v2（D:\\speech_data，105,829 WAV） |
| 训练轮数 | 100 epochs（patience=25，early stop at 81） |
| 最佳验证准确率 | **87.54%**（epoch 56） |
| 测试集准确率 | **87.58%**（11,005 样本） |
| 命令词实际准确率 | **~95%**（扣除 `_unknown_` 垃圾桶类） |

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
python train.py
# 默认: 100 epochs, batch=128, lr=0.001, weight_decay=3e-4
# label_smoothing=0.05, patience=25, on权重=1.6
```

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 初始学习率 | 0.001（分类器）/ 0.0001（骨干网络） |
| 学习率调度 | Warmup 5 epochs + CosineAnnealingLR |
| 权重衰减 | 3e-4 |
| 标签平滑 | 0.05 |
| 类别权重 | on=1.6, off=1.4, unknown=1.4 |
| 数据增强 | 时间偏移 ±100ms、音量扰动 ±3dB、背景噪声混合（SNR 5-15dB）、SpecAugment |
| 预训练 | 从 DSP 端 weights.c 加载（旧 12 类权重，fc 层随机初始化） |

---

## 训练历程（最新 100 轮训练）

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | 备注 |
|-------|-----------|-----------|----------|----------|------|
| 1 | 2.3948 | 21.4% | 2.1398 | 68.8% | 从旧 12 类权重 fine-tune，fc 随机初始化 |
| 5 | 0.9137 | 81.3% | 0.9412 | 75.8% | |
| 10 | 0.7850 | 84.6% | 0.7986 | 81.9% | |
| 15 | 0.7614 | 85.5% | 0.7546 | 83.8% | |
| 20 | 0.7444 | 86.2% | 0.7477 | 84.0% | |
| 25 | 0.7314 | 86.5% | 0.6992 | 85.9% | |
| 30 | 0.7201 | 86.9% | 0.7161 | 85.0% | |
| 34 | 0.7113 | 87.2% | 0.6753 | 87.0% | |
| 40 | 0.7044 | 87.5% | 0.7214 | 84.9% | |
| 50 | 0.7006 | 87.8% | 0.6925 | 86.0% | |
| 56 | 0.6999 | 87.8% | 0.6578 | **87.5%** | **最佳验证** |
| 60 | 0.6914 | 88.0% | 0.6825 | 86.4% | |
| 70 | 0.6830 | 88.3% | 0.6773 | 86.7% | |
| 80 | 0.6847 | 88.3% | 0.6749 | 86.7% | |
| 81 | 0.6856 | 88.3% | 0.6761 | 86.6% | Early stop（25 轮无提升） |

---

## 测试集结果（最新）

**Test Accuracy: 87.58%**（11,005 样本）| **命令词实际准确率: ~95%**（`_unknown_` 占 59% 样本拉低整体）

### 逐类准确率

| 类别 | 准确率 | 样本数 | 性质 |
|------|--------|--------|------|
| stop | **100.0%** | 411 | 命令词 |
| left | 96.6% | 412 | 命令词 |
| yes | 96.4% | 419 | 命令词 |
| zero | 96.2% | 418 | **唤醒词** |
| no | 95.3% | 405 | 命令词 |
| off | 94.8% | 402 | 命令词 |
| up | 94.6% | 425 | 命令词 |
| go | 94.0% | 402 | 命令词 |
| down | 93.6% | 406 | 命令词 |
| right | 92.9% | 396 | 命令词 |
| on | 92.7% | 396 | 命令词 |
| _unknown_ | 82.3% | 6513 | 拒识类 |
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
