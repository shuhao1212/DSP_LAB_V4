# DSP_LAB_V4 — 语音关键词识别 (KWS) 工程文档

> **给接手此工程的人 / AI 的完整说明**
> 目标芯片: TMS320C6748 DSP (TI C6000 系列)
> 开发环境: Code Composer Studio (CCS) 20.5.1, TI CGT 8.3.10
> GitHub: https://github.com/shuhao1212/DSP_LAB_V4

---

## 1. 工程目录结构

```
D:\DSP_LAB_V4\
├── DSP_LAB_V4\                         ← CCS 工程根目录（用 CCS 打开这个）
│   ├── .cproject                       ← CCS 工程配置（编译器选项、include路径）
│   ├── .project                        ← Eclipse 工程描述
│   ├── .ccsproject                     ← CCS 专用（芯片型号 C6748、仿真器）
│   │
│   ├── Code\                           ← 应用层源代码
│   │   ├── project3.c                  ← 主函数入口 main()
│   │   ├── project3.h                  ← ★ 核心文件：所有 KWS 逻辑都在这里
│   │   ├── Driver\                     ← 底层驱动
│   │   │   ├── 00_sys\                 ← 系统初始化
│   │   │   ├── 01_led\                 ← LED 驱动
│   │   │   ├── 02_key\                 ← 按键驱动
│   │   │   ├── 03_timer\               ← 定时器
│   │   │   ├── 04_adc\                 ← ADC 采集（音频输入，20kHz）
│   │   │   ├── 05_dac\                 ← DAC 输出（音频回放）
│   │   │   ├── 11_lcd\                 ← LCD 显示（800×480）
│   │   │   ├── 12_touch\               ← 触摸屏
│   │   │   ├── 20_platform\            ← 平台外设（GPIO/EDMA/I2C/SPI/UART...）
│   │   │   └── 24_pru\                 ← PRU 协处理器（ADC/DAC 时序控制）
│   │   └── User\
│   │       ├── weights.c               ← ★ 神经网络权重（13类 BC-ResNet）
│   │       ├── weights.h               ← ★ 权重 extern 声明
│   │       ├── weights.c.bak_20260717  ← 旧版权重备份（12 类，已废弃）
│   │       ├── weights.h.bak_20260717  ← 旧版权重头文件备份
│   │       ├── user_include.h          ← 用户驱动汇总头文件
│   │       └── user_driver\            ← 用户层驱动封装
│   │
│   ├── Library\                        ← 静态库（不用管）
│   │   ├── StarterWare\                ← TI 外设库（Grlib/Drivers/SystemConfig...）
│   │   ├── DSPLib\                     ← TI DSP 数学库
│   │   ├── MathLib\                    ← TI 数学库
│   │   └── IMGLib\                     ← TI 图像处理库
│   │
│   ├── Include\                        ← 头文件
│   │   └── StarterWare\Grlib\grlib.h  ← Grlib 图形库 API（LCD 绘图用）
│   │
│   ├── TargetConfig\                   ← 链接器脚本
│   │   ├── C6748.cmd                   ← DSP 内存布局
│   │   └── User.cmd                    ← 用户自定义内存段
│   │
│   ├── Debug\                          ← 编译输出（.out, .obj, .map）
│   │   ├── makefile                    ← ★ 自动生成的 Makefile（列出了所有 .obj）
│   │   └── DSP_LAB_V3.out             ← 最终烧录文件
│   │
│   └── training\                       ← ★ PC 端 PyTorch 训练代码
│       ├── train.py                    ← 训练入口（fine-tune BC-ResNet）
│       ├── model.py                    ← PyTorch 模型定义（与 DSP 逐层对应）
│       ├── data_loader.py              ← 数据加载 + Mel 频谱提取（与 DSP 一致）
│       ├── export_weights.py           ← 导出权重为 .c/.h（给 DSP 用）
│       ├── load_weights.py             ← 从 DSP 的 weights.c 解析权重回 PyTorch
│       ├── eval_model.py               ← 模型评估
│       ├── verify.py                   ← 验证导出的权重
│       ├── train_log.txt               ← 上次训练的完整日志
│       ├── checkpoints\
│       │   ├── best_model.pt           ← 训练最佳模型 (val_acc=86.64%, epoch 33)
│       │   ├── final_weights.h         ← 从 best_model 导出的头文件
│       │   └── final_weights.c.bak     ← 从 best_model 导出的权重（备份）
│       └── __pycache__\
│
└── data\                               ← Google Speech Commands v2 数据集
    ├── down/ go/ left/ no/ off/ on/ right/ zero/ stop/ up/ yes/  ← 命令词 WAV
    ├── _background_noise_/             ← 背景噪声
    ├── sheila/ ...                     ← 其他非命令词（用作 _unknown_ 类）
    ├── validation_list.txt             ← 官方验证集划分
    └── testing_list.txt                ← 官方测试集划分
```

---

## 2. 核心架构：KWS 语音识别流水线

```
麦克风 → ADC(20kHz) → 1024采样块 → VAD(能量检测)
                                         ↓ 检测到语音
                                    累积1秒语音 → 降采样20k→16k
                                         ↓
                                    Mel频谱提取 (40频带×101帧)
                                         ↓
                                    BC-ResNet 前向推理
                                         ↓
                                    13类 logits → softmax → 置信度
                                         ↓
                                    唤醒词检查 + 阈值过滤 → LCD显示
```

### 2.1 音频参数（project3.h）

| 宏 | 值 | 含义 |
|----|-----|------|
| `PROJECT3_HW_SAMPLE_RATE` | 20000 | ADC 采样率 (Hz) |
| `PROJECT3_MODEL_SAMPLE_RATE` | 16000 | 模型期望采样率 (Hz) |
| `PROJECT3_BLOCK_SAMPLES` | 1024 | 每次 DMA 传输的采样数 |
| `PROJECT3_WIN_SIZE` | 480 | FFT 窗长 (30ms @ 16kHz) |
| `PROJECT3_HOP_SIZE` | 160 | 帧移 (10ms @ 16kHz) |
| `PROJECT3_FFT_LEN` | 512 | FFT 点数 |
| `PROJECT3_MELS_NUM` | 40 | Mel 频带数 |
| `PROJECT3_MODEL_FRAMES` | 101 | 时间帧数（≈1秒） |
| `PROJECT3_INPUT_GAIN` | 2.0f | ADC 增益补偿（可调） |

### 2.2 识别词列表（13 类）

| class_id | 标签 | 备注 |
|----------|------|------|
| 0 | `_silence_` | 静音（不显示，直接忽略） |
| 1 | `_unknown_` | 未知词（激活态下显示 "Unknown"） |
| **2** | **down** | |
| **3** | **go** | |
| **4** | **left** | |
| **5** | **no** | |
| **6** | **off** | |
| **7** | **on** | |
| **8** | **right** | |
| **9** | **zero** | ★ 唤醒词 |
| **10** | **stop** | |
| **11** | **up** | |
| **12** | **yes** | |

**enum 定义位置：** `project3.h` 第 127-141 行 `PROJECT3_CLASS_ID`
**标签数组：** `project3.h` 第 199-213 行 `g_project3_labels`
**权重声明：** `weights.h` 第 109-110 行 `fc_weight[416]` (13×32)、`fc_bias[13]`

---

## 3. 当前功能状态

### 3.1 唤醒词机制

```
        上电 → 休眠态 (wake_active=0)
                   │
         说 "zero" (置信度≥50%) → 激活态 (wake_active=1)
                   │                  │
                   │          说其他词 → 识别并显示
                   │          说 "zero" → 回到休眠态
                   │                  │
                   └──────────────────┘
```

- **休眠态：** LCD 显示 "Say Zero..."，LED 熄灭。除 "zero" 外所有语音被忽略。
- **激活态：** LCD 显示 "Listening..."，LED 常亮。识别所有 13 个词。
- **再次说 "zero"：** 回到休眠态，LED 熄灭。
- **唤醒词阈值：** `PROJECT3_CONFIDENCE_DISPLAY_THRESHOLD` (0.50，即 50%)

### 3.2 识别结果显示规则

| 条件 | 休眠态 | 激活态 |
|------|:------:|:------:|
| `_silence_` | 忽略 | 忽略 |
| `_unknown_` | 忽略 | 显示 "Unknown" |
| `zero` + 置信度≥50% | 激活系统 | 回休眠态 |
| 其他命令词 + 置信度≥50% | 忽略 | 显示识别词 |
| 置信度 < 50% | — | 显示 "Please say again" |
| 置信度 < 35% | — | 判为无效 |

### 3.3 DSP 推理加速优化

以下优化已实现在 `project3.h` 中，总计约 **1.5-2x 推理加速**，准确率无损：

| 优化 | 方法 | 加速 |
|------|------|:--:|
| BN 参数预合并 | 启动时预计算 `merged_w = gamma/sqrt(var+eps)`，消除推理时 17 次 sqrt/div | ~20% |
| ScaleBiasReLU | 替代原始 BatchNorm，简化为单次乘加 + ReLU | ~15% |
| 循环展开 | Conv2d/DWConv 内层 `switch-case` 展开 k_w=1/3/5 路径 | ~10% |
| restrict 指针 + pragma | `#pragma MUST_ITERATE` + `restrict` 关键字辅助 VLIW 编译优化 | ~10% |
| AvgPool + FC 优化 | 预计算 `inv_frames`，展开循环 | ~5% |

> **注意：** 曾尝试将帧数从 101 减半到 51 以进一步加速，但准确率从 88% 暴跌至 49%，已否决。

### 3.4 已知问题

| 问题 | 严重程度 | 原因 | 调优方向 |
|------|---------|------|---------|
| **on/down/no 识别错误** | 高 | VAD 切掉了鼻音词尾 (/n/, /m/ 低能量，被误判为静音) | 调大 `stop_frames`(15)、调小 `stop_ratio`(1.3) |
| **置信度整体偏低 (~70%)** | 中 | DSP 音频电平比 PC 训练数据弱 | 调 `PROJECT3_INPUT_GAIN`(当前2.0) |
| **休眠态说话屏幕闪** | 低 | 每次 VAD 触发都会重绘（即使文字不变） | `last_main_text[0]='\0'` 强制了重绘，去掉即可 |
| **屏幕闪烁** | 低 | `Project3_RenderScreen` 是全屏清空+重绘 | 改成只更新变化区域 |
| **VAD 对环境噪声敏感** | 低 | 能量阈值固定 | 按键 KEY2/KEY3 可动态调整 |

### 3.5 PC 端训练结果（最新）

| 指标 | 值 |
|------|-----|
| 最佳模型 | epoch 56, val_acc = 87.54% |
| 测试集准确率 | 87.58% |
| 各词准确率 (PC) | stop: 100%, left/yes: 96%+, zero: 96.2%, on: 92.7% |
| 模型参数量 | 5,793 (22.6 KB float32) |
| 训练参数 | lr=0.001, batch=128, weight_decay=3e-4, label_smoothing=0.05, on权重=1.6 |

---

## 4. 关键可调参数（project3.h）

### 4.1 音频前端

```c
#define PROJECT3_INPUT_GAIN   2.0f    // ADC增益补偿。DSP信号偏弱时调大(1.5~3.0)
```

### 4.2 VAD（语音活动检测，project3.h 第 742-782 行）

```c
stop_ratio  = 1.3f;   // 静音判定比：能量 < noise_floor*1.3 算静音
                       //   减小→更不容易判静音（保留鼻音词尾）
                       //   增大→更敏感（容易截断短词）
stop_frames = 15;     // 需要连续15帧(150ms)静音才结束录音
                       //   增大→保留长尾音（鼻音/m/n）
                       //   减小→响应更快但容易截断
start_ratio = 4.5f;   // 语音起始判定比：能量 > noise_floor*4.5
start_frames = 4;     // 需要连续4帧语音才触发
```

### 4.3 置信度阈值

```c
PROJECT3_INFERENCE_CONF_THRESHOLD   0.35f  // 低于此值→判为无效
PROJECT3_CONFIDENCE_DISPLAY_THRESHOLD  0.50f  // 低于此值→显示"Please say again"
```

### 4.4 UI 时序

```c
PROJECT3_RESULT_HOLD_BLOCKS      20  // 识别结果保持时间 (20×51.2ms≈1秒)
PROJECT3_POST_INFER_IGNORE_BLOCKS 8  // 推理后忽略时间 (防止重复触发)
```

---

## 5. 编译和烧录

### 5.1 用 CCS 编译

1. 用 CCS 20.5.1 打开本工程（Import CCS Project）
2. 选择 **Debug** 配置
3. Project → Build All
4. 输出文件：`Debug/DSP_LAB_V4.out`

### 5.2 烧录到 DSP

1. 用仿真器（XDS100/XDS200）连接 C6748 开发板
2. CCS → Run → Debug → 下载 .out 到 DSP
3. 运行

### 5.3 常见编译问题

- **symbol redefined 错误：** 有重复的 `.c` 文件定义了相同符号。检查是否有 `final_weights.c` 和 `weights.c` 同时存在（目前已处理）。
- **Makefile 是自动生成的：** `Debug/makefile`、`subdir_vars.mk` 都是 CCS 自动生成，不要手动改。如果有新的 `.c` 文件加入工程，CCS 会自动添加到 makefile。

---

## 6. 模型训练（PC 端）

### 6.1 重新训练

```bash
cd training
python train.py
# 默认: 100 epochs, batch=128, lr=0.001, weight_decay=3e-4, patience=25, label_smoothing=0.05
# 从头训练: python train.py --from-scratch
# 恢复训练: python train.py --resume checkpoints/best_model.pt
```

> **数据集路径：** `D:\speech_data`（Google Speech Commands v2，约 105k WAV 文件）
> **Mel 缓存：** 首次运行自动预计算到 `D:\mel_cache`（约 30 分钟），后续秒开。

### 6.2 训练流程

1. `load_weights.py` → 从 `Code/User/weights.c` 解析现有权重
2. `data_loader.py` → 从 `../data/` 加载 Google Speech Commands，提取 Mel 频谱（**与 DSP 端 `Project3_ExtractLogMel()` 逐位一致**）
3. `model.py` → BC-ResNet（**与 DSP 端 `Project3_BCResNetForward()` 逐层一致**）
4. `train.py` → Fine-tune，保存最佳模型到 `checkpoints/best_model.pt`
5. `export_weights.py` → 将模型导出为 `weights.c` + `weights.h`

### 6.3 部署新权重到 DSP

训练完成后自动部署到 `Code/User/weights.c` 和 `Code/User/weights.h`（旧权重自动备份为 `.bak`）。在 CCS 中重新 Build 即可。

> **注意：** 不要在 `training/checkpoints/` 中保留 `.c` 文件，否则 CCS 会重复编译导致 symbol redefined 错误。

### 6.4 模型架构对照

| PyTorch (model.py) | DSP (project3.h) |
|---------------------|------------------|
| `self.conv1` (1→16, 3×3, stride=2×1) | `Project3_Conv2d(logmel→act_a, 1→16, ...)` |
| `self.block1` (16→8) | `Project3_BCResBlock(..., 16→8, stride=1)` |
| `self.block2` (8→12, stride=2) | `Project3_BCResBlock(..., 8→12, stride=2)` |
| `self.block3` (12→16, stride=2) | `Project3_BCResBlock(..., 12→16, stride=2)` |
| `self.dwconv` (16→16, 3×3) | `Project3_DepthwiseConv2d(..., 16)` |
| `self.pwconv` (16→20, 1×1) | `Project3_Conv2d(..., 16→20, 1×1)` |
| `self.conv2` (20→20, 5×1) | `Project3_Conv2d(..., 20→20, 5×1)` |
| `self.expand` (20→32, 1×1) | `Project3_Conv2d(..., 20→32, 1×1)` |
| AvgPool over time → fc(32→13) | 手动 avg + `fc_weight[416]` / `fc_bias[13]` |

---

## 7. 文件清单：哪些需要改、哪些不要动

### 常改文件

| 文件 | 改什么 |
|------|--------|
| `Code/project3.h` | 所有 KWS 逻辑、参数、UI、VAD、状态机 |
| `Code/User/weights.c` | 模型权重（训练后替换） |
| `Code/User/weights.h` | 权重头文件（训练后替换） |

### 偶尔改

| 文件 | 改什么 |
|------|--------|
| `training/train.py` | 训练超参（epochs/lr/batch_size） |
| `training/model.py` | 模型架构（需同步改 DSP 端） |
| `training/data_loader.py` | 数据增强、Mel 参数（需同步改 DSP 端） |

### 不要动

| 文件 | 原因 |
|------|------|
| `Code/Driver/**` | TI StarterWare 驱动，平台相关 |
| `Library/**` | 预编译库 |
| `TargetConfig/**` | 链接器脚本，芯片内存布局 |
| `Debug/makefile` | CCS 自动生成 |
| `.cproject` / `.project` / `.ccsproject` | CCS 工程配置 |

---

## 8. 硬件按键功能

| 按键 | 功能 |
|------|------|
| KEY1 | 清除当前结果，回到监听态 |
| KEY2 | 降低 VAD 灵敏度（噪声基底 ×1.15） |
| KEY3 | 提高 VAD 灵敏度（噪声基底 ×0.85） |
| KEY4 | 重置 VAD 噪声基线 |
| KEY5 | 系统重启（重新初始化） |

---

## 9. LCD 布局（800×480）

```
┌──────────────────────────────────────────────┐ Y=0
│  Voice Command Recognition        [状态灯]   │ Header (56px)
├──────────────────────────────────────────────┤ Y=55
│                                              │
│   ┌──────────────────────────────────┐       │ Y=90
│   │                                  │       │
│   │        "识别结果文字"             │      │ Card (240px)
│   │        (cm48 字体)               │       │
│   │                                  │       │
│   │   Confidence: 85%  ████████░░    │       │
│   └──────────────────────────────────┘       │ Y=330
│                                              │
│                                              │ (空白区域)
├──────────────────────────────────────────────┤ Y=440
│  State: Listening | Recog: 3 | ACTIVE        │ Bottom Bar
│  VAD floor: 1.2e-06 | Energy: 3.5e-04       │ (40px)
└──────────────────────────────────────────────┘ Y=480
```

---

## 10. Bug 排查速查表

| 现象 | 可能原因 | 查哪里 |
|------|---------|--------|
| 编译报 symbol redefined | 两个 .c 定义了相同全局变量 | 检查工程里有没有重复的权重文件 |
| 所有词识别不到 | ADC 没工作或增益太低 | 查 `PROJECT3_INPUT_GAIN`、ADC 初始化 |
| 某几个词总识别错 | VAD 截断了词尾 | 调 VAD 的 `stop_ratio`/`stop_frames` |
| 置信度整体偏低 | 音频电平不够 | 调大 `PROJECT3_INPUT_GAIN` |
| 屏幕闪 | 重绘太频繁 | `last_main_text[0]='\0'` 强制了重绘 |
| 唤醒词不灵敏 | 阈值太高 | 调低 `CONFIDENCE_DISPLAY_THRESHOLD` |
| 不唤醒也能识别 | `wake_active` 没正确初始化 | 查 `Project3_ModelInit` |

---

*文档更新日期: 2026-07-22*
