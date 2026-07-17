# KWS 项目状态 — 换手文档

## 当前状态：能跑，准确率 ~58%

同学的正确工程已跑通。LCD 显示 + 麦克风采集 + VAD 自动检测 + 模型推理。

## 文件结构（有效文件）

```
Code/
  project3.c       ← main()，同学的原始代码
  project3.h       ← 全部实现（FFT/Mel/VAD/推理/LCD），同学的原始代码
  User/
    weights.c      ← 模型权重（float，119KB）
    weights.h      ← 权重声明
    user_include.h ← 驱动头文件（已清理干净）
    user_driver/   ← 各种驱动（ADC/LCD/Key/Touch...）

First Commit_Project3/  ← 全部 .ref 后缀，仅作参考，不参与编译
LESSONS_LEARNED.md      ← 经验教训
```

## 模型信息

- **架构**: DS-CNN（深度可分离卷积），3×3 head → 3 blocks → classifier
- **权重**: `Code/User/weights.c`（同学提供的已训练 float 权重）
- **ADC**: 20kHz
- **重采样**: 5:4 抽取→16kHz（`Project3_BuildModelWave` 中实现）
- **特征**: 101 帧 × 40 Mel，镜像填充
- **VAD**: 能量检测，自动识别语音起止

## 测试结果（眼睛看 LCD）

| 词 | 结果 |
|----|------|
| go, up, yes, off | ✅ 100% 准确 |
| left, right | ⚠️ ~60% |
| down, on, no, stop | ❌ 几乎全错 |

## 下一步优化方向

1. **提 VAD 灵敏度**：按板子 KEY3，修复 "down" 漏识别
2. **重新训练模型**：用我们写的 `train_bcresnet.py`（在 Code\User\kws\ 的备份里，或重写）+ 数据增强
3. **导出权重替换**：训完用 `export_weights.py.bak` 导出 → 替换 `weights.c/h`

## 关键约束

- **不要动同学代码**：`project3.c/h` 是正确基准，别改
- **UARTprintf 不支持 %f、%l、宽度对齐**
- **20kHz ADC 是正确选择**，不要改回 50kHz
- **float 不是 double**

## 历史对话

上一个 AI 花了一天半在信号处理上折腾（FIR、IIR、CMVN、线性插值 vs 抽取），根因是用了 50kHz ADC 导致频谱混叠。换成同学的 20kHz 方案后直接跑通。详见 `LESSONS_LEARNED.md`。
