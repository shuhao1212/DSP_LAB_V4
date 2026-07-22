# 训练与部署会话记录 — 2026-07-21

## 本次完成的工作

### 1. 13 类训练适配
- 将模型从 12 类扩展到 13 类（新增 `zero` 类别）
- 原始计划用 `start`，但 Google Speech Commands v2 数据集中无此词，改为 `zero`
- 修改文件：
  - `training/data_loader.py` — `start` → `zero`，`DATA_ROOT` 指向项目内 `data/` 目录
  - `training/model.py` — 默认参数 12 → 13
  - `training/train.py` — 模型创建 12 → 13，增加 `--resume` 续训功能
  - `training/verify.py` — fc 层形状 (12,32) → (13,32)
  - `training/eval_model.py` — 所有硬编码 12 → 13
  - `training/deploy_weights.py` — 12 → 13
  - `training/load_weights.py` — `__main__` 测试块 12 → 13
  - `training/export_weights.py` — `__main__` 测试块 12 → 13

### 2. 训练
- 环境：PyTorch 2.5.1 + CUDA 12.1，GeForce RTX 4070 Laptop GPU
- 数据集：DSP_LAB_V3/data/（Google Speech Commands v2）
- 首次运行预计算 Mel 频谱缓存（~130K 个，约 30 分钟）
- 训练 50 epochs，中途因 emoji 编码 bug 中断（epoch 10），修复后续训
- 修复 `train.py` 中 emoji 字符（⚠️✅❌→ ASCII）导致的 Windows GBK 编码崩溃

### 3. 最终训练结果
- 测试集准确率：**86.44%**
- 最佳验证准确率：**86.64%**（epoch 33）
- 11 个命令词全部 >91%，8 个 >95%
- 模型参数：5,793（22.6 KB）
- 权重已导出：`training/checkpoints/final_weights.c` + `final_weights.h`

### 4. DSP 工程更新
- `Code/project3.h` — `PROJECT3_CMD_COUNT` 12 → 13，标签数组增加 "zero"
- `Code/User/weights.c` + `Code/User/weights.h` — 替换为 13 类新权重
- `Code/project3.c` — 增加 `Led_Init()`
- `Code/project3.h` — 识别成功点亮 LED，超时自动熄灭

### 5. LED 反馈
- 使用 `Led_Control(LED1_CORE, LED_ON/OFF)` API
- 识别到命令词 → LED 亮；显示超时/KEY1 清除 → LED 灭

### 6. 产出文件
- `TRAINING_REPORT.md` — 完整训练报告（架构、历程、混淆矩阵、部署步骤）
- `SESSION_LOG.md` — 本文件

## 当前状态
CCS 工程已就绪，可直接编译烧录到 TMS320C6748 DSP。
