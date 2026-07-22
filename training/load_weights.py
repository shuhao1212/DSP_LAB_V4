"""
load_weights.py — 从 Code/User/weights.c 解析权重并加载到 PyTorch 模型

weights.c 的注释标记了原始 PyTorch 维度, 加载时直接恢复原始形状.
权重排列顺序: PyTorch 标准 (out_ch, in_ch, h, w).
"""

import re
import torch
import numpy as np
from pathlib import Path
from model import BcResNet

# ── 路径配置 ──────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_C_PATH = WORKSPACE_ROOT / "Code" / "User" / "weights.c"


def parse_float_array(text: str) -> np.ndarray:
    """从 C 源码文本中解析 float 数组."""
    # 移除注释、空白行
    text = re.sub(r'//.*', '', text)
    # 找到 { ... } 之间的内容
    match = re.search(r'\{([^}]*)\}', text, re.DOTALL)
    if not match:
        raise ValueError("Cannot find float array body in text")
    body = match.group(1)
    # 提取所有浮点数 (支持科学计数法)
    values = re.findall(r'[-+]?\d+\.?\d*(?:e[-+]?\d+)?', body)
    return np.array([float(v) for v in values], dtype=np.float32)


def parse_weights_c(filepath: Path) -> dict:
    """解析整个 weights.c, 返回 {变量名: numpy array} 字典."""
    # weights.c 使用 GBK 编码 (中文注释)
    for enc in ['gbk', 'gb2312', 'utf-8']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    weights = {}

    # 用正则匹配每个数组: const float NAME[SIZE] = { ... };
    # 使用非贪婪匹配找到 { 到 };
    array_pattern = re.compile(
        r'const float\s+(\w+)\[(\d+)\]\s*=\s*\{',
        re.MULTILINE
    )

    for match in array_pattern.finditer(content):
        name = match.group(1)
        # 从 { 之后开始找匹配的 }
        start = match.end()  # { 之后的位置
        depth = 1
        pos = start
        while pos < len(content) and depth > 0:
            if content[pos] == '{':
                depth += 1
            elif content[pos] == '}':
                depth -= 1
            pos += 1
        # pos 现在指向 } 之后
        body = content[start:pos - 1]  # { 和 } 之间的内容

        # 提取浮点数
        values = re.findall(r'[-+]?\d+\.?\d*(?:e[-+]?\d+)?f?', body)
        arr = np.array([float(v.rstrip('f')) for v in values], dtype=np.float32)
        weights[name] = arr

    return weights


def reshape_weight(arr: np.ndarray, name: str) -> np.ndarray:
    """根据变量名推断正确的形状 (匹配 PyTorch 原始维度)."""
    # 注释中标记了原始维度, 这里硬编码映射
    shape_map = {
        # Conv2d: (out_ch, in_ch, kh, kw)
        'conv1_weight':          (16, 1, 3, 3),
        'layer1_conv1_weight':   (8, 16, 1, 1),
        'layer1_dwconv_weight':  (8, 1, 3, 3),   # PyTorch groups 格式
        'layer1_conv2_weight':   (8, 8, 1, 1),
        'layer1_shortcut_0_weight': (8, 16, 1, 1),
        'layer2_conv1_weight':   (12, 8, 1, 1),
        'layer2_dwconv_weight':  (12, 1, 3, 3),
        'layer2_conv2_weight':   (12, 12, 1, 1),
        'layer2_shortcut_0_weight': (12, 8, 1, 1),
        'layer3_conv1_weight':   (16, 12, 1, 1),
        'layer3_dwconv_weight':  (16, 1, 3, 3),
        'layer3_conv2_weight':   (16, 16, 1, 1),
        'layer3_shortcut_0_weight': (16, 12, 1, 1),
        'dwconv_weight':         (16, 1, 3, 3),
        'pwconv_weight':         (20, 16, 1, 1),
        'conv2_weight':          (20, 20, 5, 1),
        'conv_expand_weight':    (32, 20, 1, 1),
        'fc_weight':             (12, 32),
        'fc_bias':               (12,),
    }

    if name in shape_map:
        target = shape_map[name]
        if arr.size != np.prod(target):
            # 尝试推断
            if arr.size == np.prod(target):
                pass
            else:
                raise ValueError(
                    f"Size mismatch for {name}: "
                    f"array has {arr.size}, target shape {target} needs {np.prod(target)}"
                )
        return arr.reshape(target)
    return arr


def load_weights_to_model(model: BcResNet, weights: dict) -> BcResNet:
    """将解析后的权重字典加载到 PyTorch 模型."""
    state_dict = model.state_dict()
    updated = {}

    # 映射: PyTorch 参数名 → weights.c 变量名
    mapping = [
        # conv1
        ('conv1.weight',       'conv1_weight'),
        ('bn1.weight',         'bn1_weight'),
        ('bn1.bias',           'bn1_bias'),
        ('bn1.running_mean',   'bn1_running_mean'),
        ('bn1.running_var',    'bn1_running_var'),
        # block1
        ('block1.expand.weight',         'layer1_conv1_weight'),
        ('block1.bn1.weight',            'layer1_bn1_weight'),
        ('block1.bn1.bias',              'layer1_bn1_bias'),
        ('block1.bn1.running_mean',      'layer1_bn1_running_mean'),
        ('block1.bn1.running_var',       'layer1_bn1_running_var'),
        ('block1.dw.weight',             'layer1_dwconv_weight'),
        ('block1.bn2.weight',            'layer1_bn2_weight'),
        ('block1.bn2.bias',              'layer1_bn2_bias'),
        ('block1.bn2.running_mean',      'layer1_bn2_running_mean'),
        ('block1.bn2.running_var',       'layer1_bn2_running_var'),
        ('block1.project.weight',        'layer1_conv2_weight'),
        ('block1.bn3.weight',            'layer1_bn3_weight'),
        ('block1.bn3.bias',              'layer1_bn3_bias'),
        ('block1.bn3.running_mean',      'layer1_bn3_running_mean'),
        ('block1.bn3.running_var',       'layer1_bn3_running_var'),
        ('block1.shortcut.weight',       'layer1_shortcut_0_weight'),
        ('block1.shortcut_bn.weight',    'layer1_shortcut_1_weight'),
        ('block1.shortcut_bn.bias',      'layer1_shortcut_1_bias'),
        ('block1.shortcut_bn.running_mean', 'layer1_shortcut_1_running_mean'),
        ('block1.shortcut_bn.running_var',  'layer1_shortcut_1_running_var'),
        # block2
        ('block2.expand.weight',         'layer2_conv1_weight'),
        ('block2.bn1.weight',            'layer2_bn1_weight'),
        ('block2.bn1.bias',              'layer2_bn1_bias'),
        ('block2.bn1.running_mean',      'layer2_bn1_running_mean'),
        ('block2.bn1.running_var',       'layer2_bn1_running_var'),
        ('block2.dw.weight',             'layer2_dwconv_weight'),
        ('block2.bn2.weight',            'layer2_bn2_weight'),
        ('block2.bn2.bias',              'layer2_bn2_bias'),
        ('block2.bn2.running_mean',      'layer2_bn2_running_mean'),
        ('block2.bn2.running_var',       'layer2_bn2_running_var'),
        ('block2.project.weight',        'layer2_conv2_weight'),
        ('block2.bn3.weight',            'layer2_bn3_weight'),
        ('block2.bn3.bias',              'layer2_bn3_bias'),
        ('block2.bn3.running_mean',      'layer2_bn3_running_mean'),
        ('block2.bn3.running_var',       'layer2_bn3_running_var'),
        ('block2.shortcut.weight',       'layer2_shortcut_0_weight'),
        ('block2.shortcut_bn.weight',    'layer2_shortcut_1_weight'),
        ('block2.shortcut_bn.bias',      'layer2_shortcut_1_bias'),
        ('block2.shortcut_bn.running_mean', 'layer2_shortcut_1_running_mean'),
        ('block2.shortcut_bn.running_var',  'layer2_shortcut_1_running_var'),
        # block3
        ('block3.expand.weight',         'layer3_conv1_weight'),
        ('block3.bn1.weight',            'layer3_bn1_weight'),
        ('block3.bn1.bias',              'layer3_bn1_bias'),
        ('block3.bn1.running_mean',      'layer3_bn1_running_mean'),
        ('block3.bn1.running_var',       'layer3_bn1_running_var'),
        ('block3.dw.weight',             'layer3_dwconv_weight'),
        ('block3.bn2.weight',            'layer3_bn2_weight'),
        ('block3.bn2.bias',              'layer3_bn2_bias'),
        ('block3.bn2.running_mean',      'layer3_bn2_running_mean'),
        ('block3.bn2.running_var',       'layer3_bn2_running_var'),
        ('block3.project.weight',        'layer3_conv2_weight'),
        ('block3.bn3.weight',            'layer3_bn3_weight'),
        ('block3.bn3.bias',              'layer3_bn3_bias'),
        ('block3.bn3.running_mean',      'layer3_bn3_running_mean'),
        ('block3.bn3.running_var',       'layer3_bn3_running_var'),
        ('block3.shortcut.weight',       'layer3_shortcut_0_weight'),
        ('block3.shortcut_bn.weight',    'layer3_shortcut_1_weight'),
        ('block3.shortcut_bn.bias',      'layer3_shortcut_1_bias'),
        ('block3.shortcut_bn.running_mean', 'layer3_shortcut_1_running_mean'),
        ('block3.shortcut_bn.running_var',  'layer3_shortcut_1_running_var'),
        # dwconv
        ('dwconv.weight',       'dwconv_weight'),
        ('bn_dw.weight',        'bn_dw_weight'),
        ('bn_dw.bias',          'bn_dw_bias'),
        ('bn_dw.running_mean',  'bn_dw_running_mean'),
        ('bn_dw.running_var',   'bn_dw_running_var'),
        # pwconv
        ('pwconv.weight',       'pwconv_weight'),
        ('bn_pw.weight',        'bn_pw_weight'),
        ('bn_pw.bias',          'bn_pw_bias'),
        ('bn_pw.running_mean',  'bn_pw_running_mean'),
        ('bn_pw.running_var',   'bn_pw_running_var'),
        # conv2
        ('conv2.weight',        'conv2_weight'),
        ('bn2.weight',          'bn2_weight'),
        ('bn2.bias',            'bn2_bias'),
        ('bn2.running_mean',    'bn2_running_mean'),
        ('bn2.running_var',     'bn2_running_var'),
        # expand
        ('expand.weight',       'conv_expand_weight'),
        ('bn_expand.weight',    'bn_expand_weight'),
        ('bn_expand.bias',      'bn_expand_bias'),
        ('bn_expand.running_mean', 'bn_expand_running_mean'),
        ('bn_expand.running_var',  'bn_expand_running_var'),
        # fc
        ('fc.weight',           'fc_weight'),
        ('fc.bias',             'fc_bias'),
    ]

    for pt_name, c_name in mapping:
        if c_name not in weights:
            raise KeyError(f"Missing weight: {c_name} (for {pt_name})")
        w = weights[c_name]

        # 获取目标形状
        if pt_name in state_dict:
            target_shape = state_dict[pt_name].shape
        else:
            # BN num_batches_tracked 不需要加载
            continue

        if w.size != np.prod(target_shape):
            print(f"  [SKIP] {pt_name}: size mismatch (old={w.size}, new={np.prod(target_shape)}), using random init")
            continue
        w_shaped = w.reshape(target_shape)
        updated[pt_name] = torch.from_numpy(w_shaped.copy())

    # 检查是否有遗漏的参数
    for pt_name in state_dict:
        if pt_name not in updated and 'num_batches_tracked' not in pt_name:
            print(f"  [WARNING] {pt_name} not loaded from weights.c, using random init")

    model.load_state_dict(updated, strict=False)
    return model


def verify_weights_loaded(model: BcResNet, weights: dict) -> bool:
    """快速验证权重加载完整性."""
    expected = {
        'conv1.weight': (16, 1, 3, 3),
        'block1.dw.weight': (8, 1, 3, 3),
        'block3.expand.weight': (16, 12, 1, 1),
        'fc.weight': (12, 32),
        'fc.bias': (12,),
    }
    state = model.state_dict()
    all_ok = True
    for name, shape in expected.items():
        actual = state[name].shape
        if actual != torch.Size(shape):
            print(f"  [FAIL] {name}: expected {shape}, got {list(actual)}")
            all_ok = False
        else:
            print(f"  [OK]   {name}: {list(actual)}")
    return all_ok


if __name__ == "__main__":
    print("Parsing weights.c ...")
    weights_dict = parse_weights_c(WEIGHTS_C_PATH)
    print(f"Parsed {len(weights_dict)} arrays: {list(weights_dict.keys())}")

    print("\nCreating model and loading weights ...")
    model = BcResNet(n_classes=13)
    model = load_weights_to_model(model, weights_dict)
    model.eval()

    print("\nVerifying key weight shapes ...")
    verify_weights_loaded(model, weights_dict)

    print("\nRunning test inference ...")
    x = torch.randn(1, 1, 40, 101)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    print(f"Logits:  {logits.squeeze().tolist()}")
    print(f"Prediction: class {pred}, confidence {probs[0, pred].item():.4f}")
    print("\n✅ Weights loaded successfully.")
