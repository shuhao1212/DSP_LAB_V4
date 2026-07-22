"""
export_weights.py — 将 PyTorch 模型权重导出为 C 数组格式 (weights.c / weights.h)

输出格式完全匹配现有 Code/User/weights.c 和 Code/User/weights.h.
"""

import numpy as np
import torch
from pathlib import Path
from model import BcResNet

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_C = WORKSPACE_ROOT / "Code" / "User" / "weights.c"
OUTPUT_H = WORKSPACE_ROOT / "Code" / "User" / "weights.h"
BACKUP_C = WORKSPACE_ROOT / "Code" / "User" / "weights.c.bak"
BACKUP_H = WORKSPACE_ROOT / "Code" / "User" / "weights.h.bak"


def tensor_to_c_array(tensor: torch.Tensor, name: str, comment: str = "") -> str:
    """将 PyTorch tensor 转为 C const float 数组字符串."""
    arr = tensor.detach().cpu().numpy().flatten()
    shape_str = " × ".join(str(s) for s in tensor.shape)

    lines = []
    lines.append(f"// {comment}")
    lines.append(f"// 原始维度: [{', '.join(str(s) for s in tensor.shape)}]")
    lines.append(f"// 元素数量: {arr.size}")
    lines.append(f"#pragma DATA_ALIGN({name}, 8)")
    lines.append(f"const float {name}[{arr.size}] = {{")

    # 每行 8 个值, 科学计数法
    vals_per_line = 8
    for i in range(0, len(arr), vals_per_line):
        chunk = arr[i:i + vals_per_line]
        val_strs = [f"{v: .6e}f" for v in chunk]
        line = "    " + ", ".join(val_strs)
        if i + vals_per_line < len(arr):
            line += ","
        lines.append(line)

    lines.append("};")
    return "\n".join(lines)


def export_weights(model: BcResNet, output_c: Path = OUTPUT_C,
                   output_h: Path = OUTPUT_H,
                   backup: bool = True):
    """导出模型权重到 weights.c 和 weights.h."""
    state = model.state_dict()

    # 备份原有文件
    if backup and output_c.exists():
        import shutil
        shutil.copy(output_c, BACKUP_C)
        print(f"Backed up {output_c} → {BACKUP_C}")
    if backup and output_h.exists():
        import shutil
        shutil.copy(output_h, BACKUP_H)
        print(f"Backed up {output_h} → {BACKUP_H}")

    # 权重映射: (pt_name, c_name, comment)
    weight_list = [
        # conv1
        ('conv1.weight', 'conv1_weight', 'conv1.weight'),
        ('bn1.weight', 'bn1_weight', 'bn1.weight'),
        ('bn1.bias', 'bn1_bias', 'bn1.bias'),
        ('bn1.running_mean', 'bn1_running_mean', 'bn1.running_mean'),
        ('bn1.running_var', 'bn1_running_var', 'bn1.running_var'),
        ('bn1.num_batches_tracked', 'bn1_num_batches_tracked', 'bn1.num_batches_tracked'),
        # block1
        ('block1.expand.weight', 'layer1_conv1_weight', 'layer1.conv1.weight'),
        ('block1.bn1.weight', 'layer1_bn1_weight', 'layer1.bn1.weight'),
        ('block1.bn1.bias', 'layer1_bn1_bias', 'layer1.bn1.bias'),
        ('block1.bn1.running_mean', 'layer1_bn1_running_mean', 'layer1.bn1.running_mean'),
        ('block1.bn1.running_var', 'layer1_bn1_running_var', 'layer1.bn1.running_var'),
        ('block1.bn1.num_batches_tracked', 'layer1_bn1_num_batches_tracked', 'layer1.bn1.num_batches_tracked'),
        ('block1.dw.weight', 'layer1_dwconv_weight', 'layer1.dwconv.weight'),
        ('block1.bn2.weight', 'layer1_bn2_weight', 'layer1.bn2.weight'),
        ('block1.bn2.bias', 'layer1_bn2_bias', 'layer1.bn2.bias'),
        ('block1.bn2.running_mean', 'layer1_bn2_running_mean', 'layer1.bn2.running_mean'),
        ('block1.bn2.running_var', 'layer1_bn2_running_var', 'layer1.bn2.running_var'),
        ('block1.bn2.num_batches_tracked', 'layer1_bn2_num_batches_tracked', 'layer1.bn2.num_batches_tracked'),
        ('block1.project.weight', 'layer1_conv2_weight', 'layer1.conv2.weight'),
        ('block1.bn3.weight', 'layer1_bn3_weight', 'layer1.bn3.weight'),
        ('block1.bn3.bias', 'layer1_bn3_bias', 'layer1.bn3.bias'),
        ('block1.bn3.running_mean', 'layer1_bn3_running_mean', 'layer1.bn3.running_mean'),
        ('block1.bn3.running_var', 'layer1_bn3_running_var', 'layer1.bn3.running_var'),
        ('block1.bn3.num_batches_tracked', 'layer1_bn3_num_batches_tracked', 'layer1.bn3.num_batches_tracked'),
        ('block1.shortcut.weight', 'layer1_shortcut_0_weight', 'layer1.shortcut.weight'),
        ('block1.shortcut_bn.weight', 'layer1_shortcut_1_weight', 'layer1.shortcut_bn.weight'),
        ('block1.shortcut_bn.bias', 'layer1_shortcut_1_bias', 'layer1.shortcut_bn.bias'),
        ('block1.shortcut_bn.running_mean', 'layer1_shortcut_1_running_mean', 'layer1.shortcut_bn.running_mean'),
        ('block1.shortcut_bn.running_var', 'layer1_shortcut_1_running_var', 'layer1.shortcut_bn.running_var'),
        ('block1.shortcut_bn.num_batches_tracked', 'layer1_shortcut_1_num_batches_tracked', 'layer1.shortcut_bn.num_batches_tracked'),
        # block2
        ('block2.expand.weight', 'layer2_conv1_weight', 'layer2.conv1.weight'),
        ('block2.bn1.weight', 'layer2_bn1_weight', 'layer2.bn1.weight'),
        ('block2.bn1.bias', 'layer2_bn1_bias', 'layer2.bn1.bias'),
        ('block2.bn1.running_mean', 'layer2_bn1_running_mean', 'layer2.bn1.running_mean'),
        ('block2.bn1.running_var', 'layer2_bn1_running_var', 'layer2.bn1.running_var'),
        ('block2.bn1.num_batches_tracked', 'layer2_bn1_num_batches_tracked', 'layer2.bn1.num_batches_tracked'),
        ('block2.dw.weight', 'layer2_dwconv_weight', 'layer2.dwconv.weight'),
        ('block2.bn2.weight', 'layer2_bn2_weight', 'layer2.bn2.weight'),
        ('block2.bn2.bias', 'layer2_bn2_bias', 'layer2.bn2.bias'),
        ('block2.bn2.running_mean', 'layer2_bn2_running_mean', 'layer2.bn2.running_mean'),
        ('block2.bn2.running_var', 'layer2_bn2_running_var', 'layer2.bn2.running_var'),
        ('block2.bn2.num_batches_tracked', 'layer2_bn2_num_batches_tracked', 'layer2.bn2.num_batches_tracked'),
        ('block2.project.weight', 'layer2_conv2_weight', 'layer2.conv2.weight'),
        ('block2.bn3.weight', 'layer2_bn3_weight', 'layer2.bn3.weight'),
        ('block2.bn3.bias', 'layer2_bn3_bias', 'layer2.bn3.bias'),
        ('block2.bn3.running_mean', 'layer2_bn3_running_mean', 'layer2.bn3.running_mean'),
        ('block2.bn3.running_var', 'layer2_bn3_running_var', 'layer2.bn3.running_var'),
        ('block2.bn3.num_batches_tracked', 'layer2_bn3_num_batches_tracked', 'layer2.bn3.num_batches_tracked'),
        ('block2.shortcut.weight', 'layer2_shortcut_0_weight', 'layer2.shortcut.weight'),
        ('block2.shortcut_bn.weight', 'layer2_shortcut_1_weight', 'layer2.shortcut_bn.weight'),
        ('block2.shortcut_bn.bias', 'layer2_shortcut_1_bias', 'layer2.shortcut_bn.bias'),
        ('block2.shortcut_bn.running_mean', 'layer2_shortcut_1_running_mean', 'layer2.shortcut_bn.running_mean'),
        ('block2.shortcut_bn.running_var', 'layer2_shortcut_1_running_var', 'layer2.shortcut_bn.running_var'),
        ('block2.shortcut_bn.num_batches_tracked', 'layer2_shortcut_1_num_batches_tracked', 'layer2.shortcut_bn.num_batches_tracked'),
        # block3
        ('block3.expand.weight', 'layer3_conv1_weight', 'layer3.conv1.weight'),
        ('block3.bn1.weight', 'layer3_bn1_weight', 'layer3.bn1.weight'),
        ('block3.bn1.bias', 'layer3_bn1_bias', 'layer3.bn1.bias'),
        ('block3.bn1.running_mean', 'layer3_bn1_running_mean', 'layer3.bn1.running_mean'),
        ('block3.bn1.running_var', 'layer3_bn1_running_var', 'layer3.bn1.running_var'),
        ('block3.bn1.num_batches_tracked', 'layer3_bn1_num_batches_tracked', 'layer3.bn1.num_batches_tracked'),
        ('block3.dw.weight', 'layer3_dwconv_weight', 'layer3.dwconv.weight'),
        ('block3.bn2.weight', 'layer3_bn2_weight', 'layer3.bn2.weight'),
        ('block3.bn2.bias', 'layer3_bn2_bias', 'layer3.bn2.bias'),
        ('block3.bn2.running_mean', 'layer3_bn2_running_mean', 'layer3.bn2.running_mean'),
        ('block3.bn2.running_var', 'layer3_bn2_running_var', 'layer3.bn2.running_var'),
        ('block3.bn2.num_batches_tracked', 'layer3_bn2_num_batches_tracked', 'layer3.bn2.num_batches_tracked'),
        ('block3.project.weight', 'layer3_conv2_weight', 'layer3.conv2.weight'),
        ('block3.bn3.weight', 'layer3_bn3_weight', 'layer3.bn3.weight'),
        ('block3.bn3.bias', 'layer3_bn3_bias', 'layer3.bn3.bias'),
        ('block3.bn3.running_mean', 'layer3_bn3_running_mean', 'layer3.bn3.running_mean'),
        ('block3.bn3.running_var', 'layer3_bn3_running_var', 'layer3.bn3.running_var'),
        ('block3.bn3.num_batches_tracked', 'layer3_bn3_num_batches_tracked', 'layer3.bn3.num_batches_tracked'),
        ('block3.shortcut.weight', 'layer3_shortcut_0_weight', 'layer3.shortcut.weight'),
        ('block3.shortcut_bn.weight', 'layer3_shortcut_1_weight', 'layer3.shortcut_bn.weight'),
        ('block3.shortcut_bn.bias', 'layer3_shortcut_1_bias', 'layer3.shortcut_bn.bias'),
        ('block3.shortcut_bn.running_mean', 'layer3_shortcut_1_running_mean', 'layer3.shortcut_bn.running_mean'),
        ('block3.shortcut_bn.running_var', 'layer3_shortcut_1_running_var', 'layer3.shortcut_bn.running_var'),
        ('block3.shortcut_bn.num_batches_tracked', 'layer3_shortcut_1_num_batches_tracked', 'layer3.shortcut_bn.num_batches_tracked'),
        # dwconv
        ('dwconv.weight', 'dwconv_weight', 'dwconv.weight'),
        ('bn_dw.weight', 'bn_dw_weight', 'bn_dw.weight'),
        ('bn_dw.bias', 'bn_dw_bias', 'bn_dw.bias'),
        ('bn_dw.running_mean', 'bn_dw_running_mean', 'bn_dw.running_mean'),
        ('bn_dw.running_var', 'bn_dw_running_var', 'bn_dw.running_var'),
        ('bn_dw.num_batches_tracked', 'bn_dw_num_batches_tracked', 'bn_dw.num_batches_tracked'),
        # pwconv
        ('pwconv.weight', 'pwconv_weight', 'pwconv.weight'),
        ('bn_pw.weight', 'bn_pw_weight', 'bn_pw.weight'),
        ('bn_pw.bias', 'bn_pw_bias', 'bn_pw.bias'),
        ('bn_pw.running_mean', 'bn_pw_running_mean', 'bn_pw.running_mean'),
        ('bn_pw.running_var', 'bn_pw_running_var', 'bn_pw.running_var'),
        ('bn_pw.num_batches_tracked', 'bn_pw_num_batches_tracked', 'bn_pw.num_batches_tracked'),
        # conv2
        ('conv2.weight', 'conv2_weight', 'conv2.weight'),
        ('bn2.weight', 'bn2_weight', 'bn2.weight'),
        ('bn2.bias', 'bn2_bias', 'bn2.bias'),
        ('bn2.running_mean', 'bn2_running_mean', 'bn2.running_mean'),
        ('bn2.running_var', 'bn2_running_var', 'bn2.running_var'),
        ('bn2.num_batches_tracked', 'bn2_num_batches_tracked', 'bn2.num_batches_tracked'),
        # expand
        ('expand.weight', 'conv_expand_weight', 'conv_expand.weight'),
        ('bn_expand.weight', 'bn_expand_weight', 'bn_expand.weight'),
        ('bn_expand.bias', 'bn_expand_bias', 'bn_expand.bias'),
        ('bn_expand.running_mean', 'bn_expand_running_mean', 'bn_expand.running_mean'),
        ('bn_expand.running_var', 'bn_expand_running_var', 'bn_expand.running_var'),
        ('bn_expand.num_batches_tracked', 'bn_expand_num_batches_tracked', 'bn_expand.num_batches_tracked'),
        # fc
        ('fc.weight', 'fc_weight', 'fc.weight'),
        ('fc.bias', 'fc_bias', 'fc.bias'),
    ]

    # 生成 weights.c
    lines_c = [
        "// ============================================================",
        "//  BC-ResNet 网络权重参数文件",
        "//  由 PC 端 PyTorch 训练后自动导出",
        "//  目标平台: TMS320C6748 DSP",
        "// ============================================================",
        "",
        '#include <stdint.h>',
        "",
    ]

    # 生成 weights.h
    lines_h = [
        "#ifndef _WEIGHTS_H_",
        "#define _WEIGHTS_H_",
        "",
        "/* Auto-generated extern declarations for weights.c. */",
        "/* Add weights.c to the CCS project together with project3.c. */",
        "",
    ]

    for pt_name, c_name, comment in weight_list:
        tensor = state[pt_name]
        lines_c.append(tensor_to_c_array(tensor, c_name, comment))
        lines_c.append("")
        lines_h.append(f"extern const float {c_name}[{tensor.numel()}];")

    lines_h.append("")
    lines_h.append("#endif")

    with open(output_c, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_c))
    print(f"Exported {output_c} ({output_c.stat().st_size:,} bytes)")

    with open(output_h, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_h))
    print(f"Exported {output_h} ({output_h.stat().st_size:,} bytes)")


if __name__ == "__main__":
    from load_weights import parse_weights_c, load_weights_to_model
    print("Loading existing model for export test ...")
    model = BcResNet(n_classes=13)
    weights_dict = parse_weights_c(WORKSPACE_ROOT / "Code" / "User" / "weights.c")
    model = load_weights_to_model(model, weights_dict)
    model.eval()

    # 导出到临时目录测试
    export_weights(model,
                   output_c=WORKSPACE_ROOT / "training" / "test_weights.c",
                   output_h=WORKSPACE_ROOT / "training" / "test_weights.h",
                   backup=False)
    print("Export test passed.")
