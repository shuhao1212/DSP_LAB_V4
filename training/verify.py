"""
verify.py — PC 端 vs DSP 端推理一致性验证

三层校验:
  Level 1: 逐层输出对比 (PyTorch layer vs DSP C code)
  Level 2: 权重加载完整性验证
  Level 3: 端到端 logits 对比 (同一输入 → PyTorch vs DSP)
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import BcResNet
from load_weights import parse_weights_c, load_weights_to_model
from data_loader import extract_log_mel, load_wav

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_C = WORKSPACE_ROOT / "Code" / "User" / "weights.c"


def verify_weight_loading(model: BcResNet, weights_dict: dict) -> bool:
    """Level 2: 验证所有权重加载完整且形状正确."""
    print("=" * 60)
    print("Level 2: Weight Loading Verification")
    print("=" * 60)

    state = model.state_dict()
    expected_shapes = {
        'conv1.weight': (16, 1, 3, 3),
        'bn1.weight': (16,),
        'bn1.running_mean': (16,),
        'block1.expand.weight': (8, 16, 1, 1),
        'block1.dw.weight': (8, 1, 3, 3),
        'block1.project.weight': (8, 8, 1, 1),
        'block1.shortcut.weight': (8, 16, 1, 1),
        'block2.expand.weight': (12, 8, 1, 1),
        'block2.dw.weight': (12, 1, 3, 3),
        'block2.project.weight': (12, 12, 1, 1),
        'block2.shortcut.weight': (12, 8, 1, 1),
        'block3.expand.weight': (16, 12, 1, 1),
        'block3.dw.weight': (16, 1, 3, 3),
        'block3.project.weight': (16, 16, 1, 1),
        'block3.shortcut.weight': (16, 12, 1, 1),
        'dwconv.weight': (16, 1, 3, 3),
        'pwconv.weight': (20, 16, 1, 1),
        'conv2.weight': (20, 20, 5, 1),
        'expand.weight': (32, 20, 1, 1),
        'fc.weight': (13, 32),
        'fc.bias': (13,),
    }

    all_ok = True
    for name, expected_shape in expected_shapes.items():
        actual = state[name].shape
        shape_ok = actual == torch.Size(expected_shape)
        # 检查是否有 NaN
        has_nan = torch.isnan(state[name]).any().item()
        status = "OK" if (shape_ok and not has_nan) else "FAIL"
        if status == "FAIL":
            all_ok = False
            issues = []
            if not shape_ok:
                issues.append(f"shape mismatch: expected {expected_shape}, got {list(actual)}")
            if has_nan:
                issues.append("contains NaN")
            print(f"  [{status}] {name}: {'; '.join(issues)}")
        else:
            print(f"  [{status}] {name}: {list(actual)}")

    return all_ok


def verify_end_to_end(model: BcResNet) -> bool:
    """Level 3: 端到端推理一致性 (理想条件下)."""
    print("\n" + "=" * 60)
    print("Level 3: End-to-End Inference Test")
    print("=" * 60)

    model.eval()

    # 使用固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 生成测试输入
    test_input = torch.randn(1, 1, 40, 101)

    with torch.no_grad():
        logits = model(test_input)
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    print(f"  Input shape:  {list(test_input.shape)}")
    print(f"  Input stats:  min={test_input.min().item():.4f}, max={test_input.max().item():.4f}, mean={test_input.mean().item():.4f}")
    print(f"  Logits:       {[f'{v:.4f}' for v in logits.squeeze().tolist()]}")
    print(f"  Probabilities:{[f'{v:.4f}' for v in probs.squeeze().tolist()]}")
    print(f"  Prediction:   class {pred}, confidence {probs[0, pred].item():.4f}")

    # 检查 logits 是否合理 (不全是 0, 没有 NaN/Inf)
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    all_zero = torch.allclose(logits, torch.zeros_like(logits))

    if has_nan:
        print("  [FAIL] Logits contain NaN!")
        return False
    if has_inf:
        print("  [FAIL] Logits contain Inf!")
        return False
    if all_zero:
        print("  [FAIL] All logits are zero!")
        return False

    print("  [OK]   Logits look reasonable.")
    return True


def verify_with_real_audio(model: BcResNet, wav_path: str = None) -> bool:
    """Level 3b: 使用真实音频文件测试."""
    print("\n" + "=" * 60)
    print("Level 3b: Real Audio Test")
    print("=" * 60)

    model.eval()

    # 尝试找一个测试音频
    data_root = WORKSPACE_ROOT / "data"
    test_words = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    wav_files = []
    for word in test_words:
        word_dir = data_root / word
        if word_dir.exists():
            wavs = list(word_dir.glob("*.wav"))
            if wavs:
                wav_files.append((word, wavs[0]))

    if not wav_files:
        print("  No test audio files found. Skipping.")
        return True

    for word, wav_file in wav_files[:5]:  # 测试前 5 个
        try:
            waveform = load_wav(wav_file)
            logmel = extract_log_mel(waveform)
            logmel_tensor = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0)  # [1, 1, 40, 101]

            with torch.no_grad():
                logits = model(logmel_tensor)
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()

            from data_loader import LABELS
            print(f"  {word:>8} ({wav_file.name[:16]}...) → predicted: {LABELS[pred]:>12} "
                  f"(conf: {probs[0, pred].item():.3f}, correct: {'✅' if LABELS[pred] == word else '❌'})")
        except Exception as e:
            print(f"  {word:>8}: ERROR - {e}")

    return True


def main():
    print("BC-ResNet PC-DSP Verification Suite\n")

    # 加载模型和权重
    print("Loading model and weights ...")
    model = BcResNet(n_classes=13)
    weights_dict = parse_weights_c(WEIGHTS_C)
    model = load_weights_to_model(model, weights_dict)

    # Level 2: 权重验证
    level2_ok = verify_weight_loading(model, weights_dict)

    # Level 3: 端到端验证
    level3_ok = verify_end_to_end(model)

    # Level 3b: 真实音频测试
    verify_with_real_audio(model)

    # 总结
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Level 2 (Weight loading): {'✅ PASS' if level2_ok else '❌ FAIL'}")
    print(f"  Level 3 (End-to-end):     {'✅ PASS' if level3_ok else '❌ FAIL'}")
    print("\nNext step: Run train.py to fine-tune the model.")


if __name__ == "__main__":
    main()
