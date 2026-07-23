"""
train.py — Fine-tune BC-ResNet 模型

从 DSP 端现有权重出发, 在 Google Speech Commands v2 上 fine-tune.
目标: 12 类识别准确率 > 85%, 重点关注弱项词 (down, on, no, stop, left, right).
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# 添加 training 目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import BcResNet, count_parameters
from load_weights import parse_weights_c, load_weights_to_model
from data_loader import create_precomputed_dataloaders, LABELS, LABEL_TO_ID
from export_weights import export_weights

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_C = WORKSPACE_ROOT / "Code" / "User" / "weights.c"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """训练一个 epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader):4d} | "
                  f"Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.1f}%")

    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_preds.extend(predicted.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    return running_loss / len(loader), 100.0 * correct / total, all_preds, all_targets


def compute_per_class_accuracy(preds, targets, n_classes):
    """计算每类准确率."""
    preds = np.array(preds)
    targets = np.array(targets)
    per_class = {}
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() > 0:
            per_class[LABELS[c]] = 100.0 * (preds[mask] == c).sum() / mask.sum()
        else:
            per_class[LABELS[c]] = 0.0
    return per_class


def compute_confusion_matrix(preds, targets, n_classes):
    """计算混淆矩阵."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for p, t in zip(preds, targets):
        cm[t][p] += 1
    return cm


def print_confusion_matrix(cm, labels):
    """打印混淆矩阵."""
    n = len(labels)
    # 表头
    header = "        " + " ".join(f"{l[:5]:>6}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row = f"{label:>7} " + " ".join(f"{cm[i][j]:6d}" for j in range(n))
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BC-ResNet for KWS")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=3e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch (ignore existing weights)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # 创建模型
    model = BcResNet(n_classes=13)
    if not args.from_scratch:
        print(f"Loading pretrained weights from {WEIGHTS_C} ...")
        weights_dict = parse_weights_c(WEIGHTS_C)
        model = load_weights_to_model(model, weights_dict)
        print("Weights loaded successfully.")
    else:
        print("Training from scratch (random initialization).")

    model = model.to(device)
    print(f"Model parameters: {count_parameters(model)}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) * 4 / 1024:.1f} KB (float32)")

    # 数据加载
    print("\nLoading data (precomputing Mel spectrograms on first run) ...")
    train_loader, val_loader, test_loader = create_precomputed_dataloaders(
        batch_size=args.batch_size,
        augment=not args.no_augment
    )

    # 损失函数: 给弱项词加权 + 标签平滑
    class_weights = torch.ones(len(LABELS))
    class_weights[LABEL_TO_ID['on']] = 2.0     # 最弱命令词 /ɔn/，短词+元音开头无辅音锚点
    class_weights[LABEL_TO_ID['off']] = 1.4    # 上次 93.6%
    class_weights[LABEL_TO_ID['_unknown_']] = 1.4  # 上次 80.5%
    class_weights[LABEL_TO_ID['_silence_']] = 0.5
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)

    # 分层学习率: 分类器用更高学习率
    fc_params = list(model.fc.parameters()) + list(model.bn_expand.parameters()) + list(model.expand.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if 'fc' not in n and 'bn_expand' not in n and 'expand' not in n]

    optimizer = AdamW([
        {'params': other_params, 'lr': args.lr * 0.1},
        {'params': fc_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    # Warmup (5 epochs) + CosineAnnealing
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr * 1e-4)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # 训练
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('val_acc', 0.0)
        best_epoch = ckpt['epoch']
        print(f"Resumed at epoch {start_epoch}, best val_acc={best_val_acc:.2f}%")

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | Time: {elapsed:.1f}s")

        # 保存最佳模型 + Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  >>> Saved best model (val_acc={val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping: no improvement for {args.patience} epochs.")
                break

        # 每 10 个 epoch 打印一次每类准确率
        if epoch % 10 == 0:
            per_class = compute_per_class_accuracy(val_preds, val_targets, len(LABELS))
            print("  Per-class accuracy (val):")
            for label, acc in per_class.items():
                marker = "!! " if acc < 50 else "  "
                print(f"    {marker}{label:>12}: {acc:5.1f}%")

    # 最终测试
    print(f"\n{'='*60}")
    print(f"Training complete. Best val_acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    print(f"{'='*60}")

    # 加载最佳模型
    checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 测试集评估
    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # 混淆矩阵
    print(f"\nConfusion Matrix (test set):")
    cm = compute_confusion_matrix(test_preds, test_targets, len(LABELS))
    print_confusion_matrix(cm, LABELS)

    # 每类准确率
    print(f"\nPer-class accuracy (test set):")
    per_class = compute_per_class_accuracy(test_preds, test_targets, len(LABELS))
    for label, acc in per_class.items():
        marker = "FAIL " if acc < 50 else ("LOW  " if acc < 80 else "OK   ")
        print(f"  {marker}{label:>12}: {acc:5.1f}%")

    # 导出最终权重 (直接部署到 Code/User/, 自动备份旧权重)
    final_c = WORKSPACE_ROOT / "Code" / "User" / "weights.c"
    final_h = WORKSPACE_ROOT / "Code" / "User" / "weights.h"
    export_weights(model, output_c=final_c, output_h=final_h, backup=True)
    print(f"\nWeights deployed to:")
    print(f"  {final_c}")
    print(f"  {final_h}")
    print(f"  cp {final_c} {WEIGHTS_C}")
    print(f"  cp {final_h} {WORKSPACE_ROOT / 'Code' / 'User' / 'weights.h'}")


if __name__ == "__main__":
    main()
