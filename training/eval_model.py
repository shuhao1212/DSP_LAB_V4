"""评估最佳模型"""
import sys, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BcResNet
from data_loader import create_precomputed_dataloaders, LABELS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

print('Loading best model...')
model = BcResNet(n_classes=13).to(device)
ckpt_path = Path(__file__).resolve().parent / 'checkpoints' / 'best_model.pt'
ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Best epoch: {ckpt['epoch']}, val_acc: {ckpt['val_acc']:.2f}%")

print('Loading test data...')
_, _, test_loader = create_precomputed_dataloaders(batch_size=128, augment=False)

correct = 0
total = 0
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.extend(predicted.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

print(f'\nTest Acc: {100.0*correct/total:.2f}% ({correct}/{total})')

print('\nPer-class accuracy:')
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
for c in range(13):
    mask = all_targets == c
    if mask.sum() > 0:
        acc = 100.0 * (all_preds[mask] == c).sum() / mask.sum()
        marker = 'X' if acc < 50 else ('~' if acc < 80 else 'OK')
        print(f'  [{marker}] {LABELS[c]:>12}: {acc:5.1f}% ({mask.sum()} samples)')

print('\nConfusion Matrix:')
cm = np.zeros((13, 13), dtype=np.int32)
for p, t in zip(all_preds, all_targets):
    cm[t][p] += 1
header = '         ' + ''.join(f'{l[:4]:>6}' for l in LABELS)
print(header)
for i, label in enumerate(LABELS):
    row = f'{label:>8} ' + ''.join(f'{cm[i][j]:6d}' for j in range(13))
    print(row)
