"""部署训练好的权重到 DSP 工程"""
import sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BcResNet
from export_weights import export_weights

ws = Path(r'c:\Users\Haoxuan Zuo\DSP_LAB_V3')

print('Loading best model...')
model = BcResNet(n_classes=13)
ckpt = torch.load(str(Path(__file__).resolve().parent / 'checkpoints' / 'best_model.pt'),
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model from epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%")

print('Exporting to Code/User/weights.c and weights.h ...')
export_weights(model,
    output_c=ws / 'Code' / 'User' / 'weights.c',
    output_h=ws / 'Code' / 'User' / 'weights.h',
    backup=False)
print('Done! Ready for CCS compile.')
