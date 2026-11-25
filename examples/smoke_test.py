"""Run from repository root: python -m examples.smoke_test"""
import torch
from losses import ClasswiseNNPU, sshr_classification_loss

# Synthetic numbers only. Not estimates for LUAD or BCSS.
torch.manual_seed(11)
criterion = ClasswiseNNPU([.7,.6,.3,.15], [.42,.36,.18,.09])
heads = [torch.randn(4,4,requires_grad=True) for _ in range(4)]
labels = torch.tensor([[1.,1.,0.,0.],[1.,0.,0.,0.],[0.,1.,1.,0.],[1.,0.,0.,0.]])
loss, stats = sshr_classification_loss(heads, labels, criterion, return_stats=True)
loss.backward()
print('Synthetic loss:', loss.item())
print('Deep head positive counts:', stats[-1]['positive_count'].tolist())
print('Deep head raw negative risk:', stats[-1]['negative_risk_raw'].tolist())
print('All four heads have finite gradients:', all(torch.isfinite(h.grad).all().item() for h in heads))
