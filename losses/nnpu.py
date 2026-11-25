"""Class-wise non-negative PU risk for a single partially labeled dataset."""
import torch
from torch import nn
from torch.nn import functional as F


class ClasswiseNNPU(nn.Module):
    """Marginal-risk nnPU with fixed observed-positive frequencies.

    prior[c] estimates P(Y_c=1); observed_rate[c] is the empirical P(S_c=1)
    from the *corrupted training manifest*. SCAR within each class is assumed.
    Uses a uniform minibatch estimate E[S*l]/observed_rate, not division by
    the random batch positive count. Clamping still introduces minibatch bias.
    This is a clamped-risk objective, not Kiryo's gradient-correction algorithm.
    """
    def __init__(self, prior, observed_rate, *, loss='logistic', class_weights=None):
        super().__init__()
        p = torch.as_tensor(prior, dtype=torch.float32).detach().clone()
        q = torch.as_tensor(observed_rate, dtype=torch.float32).detach().clone()
        if p.ndim != 1 or not p.numel() or q.shape != p.shape:
            raise ValueError('prior and observed_rate must be equal-length vectors')
        if not (torch.isfinite(p).all() and torch.isfinite(q).all()
                and ((q > 0) & (q <= p) & (p <= 1)).all()):
            raise ValueError('Require 0 < observed_rate <= prior <= 1 for each class')
        if loss not in ('logistic', 'sigmoid'):
            raise ValueError('loss must be logistic or sigmoid')
        w = torch.ones_like(p) if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32).detach().clone()
        if w.shape != p.shape or not (torch.isfinite(w).all() and (w >= 0).all() and (w.sum() > 0)):
            raise ValueError('Invalid class_weights')
        self.register_buffer('prior', p)
        self.register_buffer('observed_rate', q)
        self.register_buffer('class_weights', w)
        self.loss = loss

    def forward(self, logits, observed_labels, *, return_stats=False):
        if logits.ndim != 2 or logits.shape != observed_labels.shape or logits.shape[1] != self.prior.numel() or logits.shape[0] == 0:
            raise ValueError('Expected nonempty matching [batch, classes] tensors')
        if not logits.is_floating_point():
            raise ValueError('logits must be floating point')
        if not ((observed_labels == 0) | (observed_labels == 1)).all():
            raise ValueError('Observed labels must be binary 0/1')
        # Keep the risk subtraction in FP32 even for BF16/FP16 model outputs.
        z = logits.float()
        s = observed_labels.to(device=z.device, dtype=z.dtype)
        if not torch.isfinite(z).all():
            raise ValueError('Nonfinite logits')
        if self.loss == 'logistic':
            lpos, lneg = F.softplus(-z), F.softplus(z)
        else:
            lpos, lneg = torch.sigmoid(-z), torch.sigmoid(z)
        ratio = self.prior.float() / self.observed_rate.float()
        rpos = ratio * (s * lpos).mean(dim=0)
        rneg = lneg.mean(dim=0) - ratio * (s * lneg).mean(dim=0)
        per_class = rpos + rneg.clamp_min(0)
        # Match SSHR's class reduction: divide by number of classes, not sum of weights.
        result = (per_class * self.class_weights.float()).mean()
        if not return_stats:
            return result
        return result, {'positive_risk': rpos.detach(), 'negative_risk_raw': rneg.detach(),
                        'negative_risk_clipped': rneg.detach().clamp_min(0),
                        'positive_count': s.sum(dim=0).detach(),
                        'clipped': (rneg < 0).detach()}


def sshr_classification_loss(heads, labels, criterion, head_weights=(.10, .15, .25, .50), *, return_stats=False):
    """Apply the same PU criterion to each raw SSHR classification head."""
    if len(heads) != 4 or len(head_weights) != 4:
        raise ValueError('SSHR requires four classification heads')
    import math
    if not all(math.isfinite(w) and w >= 0 for w in head_weights) or sum(head_weights) <= 0:
        raise ValueError('Invalid head weights')
    if return_stats:
        results = [criterion(z, labels, return_stats=True) for z in heads]
        return sum(w * r[0] for w, r in zip(head_weights, results)), [r[1] for r in results]
    return sum(w * criterion(z, labels) for w, z in zip(head_weights, heads))
