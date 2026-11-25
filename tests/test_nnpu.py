import pytest
import torch
from torch.nn import functional as F
from losses import ClasswiseNNPU, sshr_classification_loss


def test_clean_full_batch_matches_bce():
    z = torch.tensor([[1., -2.], [2., 1.], [-1., 3.], [0., -1.]])
    s = torch.tensor([[1., 0.], [1., 1.], [0., 1.], [0., 0.]])
    q = s.mean(0)
    assert torch.allclose(ClasswiseNNPU(q, q)(z, s), F.binary_cross_entropy_with_logits(z, s))


def test_exact_balanced_corruption_matches_clean_risk():
    z = torch.tensor([[2.], [2.], [-1.], [-1.]])
    y = torch.tensor([[1.], [1.], [0.], [0.]])
    s = torch.tensor([[1.], [0.], [0.], [0.]])
    assert torch.allclose(ClasswiseNNPU([.5], [.25])(z, s), F.binary_cross_entropy_with_logits(z, y))


def test_negative_risk_clamped():
    z = torch.tensor([[5.], [-5.], [-5.], [-5.]], requires_grad=True)
    s = torch.tensor([[1.], [0.], [0.], [0.]])
    loss, stats = ClasswiseNNPU([.75], [.25])(z, s, return_stats=True)
    assert stats['negative_risk_raw'].item() < 0
    assert stats['negative_risk_clipped'].item() == 0
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_no_positive_in_batch():
    z = torch.randn(4, 4, requires_grad=True)
    loss, stats = ClasswiseNNPU([.5]*4, [.3]*4)(z, torch.zeros_like(z), return_stats=True)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert (stats['positive_count'] == 0).all()


def test_four_heads_backward():
    heads = [torch.randn(4, 4, requires_grad=True) for _ in range(4)]
    s = torch.eye(4)
    criterion = ClasswiseNNPU([.5]*4, [.25]*4)
    loss, stats = sshr_classification_loss(heads, s, criterion, return_stats=True)
    expected = sum(w * criterion(z, s) for w,z in zip([.1,.15,.25,.5],heads))
    assert torch.allclose(loss, expected) and len(stats) == 4
    loss.backward()
    assert all(torch.isfinite(z.grad).all() for z in heads)


@pytest.mark.parametrize('p,q', [([.1],[.2]), ([.5],[0.]), ([1.1],[.2]), ([float('nan')],[.2])])
def test_bad_priors(p,q):
    with pytest.raises(ValueError): ClasswiseNNPU(p,q)


def test_invalid_labels():
    with pytest.raises(ValueError): ClasswiseNNPU([.5],[.2])(torch.zeros(2,1), torch.tensor([[0.],[2.]]))


def test_bfloat16_large_logits():
    z = torch.tensor([[100., -100.],[-100.,100.]], dtype=torch.bfloat16, requires_grad=True)
    loss = ClasswiseNNPU([.5,.5],[.5,.5])(z, torch.eye(2))
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(z.grad).all()


def test_sigmoid_loss():
    z = torch.tensor([[1.],[-1.]])
    s = torch.tensor([[1.],[0.]])
    assert torch.allclose(ClasswiseNNPU([.5],[.5],loss='sigmoid')(z,s), torch.sigmoid(torch.tensor(-1.)))


def test_class_weights_match_sshr_clean():
    z = torch.randn(4, 2)
    s = torch.tensor([[1.,0.],[0.,1.],[1.,1.],[0.,0.]])
    q = s.mean(0)
    w = torch.tensor([1., 2.])
    assert torch.allclose(ClasswiseNNPU(q,q,class_weights=w)(z,s), F.multilabel_soft_margin_loss(z,s,weight=w))
