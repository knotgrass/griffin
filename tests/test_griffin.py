import torch

from griffin import (
    RGLRU,
    Local_MQA_block,
    Recurrent_block,
    Residual_block,
    RMSNorm,
    SqrtBoundDerivative,
    Temporal_Conv1D,
    apply_rope,
)


def test_causal():
    """Temporal_Conv1D keeps the length T, and y[t] never uses x[s] with s > t."""
    torch.manual_seed(0)
    B, T, D = 2, 10, 8
    conv = Temporal_Conv1D(D, kernel_size=4)
    x = torch.randn(B, T, D, requires_grad=True)

    y = conv(x)
    assert y.shape == (B, T, D)

    for t in range(T):
        # grad[:, s] = d(sum of y[:, t]) / d(x[:, s]): non-zero only if y[t] uses x[s].
        (grad,) = torch.autograd.grad(y[:, t].sum(), x, retain_graph=True)
        assert grad[:, t + 1:].abs().sum() == 0, f"y[{t}] uses a future step"
        assert grad[:, t].abs().sum() > 0, f"y[{t}] ignores the current step"


def test_residual_block():
    """Residual_block runs with D_rnn != D and keeps the fixes of C3-C9 and S1-S3."""
    torch.manual_seed(0)
    B, T, D = 2, 10, 6  # D_rnn = 8 != D, so a D / D_rnn mix-up fails
    blk = Residual_block(D)
    rglru = blk.tmb.rglru

    # C3: Lambda stays an nn.Parameter
    assert isinstance(rglru.Lambda, torch.nn.Parameter)
    # C4: Recurrent_block(D) without D_rnn gets the same width
    assert Recurrent_block(D).D_rnn == blk.tmb.D_rnn == 8
    # S2: reset_parameters() sets the gate biases to 0, whatever was in memory
    with torch.no_grad():
        rglru.ba.fill_(1.0)
        rglru.bx.fill_(1.0)
    rglru.reset_parameters()
    assert (rglru.ba == 0).all() and (rglru.bx == 0).all()
    # S3: one RMSNorm per branch
    assert blk.tmb_norm is not blk.mlp_norm

    # C5-C8: forward runs and keeps the shape
    x = torch.randn(B, T, D, requires_grad=True)
    y = blk(x)
    assert y.shape == (B, T, D)

    # S1: y[t] never uses x[s] with s > t
    for t in range(T):
        (grad,) = torch.autograd.grad(y[:, t].sum(), x, retain_graph=True)
        assert grad[:, t + 1:].abs().sum() == 0, f"y[{t}] uses a future step"

    y.sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name

    # C9: h_0 follows the dtype and the device of x
    blk_bf16 = Residual_block(D).to(torch.bfloat16)
    y_bf16 = blk_bf16(x.detach().to(torch.bfloat16))
    assert y_bf16.dtype == torch.bfloat16
    # N1, N2: no inf or NaN gradients in bfloat16
    y_bf16.float().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in blk_bf16.parameters())
    # the meta device holds no data, so it checks the device without a GPU
    blk_meta = Residual_block(D).to("meta")
    assert blk_meta(x.detach().to("meta")).device.type == "meta"


def test_input_term():
    """N2, N1: with numerically_stable=True, input_term keeps its digits near a_t = 1 and its gradient finite at 1 - a_t^2 = 0."""
    stable, plain = RGLRU(1), RGLRU(1, numerically_stable=False)
    log_a = -torch.logspace(-8, 0, 9, dtype=torch.float64)  # -1e-8 ... -1
    a, one = torch.exp(log_a), torch.ones_like(log_a)  # i = x = 1, so the term is sqrt(1 - a**2)

    # float64: both formulas give the same value, up to the rounding of 1 - a**2 (about 1e-16)
    ref = stable.input_term(log_a, a, one, one)
    torch.testing.assert_close(plain.input_term(log_a, a, one, one), ref, rtol=1e-7, atol=0)

    # float32: as written in (4) the term is exactly 0 at log_a = -1e-8; the stable form stays within 1e-6
    f32 = [t.float() for t in (log_a, a, one, one)]
    assert plain.input_term(*f32)[0] == 0
    rel = ((stable.input_term(*f32).double() - ref) / ref).abs()
    assert rel.max() < 1e-6

    # N1: the derivative of the sqrt is bounded at 1000, the plain sqrt gives inf at 0
    v = torch.tensor([0.0, 1e-8, 1.0], requires_grad=True)
    SqrtBoundDerivative.apply(v).sum().backward()
    torch.testing.assert_close(v.grad, torch.tensor([1000.0, 1000.0, 0.5]))
    # r_t = 0 exactly gives log_a = 0 and 1 - a**2 = 0: the gradient stays finite only with the flag
    for m, finite in ((stable, True), (plain, False)):
        la0 = torch.zeros(3, requires_grad=True)
        m.input_term(la0, torch.exp(la0), one[:3], one[:3]).sum().backward()
        assert torch.isfinite(la0.grad).all() == finite

    # both settings of the flag give the same RG-LRU output for ordinary inputs
    torch.manual_seed(0)
    stable, plain = RGLRU(8), RGLRU(8, numerically_stable=False)
    plain.load_state_dict(stable.state_dict())
    x = torch.randn(2, 5, 8)
    torch.testing.assert_close(stable(x), plain(x), rtol=1e-5, atol=1e-6)


def test_rmsnorm():
    """N3: RMSNorm follows the official code: eps = 1e-6 inside the sqrt, gain 1 + scale with scale = 0."""
    torch.manual_seed(0)
    D = 6
    norm = RMSNorm(D)
    assert norm.eps == 1e-6 and (norm.scale == 0).all()
    x = torch.randn(2, 10, D, dtype=torch.float64)
    ref = x / torch.sqrt((x**2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(norm.double()(x), ref)

    # float16, zero vector (a padding token): the old eps = 1e-8 rounded to 0 and gave 0 / 0 = NaN
    x16 = torch.zeros(2, 3, D, dtype=torch.float16, requires_grad=True)
    y16 = RMSNorm(D).half()(x16)
    assert y16.dtype == torch.float16 and (y16 == 0).all()
    y16.float().sum().backward()
    assert torch.isfinite(x16.grad).all()


def test_local_mqa():
    """Local MQA: one shared K/V head, a causal window of window_size tokens back, and RoPE."""
    torch.manual_seed(0)
    B, T, D, D_head, W = 2, 12, 12, 4, 3  # H = 3 heads, each token sees itself and 3 tokens back
    blk = Local_MQA_block(D, D_head=D_head, window_size=W)
    assert blk.H == 3 and blk.Wk.weight.shape == blk.Wv.weight.shape == (D_head, D)  # MQA: one K and one V head
    x = torch.randn(B, T, D, requires_grad=True)
    y = blk(x)
    assert y.shape == (B, T, D)

    # window: y[t] uses x[s] only for t - W <= s <= t
    for t in range(T):
        (grad,) = torch.autograd.grad(y[:, t].sum(), x, retain_graph=True)
        used = grad.abs().sum(dim=(0, 2)) > 0  # (T,)
        expected = torch.zeros(T, dtype=torch.bool)
        expected[max(0, t - W):t + 1] = True
        assert torch.equal(used, expected), f"y[{t}] uses {used.nonzero().flatten().tolist()}"

    # the attention core equals F.scaled_dot_product_attention with the same mask (K, V repeated over the heads)
    positions = torch.arange(T)
    q = apply_rope(blk.Wq(x).view(B, T, blk.H, D_head).transpose(1, 2), positions)
    k = apply_rope(blk.Wk(x).unsqueeze(1), positions).expand(B, blk.H, T, D_head)
    v = blk.Wv(x).unsqueeze(1).expand(B, blk.H, T, D_head)
    mask = (positions[None, :] <= positions[:, None]) & (positions[None, :] >= positions[:, None] - W)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(y, blk.Wo(ref.transpose(1, 2).reshape(B, T, D)), rtol=1e-5, atol=1e-6)

    # RoPE: keeps the norm, and q_m . k_n depends only on m - n
    qr, kr = torch.randn(T, D_head), torch.randn(T, D_head)
    torch.testing.assert_close(apply_rope(qr, positions).norm(dim=-1), qr.norm(dim=-1))
    scores = apply_rope(qr, positions) @ apply_rope(kr, positions).T
    shifted = apply_rope(qr, positions + 100) @ apply_rope(kr, positions + 100).T
    torch.testing.assert_close(shifted, scores, rtol=1e-5, atol=1e-5)
