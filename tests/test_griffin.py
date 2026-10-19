import pytest
import torch

from griffin import (
    RGLRU,
    TABLE_2,
    Griffin,
    GriffinConfig,
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

    y, _ = conv(x)
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
    rglru = blk.temporal_block.rglru

    # C3: Lambda stays an nn.Parameter
    assert isinstance(rglru.Lambda, torch.nn.Parameter)
    # C4: Recurrent_block(D) without D_rnn gets the same width
    assert Recurrent_block(D).D_rnn == blk.temporal_block.D_rnn == 8
    # S2: reset_parameters() sets the gate biases to 0, whatever was in memory
    with torch.no_grad():
        rglru.ba.fill_(1.0)
        rglru.bx.fill_(1.0)
    rglru.reset_parameters()
    assert (rglru.ba == 0).all() and (rglru.bx == 0).all()
    # S3: one RMSNorm per branch
    assert blk.temporal_norm is not blk.mlp_norm

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


def test_residual_block_attention():
    """Residual_block(temporal_block_type="attention") uses the local MQA block and stays causal."""
    torch.manual_seed(0)
    B, T, D = 2, 10, 256  # D must be a multiple of D_head = 128: H = 2 heads
    blk = Residual_block(D, temporal_block_type="attention")
    assert isinstance(blk.temporal_block, Local_MQA_block) and isinstance(Residual_block(6).temporal_block, Recurrent_block)
    x = torch.randn(B, T, D, requires_grad=True)
    y = blk(x)
    assert y.shape == (B, T, D)
    for t in range(T):
        (grad,) = torch.autograd.grad(y[:, t].sum(), x, retain_graph=True)
        assert grad[:, t + 1:].abs().sum() == 0, f"y[{t}] uses a future step"
    y.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in blk.parameters())
    with pytest.raises(ValueError):
        Residual_block(D, temporal_block_type="unknown")


def test_griffin_model():
    """Griffin: embedding -> N residual blocks (recurrent, recurrent, attention, ...) -> RMSNorm -> shared linear -> softmax."""
    torch.manual_seed(0)
    B, T, V, D, N = 2, 7, 50, 256, 4
    config = GriffinConfig(V=V, D=D, N=N, D_rnn=300, M=2, H=2, window_size=3)  # every column of Table 2
    model = Griffin(config)
    assert [type(b.temporal_block).__name__ for b in model.blocks] == [
        "Recurrent_block", "Recurrent_block", "Local_MQA_block", "Recurrent_block"]
    assert model.blocks[0].temporal_block.D_rnn == 300 and model.blocks[0].mlp.M == 2
    mqa = model.blocks[2].temporal_block
    assert (mqa.H, mqa.D_head, mqa.window_size) == (2, 128, 3)
    tokens = torch.randint(0, V, (B, T))
    logits = model(tokens)
    assert logits.shape == (B, T, V)
    # the output layer is the embedding matrix: the model holds one (V, D) parameter, not two
    assert sum(p.numel() for p in model.parameters() if p.shape == (V, D)) == V * D
    probs = model.probabilities(tokens)
    torch.testing.assert_close(probs.sum(-1), torch.ones(B, T))

    # causal: changing the tokens from position 4 on leaves the logits before 4 unchanged
    tokens2 = tokens.clone()
    tokens2[:, 4:] = (tokens2[:, 4:] + 1) % V
    logits2 = model(tokens2)
    torch.testing.assert_close(logits2[:, :4], logits[:, :4])
    assert not torch.allclose(logits2[:, 4:], logits[:, 4:])

    # one training step: next-token loss with finite gradients everywhere
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, V), tokens[:, 1:].reshape(-1))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_griffin_config():
    """GriffinConfig holds Table 2; from_table2 fills a row, and D_head = D / H = 128 in every row."""
    cfg = GriffinConfig.from_table2("7B", V=32_000)
    assert (cfg.V, cfg.D, cfg.D_rnn, cfg.N, cfg.M, cfg.H, cfg.window_size) == (32_000, 4096, 5632, 32, 3, 32, 1024)
    assert all(row["D"] // row["H"] == 128 and row["D"] % row["H"] == 0 for row in TABLE_2.values())
    assert [TABLE_2[s]["N"] for s in TABLE_2] == [12, 12, 12, 24, 24, 32, 40]
    # the defaults follow the rules of the paper: D_rnn = 4D/3, M = 3, D_head = 128, window 1024
    blk = Residual_block(256)
    assert (blk.temporal_block.D_rnn, blk.mlp.M) == (round(4 * 256 / 3), 3)
    blk = Residual_block(256, temporal_block_type="attention")
    assert (blk.temporal_block.D_head, blk.temporal_block.window_size) == (128, 1024)


def test_rglru_cache():
    """RG-LRU cache: prefill returns h_T, decode continues from it, and the result equals one forward call."""
    torch.manual_seed(0)
    B, T, Tp, D = 2, 9, 6, 8
    m = RGLRU(D)
    x = torch.randn(B, T, D)
    y_full, h_full = m(x)
    # y_t = h_t, but the cache is its own tensor, not a view of y
    assert torch.equal(h_full, y_full[:, -1])
    assert h_full.untyped_storage().data_ptr() != y_full.untyped_storage().data_ptr()

    # prefill on the prompt, then one decode call per token
    y_pre, h = m(x[:, :Tp])
    assert y_pre.shape == (B, Tp, D) and h.shape == (B, D)
    ys = [y_pre]
    for t in range(Tp, T):
        y_t, h = m(x[:, t:t + 1], h0=h)
        assert y_t.shape == (B, 1, D)
        ys.append(y_t)
    torch.testing.assert_close(torch.cat(ys, dim=1), y_full)
    torch.testing.assert_close(h, h_full)

    # a prompt in two chunks gives the same result as one prefill
    y_a, h_a = m(x[:, :4])
    y_b, h_b = m(x[:, 4:], h0=h_a)
    torch.testing.assert_close(torch.cat([y_a, y_b], dim=1), y_full)
    torch.testing.assert_close(h_b, h_full)


def test_conv1d_cache():
    """Conv1D cache: prefill returns the last 3 inputs, decode continues from them, and the result equals one forward call."""
    torch.manual_seed(0)
    B, T, Tp, D = 2, 9, 6, 8
    conv = Temporal_Conv1D(D, kernel_size=4)
    x = torch.randn(B, T, D)
    y_full, state_full = conv(x)
    # the cache is the last kernel_size - 1 = 3 inputs, in its own memory, not a view of the padded input
    assert torch.equal(state_full, x[:, -3:])
    assert state_full.untyped_storage().nbytes() == state_full.nbytes

    # prefill on the prompt, then one decode call per token
    y_pre, state = conv(x[:, :Tp])
    assert y_pre.shape == (B, Tp, D)
    assert state.shape == (B, 3, D)
    ys = [y_pre]
    for t in range(Tp, T):
        y_t, state = conv(x[:, t:t + 1], state=state)
        assert y_t.shape == (B, 1, D)
        ys.append(y_t)
    torch.testing.assert_close(torch.cat(ys, dim=1), y_full)
    torch.testing.assert_close(state, state_full)

    # a prompt in two chunks gives the same result as one prefill
    y_a, state_a = conv(x[:, :4])
    y_b, state_b = conv(x[:, 4:], state=state_a)
    torch.testing.assert_close(torch.cat([y_a, y_b], dim=1), y_full)
    torch.testing.assert_close(state_b, state_full)

    # a prompt shorter than 3 tokens: the cache keeps zeros on the left
    _, state_2 = conv(x[:, :2])
    assert torch.equal(state_2[:, :1], torch.zeros(B, 1, D))
    assert torch.equal(state_2[:, 1:], x[:, :2])
