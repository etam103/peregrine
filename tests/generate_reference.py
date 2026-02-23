"""Generate PyTorch reference data for numerical parity tests.

Binary format per tensor: [ndim: u32][shape[0]: u32]...[data: f32 * N]

Usage: python tests/generate_reference.py
"""

import struct
import os
import torch
import torch.nn.functional as F
import numpy as np

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURES, exist_ok=True)

torch.manual_seed(42)


def save_tensor(path: str, t: torch.Tensor):
    """Save tensor in binary format: [ndim][shape...][f32 data]."""
    t = t.detach().float().contiguous()
    shape = t.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(shape)))
        for s in shape:
            f.write(struct.pack("<I", s))
        f.write(t.numpy().tobytes())


def save_vec(path: str, v: list):
    """Save a flat list of usize as [len: u32][v[0]: u32]..."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(v)))
        for x in v:
            f.write(struct.pack("<I", x))


# ---- 1. MatMul ----
print("Generating matmul reference...")
a = torch.randn(32, 64)
b = torch.randn(64, 16)
c = a @ b
save_tensor(f"{FIXTURES}/matmul_a.bin", a)
save_tensor(f"{FIXTURES}/matmul_b.bin", b)
save_tensor(f"{FIXTURES}/matmul_c.bin", c)

# Rectangular matmul
a2 = torch.randn(8, 128)
b2 = torch.randn(128, 32)
c2 = a2 @ b2
save_tensor(f"{FIXTURES}/matmul_rect_a.bin", a2)
save_tensor(f"{FIXTURES}/matmul_rect_b.bin", b2)
save_tensor(f"{FIXTURES}/matmul_rect_c.bin", c2)

# ---- 2. Softmax ----
print("Generating softmax reference...")
x_sm = torch.randn(16, 10)
y_sm = F.softmax(x_sm, dim=-1)
save_tensor(f"{FIXTURES}/softmax_input.bin", x_sm)
save_tensor(f"{FIXTURES}/softmax_output.bin", y_sm)

# Softmax with large values (numerical stability test)
x_sm_large = torch.randn(16, 10) * 100.0
y_sm_large = F.softmax(x_sm_large, dim=-1)
save_tensor(f"{FIXTURES}/softmax_large_input.bin", x_sm_large)
save_tensor(f"{FIXTURES}/softmax_large_output.bin", y_sm_large)

# ---- 3. Log-softmax ----
print("Generating log_softmax reference...")
x_lsm = torch.randn(16, 10)
y_lsm = F.log_softmax(x_lsm, dim=-1)
save_tensor(f"{FIXTURES}/log_softmax_input.bin", x_lsm)
save_tensor(f"{FIXTURES}/log_softmax_output.bin", y_lsm)

# ---- 4. LayerNorm ----
print("Generating layernorm reference...")
x_ln = torch.randn(8, 64)
ln = torch.nn.LayerNorm(64, elementwise_affine=True)
ln.weight.data = torch.randn(64)
ln.bias.data = torch.randn(64)
y_ln = ln(x_ln)
save_tensor(f"{FIXTURES}/layernorm_input.bin", x_ln)
save_tensor(f"{FIXTURES}/layernorm_gamma.bin", ln.weight.data)
save_tensor(f"{FIXTURES}/layernorm_beta.bin", ln.bias.data)
save_tensor(f"{FIXTURES}/layernorm_output.bin", y_ln)

# ---- 5. Cross-entropy loss ----
print("Generating cross_entropy reference...")
logits_ce = torch.randn(32, 10)
targets_ce = torch.randint(0, 10, (32,))
loss_ce = F.cross_entropy(logits_ce, targets_ce)
save_tensor(f"{FIXTURES}/ce_logits.bin", logits_ce)
save_vec(f"{FIXTURES}/ce_targets.bin", targets_ce.tolist())
save_tensor(f"{FIXTURES}/ce_loss.bin", loss_ce.unsqueeze(0))

# ---- 6. Element-wise ops ----
print("Generating element-wise reference...")
x_ew = torch.randn(8, 16)
y_ew = torch.randn(8, 16)

save_tensor(f"{FIXTURES}/ew_x.bin", x_ew)
save_tensor(f"{FIXTURES}/ew_y.bin", y_ew)
save_tensor(f"{FIXTURES}/ew_add.bin", x_ew + y_ew)
save_tensor(f"{FIXTURES}/ew_sub.bin", x_ew - y_ew)
save_tensor(f"{FIXTURES}/ew_mul.bin", x_ew * y_ew)
save_tensor(f"{FIXTURES}/ew_div.bin", x_ew / y_ew)
save_tensor(f"{FIXTURES}/ew_neg.bin", -x_ew)
save_tensor(f"{FIXTURES}/ew_exp.bin", x_ew.exp())
# Use abs to avoid log of negative
x_pos = x_ew.abs() + 0.01
save_tensor(f"{FIXTURES}/ew_x_pos.bin", x_pos)
save_tensor(f"{FIXTURES}/ew_log.bin", x_pos.log())
save_tensor(f"{FIXTURES}/ew_sqrt.bin", x_pos.sqrt())
save_tensor(f"{FIXTURES}/ew_abs.bin", x_ew.abs())
save_tensor(f"{FIXTURES}/ew_sin.bin", x_ew.sin())
save_tensor(f"{FIXTURES}/ew_cos.bin", x_ew.cos())
save_tensor(f"{FIXTURES}/ew_tanh.bin", x_ew.tanh())
save_tensor(f"{FIXTURES}/ew_relu.bin", F.relu(x_ew))
save_tensor(f"{FIXTURES}/ew_sigmoid.bin", torch.sigmoid(x_ew))

# ---- 7. Adam optimizer step ----
print("Generating Adam optimizer reference...")
w = torch.randn(16, 8, requires_grad=True)
w_init = w.data.clone()

# Fake a forward/backward to get gradients
target = torch.randn(4, 8)
x_adam = torch.randn(4, 16)
opt = torch.optim.Adam([w], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
opt.zero_grad()
pred = x_adam @ w
loss = ((pred - target) ** 2).mean()
loss.backward()
grad_snapshot = w.grad.data.clone()
opt.step()
w_after = w.data.clone()

save_tensor(f"{FIXTURES}/adam_w_init.bin", w_init)
save_tensor(f"{FIXTURES}/adam_grad.bin", grad_snapshot)
save_tensor(f"{FIXTURES}/adam_w_after.bin", w_after)

# ---- 8. Full MLP training comparison ----
print("Generating MLP training reference...")

torch.manual_seed(123)

# Weights for a 4->8->4->2 MLP
w1 = torch.randn(4, 8, requires_grad=True)
b1 = torch.zeros(1, 8, requires_grad=True)
w2 = torch.randn(8, 4, requires_grad=True)
b2 = torch.zeros(1, 4, requires_grad=True)
w3 = torch.randn(4, 2, requires_grad=True)
b3 = torch.zeros(1, 2, requires_grad=True)

save_tensor(f"{FIXTURES}/mlp_w1.bin", w1.data)
save_tensor(f"{FIXTURES}/mlp_b1.bin", b1.data)
save_tensor(f"{FIXTURES}/mlp_w2.bin", w2.data)
save_tensor(f"{FIXTURES}/mlp_b2.bin", b2.data)
save_tensor(f"{FIXTURES}/mlp_w3.bin", w3.data)
save_tensor(f"{FIXTURES}/mlp_b3.bin", b3.data)

# Fixed training data
x_train = torch.randn(16, 4)
targets_train = torch.randint(0, 2, (16,))
save_tensor(f"{FIXTURES}/mlp_x_train.bin", x_train)
save_vec(f"{FIXTURES}/mlp_targets_train.bin", targets_train.tolist())

opt = torch.optim.Adam([w1, b1, w2, b2, w3, b3], lr=1e-3)

losses = []
for step in range(10):
    opt.zero_grad()
    h1 = F.relu(x_train @ w1 + b1)
    h2 = F.relu(h1 @ w2 + b2)
    logits = h2 @ w3 + b3
    loss = F.cross_entropy(logits, targets_train)
    losses.append(loss.item())
    loss.backward()
    opt.step()

print(f"  MLP losses: {[f'{l:.6f}' for l in losses]}")
save_tensor(f"{FIXTURES}/mlp_losses.bin", torch.tensor(losses))
save_tensor(f"{FIXTURES}/mlp_w1_final.bin", w1.data)
save_tensor(f"{FIXTURES}/mlp_w2_final.bin", w2.data)
save_tensor(f"{FIXTURES}/mlp_w3_final.bin", w3.data)

print("Done! Reference data saved to tests/fixtures/")
