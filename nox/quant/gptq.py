"""RTN and GPTQ (Hessian error feedback) with group-wise symmetric quantization.

Convention: W has shape (d_out, d_in) and the layer computes y = W x, so the
calibration Hessian H = E[x x^T] has shape (d_in, d_in). The objective is
||(W - W_hat) X||_F^2 = tr((W - W_hat) H (W - W_hat)^T) up to the token count.
"""

import torch


def bits_per_weight(bits, group_size, scale_bits=16):
    """Payload bits per weight including one fp16 scale per group."""
    return bits + scale_bits / group_size


def rtn_quantize(weight, bits=4, group_size=128):
    """Round-to-nearest, groups along d_in for each output row."""
    w = weight.float()
    d_out, d_in = w.shape
    assert d_in % group_size == 0, "d_in must be a multiple of group_size"
    groups = w.reshape(d_out, d_in // group_size, group_size)
    scale = groups.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / max(2 ** (bits - 1) - 1, 1)
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return (torch.clamp(torch.round(groups / scale), qmin, qmax) * scale).reshape(d_out, d_in)


@torch.no_grad()
def gptq_quantize(weight, hessian, bits=4, group_size=128, damp=0.01, act_order=False,
                  block_size=128):
    """GPTQ: quantize columns in order, pushing each column's error onto later ones.

    Scales for each group are fixed from the error-updated weights when the
    group is first reached, as in the reference implementation.
    """
    w = weight.clone().float()
    d_out, d_in = w.shape
    assert d_in % group_size == 0, "d_in must be a multiple of group_size"
    assert block_size % group_size == 0, "block_size must be a multiple of group_size"
    h = hessian.clone().float().to(w.device)
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(h), descending=True)
        w, h = w[:, perm], h[perm][:, perm]
    h = h + damp * torch.mean(torch.diag(h)) * torch.eye(d_in, device=h.device)
    chol = torch.linalg.cholesky(h)
    h_inv = torch.cholesky_inverse(chol)
    u = torch.linalg.cholesky(h_inv, upper=True)

    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    q_out = torch.zeros_like(w)
    scale = None
    for start in range(0, d_in, block_size):
        end = min(start + block_size, d_in)
        w_blk = w[:, start:end].clone()
        err_blk = torch.zeros_like(w_blk)
        u_blk = u[start:end, start:end]
        for i in range(end - start):
            col = start + i
            if col % group_size == 0:
                group = w_blk[:, i:i + group_size]
                scale = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / max(qmax, 1)
            wc = w_blk[:, i:i + 1]
            qc = torch.clamp(torch.round(wc / scale), qmin, qmax) * scale
            q_out[:, col:col + 1] = qc
            e = (wc - qc) / u_blk[i, i]
            w_blk[:, i:] -= e @ u_blk[i:i + 1, i:]
            err_blk[:, i:i + 1] = e
        w[:, end:] -= err_blk @ u[start:end, end:]
    if perm is not None:
        inv = torch.argsort(perm)
        q_out = q_out[:, inv]
    return q_out
