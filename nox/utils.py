"""Utility functions for tensor operations, quantization, and evaluation metrics."""

import torch
import torch.nn.functional as F


def quantize_dequantize_groupwise(tensor, bits=4, group_size=64):
    """
    Simulates uniform group-wise symmetric quantization and dequantization.
    Used for core quantization in FP16/FP32 simulation.
    """
    if bits >= 16:
        return tensor
    orig_shape = tensor.shape
    flat = tensor.reshape(-1)
    pad = (group_size - (flat.numel() % group_size)) % group_size
    if pad > 0:
        flat = F.pad(flat, (0, pad))
    groups = flat.reshape(-1, group_size)

    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    max_val = torch.amax(torch.abs(groups), dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_val / qmax
    q = torch.clamp(torch.round(groups / scale), qmin, qmax)
    deq = (q * scale).reshape(-1)
    if pad > 0:
        deq = deq[:-pad]
    return deq.reshape(orig_shape)


def quantize_4bit_per_channel(tensor, dim=0, bits=4):
    """
    Symmetric per-channel quantization/dequantization along the specified channel dimension.
    """
    if bits >= 16:
        return tensor
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    dims_to_reduce = [d for d in range(tensor.ndim) if d != dim]
    max_val = torch.amax(torch.abs(tensor), dim=dims_to_reduce, keepdim=True).clamp(min=1e-8)
    scale = max_val / qmax
    q = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return q * scale


def contract_mpo_4site(cores, X):
    """
    Contracts a 4-site Matrix Product Operator with an input activation matrix X.
    
    Args:
        cores: list of 4 tensor cores:
            c0: [1, o0, i0, r1]
            c1: [r1, o1, i1, r2]
            c2: [r2, o2, i2, r3]
            c3: [r3, o3, i3, 1]
        X: input activations of shape [d_in, N]
        
    Returns:
        Contracted output matrix Y of shape [d_out, N]
    """
    c0 = cores[0].squeeze(0)
    c1 = cores[1]
    c2 = cores[2]
    c3 = cores[3].squeeze(-1)

    o0, i0, _ = c0.shape
    _, o1, i1, _ = c1.shape
    _, o2, i2, _ = c2.shape
    _, o3, i3 = c3.shape

    N = X.shape[1]
    X_reshaped = X.reshape(i0, i1, i2, i3, N)
    Y = torch.einsum("aib,bcjd,dekf,fgl,ijklN->acegN", c0, c1, c2, c3, X_reshaped)
    return Y.reshape(o0 * o1 * o2 * o3, N)


def evaluate_nox_prediction(cores, bridge=None, perm_info=None, S_sparse=None, X=None):
    """
    Evaluates Nox MPO prediction with optional channel permutation,
    low-rank activation bridge, and sparse outlier residual.
    
    Args:
        cores: list of 4 MPO cores
        bridge: optional tuple (A_q, B_q) for low-rank outlier bridge
        perm_info: optional tuple (perm_in, perm_out, inv_perm_in, inv_perm_out)
        S_sparse: optional sparse outlier coordinate tensor
        X: input activation matrix [d_in, N]
        
    Returns:
        Predicted output activation matrix Y_pred [d_out, N]
    """
    if X is None:
        raise ValueError("Input activation matrix X cannot be None")

    if perm_info is not None:
        perm_in, perm_out, inv_perm_in, inv_perm_out = perm_info
        X_core = X[perm_in, :]
        Y_core_perm = contract_mpo_4site(cores, X_core)
        Y_pred = Y_core_perm[inv_perm_out, :]
    else:
        Y_pred = contract_mpo_4site(cores, X)

    if bridge is not None and isinstance(bridge, (tuple, list)) and len(bridge) == 2:
        A_q, B_q = bridge
        Y_pred = Y_pred + (A_q @ (B_q @ X))

    if S_sparse is not None and torch.is_tensor(S_sparse):
        Y_pred = Y_pred + (S_sparse @ X)

    return Y_pred


def mpo_to_dense(cores):
    """
    Reconstructs the full 2D dense matrix [d_out, d_in] from 4-site MPO cores.
    Useful for direct matrix inspection and norm comparisons.
    """
    c0 = cores[0].squeeze(0)
    c1 = cores[1]
    c2 = cores[2]
    c3 = cores[3].squeeze(-1)

    o0, i0, _ = c0.shape
    _, o1, i1, _ = c1.shape
    _, o2, i2, _ = c2.shape
    _, o3, i3 = c3.shape

    T = torch.einsum("aib,bcjd,dekf,fgl->acegijkl", c0, c1, c2, c3)
    return T.permute(0, 1, 2, 3, 4, 5, 6, 7).reshape(o0 * o1 * o2 * o3, i0 * i1 * i2 * i3)


def count_parameters(cores, bridge=None, S_sparse=None, bits=4):
    """
    Computes total non-zero parameters and estimated byte footprint.
    
    For 4-bit cores, elements occupy 0.5 bytes.
    For 4-bit low-rank bridge (A_q, B_q), elements occupy 0.5 bytes.
    For sparse CSR matrix S, stores non-zero float16 values (2 bytes) plus 16-bit column indices (2 bytes).
    """
    core_elements = sum(c.numel() for c in cores)
    core_bytes = core_elements * (bits / 8.0)

    bridge_elements = 0
    bridge_bytes = 0.0
    if bridge is not None and isinstance(bridge, (tuple, list)) and len(bridge) == 2:
        A_q, B_q = bridge
        bridge_elements = A_q.numel() + B_q.numel()
        bridge_bytes = bridge_elements * (bits / 8.0)

    sparse_elements = 0
    sparse_bytes = 0.0
    if S_sparse is not None and torch.is_tensor(S_sparse):
        sparse_elements = int((S_sparse != 0).sum().item())
        sparse_bytes = sparse_elements * 4.0

    total_elements = core_elements + bridge_elements + sparse_elements
    total_bytes = core_bytes + bridge_bytes + sparse_bytes
    return {
        "core_elements": core_elements,
        "bridge_elements": bridge_elements,
        "sparse_elements": sparse_elements,
        "total_elements": total_elements,
        "total_bytes": total_bytes,
        "total_kb": total_bytes / 1024.0,
    }


def compute_metrics(Y_true, Y_pred):
    """
    Calculates reconstruction fidelity metrics:
      - Relative Frobenius Error: ||Y_true - Y_pred||_F / ||Y_true||_F
      - Angular Drift: 1 - cosine_similarity(flattened Y_true, flattened Y_pred)
      - Norm Ratio: ||Y_pred||_F / ||Y_true||_F
    """
    diff_norm = torch.norm(Y_true - Y_pred, p="fro").item()
    true_norm = torch.norm(Y_true, p="fro").item() + 1e-12
    pred_norm = torch.norm(Y_pred, p="fro").item() + 1e-12

    relative_error = diff_norm / true_norm

    flat_true = Y_true.reshape(-1)
    flat_pred = Y_pred.reshape(-1)
    cos_sim = F.cosine_similarity(flat_true.unsqueeze(0), flat_pred.unsqueeze(0)).item()
    angular_drift = 1.0 - cos_sim

    norm_ratio = pred_norm / true_norm

    return {
        "relative_error": relative_error,
        "angular_drift": angular_drift,
        "norm_ratio": norm_ratio,
        "cosine_similarity": cos_sim,
    }
