"""Experimental disk encoding for existing simulated-int4 Nox artifacts.

Factors are requantized onto a groupwise int4 grid with FP32 scales. Sparse
values remain FP16, with CSR row pointers and compact column indices. This is
a storage control: factor rounding can change and requires quality validation.
"""
import math

import torch


def pack_factor(tensor, group_size=64):
    if group_size <= 0 or group_size % 2:
        raise ValueError("Group size must be positive and even")
    if tensor.numel() == 0 or not torch.isfinite(tensor).all():
        raise ValueError("Expected a nonempty finite factor")
    flat = tensor.detach().cpu().float().reshape(-1)
    padding = (-flat.numel()) % group_size
    groups = torch.nn.functional.pad(flat, (0, padding)).reshape(-1, group_size)
    scales = groups.abs().amax(dim=1) / 7
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = (torch.round(groups / scales[:, None]).clamp(-8, 7).to(torch.int16) + 8).to(torch.uint8).flatten()
    packed = codes[::2] | (codes[1::2] << 4)
    return {"codes": packed, "scales": scales, "shape": tuple(tensor.shape), "group_size": group_size}


def unpack_factor(packed):
    size = math.prod(packed["shape"])
    group_size = packed["group_size"]
    codes = packed["codes"]
    values = torch.stack((codes & 15, codes >> 4), dim=1).flatten().to(torch.float32) - 8
    values = values.reshape(-1, group_size) * packed["scales"][:, None]
    return values.flatten()[:size].reshape(packed["shape"]).half()


def tensor_payload_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_payload_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_payload_bytes(v) for v in value)
    return 0


def pack_artifact(artifact):
    if artifact.get("quantization_bits", 4) != 4:
        raise ValueError("Packing control expects an existing simulated-int4 artifact")
    if "encoding" in artifact:
        raise ValueError("Artifact is already encoded")
    packed = dict(artifact)
    packed["encoding"] = "nox_int4_csr_v1"
    packed["cores"] = [pack_factor(core) for core in artifact["cores"]]
    for key in ("A_bridge", "B_bridge"):
        if artifact.get(key) is not None:
            packed[key] = pack_factor(artifact[key])
    indices = packed.pop("sparse_indices", None)
    if indices is not None:
        rows, columns = artifact["orig_shape"]
        indices = indices.cpu().long()
        if len(indices) != len(artifact["sparse_values"]):
            raise ValueError("Sparse indices/values length mismatch")
        if len(indices):
            flat = indices[:, 0] * columns + indices[:, 1]
            if (indices < 0).any() or (indices[:, 0] >= rows).any() or (indices[:, 1] >= columns).any():
                raise ValueError("Sparse coordinates out of bounds")
            if not torch.all(flat[1:] > flat[:-1]):
                raise ValueError("Sparse coordinates must be unique and row-sorted")
        counts = torch.bincount(indices[:, 0], minlength=rows)
        packed["sparse_row_ptr"] = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).int()
        packed["sparse_columns"] = indices[:, 1].to(torch.int16 if columns <= 32768 else torch.int32)
    packed["total_kb"] = tensor_payload_bytes(packed) / 1024
    return packed


def unpack_artifact(packed):
    if packed.get("encoding") != "nox_int4_csr_v1":
        raise ValueError("Unknown packed artifact encoding")
    artifact = dict(packed)
    artifact.pop("encoding")
    artifact["cores"] = [unpack_factor(core) for core in packed["cores"]]
    for key in ("A_bridge", "B_bridge"):
        if packed.get(key) is not None:
            artifact[key] = unpack_factor(packed[key])
    if "sparse_row_ptr" in packed:
        pointers = artifact.pop("sparse_row_ptr").long()
        columns = artifact.pop("sparse_columns").long()
        rows = torch.repeat_interleave(torch.arange(len(pointers) - 1), pointers.diff())
        artifact["sparse_indices"] = torch.stack((rows, columns), dim=1).int()
    else:
        artifact["sparse_indices"] = None
    artifact["total_kb"] = tensor_payload_bytes(artifact) / 1024
    return artifact
