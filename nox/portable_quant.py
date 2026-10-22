"""Portable activation-aware packed operators for NEC model families."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from nox.model_adapter import CompressionTarget, resolve_module
from nox.residual_codec import decode_groupwise, encode_groupwise


def artifact_filename(module_name: str) -> str:
    readable = module_name.replace(".", "_")[-96:]
    digest = hashlib.sha256(module_name.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}.pt"


def effective_weight(module: nn.Module) -> tuple[torch.Tensor, bool]:
    """Return a conventional [out, in] matrix and whether storage is transposed."""

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("module does not expose a matrix weight")
    if type(module).__name__ == "Conv1D":
        return weight.T, True
    return weight, False


class ActivationRMSCollector:
    """Collect input-channel RMS values through module pre-forward hooks."""

    def __init__(self, model: nn.Module, targets: Iterable[CompressionTarget]):
        self.model = model
        self.targets = tuple(targets)
        self.sum_squares: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.handles = []

    def _hook(self, name: str, in_features: int):
        def collect(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            value = inputs[0].detach()
            if not value.is_floating_point() or value.shape[-1] != in_features:
                return
            flat = value.reshape(-1, in_features).float()
            update = flat.square().sum(dim=0)
            if name not in self.sum_squares:
                self.sum_squares[name] = update
                self.counts[name] = int(flat.shape[0])
            else:
                self.sum_squares[name].add_(update)
                self.counts[name] += int(flat.shape[0])
        return collect

    def __enter__(self):
        for target in self.targets:
            if target.module_kind == "Embedding":
                continue
            module = resolve_module(self.model, target.name)
            matrix, _ = effective_weight(module)
            self.handles.append(module.register_forward_pre_hook(
                self._hook(target.name, int(matrix.shape[1])),
            ))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def rms(self, name: str) -> torch.Tensor | None:
        if name not in self.sum_squares:
            return None
        return (self.sum_squares[name] / self.counts[name]).sqrt()


@torch.no_grad()
def encode_target(
    model: nn.Module,
    target: CompressionTarget,
    *,
    bits: int,
    group_size: int,
    activation_rms: torch.Tensor | None,
    activation_alpha: float,
) -> dict:
    module = resolve_module(model, target.name)
    matrix, transposed_storage = effective_weight(module)
    artifact = encode_groupwise(
        matrix,
        bits=bits,
        group_size=group_size,
        activation_rms=activation_rms,
        alpha=activation_alpha if activation_rms is not None else 0.0,
    )
    artifact.update({
        "codec_kind": "portable_direct_quant",
        "module_name": target.name,
        "module_kind": target.module_kind,
        "role": target.role,
        "transposed_storage": transposed_storage,
        "source_shape": tuple(module.weight.shape),
    })
    return artifact


@torch.no_grad()
def inject_portable_package(
    model: nn.Module,
    package_dir: str | Path,
    *,
    device: str | torch.device | None = None,
) -> list[str]:
    """Decode a portable package into a compatible loaded model."""

    package_dir = Path(package_dir)
    manifest = __import__("json").loads((package_dir / "nec_manifest.json").read_text())
    injected = []
    for record in manifest["compressed_modules"]:
        artifact = torch.load(
            package_dir / "operators" / record["artifact_file"],
            map_location="cpu",
            weights_only=True,
        )
        module = resolve_module(model, record["name"])
        target_device = module.weight.device if device is None else torch.device(device)
        matrix = decode_groupwise(artifact, device=target_device, dtype=module.weight.dtype)
        stored = matrix.T if artifact.get("transposed_storage") else matrix
        if tuple(stored.shape) != tuple(module.weight.shape):
            raise ValueError(f"{record['name']}: decoded shape differs from destination")
        module.weight.data.copy_(stored)
        injected.append(record["name"])
        for alias_name in record.get("aliases", ()):
            alias = resolve_module(model, alias_name)
            if alias.weight is not module.weight:
                if tuple(alias.weight.shape) != tuple(stored.shape):
                    raise ValueError(f"{alias_name}: tied alias shape differs from decoded operator")
                alias.weight.data.copy_(stored)
            injected.append(alias_name)
    return injected


def load_portable_model(
    package_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = False,
):
    """Construct a causal LM from a self-contained NEC package.

    The original dense weight shards are never loaded.  The current reference
    runtime materializes decoded matrices before moving the model to its final
    device; packed execution kernels can replace that boundary independently.
    """

    import json
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    package_dir = Path(package_dir)
    metadata_dir = package_dir / "hf_metadata"
    manifest = json.loads((package_dir / "nec_manifest.json").read_text())
    config = AutoConfig.from_pretrained(metadata_dir, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_config(
        config,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )
    base_state = torch.load(package_dir / "base_state.pt", map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(base_state, strict=False)
    expected_missing = {
        f"{name}.weight"
        for record in manifest["compressed_modules"]
        for name in (record["name"], *record.get("aliases", ()))
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "portable base state differs from manifest: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    inject_portable_package(model, package_dir, device="cpu")
    model.to(device=device, dtype=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(metadata_dir, trust_remote_code=trust_remote_code)
    return model, tokenizer
