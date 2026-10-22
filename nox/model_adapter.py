"""Architecture-neutral discovery for NEC compression targets.

The compiler must reason about the model that is actually loaded instead of
assuming Llama projection names.  This module deliberately depends only on
PyTorch.  Hugging Face model families can expose different attention and MLP
layouts, but they all ultimately contain repeated blocks and matrix-valued
operators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class CompressionTarget:
    """One matrix-valued module that NEC can replace."""

    name: str
    module_kind: str
    shape: tuple[int, int]
    block_index: int | None
    block_path: str | None
    role: str
    tied_group: int
    weight_bytes: int
    bias_bytes: int

    @property
    def scalar_count(self) -> int:
        return self.shape[0] * self.shape[1]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True)
class ModelInventory:
    """Portable description of a model's repeated blocks and matrix weights."""

    model_class: str
    model_type: str | None
    block_container: str | None
    block_count: int
    targets: tuple[CompressionTarget, ...]
    untargeted_parameter_bytes: int
    total_unique_parameter_bytes: int

    @property
    def target_weight_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for target in self.targets:
            if target.tied_group not in seen:
                seen.add(target.tied_group)
                total += target.weight_bytes
            total += target.bias_bytes
        return total

    def to_dict(self) -> dict:
        return {
            "model_class": self.model_class,
            "model_type": self.model_type,
            "block_container": self.block_container,
            "block_count": self.block_count,
            "target_weight_bytes": self.target_weight_bytes,
            "untargeted_parameter_bytes": self.untargeted_parameter_bytes,
            "total_unique_parameter_bytes": self.total_unique_parameter_bytes,
            "targets": [target.to_dict() for target in self.targets],
        }


def _matrix_module(module: nn.Module) -> bool:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        return True
    # Transformers' GPT-style Conv1D stores a transposed two-dimensional
    # weight and is not a torch.nn.Linear subclass.
    weight = getattr(module, "weight", None)
    return type(module).__name__ == "Conv1D" and isinstance(weight, torch.Tensor) and weight.ndim == 2


def _matrix_bytes(module: nn.Module) -> int:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return 0
    return weight.numel() * weight.element_size()


def _child_signature(module: nn.Module) -> tuple[str, ...]:
    return tuple(type(child).__name__ for child in module.children())


def discover_block_container(model: nn.Module) -> tuple[str | None, nn.ModuleList | None]:
    """Find the repeated block stack without a model-family allowlist.

    Candidate ``ModuleList`` objects are ranked by the matrix bytes they own.
    Requiring a repeated structural signature filters small unrelated lists
    such as task heads or auxiliary adapters.
    """

    candidates: list[tuple[int, int, str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) < 2:
            continue
        signatures = [_child_signature(child) for child in module]
        dominant = max((signatures.count(signature) for signature in set(signatures)), default=0)
        if dominant < max(2, int(0.75 * len(module))):
            continue
        matrix_bytes = sum(
            _matrix_bytes(child)
            for block in module
            for child in block.modules()
            if _matrix_module(child)
        )
        if matrix_bytes:
            candidates.append((matrix_bytes, len(module), name, module))
    if not candidates:
        return None, None
    _, _, name, module = max(candidates, key=lambda item: (item[0], item[1]))
    return name, module


def _embedding_identity(model: nn.Module, getter: str) -> int | None:
    method = getattr(model, getter, None)
    if not callable(method):
        return None
    try:
        module = method()
    except (AttributeError, NotImplementedError):
        return None
    weight = getattr(module, "weight", None)
    return id(weight) if isinstance(weight, torch.Tensor) else None


def _parameter_nbytes(parameter: torch.Tensor) -> int:
    return parameter.numel() * parameter.element_size()


def _unique_parameters(model: nn.Module) -> Iterable[torch.Tensor]:
    seen: set[int] = set()
    for parameter in model.parameters():
        identity = id(parameter)
        if identity not in seen:
            seen.add(identity)
            yield parameter


def inventory_model(model: nn.Module) -> ModelInventory:
    """Inventory compressible matrices, tied weights, and repeated blocks."""

    block_path, blocks = discover_block_container(model)
    block_prefixes = (
        tuple(f"{block_path}.{index}." for index in range(len(blocks)))
        if block_path is not None and blocks is not None else ()
    )
    input_embedding = _embedding_identity(model, "get_input_embeddings")
    output_embedding = _embedding_identity(model, "get_output_embeddings")

    group_for_weight: dict[int, int] = {}
    next_group = 0
    targets: list[CompressionTarget] = []
    target_parameter_ids: set[int] = set()

    for name, module in model.named_modules():
        if not name or not _matrix_module(module):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        identity = id(weight)
        if identity not in group_for_weight:
            group_for_weight[identity] = next_group
            next_group += 1
        target_parameter_ids.add(identity)

        bias = getattr(module, "bias", None)
        bias_bytes = _parameter_nbytes(bias) if isinstance(bias, torch.Tensor) else 0
        if isinstance(bias, torch.Tensor):
            target_parameter_ids.add(id(bias))

        block_index = None
        block_owner = None
        for index, prefix in enumerate(block_prefixes):
            if name == prefix[:-1] or name.startswith(prefix):
                block_index = index
                block_owner = f"{block_path}.{index}"
                break

        if identity == input_embedding and identity == output_embedding:
            role = "tied_input_output_embedding"
        elif identity == input_embedding:
            role = "input_embedding"
        elif identity == output_embedding:
            role = "output_head"
        elif isinstance(module, nn.Embedding):
            role = "embedding"
        else:
            role = "block_linear" if block_index is not None else "linear"

        targets.append(CompressionTarget(
            name=name,
            module_kind=type(module).__name__,
            shape=(int(weight.shape[0]), int(weight.shape[1])),
            block_index=block_index,
            block_path=block_owner,
            role=role,
            tied_group=group_for_weight[identity],
            weight_bytes=_parameter_nbytes(weight),
            bias_bytes=bias_bytes,
        ))

    total = sum(_parameter_nbytes(parameter) for parameter in _unique_parameters(model))
    targeted_unique = sum(
        _parameter_nbytes(parameter)
        for parameter in _unique_parameters(model)
        if id(parameter) in target_parameter_ids
    )
    config = getattr(model, "config", None)
    return ModelInventory(
        model_class=type(model).__name__,
        model_type=getattr(config, "model_type", None),
        block_container=block_path,
        block_count=0 if blocks is None else len(blocks),
        targets=tuple(targets),
        untargeted_parameter_bytes=total - targeted_unique,
        total_unique_parameter_bytes=total,
    )


def resolve_module(model: nn.Module, qualified_name: str) -> nn.Module:
    """Resolve a dotted module path, including numeric ModuleList indices."""

    current = model
    for part in qualified_name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"{qualified_name!r} did not resolve to a torch module")
    return current
