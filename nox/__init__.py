"""Nox: High-efficiency closed-form tensor network compression for LLMs."""

from nox.compiler import (
    nox_4step_compiler,
    nox_hybrid_dmrg_decompose,
    adaptive_tt_svd_4site,
    local_ridge_solve,
    solve_site_anchored,
    solve_accumulated_system,
)
from nox.utils import (
    quantize_dequantize_groupwise,
    quantize_4bit_per_channel,
    contract_mpo_4site,
    evaluate_nox_prediction,
    mpo_to_dense,
    count_parameters,
    compute_metrics,
    get_modes_for_projection,
)
from nox.block_compiler import (
    RMSNorm,
    scaled_dot_product_attention_2d,
    compile_attention_block,
    compile_mlp_block,
    compile_transformer_block,
    compile_multiverse_block,
    compute_isobyte_multiverse_ranks,
    evaluate_compiled_block,
)

__all__ = [
    "nox_4step_compiler",
    "nox_hybrid_dmrg_decompose",
    "adaptive_tt_svd_4site",
    "local_ridge_solve",
    "solve_site_anchored",
    "solve_accumulated_system",
    "quantize_dequantize_groupwise",
    "quantize_4bit_per_channel",
    "contract_mpo_4site",
    "evaluate_nox_prediction",
    "mpo_to_dense",
    "count_parameters",
    "compute_metrics",
    "get_modes_for_projection",
    "RMSNorm",
    "scaled_dot_product_attention_2d",
    "compile_attention_block",
    "compile_mlp_block",
    "compile_transformer_block",
    "compile_multiverse_block",
    "compute_isobyte_multiverse_ranks",
    "evaluate_compiled_block",
]
