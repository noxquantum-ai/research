# NEC portable Schmidt redesign

## Decision

The full 32-layer allocation run proved that allocation cannot rescue a weak
codec.  The selected 28-layer assembly reached 3.45x serialized compression
and WikiText-2 PPL 712.18.  NEC therefore needs a new representation before
another allocation search.

The new path is an adaptive binary Schmidt compiler:

1. Discover repeated blocks and matrix operators from the loaded module graph.
   Do not encode Llama projection names into the codec.
2. Treat every matrix as a sum of operators with adaptive Schmidt bonds.  Pack
   factor signs at one bit and store explicit magnitude coordinates.  Less
   aggressive frontier points spend bytes on independently canonicalized
   residual paths instead of padding a single saturated path.
3. Allocate bond dimensions under an actual whole-package byte constraint.
   Operator spectra and held-out functional loss determine how rank moves
   between modules after the uniform preflight.
4. Optimize the assembled student against teacher logits and hidden
   trajectories.  Local matrix error is a diagnostic, not the final objective.
5. Serialize the packed representation and measure the resulting directory.
   No coefficient estimate qualifies as achieved compression.

This retains NEC's quantum-inspired basis: Schmidt spectra measure operator
entanglement, adaptive bond dimensions distribute representational capacity,
and canonical tensor factors define the stored operator.  It removes the
fixed four-site chain whose central cut discarded 80–93% of residual energy
in the recorded audit.

## Architecture portability contract

`nox.model_adapter` discovers the largest repeated `ModuleList`, inventories
`Linear`, `Embedding`, and GPT-style `Conv1D` matrices, maps them to block
indices, and identifies shared weights.  Compression and accounting operate
on this inventory.  Model-family adapters may add faster kernels or semantic
grouping, but correctness cannot depend on Llama-only names, SwiGLU, separate
Q/K/V matrices, RoPE, a fixed vocabulary, or a fixed number of blocks.

The first portability gate covers at least one model from each distinct
layout class:

| Layout property | Required gate |
| --- | --- |
| Separate Q/K/V, gated MLP | Llama/Mistral-style decoder |
| Fused QKV | Phi- or Falcon-style decoder |
| GPT `Conv1D` operators | GPT-2-style decoder |
| Tied input/output embeddings | A tied-embedding causal LM |
| Mixture of experts | Inventory and storage preflight before compression |

Passing discovery is not a quality claim.  Each model must independently pass
serialization parity, held-out PPL/KL, task evaluation, and actual package
measurement.

## Commercial boundary

The LittleBit/LittleBit-2 repository at commit
`42d658b0c79f76450b34b6a3547462c7cdc3e1a0` is licensed CC BY-NC 4.0.  It may
be used in this research run as an attributed experimental oracle.  It is not
part of the commercial NEC code path, and its source or derived implementation
must not be copied into a commercial package without a separate commercial
license.  NEC's implementation is maintained independently in this repository.

This is an engineering provenance rule rather than a conclusion about patent
rights.  Commercial release still requires the company's normal legal review
of dependencies, model licenses, datasets, and relevant patents.

## Acceptance gates

The immediate objective is quality-constrained compression, not a fixed 10x
pass condition.  NEC produces a measured frontier at approximately 2x, 3x,
4x, 5x, 6x, 8x, and 10x, then selects the most compressed package that passes
the quality gates.  Ten times remains a long-term ambition.

For Llama-2-7B, PPL at most 10 remains the permissive outer limit inherited
from the earlier experiments.  A credible "maintained accuracy" result should
also stay substantially closer to the dense PPL, preserve at least 90% of the
dense score on each reported public task, and report teacher KL, next-token
accuracy, and dense top-1 agreement.  Results that only satisfy the permissive
PPL ceiling are labeled degraded rather than accuracy preserving.  The same
compiler interface must operate on other architectures; numerical quality
gates are evaluated relative to each model's dense baseline.

The run is infeasible if unchanged bytes plus the minimum factor payload
exceed the package budget.  The compiler reports this before GPU optimization.
If storage is feasible but quality fails, it reports a representation-quality
failure at that frontier point.  It may select a less aggressive measured
point only when that point independently passes the stated quality gates.
