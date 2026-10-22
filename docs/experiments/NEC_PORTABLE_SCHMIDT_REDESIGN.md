# NEC portable Schmidt redesign

## Decision

The full 32-layer allocation run proved that allocation cannot rescue a weak
codec.  The selected 28-layer assembly reached 3.45x serialized compression
and WikiText-2 PPL 712.18.  NEC therefore needs a new representation before
another allocation search.

The new path is an adaptive binary Schmidt compiler:

1. Discover repeated blocks and matrix operators from the loaded module graph.
   Do not encode Llama projection names into the codec.
2. Treat every matrix as an operator with an adaptive Schmidt bond.  Pack both
   factor signs at one bit and store explicit magnitude coordinates.
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

The Llama-2-7B research target remains at least 10x actual serialized
whole-model compression with PPL at most 10.  A run must also report teacher
KL, next-token accuracy, dense top-1 agreement, and public task results.  The
same compiler interface must operate on other architectures; quality targets
are evaluated per model against its dense baseline.

The run is infeasible if unchanged bytes plus the minimum factor payload
exceed the package budget.  The compiler reports this before GPU optimization.
If storage is feasible but quality fails, it reports a representation-quality
failure and does not relax the ratio or PPL target silently.
