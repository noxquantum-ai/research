# NEC full-32-layer allocation run

This run uses the full original 32-layer Llama-2-7B for evaluation and treats
all layers 0–31 as allocation candidates. The allocator may retain at most four
non-NEC layers. Each retained layer can use either FP16 or per-output-channel
INT8; the remaining candidates use their existing or newly compiled NEC
checkpoints. The model architecture and layers outside the selected NEC set
remain present during PPL, teacher-KL, and next-token-accuracy evaluation.

## Staged compile and quality checkpoints

The compiler is resumed to inclusive end-layer 7, 15, 23, and 31. At each
stage, the full 32-layer model is evaluated with the compiled prefix replaced
by NEC and the uncompiled suffix left dense. These are cumulative diagnostics,
not final adaptive selections. Each milestone reports student and teacher PPL,
teacher KL, next-token accuracy, dense top-1 agreement, and storage preflight.

The staging runner stops cleanly at each end-layer so it does not signal the
compiler while a checkpoint may be written:

```bash
python scripts/run_nec_full32_milestone_stages.py \
  --checkpoint-dir /workspace/runs/nec-streaming-qualification/layer0/reference-post-sweep-q4-48cg \
  --output-dir /workspace/runs/nec-adaptive-layer-allocation-32layer_a40_20261022/milestones \
  --log-dir /workspace/runs/nec-streaming-qualification/logs
```

The allocator runs after all 32 checkpoints are available:

```bash
python scripts/search_nec_layer_allocation.py \
  --checkpoint-dir /workspace/runs/nec-streaming-qualification/layer0/reference-post-sweep-q4-48cg \
  --output /workspace/runs/nec-adaptive-layer-allocation-32layer_a40_20261022/final_allocation \
  --optimize-storage --max-dense-layers 4 --max-ppl 10
```

It profiles one-layer NEC and INT8 substitutions, builds separate and combined
Fisher interaction estimates, searches NEC masks at each allowed compression
count, then refines a bounded shortlist of mixed NEC/INT8/FP16 allocations with
full-model inference. Refinement and held-out data remain separate from the
single-layer proxy sample.

## Storage accounting

The preflight in [`storage_preflight.md`](storage_preflight.md) is an estimate
from the first four NEC artifact sizes. It shows that the current representation
cannot approach 10× whole-model compression: even the optimistic 32-layer
estimate is about 4.87× before rebuilt-package overhead.

The final allocation run writes a custom mixed package under
`final_allocation/deployable_model/`. It removes replaced projection and norm
tensors from the original safetensors shards, retains the rest of the model and
metadata, adds the selected NEC and INT8 artifacts, includes the loader and its
NEC runtime code, and reports the byte sum of every package file. The
pre-refinement search uses exact serialized layer-artifact sizes; its package
ratio is a preflight bound. The final report's measured package byte count is
the result used for the whole-model ratio.

The loader is custom because NEC layers are reconstructed at load time and
INT8 dense layers are dequantized at load time. FP16 is preserved for the small
normalization vectors in INT8-retained layers. A result below the requested
quality threshold or the 10× aspiration is reported explicitly; 3× is not
treated as a success target.
