---
license: llama2
language:
- en
base_model: NousResearch/Llama-2-7b-hf
library_name: pytorch
pipeline_tag: text-generation
inference: false
tags:
- nec
- nox-quantum
- quantum-inspired
- tensor-network
- structural-compression
- training-free
- custom-code
---

# NEC Llama 2 7B Hybrid

Research checkpoint produced by the Nox Entanglement Compiler (NEC).
The complete original 32-layer Llama-2-7B architecture is retained. All seven
attention/MLP projections in every layer use a Schmidt MPO backbone plus
sparse residual amplitudes. Embeddings, output head and normalization tensors
remain FP16. Construction uses no gradient-based fine-tuning or healing.

Source model: `NousResearch/Llama-2-7b-hf`, revision
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d`. Llama 2 was developed by Meta.

## Measured results

| Metric | Dense source | NEC hybrid |
| --- | ---: | ---: |
| Serialized model | 13.479 GB | approximately 2.490 GB |
| Storage compression | 1x | approximately 5.413x |
| Stored weight/factor coefficients, including scales and all branches | 6,738,417,664 | 3,323,032,892 |
| Structural coefficient reduction | 1x | 2.028x |
| Held-out perplexity | 4.867285 | 8.348960 |
| Teacher KL, nats/token | 0 | 0.543950 |
| Next-token accuracy | 62.820% | 53.999% |
| Dense top-1 agreement | 100% | 71.026% |
| Dense transformer layers retained | 32 | 0 |

Quality is measured on 308,700 scored WikiText-2 raw test
tokens after token offset 32,768, with context 2048 and
stride 512. Calibration uses 32768 training tokens in 2048-token windows.
This checkpoint meets the declared PPL <=10, storage >=5x and coefficient
reduction >=2x targets together. It still has a substantial quality gap from
the dense teacher. Dense-equivalent task accuracy and 10x compression are
not demonstrated. The example completions are qualitative checks, not a
downstream benchmark score.

The gzip containers are lossless and preserve the original NEC checkpoint
bytes exactly. They improve serialized storage only; coefficient counts and
model predictions are unchanged. Package measurements include retained
weights, compressed artifacts, masks, scales, loader code and metadata.
Integer coordinate metadata and deterministic padding are excluded from
coefficient counts but included in storage bytes.

## Load the complete model

Use the bundled loader to reconstruct all NEC projections. Loading only the
retained safetensors with the ordinary Transformers constructor leaves the
NEC projections uninitialized.

```python
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import sys

path = snapshot_download("Fluxx08/NEC-Llama-2-7B-Hybrid")  # authenticate for a private repository
sys.path.insert(0, path)
from load_nec_package import load_model

tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = load_model(device_map="auto")
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Tested runtime versions are in `requirements.txt`. The loader reconstructs
dense inference matrices. These storage savings do not imply equivalent
runtime VRAM or latency reductions. The generic Hub safetensors parameter
counter may count only the retained tensors; the correct complete stored
coefficient count includes the NEC files and is **3,323,032,892**.

## Reproducibility and scope

The bundled decoder passed a separate-interpreter test with bitwise-equal
logits against the evaluated checkpoint assembly. Each gzip file has a
verified source/round-trip SHA-256 pair in `nec_allocation_manifest.json`.
Raw held-out metrics are in `nec_quality_evaluation.json`; component counts
and qualitative examples, where present, are in `verification/`.

NEC is explicitly a tensor-network/sparsity hybrid. Conventional sparsity
and packing are not claimed as novel quantum methods. Component ablations
zero the main MPO cores without reoptimizing the remaining representation;
they are contribution checks, not matched reoptimized sparse baselines or
evidence of quantum advantage. Generalization to other base models has not
yet been demonstrated by this experiment.

Compiler source and experiment reports:
https://github.com/noxquantum-ai/research/tree/main/results/nec_sparse_excitation_a40_20261023

Base-model terms and attribution are included in `LICENSE.txt` and `NOTICE`.
