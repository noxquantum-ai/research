# Layer-0 NEC operator-Schmidt diagnostic

Status: initial GPU diagnostic complete. No prefix compilation, perplexity run,
MMLU evaluation, or full-model run was started. The saved NEC checkpoint was
read-only; candidate factors were constructed in memory and discarded after
measurement. Metrics and byte counts were written to diagnostic JSON reports.
No gradient fitting or post-compression healing was used.

## Environment and inputs

- SSH endpoint: `shadeform@129.213.125.12`; connected host `shadecloud`
  (private address `10.19.89.48`). The A10 was idle before and after the
  experiment: 0/23,028 MiB used after completion.
- Remote checkout: `main`, commit `786409e3ead2432ffc13c425fccd57a2ceb3e67a`.
- Runtime: Python 3.10, PyTorch 2.7.0 with CUDA available, Transformers 5.20.1,
  Hugging Face Hub 2.2.0, safetensors 0.8.0, PyArrow 25.0.1; NVIDIA driver
  570.148.08 / CUDA 12.8.
- Existing source checkpoint:
  `compressed_models/tiers/high_fidelity_10x/compressed_layer_0.pt`,
  65,739,309 bytes, SHA-256
  `81d44f6b676853cbd218f0c2b22bd0a388eed7195b88153045f5c00ead2de6e9`.
  It was copied unchanged to the instance; no NEC layer was recompiled.
- The new instance had no prior research checkout or activation cache. Dense
  layer-0 tensors were fetched from `NousResearch/Llama-2-7b-hf` at revision
  `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`. Layer-0 inputs were regenerated
  from the pinned tokenizer and canonical raw WikiText-2 train/validation
  splits: 4,096 tokens per split, 128-token windows, uniform sampling, seeds
  42/43. The raw split SHA-256 values are recorded in
  `diagnostic_data_manifest.json`. These are fresh matched inputs, not the
  missing historical calibration tensors.
- The prep stage did not load or run the full 7B transformer. It fetched the
  layer-0 projection tensors and embedding table, then formed layer-0 inputs
  by embedding the sampled tokens.

The CPU synthetic suite on the A10 checkout passed (7/7 operator-Schmidt
tests). The full CPU suite was not rerun on that clean host; the local suite
had previously passed 83 tests with 3 CUDA-only skips.

## Results

All error values below are relative Frobenius errors on the same 4,096 held-out
WikiText activation tokens. Restoration rows replace selected saved NEC
projections with the corresponding dense projections; they are diagnostic
upper bounds, not candidate compression results.

| Layer-0 block variant | Active-transform error | Full-block output error |
| --- | ---: | ---: |
| Saved NEC baseline | 0.43084 | 0.36374 |
| Restore all attention projections | 0.40633 | 0.34305 |
| Restore all MLP projections | 0.22219 | 0.18759 |
| Restore Gate only | 0.42111 | 0.35553 |
| Restore Up only | 0.41357 | 0.34916 |
| Restore Down only | 0.43132 | 0.36415 |
| Restore Gate + Up | 0.40747 | 0.34402 |
| Operator-Schmidt Gate + Up | 0.41252 | 0.34828 |

The MLP group is the dominant layer-0 failure source: restoring all three MLP
projections cuts block error by about 48% relative to the saved baseline,
whereas restoring attention cuts it by about 6%. No single MLP projection
restoration explains the group result. Gate + Up together still recover only a
small part, pointing to the nonlinear Gate/Up/Down interaction rather than a
single isolated matrix.

The targeted operator-Schmidt candidate used only Gate and Up. It reduced the
individual held-out projection errors substantially, but had only a modest
block effect:

| Projection | Saved NEC full error | New sparse + bridge | New full candidate | Improvement from new MPO core | Old/new core coefficients | Old/new serialized artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate | 0.63772 | 0.42916 | 0.42731 | 0.00185 | 812,544 / 808,064 | 10,741,513 / 10,728,593 |
| Up | 0.75046 | 0.51439 | 0.50911 | 0.00528 | 812,544 / 808,064 | 10,736,521 / 10,717,457 |

Both layouts selected `balanced_direct`; the head-aware/crossed/SwiGLU-group
alternatives did not win this screen. The new core contributes a small positive
held-out change beyond sparse + bridge (0.00185 and 0.00528 absolute relative
error), but most of the projection-level improvement is already present in
the newly constructed sparse + bridge components. In the saved NEC artifacts,
the MPO core instead worsened held-out error relative to sparse + bridge alone
for Gate (0.63772 vs. 0.63305), Up (0.75046 vs. 0.74329), and Down (0.61362 vs.
0.58159).

Replacing only Gate and Up yielded a serialized layer checkpoint of 65,705,580
bytes versus 65,739,309 bytes for the saved layer (33,729 bytes smaller, about
0.05%). The new candidate artifacts totaled 21,446,050 serialized bytes. This
is not a meaningful layer-size win and says nothing about complete-model size.

The candidate diagnostic took 17.83 seconds total (7.27 seconds for Gate and
7.20 seconds for Up candidate construction); peak allocated GPU memory was
8,405,942,272 bytes (about 7.83 GiB). Group-only, MLP-individual, and Gate/Up
pair restoration diagnostics took 0.65, 3.72, and 1.03 seconds respectively.

## Interpretation and stop decision

The redesigned representation improved held-out error for both selected
matrices at almost unchanged coefficient and serialized-byte budgets. That is
a real projection-level signal. However, the MPO's incremental benefit over
the newly built sparse + bridge channel is small, the selected topology was the
plain balanced layout, and the Gate/Up candidate improves full-block error by
only 4.25% relative (0.36374 to 0.34828). The much larger recovery from
restoring the full MLP group shows that independent per-projection MPOs are
not yet addressing the dominant nonlinear interaction.

Recommendation: stop GPU scaling here. Do not launch layer-1/2/4 compilation
or perplexity/MMLU runs on this result. If further work is approved, first
design a CPU-testable joint SwiGLU representation/fitting experiment that
models the Gate/Up/Down product together, then return to a small GPU block
comparison. These results do not establish downstream capability retention.

## Reproduction

The new instance was initialized from `main`; the current SSH private key was
used locally and was not copied to or committed on the instance. On the remote
host, the checkout/environment setup was:

```bash
git clone --branch main --single-branch https://github.com/noxquantum-ai/research.git \
  /home/shadeform/nec-crosslayer-research
cd /home/shadeform/nec-crosslayer-research
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

From the local workspace, transfer only the saved layer-0 checkpoint and the
two diagnostic scripts (the private key is never transferred):

```bash
scp -i A10-mvc1exyr_key.pem \
  compressed_models/tiers/high_fidelity_10x/compressed_layer_0.pt \
  shadeform@129.213.125.12:/home/shadeform/nec-crosslayer-research/checkpoints/diagnostic_36h/operator_schmidt/layer0_initial/
scp -i A10-mvc1exyr_key.pem \
  scripts/diagnose_nec_components.py scripts/prepare_operator_schmidt_layer0.py \
  shadeform@129.213.125.12:/home/shadeform/nec-crosslayer-research/scripts/
```

On the A10, from the repository root, prepare the pinned data and weights:

```bash
cd /home/shadeform/nec-crosslayer-research
CUDA_VISIBLE_DEVICES="" .venv/bin/python -u scripts/prepare_operator_schmidt_layer0.py \
  --output-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache \
  --n-calib 4096 --n-validation 4096 --seq-len 128 \
  --calibration-seed 42 --validation-seed 43
```

Still on the A10, run the coarse group screen, followed by MLP-only projection
diagnosis, the targeted Gate/Up restoration check, and finally the two
candidate fits:

```bash
.venv/bin/python -u scripts/diagnose_nec_components.py \
  --checkpoint-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache --layer 0 --device cuda \
  --seq-len 128 --n-heads 32 --n-kv-heads 32 \
  --analysis-only --skip-projection-components \
  --output experiment_artifacts/diagnostic_36h/operator_schmidt/layer0/layer0_group_restore.json

.venv/bin/python -u scripts/diagnose_nec_components.py \
  --checkpoint-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache --layer 0 --device cuda \
  --seq-len 128 --n-heads 32 --n-kv-heads 32 \
  --analysis-only --projections gate_proj,up_proj,down_proj \
  --restore-individual-projections \
  --output experiment_artifacts/diagnostic_36h/operator_schmidt/layer0/layer0_mlp_individual.json

.venv/bin/python -u scripts/diagnose_nec_components.py \
  --checkpoint-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache --layer 0 --device cuda \
  --seq-len 128 --n-heads 32 --n-kv-heads 32 \
  --analysis-only --skip-projection-components --projections gate_proj,up_proj \
  --restore-individual-projections --restore-projection-set gate_proj,up_proj \
  --output experiment_artifacts/diagnostic_36h/operator_schmidt/layer0/layer0_gate_up_pair.json

.venv/bin/python -u scripts/diagnose_nec_components.py \
  --checkpoint-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache --layer 0 --device cuda \
  --seq-len 128 --n-heads 32 --n-kv-heads 32 \
  --projections gate_proj,up_proj --restore-individual-projections \
  --restore-projection-set gate_proj,up_proj \
  --output experiment_artifacts/diagnostic_36h/operator_schmidt/layer0/layer0_gate_up_operator_schmidt.json
```

The raw JSON results and data/token checksums are kept alongside this report.
