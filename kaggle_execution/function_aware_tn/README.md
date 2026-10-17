# Function-aware TN Stage A/B Kaggle batch

This private script runs the bounded pinned Llama-2-7B module diagnostics on one Kaggle T4. It streams one transformer layer at a time, captures only layers 0, 15, and 31, checkpoints their CPU inputs for resumption, and writes compact machine-readable output under `/kaggle/working/function_aware_tn/`.

The configured cap is 21,600 seconds (six hours), the measured GPU memory ceiling is recorded in `results.json`, and the preflight estimates 14 GB peak GPU allocation plus at most 1.1 GB of temporary layer-input checkpoints. The final outputs target 100 MB. The private kernel has internet enabled for the pinned public artifacts and the exact model revision; it reads an optional Kaggle secret named `HF_TOKEN` without printing or saving its value. If access fails, the JSON output records the exact blocker and does not substitute a model or dataset.

The driver reuses the Kaggle image's Python packages when they meet the minimum versions listed in `requirements.txt`; if a dependency is missing or too old, it installs those pinned packages. PyTorch is provided by the Kaggle image and is not replaced. Exact versions are saved in `results.json`. After pushing, inspect `kaggle kernels status toheebogunade/function-aware-tn-stage-a-and-b-diagnostics`, retrieve artifacts with `kaggle kernels output`, and preserve the verified output locally before starting another GPU batch.

For a resumed version, attach the prior private kernel output as a Kaggle input or set `NOX_RESUME_DIR` to its extracted `function_aware_tn` output directory. The driver checks model and dataset revisions, sampled-token hashes, code-bundle digest, and checkpoint shapes before loading saved layer inputs; it also restores completed Stage A records and completed per-calibration Stage B records. Partial layer/calibration progress is written atomically to `results.json`.

A second batch is reserved for 1/2/4-layer prefix evaluation only if the full Stage A and Stage B gates pass. Full-model compilation and benchmark evaluation are not part of this batch.
