# Coupled NEC function construction: development results

The acceptance gate remains full-32-layer PPL <= 10, actual whole-package
compression >= 5x and structural coefficient reduction >= 2x together, with
no gradient-based healing. **These eight-layer results do not meet that gate.**

The paired experiment compresses all seven projections in layers 0-7 and
leaves layers 8-31 dense for evaluation through the original model. It uses
32768 training tokens in 2048-token windows for construction. The new solver
fits Up and Down MPO cores through their exact linear effect on the nonlinear
MLP output; Gate proposals reconstruct teacher preactivations linearly.
Packed candidates must improve both construction and a calibration guard.
There is no nonlinear derivative, autograd or added inference coefficient.

| Matched configuration | Validation PPL | Teacher KL (nats/token) | Next-token accuracy | Dense top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| Dense teacher | 4.083 | 0 | 66.37% | 100% |
| Packed Schmidt initialization | 48.841 | 2.4828 | 36.24% | 43.71% |
| Coupled function construction | 26.828 | 1.8835 | 41.71% | 51.09% |
| Sequential asymmetric coupled construction | 16.061 | 1.3630 | 46.96% | 58.14% |

Both candidates use the same ranks, tensor coefficient counts and tensor
payload sizes. Five-bit cores use the available bytes under the simultaneous
5x/2x full-run planning limits. Those full-run limits are projected budgets,
not actual whole-model compression achieved by an eight-layer assembly.
The raw record includes the actual evaluated assembly bounds separately.

Evaluation scores 4095 validation tokens with 2048-token context and stride
512. These tokens are outside training construction data. Repeated development
on this slice means it is not an untouched final confirmation set. A successful
candidate must subsequently pass a separately held-out full-model evaluation.

The capture manifest and paired raw measurements are in
[paired_8layer.json](paired_8layer.json). Construction code and mathematical
limitations are described in
[the redesign note](../../docs/experiments/NEC_ACCURACY_REDESIGN_20261023.md).
The sequential experiment uses the actual reconstructed compressed-prefix
activations and clean teacher block-output targets. Its raw record is in
[sequential_and_representation_probes.json](sequential_and_representation_probes.json).
Replaying saved prefix layers is necessary on resume; simply skipping their
construction would reset downstream calibration onto an incorrect trajectory.

Two separate layer-3 probes used 8192 validation activation tokens. Protected
nonlinear feature sectors reduced MLP relative error from 0.7445 to 0.7002
and block-update error from 0.6959 to 0.6598. Coupled Value/Output construction
alone reduced attention error only from 0.4981 to 0.4963. These are local
measurements, not end-to-end quality claims or proof the representation scales.
