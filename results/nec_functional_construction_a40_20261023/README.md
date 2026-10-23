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
Sequential asymmetric construction and direct-sum nonlinear feature sectors
remain separate experiments; no unmeasured improvement is attributed to them.
