# NEC bridge-quantization CPU control (2026-10-21)

This directory records a deterministic, ill-conditioned `16 x 16` numerical
control. It is not a Llama quality measurement and cannot support claims about
perplexity, MMLU, or real-model memory use.

The experiment compares the direct-SVD and streaming Gram-statistics compilers
with identical inputs, ranks, sparse budget, two sweeps, and seeds. The two new
bridge controls use post-sweep MPO Q4 and 48 CG iterations:

- `fp16_control` stores and uses the bridge factors at FP16 rather than Q4.
- `operator_refit_q4` alternates closed-form factor refits and Q4 rounding,
  retaining independent factorwise Q4 whenever no candidate lowers the exact
  Frobenius error of the represented bridge operator.

## Result

| Compiler control | Held-out direct/streaming disagreement |
|---|---:|
| Unquantized, 6 CG | 0.00192% |
| Historical in-sweep Q4, 6 CG | 3.168% |
| Independent post-sweep Q4, 48 CG | **0.3608%** |
| FP16 bridge, post-sweep Q4 MPO, 48 CG | 3.951% |
| Operator-refit bridge Q4, post-sweep Q4 MPO, 48 CG | 2.423% |

On this fixture, neither new bridge control improves cross-backend functional
parity over independent post-sweep Q4. This is a negative result for these
specific controls, not proof that the controls reduce compression fidelity:
the operator-refit objective measures each quantized bridge against its own
unquantized bridge, whereas the table measures agreement between two compilers
whose unquantized factors are slightly different. The FP16 and operator-refit
controls also happen to lower the reference compiler's held-out reconstruction
error on this synthetic fixture, but that observation is not evidence of
real-model improvement.

The next paid experiment, if run, should be the bridge-only real-activation
diagnostic in `benchmarks/diagnose_real_bridge.py`. It avoids TT-SVD and DMRG
and tests both controls on the already established layer-0 Llama-2-7B
`down_proj` setup before any full projection or layer run is authorized.

The full machine-readable report is [`diagnostic.json`](diagnostic.json). Its
recorded source commit is `d492137fe145b7f86498a09439d696ed60a227cc`; the
working tree was clean when the diagnostic ran.
