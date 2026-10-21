# Deterministic NEC MPO gauge/Q4 diagnostic (2026-10-21)

This CPU experiment holds one four-site MPO operator fixed, applies exactly
equivalent internal-bond transformations, and independently Q4-quantizes the
resulting cores. It is a numerical sensitivity test, not a Llama projection or
model-quality benchmark.

All candidate gauges reproduce the unquantized operator to relative error at
most `4.28e-7`. After Q4, their held-out output errors range from **15.83% to
96.53%**. Thus independent factorwise Q4 is strongly gauge-dependent even when
the pre-quantization operator is unchanged.

Quantizing one core at a time in the identity gauge gives:

| Q4 component | Held-out output error |
|---|---:|
| Core 0 only | 13.06% |
| Core 1 only | 9.45% |
| Core 2 only | 6.94% |
| Core 3 only | 6.29% |
| All four cores | 16.94% |

The calibration-selected candidate (`orthogonal_0`) lowers calibration error
from 15.87% to 15.57%, but worsens held-out error from 16.94% to 17.85%.
This is a small deterministic counterexample to the assumption that searching
gauges on a limited calibration set necessarily improves unseen activations.
No candidate has been integrated into production NEC.

The report records seed `20261021`, ranks `[1,4,6,4,1]`, all candidate gauges,
per-core and cumulative Q4 errors, and clean source provenance at commit
`e61a164b305a8012bc5af5a3fe66cdd8adaaa6eb`.
