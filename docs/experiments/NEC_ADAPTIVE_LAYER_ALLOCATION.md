# NEC adaptive layer allocation: first prototype

## Decision problem

For compiled candidate layers `C`, let `S` be the subset whose dense operators
are replaced by NEC checkpoints. The allocator must choose the complete mask,
not rank layers independently. For an exact dense-layer count `K`, it chooses
`|S| = number_of_model_layers - K`; if checkpoints exist for only some layers,
the request is infeasible unless the available candidates can satisfy that
count. Boundary protection is an optional constraint, not a default.

The proxy ranks allocations by estimated teacher KL, using predicted PPL as a
feasibility constraint. The bounded exact-refinement stage then selects among
the shortlisted candidates by measured PPL under the requested limit, with KL
as a tie-breaker. If none meets the limit, it returns the lowest-PPL
shortlisted candidate and marks it infeasible. On a separate held-out split,
the exact measured PPL, KL, next-token accuracy, checkpoint bytes, and dense
top-1 agreement determine whether the allocation passed. Proxy estimates are
never reported as measured model results.

For `--max-ppl P --optimize-storage`, the deployment objective is the maximum
actual checkpoint storage saved subject to the independently measured quality
constraint. That variable-count mode is deliberately separate from this first
fixed-count prototype; it must search a measured Pareto frontier and cannot
infer that adding a compressed layer always worsens quality.

## Cheap interaction signal

On a selection sample, the prototype measures the dense model's logits
`z_0(x)` and one full-model forward for each individual NEC replacement:

```text
delta_l(x) = z_{ {l} }(x) - z_0(x)
```

It does not forward any multi-layer allocation during proxy search. Given
dense probabilities `p = softmax(z_0)`, it builds the empirical Fisher
interaction matrix:

```text
F_ij = E_x[ delta_i^T (diag(p) - p p^T) delta_j ]
a_i  = E_(x,y)[ p^T delta_i - delta_i[y] ]
```

The proxy predicts:

```text
KL(dense || S)  ~= 0.5 * sum_(i,j in S) F_ij
NLL(S)          ~= NLL(dense) + sum_(i in S) a_i
                               + 0.5 * sum_(i,j in S) F_ij
```

The off-diagonal terms matter: a negative `F_ij` predicts that two single-
layer logit perturbations partly cancel, while a positive term predicts that
they reinforce each other. Conditional marginal costs are recomputed as
layers enter a partial allocation. For small candidate sets, the code can rank
all masks in this cheap proxy space. For larger sets it uses conditional beam
search and 1-for-1 swap refinement on the proxy. Neither path calls the full
model for each combination.

This signal is a second-order approximation in final-logit space. It captures
correlated and opposing logit errors, but it omits higher-order effects from
nonlinear hidden-state interactions between multiple compressed layers.
Therefore it proposes an allocation; it does not certify one. The chosen mask
must be evaluated with the actual model on untouched data. If the proxy does
not beat simple sensitivity baselines or repeatedly chooses poor masks, it
should be revised or discarded.

## Four-layer proof of concept

The prototype uses the saved checkpoints for layers 0–3. It requests two
compressed candidates (`--compress-count 2`), which means 30 dense layers in
the full 32-layer model. The four-layer pair ranking already published from
the 4,095-token validation slice is an answer key for retrospective analysis
only. The search program does not read that report or its scores.

The checkpoints were calibrated on WikiText-2 train tokens sampled at saved
token offsets. The proxy sample uses train tokens 4,096–5,120 and the exact
refinement sample uses 8,192–12,288; both ranges are checked against the
recorded calibration windows and each other. The final check uses WikiText-2
test, which is a separate dataset split. The default is 4,096 held-out input
tokens; `--heldout-tokens 0` requests the complete split.

The proxy-only run selected layers 0 and 1. That missed the known validation
winner, layers 0 and 3, and scored PPL 8.12 on the held-out test slice. The
proxy's first two candidates were layers 0+1 and 0+3, with nearly tied
estimated KL. The follow-up measured just those two combinations on the
separate refinement slice, out of six possible pairs. Measured PPLs were 8.59
for 0+1 and 6.89 for 0+3, so refinement selected 0+3. Its held-out test PPL
was 6.32, teacher KL was 0.259 nats/token, and dense top-1 agreement was
81.1%. On the prior 4,095-token validation answer key, 0+3 was rank 1 at PPL
5.079; the refined choice therefore had zero regret against that local answer
key. The proxy-only 0+1 choice was rank 4 at PPL 7.497, a 2.418 PPL regret.
Both held-out test scores use the same token slice.

The refined choice saves 669,619,324 serialized weight bytes across the four
candidate layers (41.4% of that region); the remaining 28 model layers stay
dense. The run used five configurations for the 1,023-token singleton probe,
three for refinement (dense reference plus two shortlisted pairs), then dense
and selected-student passes for held-out evaluation. It did not run the other
four mixed pairs. Total elapsed wall time was about 95 seconds, including
model loading; the measured proxy/search and refinement stages took about 22
seconds. These timings are specific to the A40 and this implementation.

The reports and raw logs are archived in
[`results/nec_adaptive_layer_allocation_poc_a40_20261022/`](../../results/nec_adaptive_layer_allocation_poc_a40_20261022/).

The run records:

- the selected pair and its rank against the previously measured 4,095-token
  validation answer key (calculated after selection, never fed to the search);
- proxy rank and predicted KL/PPL for each considered mask;
- dense plus singleton model-evaluation counts and GPU time;
- exact held-out PPL, teacher KL, next-token accuracy, and dense top-1 agreement;
- hashes for checkpoint artifacts, dataset source, and token IDs.

This is a proof of concept, not evidence that the proxy generalizes to 32
layers. A successful four-layer result means only that the proxy shortlist plus
exact refinement recovered a known local allocation without evaluating all
six pair models. The proxy alone did not recover it.
The next research gate is to freeze the proxy, then test on a larger candidate
set with an exhaustive or high-budget oracle used only for retrospective
ranking accuracy and quality regret.

## Interface and feasibility

The script accepts either `--compress-count N` among available checkpoints or
the model-wide `--keep-dense-layers K`. The latter is rejected when the
checkpoint set cannot leave exactly `K` model layers dense. For example, four
saved checkpoints cannot satisfy `--keep-dense-layers 8` on a 32-layer model;
all four candidates together still leave 28 layers dense. This prevents a
requested allocation from silently being treated as a smaller experiment.

The prototype is in [`scripts/search_nec_layer_allocation.py`](../../scripts/search_nec_layer_allocation.py),
with the Fisher interaction and conditional search in
[`nox/layer_allocator.py`](../../nox/layer_allocator.py).
