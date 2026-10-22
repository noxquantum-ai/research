# NEC adaptive layer allocation: first prototype

## Current objective and experiment scope

The deployment objective is to maximize the **measured whole-model serialized
compression ratio**, subject to an independently evaluated quality limit
(currently PPL `<= 10`, with KL reported as a complementary metric). The
long-term target remains approximately **10x**. A 3x ratio is an insufficient
result, not a target or a successful operating point. The 25% maximum dense
weight fraction is a ceiling only; the allocator must not try to use up that
allowance. It may retain fewer dense weights when quality permits.

The ratio denominator is the serialized dense model snapshot. The numerator
must be the byte sum of the **complete deployable mixed-model package**:
dense weights that remain, serialized NEC artifacts, model/config/tokenizer
metadata, rewritten weight index and shard structures, and any required
loader/runtime files or auxiliary parameters. Coefficient counts, candidate
region savings, and simulated low-bit storage are not whole-model compression
measurements. A result may claim a ratio only after the complete package is
materialized and measured.

The 16-layer experiment is a **16-candidate allocation and quality study**,
following the same structure as the four-candidate proof of concept. It is
not a claim that the entire 32-layer model has been compressed. The remaining
model layers stay dense for that study. We will report the resulting whole-
model ratio as a separate storage fact, even though this candidate scope
cannot meet the 3x or 10x deployment target. That does not cancel or invalidate
the 16-layer allocation experiment: its purpose is to test layer selection,
quality loss, and search efficiency at a larger candidate count.

Before any expensive compile or GPU evaluation, run
[`scripts/preflight_nec_compression_targets.py`](../../scripts/preflight_nec_compression_targets.py)
to check storage-only lower bounds. It compares a dense model snapshot with
the tensors left dense and the actual serialized checkpoint files available.
When candidate checkpoints are missing, it shows estimates based on the
observed artifact sizes. Its output is a feasibility preflight, not a measured
deployable ratio: generated safetensors headers/index, package loader/runtime
files, and other format overhead are omitted. A target that fails even this
lower-bound check is infeasible for that allocation scope. A target that
passes still requires building and measuring the real package.

The current search prototype has a fixed compressed-layer count and selects
for quality; it does not yet optimize complete-package bytes jointly with
per-layer artifact size. Until variable-count search and package measurement
are implemented, report its allocation result as a research result rather
than claiming it meets the deployment objective. If no measured allocation
meets both quality and compression constraints, state that it is infeasible;
do not relax the compression target silently.

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

The planned `--max-ppl P --optimize-storage` mode must maximize the measured
whole-model ratio subject to independently measured quality and the maximum
dense-weight ceiling. This variable-count mode is separate from the current
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

After subtracting the two checkpoint files from the four candidate layers'
dense projection and norm payload, the candidate-region estimate is
669,619,324 bytes smaller (41.4% of that region); the remaining
28 model layers stay dense. This is a candidate-region accounting result, not
a byte count of a complete deployable model package and not a whole-model
compression ratio. The run used five configurations for the 1,023-token
singleton probe, three for refinement (dense reference plus two shortlisted
pairs), then dense and selected-student passes for held-out evaluation. It did
not run the other four mixed pairs. Total elapsed wall time was about 95
seconds, including model loading; the measured proxy/search and refinement
stages took about 22 seconds. These timings are specific to the A40 and this
implementation.

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

For a 16-candidate study, report the full measured quality/storage frontier
for the tested compression counts: selected candidate IDs, candidate
checkpoints actually used, PPL, teacher KL, token count/split, dense-layer
count, and candidate-region bytes. Also report the complete-model serialized
ratio only if a deployable mixed-model package was materialized and measured;
otherwise show the preflight lower bound with that label. Do not describe
failure to reach 3x or 10x in this restricted study as success, and do not
cancel the allocation study because its candidate scope cannot reach those
deployment targets.

The prototype is in [`scripts/search_nec_layer_allocation.py`](../../scripts/search_nec_layer_allocation.py),
with the Fisher interaction and conditional search in
[`nox/layer_allocator.py`](../../nox/layer_allocator.py).
