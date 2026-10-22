# NEC adaptive layer allocation: first prototype

## Decision problem

For compiled candidate layers `C`, let `S` be the subset whose dense operators
are replaced by NEC checkpoints. The allocator must choose the complete mask,
not rank layers independently. For an exact dense-layer count `K`, it chooses
`|S| = number_of_model_layers - K`; if checkpoints exist for only some layers,
the request is infeasible unless the available candidates can satisfy that
count. Boundary protection is an optional constraint, not a default.

The primary selection objective is dense-teacher KL on selection data, with a
PPL limit used as a feasibility constraint. If no predicted candidate meets
the PPL limit, the allocator reports that fact and returns the lowest-PPL
candidate for inspection. On a held-out split, the exact measured PPL, KL,
next-token accuracy, checkpoint bytes, and dense top-1 agreement determine
whether the allocation passed. The selection proxy is never reported as a
measured model result.

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

It does not forward any multi-layer allocation during selection. Given dense
probabilities `p = softmax(z_0)`, it builds the empirical Fisher interaction
matrix:

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

The first run uses the saved checkpoints for layers 0–3. It requests two
compressed candidates (`--compress-count 2`), which means 30 dense layers in
the full 32-layer model. The four-layer pair ranking already published from
the 4,095-token validation slice is an answer key for analysis only. The
search program does not read that report or its scores.

The checkpoints were calibrated on 4,096 WikiText-2 train tokens sampled at
saved token offsets. The default search sample uses WikiText-2 train tokens
4,096–8,192 and checks that this range does not overlap the recorded
calibration windows. The final check uses WikiText-2 test, which is a separate
dataset split. The initial prototype defaults to 4,096 held-out input tokens;
`--heldout-tokens 0` requests the complete split.

The run records:

- the selected pair and its rank against the previously measured 4,095-token
  validation answer key (reported after selection, never fed to the search);
- proxy rank and predicted KL/PPL for each considered mask;
- dense plus singleton model-evaluation counts and GPU time;
- exact held-out PPL, teacher KL, next-token accuracy, and dense top-1 agreement;
- hashes for checkpoint artifacts, dataset source, and token IDs.

This is a proof of concept, not evidence that the proxy generalizes to 32
layers. A successful four-layer result means only that the interaction proxy
recovered a known local allocation without evaluating all six pair models.
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
