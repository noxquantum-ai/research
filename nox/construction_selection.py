"""Model-level acceptance for bounded, gradient-free NEC construction search."""
from __future__ import annotations
import math


def construction_improves(candidate, current, *, minimum_kl_gain=1e-6):
    """Require better teacher KL with no NLL/accuracy/agreement regression.

    Both scores must cover exactly the same token identities, checked by the
    caller. Integer prediction counts avoid rounding accuracy percentages.
    """
    if candidate['scored_tokens'] != current['scored_tokens']:
        raise ValueError('construction scores use different token counts')
    keys = ('mean_nll', 'teacher_kl_nats_per_token')
    if not all(math.isfinite(float(s[k])) for s in (candidate, current) for k in keys):
        return False
    return (
        candidate['teacher_kl_nats_per_token'] < current['teacher_kl_nats_per_token'] - minimum_kl_gain
        and candidate['mean_nll'] <= current['mean_nll']
        and candidate['next_token_correct'] >= current['next_token_correct']
        and candidate['dense_agreement_correct'] >= current['dense_agreement_correct']
    )
