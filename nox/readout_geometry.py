"""Training-only teacher-distribution geometry for final-block construction."""
from __future__ import annotations

import torch


@torch.no_grad()
def teacher_readout_fisher_precision(hidden, norm_weight, readout, *, fit_tokens,
                                     rms_eps=1e-6, samples_per_token=4,
                                     block_tokens=256, shrinkage=.01, seed=271828):
    """Estimate the final residual-stream KL curvature without autograd.

    For a teacher token distribution p, sample y~p and form W[y]-E_p[W].
    Apply the exact analytical RMSNorm adjoint to this score. Its outer product
    is an unbiased sample of the teacher KL Hessian at this token. Averaging
    across fit tokens gives a constant local metric, not an exact replacement
    for token-specific KL. Shrinkage ensures positive definiteness. Neither
    this metric nor its samples are stored in the compressed representation.
    """
    if not 0 < fit_tokens <= hidden.shape[1]:
        raise ValueError('invalid fit-token count')
    if samples_per_token < 1 or block_tokens < 1 or not 0 < shrinkage <= 1:
        raise ValueError('invalid Fisher construction controls')
    if readout.shape[1] != hidden.shape[0] or norm_weight.shape != (hidden.shape[0],):
        raise ValueError('readout, norm and hidden dimensions differ')
    device=hidden.device;dimension=hidden.shape[0]
    weight=readout.float();gamma=norm_weight.float()
    metric=torch.zeros(dimension,dimension,device=device)
    generator=torch.Generator(device=device).manual_seed(seed)
    for first in range(0,fit_tokens,block_tokens):
        h=hidden[:,first:min(first+block_tokens,fit_tokens)].float()
        inverse_rms=(h.square().mean(0)+rms_eps).rsqrt()
        probabilities=(weight@((h*inverse_rms)*gamma[:,None])).softmax(0)
        mean=weight.T@probabilities
        labels=torch.multinomial(probabilities.T,samples_per_token,replacement=True,generator=generator)
        score=weight[labels].permute(2,0,1)-mean[:,:,None]
        scaled=score*gamma[:,None,None]
        # J_RMSNorm(h)^T score, including the radial projection and epsilon.
        projected=scaled*inverse_rms[None,:,None]-h[:,:,None]*(h[:,:,None]*scaled).sum(0)[None,:,:]*(inverse_rms.pow(3)/dimension)[None,:,None]
        projected=projected.reshape(dimension,-1)
        metric.addmm_(projected,projected.T)
    metric.div_(fit_tokens*samples_per_token)
    average=metric.diagonal().mean().clamp_min(1e-20)
    metric.div_(average).mul_(1-shrinkage)
    metric.diagonal().add_(shrinkage)
    return metric,{'method':'sampled_teacher_readout_fisher','fit_tokens':fit_tokens,
        'samples_per_token':samples_per_token,'seed':seed,'shrinkage':shrinkage,
        'guard_used':False,'uses_autograd':False,'stored_coefficients_added':0,
        'approximates':'token-averaged local teacher KL curvature at final residual stream'}
