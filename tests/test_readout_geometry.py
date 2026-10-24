import unittest
import torch
from nox.readout_geometry import teacher_readout_fisher_precision

class ReadoutGeometryTests(unittest.TestCase):
    def test_sampled_metric_matches_exact_softmax_rms_curvature(self):
        torch.manual_seed(17)
        h=torch.randn(4,1).repeat(1,20000);gamma=torch.rand(4)+.5;w=torch.randn(7,4)
        metric,report=teacher_readout_fisher_precision(h,gamma,w,fit_tokens=h.shape[1],
            samples_per_token=8,block_tokens=512,shrinkage=.01)
        x=h[:,0];r=(x.square().mean()+1e-6).rsqrt()
        j=torch.diag(gamma)@(r*torch.eye(4)-r**3*torch.outer(x,x)/4)
        p=(w@(gamma*x*r)).softmax(0)
        exact=j.T@w.T@(torch.diag(p)-torch.outer(p,p))@w@j
        exact=exact/exact.diagonal().mean()*.99+.01*torch.eye(4)
        torch.testing.assert_close(metric,exact,atol=.025,rtol=.025)
        self.assertGreater(float(torch.linalg.eigvalsh(metric).min()),0)
        self.assertFalse(report['guard_used'])
        self.assertEqual(report['stored_coefficients_added'],0)

    def test_guard_values_do_not_change_metric(self):
        torch.manual_seed(19)
        h=torch.randn(4,24);gamma=torch.ones(4);w=torch.randn(7,4)
        a,_=teacher_readout_fisher_precision(h,gamma,w,fit_tokens=16)
        h[:,16:]*=1000
        b,_=teacher_readout_fisher_precision(h,gamma,w,fit_tokens=16)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
