import unittest

import torch
import torch.nn.functional as F

from scripts.probe_nec_swiglu_ordering import contribution_gram, permute_swiglu
from scripts.probe_nec_mlp_bases import hadamard


class SwiGLUOrderingTests(unittest.TestCase):
    def test_signed_hadamard_preserves_the_linear_function(self):
        torch.manual_seed(241)
        weight=torch.randn(12,16,dtype=torch.float64)
        inputs=torch.randn(16,23,dtype=torch.float64)
        signs=(torch.randint(0,2,(16,))*2-1).double()
        transformed_weight=hadamard(weight*signs,block=8)
        transformed_inputs=hadamard((inputs*signs[:,None]).T,block=8).T
        torch.testing.assert_close(transformed_weight@transformed_inputs,weight@inputs,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(hadamard(hadamard(weight,block=8),block=8),weight,rtol=1e-12,atol=1e-12)

    def test_shared_permutation_preserves_the_entire_nonlinear_function(self):
        torch.manual_seed(233)
        weights = {'gate_proj': torch.randn(12, 8, dtype=torch.float64),
                   'up_proj': torch.randn(12, 8, dtype=torch.float64),
                   'down_proj': torch.randn(8, 12, dtype=torch.float64)}
        x = torch.randn(8, 17, dtype=torch.float64)
        changed = permute_swiglu(weights, torch.randperm(12))
        evaluate = lambda w: w['down_proj'] @ (F.silu(w['gate_proj'] @ x) * (w['up_proj'] @ x))
        torch.testing.assert_close(evaluate(weights), evaluate(changed), rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            permute_swiglu(weights, torch.zeros(12, dtype=torch.long))

    def test_contribution_gram_matches_explicit_vector_functions(self):
        torch.manual_seed(239)
        active, down = torch.randn(12, 17, dtype=torch.float64), torch.randn(8, 12, dtype=torch.float64)
        functions = (down.T[:, :, None] * active[:, None, :]).reshape(12, -1)
        torch.testing.assert_close(contribution_gram(active, down), functions @ functions.T / 17)


if __name__ == '__main__':
    unittest.main()
