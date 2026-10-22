import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints
from scripts.diagnose_prefix_failure import NORMS, PROJECTIONS, install, module, summarize, trace
from scripts.diagnose_prefix_mlp_interactions import mlp_error_terms


class PrefixFailureDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(314)
        self.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=5,
            num_attention_heads=2, num_key_value_heads=1, rms_norm_eps=1e-5,
            attention_dropout=0.0,
        )).float().eval()
        self.model.config.use_cache = False
        self.dense = {i: {n: module(self.model.model.layers[i], n).weight.detach().clone()
                          for n in (*PROJECTIONS, *NORMS)} for i in range(4)}
        self.compressed = {i: {n: w + (0.01 * (i + 1) if n in PROJECTIONS else 0)
                               for n, w in weights.items()} for i, weights in self.dense.items()}

    def test_variant_switch_restores_previous_layers_and_preserves_suffix(self):
        suffix = copy.deepcopy(self.model.model.layers[4].state_dict())
        install(self.model, self.dense, self.compressed, {i: PROJECTIONS for i in range(4)})
        install(self.model, self.dense, self.compressed, {2: PROJECTIONS})
        for i in range(4):
            for name in (*PROJECTIONS, *NORMS):
                expected = self.compressed if i == 2 and name in PROJECTIONS else self.dense
                torch.testing.assert_close(module(self.model.model.layers[i], name).weight, expected[i][name], rtol=0, atol=0)
        for key, value in suffix.items():
            torch.testing.assert_close(self.model.model.layers[4].state_dict()[key], value, rtol=0, atol=0)

    def test_four_layer_injection_matches_independent_install_and_multisequence_trace(self):
        oracle = copy.deepcopy(self.model)
        install(oracle, self.dense, self.compressed, {i: PROJECTIONS for i in range(4)})
        with tempfile.TemporaryDirectory() as directory:
            for i in range(4):
                artifacts = {n: {"cores": [w[None, :, :, None]], "d_awt": torch.ones(w.shape[1])}
                             for n, w in self.compressed[i].items() if n in PROJECTIONS}
                torch.save({"layer_idx": i, "compressed_artifacts": artifacts,
                            **{n: self.dense[i][n] for n in NORMS}},
                           Path(directory) / f"compressed_layer_{i}.pt")
            self.assertEqual(inject_nox_checkpoints(self.model, directory, device="cpu", expected_layers=list(range(4))), list(range(4)))
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        with torch.no_grad():
            torch.testing.assert_close(self.model(ids).logits, oracle(ids).logits, rtol=0, atol=0)
        traced, _ = trace(self.model, self.dense, self.compressed, ids)
        for value in traced.values():
            self.assertLess(value["custom_dense_vs_hf"], 2e-6)
            self.assertLess(value["custom_compressed_vs_hf"], 2e-6)
            # Subtraction error scales with the perturbed operator output; this
            # synthetic fixture deliberately permits very large perturbations.
            bound = 16 * torch.finfo(torch.float32).eps * max(1., value["total_error_relative_to_clean_output"])
            self.assertLess(value["decomposition_identity_residual"], bound)

    def test_energy_weighting_can_conceal_ordinary_token_error(self):
        reference = torch.tensor([[1000., 0.], [1., 0.], [1., 0.]])
        candidate = torch.tensor([[1000., 0.], [0., 1.], [0., 1.]])
        metrics = summarize(reference, candidate)
        self.assertLess(metrics["relative_error"], 0.003)
        self.assertGreater(metrics["token_relative_error_mean"], 0.9)
        self.assertGreater(metrics["largest_token_energy_fraction"], 0.999)

    def test_swiglu_decomposition_and_intervention_controls(self):
        x = torch.randn(16, 11)
        dense, changed = self.dense[0], self.compressed[0]
        result = mlp_error_terms(dense, changed, x)
        self.assertLess(result["output_decomposition_residual"], 2e-6)
        self.assertLess(result["product_decomposition_residual"], 2e-6)
        down_only = dict(dense, down_proj=changed["down_proj"])
        result = mlp_error_terms(dense, down_only, x)
        self.assertEqual(result["product_error_through_dense_down_relative"], 0)
        self.assertEqual(result["down_error_on_changed_product_cross_term_relative"], 0)
        product_only = dict(changed, down_proj=dense["down_proj"])
        result = mlp_error_terms(dense, product_only, x)
        self.assertEqual(result["down_error_on_dense_product_relative"], 0)
        self.assertEqual(result["down_error_on_changed_product_cross_term_relative"], 0)


if __name__ == "__main__":
    unittest.main()
