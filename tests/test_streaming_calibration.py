import unittest
import json
import tempfile
from pathlib import Path

import torch

from nox.calibration import (
    CalibrationStatistics,
    CalibrationStatisticsAccumulator,
    accumulate_calibration_statistics,
)
from nox.compiler import (
    nox_hybrid_dmrg_decompose,
    nox_hybrid_dmrg_decompose_from_statistics,
)
from nox.sequential_compiler import (
    compile_independent_transformer_block_streaming,
    collect_independent_block_calibration,
    forward_transformer_block,
    forward_transformer_block_chunked,
)
from nox.block_compiler import compile_transformer_block
from benchmarks.qualify_streaming_projection import (
    accumulate_down_covariance,
    down_projection_input,
    materialize_down_inputs,
    projection_metrics_chunked,
    validate_completed,
)
from scripts.compile_high_fidelity import sha256_file
from benchmarks.compare_streaming_projection_runs import compare as compare_staged_projection


class StreamingCalibrationTests(unittest.TestCase):
    def test_completed_stage_reuse_requires_current_source_and_settings(self):
        identity = {
            "schema_version": 1,
            "backend": "streaming_statistics",
            "statistics_dtype": "float32",
            "calibration_tokens": 4096,
        }
        source = {"commit": "current", "dirty_paths": [], "clean": True}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            artifact_path = output / "artifact.pt"
            torch.save({
                "source": source,
                "qualification_identity": identity,
                "artifact": {"fixture": True},
            }, artifact_path)
            report = {
                "status": "PASS",
                "source": source,
                "qualification_identity": identity,
                "artifact": {"sha256": sha256_file(artifact_path)},
            }
            (output / "report.json").write_text(json.dumps(report))
            self.assertTrue(validate_completed(output, identity, source))
            self.assertFalse(validate_completed(
                output, identity, {**source, "commit": "newer"},
            ))
            self.assertFalse(validate_completed(
                output, {**identity, "calibration_tokens": 8192}, source,
            ))
            self.assertFalse(validate_completed(
                output, identity, {**source, "clean": False, "dirty_paths": ["changed.py"]},
            ))

    def test_chunked_statistics_match_direct_products(self):
        torch.manual_seed(101)
        x = torch.randn(11, 37, dtype=torch.float64)
        y = torch.randn(7, 37, dtype=torch.float64)
        statistics = accumulate_calibration_statistics(
            x, y, chunk_size=6, accumulation_dtype=torch.float64, include_yy=True,
        )
        torch.testing.assert_close(statistics.xx, x @ x.T, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(statistics.yx, y @ x.T, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(statistics.yy, y @ y.T, atol=1e-12, rtol=1e-12)
        self.assertEqual(statistics.sample_count, 37)

    def test_incremental_accumulator_does_not_require_equal_chunk_sizes(self):
        torch.manual_seed(102)
        x = torch.randn(9, 23)
        y = torch.randn(5, 23)
        accumulator = CalibrationStatisticsAccumulator(
            9, 5, accumulation_dtype=torch.float64,
        )
        for start, stop in ((0, 4), (4, 17), (17, 23)):
            accumulator.update(x[:, start:stop], y[:, start:stop])
        statistics = accumulator.finalize()
        torch.testing.assert_close(statistics.xx, x.double() @ x.double().T)
        torch.testing.assert_close(statistics.yx, y.double() @ x.double().T)

    def test_statistics_compiler_matches_full_matrix_compiler(self):
        torch.manual_seed(103)
        weight = torch.randn(16, 16)
        inputs = torch.randn(16, 96)
        targets = weight @ inputs
        common = dict(
            o_modes=(2, 2, 2, 2),
            i_modes=(2, 2, 2, 2),
            ranks=(1, 4, 8, 4, 1),
            r_bridge=4,
            n_outliers=12,
            n_sweeps=2,
            prior_gamma=0.01,
            alpha_awt=0.5,
            chunk_size=7,
            return_artifacts=True,
            verbose=False,
            quantize_components=False,
            diagonal_prior_strength=0.01,
        )
        reference = nox_hybrid_dmrg_decompose(
            W=weight, X_calib=inputs, Y_dense=targets, **common,
        )
        statistics = accumulate_calibration_statistics(
            inputs, targets, chunk_size=13, accumulation_dtype=torch.float64,
        )
        streamed = nox_hybrid_dmrg_decompose_from_statistics(
            W=weight, calibration_statistics=statistics, **common,
        )
        torch.testing.assert_close(streamed[0], reference[0], atol=2e-3, rtol=2e-3)
        self.assertEqual(streamed[3], reference[3])
        self.assertEqual(
            streamed[4]["compiler_hyperparameters"]["calibration_backend"],
            "streaming_sufficient_statistics",
        )
        self.assertEqual(
            reference[4]["compiler_hyperparameters"]["calibration_backend"],
            "full_matrix_svd",
        )

    def test_staged_comparison_reports_cost_and_metric_parity(self):
        torch.manual_seed(109)
        weight = torch.randn(16, 16)
        inputs = torch.randn(16, 32)
        result = nox_hybrid_dmrg_decompose(
            W=weight,
            X_calib=inputs,
            Y_dense=weight @ inputs,
            o_modes=(2, 2, 2, 2),
            i_modes=(2, 2, 2, 2),
            ranks=(1, 2, 4, 2, 1),
            r_bridge=2,
            n_outliers=4,
            n_sweeps=2,
            return_artifacts=True,
            quantize_components=True,
            verbose=False,
        )
        artifact = result[4]
        metrics = {
            "sample_count": 32,
            "relative_error": 0.2,
            "cosine_similarity": 0.98,
            "angular_drift": 0.02,
            "norm_ratio": 1.0,
        }
        base = {
            "input_data": {"calibration_token_ids_sha256": "same"},
            "hidden_state_sha256": {"calibration": "a", "validation": "b"},
            "source": {"commit": "abc", "clean": True},
            "calibration_metrics": dict(metrics),
            "validation_metrics": dict(metrics),
            "runtime_seconds": 60.0,
            "memory": {"torch_peak_cuda_allocated_bytes": 123},
        }
        compared = compare_staged_projection(base, artifact, base, artifact, 0.59)
        self.assertEqual(compared["relative_reconstructed_weight_difference"], 0.0)
        self.assertEqual(
            compared["metric_differences"]["validation"]["relative_error_absolute_difference"],
            0.0,
        )
        self.assertAlmostEqual(
            compared["planning_upper_bound_for_two_full_layer_variants"]["gpu_cost_usd"],
            7 * 120 / 3600 * 0.59,
        )

    def test_rank_deficient_statistics_are_handled_explicitly(self):
        torch.manual_seed(104)
        weight = torch.randn(16, 16)
        inputs = torch.randn(16, 7)
        statistics = accumulate_calibration_statistics(
            inputs, weight @ inputs, chunk_size=3, accumulation_dtype=torch.float64,
        )
        result = nox_hybrid_dmrg_decompose_from_statistics(
            W=weight,
            calibration_statistics=statistics,
            o_modes=(2, 2, 2, 2),
            i_modes=(2, 2, 2, 2),
            ranks=(1, 2, 4, 2, 1),
            r_bridge=2,
            n_outliers=0,
            n_sweeps=0,
            quantize_components=False,
            return_artifacts=True,
            verbose=False,
        )
        self.assertTrue(torch.isfinite(result[0]).all())
        self.assertEqual(
            result[4]["calibration_statistics_diagnostics"]["numerical_rank"], 7,
        )

    def test_fp32_does_not_silently_apply_sample_count_rank_cutoff(self):
        torch.manual_seed(107)
        weight = torch.randn(16, 16)
        singular_values = torch.linspace(1.0, 0.001, 16)
        covariance = torch.diag(singular_values.square()).float()
        statistics = CalibrationStatistics(
            xx=covariance,
            yx=weight @ covariance,
            sample_count=65_536,
        )
        result = nox_hybrid_dmrg_decompose_from_statistics(
            W=weight,
            calibration_statistics=statistics,
            o_modes=(2, 2, 2, 2),
            i_modes=(2, 2, 2, 2),
            ranks=(1, 2, 4, 2, 1),
            r_bridge=0,
            n_outliers=0,
            n_sweeps=0,
            quantize_components=False,
            return_artifacts=True,
            verbose=False,
        )
        diagnostics = result[4]["calibration_statistics_diagnostics"]
        self.assertEqual(diagnostics["numerical_rank"], 16)
        self.assertLess(diagnostics["conventional_numerical_rank"], 16)
        self.assertGreater(diagnostics["positive_modes_below_conventional_cutoff"], 0)

    def test_sequence_aligned_chunked_block_forward_matches_full_forward(self):
        torch.manual_seed(105)
        d_model = 16
        d_intermediate = 32
        weights = {
            "q_proj": torch.randn(d_model, d_model) / 4,
            "k_proj": torch.randn(d_model, d_model) / 4,
            "v_proj": torch.randn(d_model, d_model) / 4,
            "o_proj": torch.randn(d_model, d_model) / 4,
            "gate_proj": torch.randn(d_intermediate, d_model) / 4,
            "up_proj": torch.randn(d_intermediate, d_model) / 4,
            "down_proj": torch.randn(d_model, d_intermediate) / 4,
            "input_layernorm": torch.randn(d_model),
            "post_attention_layernorm": torch.randn(d_model),
        }
        inputs = torch.randn(d_model, 5 * 4)
        reference = forward_transformer_block(
            weights, inputs, n_heads=4, n_kv_heads=4, seq_len=4,
        )
        chunked = forward_transformer_block_chunked(
            weights, inputs, windows_per_chunk=2,
            n_heads=4, n_kv_heads=4, seq_len=4,
        )
        for actual, expected in zip(chunked, reference):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

        collected = collect_independent_block_calibration(
            weights, inputs, windows_per_chunk=2,
            n_heads=4, n_kv_heads=4, seq_len=4,
            accumulation_dtype=torch.float64,
        )
        torch.testing.assert_close(collected["output"], reference[0], atol=2e-6, rtol=2e-6)
        self.assertEqual(collected["sample_count"], inputs.shape[1])
        self.assertTrue(all(
            covariance.sample_count == inputs.shape[1]
            for covariance in collected["covariances"].values()
        ))

    def test_chunked_forward_rejects_partial_windows(self):
        weights = {"q_proj": torch.eye(4)}
        with self.assertRaises(ValueError):
            forward_transformer_block_chunked(
                weights, torch.randn(4, 7), windows_per_chunk=1,
                n_heads=1, n_kv_heads=1, seq_len=4,
            )

    def test_staged_down_projection_collection_and_metrics_match_direct_path(self):
        torch.manual_seed(108)
        d_model = 16
        d_intermediate = 32
        weights = {
            "q_proj": torch.randn(d_model, d_model) / 4,
            "k_proj": torch.randn(d_model, d_model) / 4,
            "v_proj": torch.randn(d_model, d_model) / 4,
            "o_proj": torch.randn(d_model, d_model) / 4,
            "gate_proj": torch.randn(d_intermediate, d_model) / 4,
            "up_proj": torch.randn(d_intermediate, d_model) / 4,
            "down_proj": torch.randn(d_model, d_intermediate) / 4,
            "input_layernorm": torch.randn(d_model),
            "post_attention_layernorm": torch.randn(d_model),
        }
        hidden = torch.randn(d_model, 7 * 4)
        options = dict(seq_len=4, n_heads=4, n_kv_heads=4, rms_norm_eps=1e-5)
        direct = down_projection_input(weights, hidden, **options)
        chunked = materialize_down_inputs(
            weights, hidden, windows_per_chunk=2, forward_options=options,
        )
        torch.testing.assert_close(chunked, direct, atol=2e-6, rtol=2e-6)
        covariance = accumulate_down_covariance(
            weights,
            hidden,
            windows_per_chunk=2,
            forward_options=options,
            accumulation_dtype=torch.float64,
        )
        torch.testing.assert_close(
            covariance.xx, chunked.double() @ chunked.double().T,
            atol=1e-9, rtol=1e-9,
        )
        self.assertEqual(covariance.sample_count, hidden.shape[1])

        reconstructed = weights["down_proj"] + 0.01 * torch.randn_like(weights["down_proj"])
        measured = projection_metrics_chunked(
            weights["down_proj"],
            reconstructed,
            weights,
            hidden,
            windows_per_chunk=2,
            forward_options=options,
        )
        target = weights["down_proj"] @ direct
        prediction = reconstructed @ direct
        expected_relative = float(
            torch.linalg.vector_norm(target - prediction)
            .div(torch.linalg.vector_norm(target))
            .item()
        )
        self.assertAlmostEqual(measured["relative_error"], expected_relative, places=6)
        self.assertEqual(measured["sample_count"], hidden.shape[1])

    def test_complete_streaming_block_compiler_matches_reference_path(self):
        torch.manual_seed(106)
        d_model = 16
        d_intermediate = 32
        weights = {
            "q_proj": torch.randn(d_model, d_model) / 4,
            "k_proj": torch.randn(d_model, d_model) / 4,
            "v_proj": torch.randn(d_model, d_model) / 4,
            "o_proj": torch.randn(d_model, d_model) / 4,
            "gate_proj": torch.randn(d_intermediate, d_model) / 4,
            "up_proj": torch.randn(d_intermediate, d_model) / 4,
            "down_proj": torch.randn(d_model, d_intermediate) / 4,
            "input_layernorm": torch.randn(d_model),
            "post_attention_layernorm": torch.randn(d_model),
        }
        inputs = torch.randn(d_model, 8 * 4)
        common = dict(
            ranks=(1, 4, 8, 4, 1),
            r_bridge=2,
            n_outliers=8,
            n_heads=4,
            n_kv_heads=4,
            seq_len=4,
            rms_norm_eps=1e-5,
            diagonal_prior_strength=0.01,
            mpo_n_sweeps=2,
            mpo_alpha_awt=0.5,
            return_artifacts=True,
            verbose=False,
        )
        reference = compile_transformer_block(
            weights,
            inputs,
            X_calib_dense=inputs,
            causal_healing=False,
            trajectory_mode="none",
            **common,
        )
        streamed = compile_independent_transformer_block_streaming(
            weights,
            inputs,
            windows_per_chunk=3,
            accumulation_dtype=torch.float64,
            **common,
        )
        for name in reference["compressed_weights"]:
            torch.testing.assert_close(
                streamed["compressed_weights"][name],
                reference["compressed_weights"][name],
                atol=5e-3,
                rtol=5e-3,
            )
        torch.testing.assert_close(
            streamed["X_out_dense"], reference["X_out_dense"], atol=2e-5, rtol=2e-5,
        )
        self.assertTrue(all(
            artifact["compiler_hyperparameters"]["calibration_backend"]
            == "streaming_sufficient_statistics"
            for artifact in streamed["compressed_artifacts"].values()
        ))
        expected_quantized_components = {
            "sparse_shield", "bridge_A", "bridge_B",
            "mpo_site_0", "mpo_site_1", "mpo_site_2", "mpo_site_3",
        }
        for artifact in streamed["compressed_artifacts"].values():
            self.assertEqual(artifact["core_quantization_bits"], 4)
            self.assertEqual(artifact["sparse_bridge_quantization_bits"], 4)
            self.assertEqual(
                set(artifact["precision_diagnostics"]), expected_quantized_components,
            )


if __name__ == "__main__":
    unittest.main()
