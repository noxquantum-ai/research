import io
import unittest

import torch
import torch.nn.functional as F

from nox.native_swiglu import (
    apply_native_swiglu,
    build_native_swiglu_artifact,
    fit_anchored_down_readout,
    native_neuron_importance,
    native_swiglu_coefficient_count,
    native_swiglu_payload_bytes,
    native_swiglu_serialized_bytes,
    NativeSwiGLUModule,
)


class NativeSwiGLUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.d = 6
        self.m = 10
        self.x = torch.randn(self.d, 16)
        self.g = torch.randn(self.m, self.d)
        self.u = torch.randn(self.m, self.d)
        self.down = torch.randn(self.d, self.m)

    def test_full_retention_matches_dense_function(self):
        artifact = build_native_swiglu_artifact(
            self.g, self.u, self.down, self.x, retain_fraction=1.0,
            storage_dtype=torch.float32,
        )
        expected = self.down @ (F.silu(self.g @ self.x) * (self.u @ self.x))
        torch.testing.assert_close(apply_native_swiglu(artifact, self.x), expected)

    def test_importance_selects_dominant_native_neuron(self):
        down = self.down.clone()
        down[:, 3] *= 1000
        score = native_neuron_importance(self.g, self.u, down, self.x)
        artifact = build_native_swiglu_artifact(
            self.g, self.u, down, self.x, retained_neurons=1,
            storage_dtype=torch.float32,
        )
        self.assertEqual(int(score.argmax()), 3)
        self.assertEqual(int(artifact["retained_indices"][0]), 3)

    def test_coefficient_and_storage_accounting(self):
        artifact = build_native_swiglu_artifact(
            self.g, self.u, self.down, self.x, retained_neurons=4,
        )
        self.assertEqual(artifact["coefficient_count"], native_swiglu_coefficient_count(6, 4))
        self.assertEqual(native_swiglu_payload_bytes(artifact), 4 * 4 + 2 * 3 * 6 * 4)
        stream = io.BytesIO()
        torch.save(artifact, stream)
        self.assertEqual(stream.tell(), native_swiglu_serialized_bytes(artifact))

    def test_anchored_readout_recovers_linear_teacher(self):
        features = torch.randn(5, 24)
        prior = torch.randn(4, 5)
        teacher = prior + 0.1 * torch.randn(4, 5)
        targets = teacher @ features
        fitted, metadata = fit_anchored_down_readout(
            features, targets, prior, sequence_length=4,
            ridge_ratios=(0.0, 1e-8),
        )
        error = torch.linalg.vector_norm(fitted @ features - targets)
        self.assertLess(float(error), 1e-3)
        self.assertFalse(metadata["construction_uses_validation"])

    def test_invalid_fraction_rejected(self):
        with self.assertRaises(ValueError):
            build_native_swiglu_artifact(
                self.g, self.u, self.down, self.x, retain_fraction=0.0,
            )

    def test_runtime_module_supports_batched_sequence_inputs(self):
        artifact = build_native_swiglu_artifact(
            self.g, self.u, self.down, self.x, retain_fraction=0.6,
            storage_dtype=torch.float32,
        )
        module = NativeSwiGLUModule(artifact)
        hidden = torch.randn(2, 3, self.d)
        expected = apply_native_swiglu(
            artifact, hidden.reshape(-1, self.d).T,
        ).T.reshape(2, 3, self.d)
        torch.testing.assert_close(module(hidden), expected)


if __name__ == "__main__":
    unittest.main()
