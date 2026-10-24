import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nox.packed_schmidt import (compile_packed_schmidt, reconstruct_packed_schmidt,
                               packed_schmidt_payload_bytes_from_artifact,
                               packed_schmidt_coefficient_count_from_artifact)
from nox.residual_codec import encode_groupwise, _pack_codes
from scripts.search_nec_layer_allocation import (
    PROJECTIONS, layer_module, materialize_deployable_allocation_package,
)


class PackedSchmidtPackageTests(unittest.TestCase):
    def test_package_preserves_unreplaced_norms_and_matches_injected_model(self):
        self._check_package_roundtrip(False)

    def test_canonical_excitation_package_matches_injected_model(self):
        self._check_package_roundtrip(True)

    def test_reconstructed_readout_preserves_coefficients_and_loads_exactly(self):
        self._check_package_roundtrip(False, override_readout=True)

    def _check_package_roundtrip(self, canonical, override_readout=False):
        torch.manual_seed(67)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32,
        )).half().eval()
        layer = model.model.layers[0]
        with torch.no_grad():
            layer.input_layernorm.weight.copy_(torch.linspace(0.7, 1.3, 8))
            layer.post_attention_layernorm.weight.copy_(torch.linspace(1.2, 0.8, 8))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkpoints, output = (root / name for name in ("source", "nec", "out"))
            checkpoints.mkdir()
            model.save_pretrained(source, max_shard_size=1024)
            artifacts = {}
            for name in PROJECTIONS:
                module = layer_module(layer, name)
                artifact, decoded, _ = compile_packed_schmidt(
                    module.weight.detach(), byte_budget=4000,
                    bits_candidates=(4,), group_size=4, als_sweeps=0,
                    n_heads=2, head_dim=4,
                )
                if canonical:
                    rows,columns=module.weight.shape
                    artifact['input_scale']=torch.linspace(.5,2.,columns).half()
                    support=torch.zeros(rows,columns,dtype=torch.bool);support[:,:2]=True
                    artifact={
                        'codec_type':'nec_sparse_excitation_schmidt_v2',
                        'orig_shape':tuple(module.weight.shape),'base_artifact':artifact,
                        'support_bits':_pack_codes(support,1),'amplitude_basis':'backbone_input_scale',
                        'amplitude_codec':encode_groupwise(torch.randn(rows,2)*.01,bits=3,group_size=4),
                        'amplitudes_per_row':2,
                    }
                    artifact['payload_bytes']=packed_schmidt_payload_bytes_from_artifact(artifact)
                    artifact['logical_coefficient_count']=packed_schmidt_coefficient_count_from_artifact(artifact)
                    decoded=reconstruct_packed_schmidt(artifact)
                artifacts[name] = artifact
                with torch.no_grad():
                    module.weight.copy_(decoded)
            torch.save({"layer_idx": 0, "compressed_artifacts": artifacts},
                       checkpoints / "compressed_layer_0.pt")
            (checkpoints / "model_manifest.json").write_text(json.dumps({
                "compiled_layers": [0],
            }))
            overrides = None
            if override_readout:
                with torch.no_grad():
                    model.lm_head.weight.mul_(.95)
                overrides = {'lm_head.weight': model.lm_head.weight.detach().clone()}
            counts = []
            for container, retained in itertools.product(("plain", "gzip"), repeat=2):
                with self.subTest(container=container, retained=retained):
                    measurement = materialize_deployable_allocation_package(
                        source, checkpoints, output / (container + retained), (0,), (0,), (), root / "unused", "tiny",
                        checkpoint_container=container, retained_container=retained,
                        retained_weight_overrides=overrides,
                        excitation_support_layout='column_major' if canonical else 'preserve',
                    )
                    package = Path(measurement["path"])
                    index = json.loads((package / "model.safetensors.index.json").read_text())
                    for norm in ("input_layernorm", "post_attention_layernorm"):
                        self.assertIn(f"model.layers.0.{norm}.weight", index["weight_map"])
                    loader_spec = importlib.util.spec_from_file_location(
                        "tiny_nec_package_loader_" + container, package / "load_nec_package.py",
                    )
                    loader = importlib.util.module_from_spec(loader_spec)
                    loader_spec.loader.exec_module(loader)
                    restored = loader.load_model(device_map=None, torch_dtype=torch.float16)
                    expected = dict(model.named_parameters())
                    for name, tensor in restored.named_parameters():
                        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
                    self.assertTrue(measurement["coefficient_accounting_complete"])
                    self.assertEqual(
                        measurement["complete_model_coefficient_count"],
                        measurement["retained_dense_coefficient_count"]
                        + measurement["replacement_coefficient_count"],
                    )
                    counts.append(measurement["complete_model_coefficient_count"])
                    self.assertEqual(measurement["serialized_bytes"], sum(
                        path.stat().st_size for path in package.rglob("*") if path.is_file()
                        and "__pycache__" not in path.parts
                    ))
                    if retained == "gzip":
                        self.assertFalse(list(package.glob("*.safetensors")))
                        records = measurement["retained_lossless_container_verification"]
                        self.assertTrue(records)
                        for record in records:
                            self.assertTrue(record["lossless_verified"])
                            self.assertEqual(record["uncompressed_sha256"], record["roundtrip_sha256"])
                    if container == "gzip":
                        self.assertFalse(list((package / "nec_checkpoints").glob("*.pt")))
                        self.assertEqual(len(list((package / "nec_checkpoints").glob("*.pt.gz"))), 1)
                        record = measurement["lossless_container_verification"][0]
                        self.assertTrue(record["lossless_verified"])
                        self.assertEqual(record["uncompressed_sha256"], record["roundtrip_sha256"])
            self.assertEqual(len(set(counts)), 1)


if __name__ == "__main__":
    unittest.main()
