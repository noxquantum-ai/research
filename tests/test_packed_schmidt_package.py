import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nox.packed_schmidt import compile_packed_schmidt
from scripts.search_nec_layer_allocation import (
    PROJECTIONS, layer_module, materialize_deployable_allocation_package,
)


class PackedSchmidtPackageTests(unittest.TestCase):
    def test_package_preserves_unreplaced_norms_and_matches_injected_model(self):
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
                artifacts[name] = artifact
                with torch.no_grad():
                    module.weight.copy_(decoded)
            torch.save({"layer_idx": 0, "compressed_artifacts": artifacts},
                       checkpoints / "compressed_layer_0.pt")
            (checkpoints / "model_manifest.json").write_text(json.dumps({
                "compiled_layers": [0],
            }))
            measurement = materialize_deployable_allocation_package(
                source, checkpoints, output, (0,), (0,), (), root / "unused", "tiny",
            )
            package = Path(measurement["path"])
            index = json.loads((package / "model.safetensors.index.json").read_text())
            for norm in ("input_layernorm", "post_attention_layernorm"):
                self.assertIn(f"model.layers.0.{norm}.weight", index["weight_map"])
            loader_spec = importlib.util.spec_from_file_location(
                "tiny_nec_package_loader", package / "load_nec_package.py",
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
            self.assertEqual(measurement["serialized_bytes"], sum(
                path.stat().st_size for path in package.rglob("*") if path.is_file()
                and "__pycache__" not in path.parts
            ))


if __name__ == "__main__":
    unittest.main()
