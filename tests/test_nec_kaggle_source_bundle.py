import hashlib
import json

import pytest

from kaggle_execution.nec_high_fidelity import driver, evaluation_driver


@pytest.mark.parametrize("module", [driver, evaluation_driver])
def test_kaggle_source_bundle_accepts_verified_extracted_dataset(tmp_path, module):
    payload = b"pinned source bytes\n"
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "module.py").write_bytes(payload)
    manifest = {
        "bundle_format": 1,
        "source_commit": "a" * 40,
        "source_files": {
            "code/module.py": {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        },
    }
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest))

    actual = module.verify_source_tree(tmp_path)

    assert actual["source_commit"] == "a" * 40


@pytest.mark.parametrize("module", [driver, evaluation_driver])
def test_kaggle_source_bundle_rejects_path_traversal(tmp_path, module):
    manifest = {
        "bundle_format": 1,
        "source_commit": "a" * 40,
        "source_files": {"../outside.py": {"bytes": 0, "sha256": ""}},
    }
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Unsafe source-bundle path"):
        module.verify_source_tree(tmp_path)


@pytest.mark.parametrize("module", [driver, evaluation_driver])
def test_kaggle_source_bundle_rejects_modified_file(tmp_path, module):
    payload = b"modified source"
    (tmp_path / "module.py").write_bytes(payload)
    manifest = {
        "bundle_format": 1,
        "source_commit": "a" * 40,
        "source_files": {"module.py": {"bytes": len(payload), "sha256": "0" * 64}},
    }
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Source-bundle checksum mismatch"):
        module.verify_source_tree(tmp_path)
