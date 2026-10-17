import hashlib
import json
from pathlib import Path

import pytest

from kaggle_execution.nec_high_fidelity import driver, evaluation_driver


def write_bundle(root: Path):
    source_root = root / "dataset"
    payload = b"pinned source bytes\n"
    (source_root / "code").mkdir(parents=True)
    (source_root / "code" / "module.py").write_bytes(payload)
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
    manifest_path = source_root / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return source_root, manifest_path, payload


@pytest.mark.parametrize("module", [driver, evaluation_driver])
def test_kaggle_source_bundle_accepts_verified_extracted_dataset(tmp_path, module):
    source_root, _, _ = write_bundle(tmp_path)

    actual = module.verify_source_tree(source_root)

    assert actual["source_commit"] == "a" * 40


def test_compile_launcher_copies_and_verifies_kaggle_extracted_dataset(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    source_root, manifest_path, payload = write_bundle(input_root)
    work_source = tmp_path / "work-source"
    monkeypatch.setattr(driver, "SOURCE", work_source)

    source_input, commit, manifest_hash = driver.get_source_bundle(input_root)

    assert source_input == manifest_path
    assert commit == "a" * 40
    assert manifest_hash == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert (work_source / "code" / "module.py").read_bytes() == payload
    assert driver.verify_source_tree(work_source)["source_commit"] == commit


def test_evaluation_launcher_copies_and_verifies_kaggle_extracted_dataset(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    _, manifest_path, payload = write_bundle(input_root)
    work_source = tmp_path / "work-source"
    monkeypatch.setattr(evaluation_driver, "SOURCE", work_source)

    source_input = evaluation_driver.find_source_input(input_root)
    manifest = evaluation_driver.extract_source(source_input)

    assert source_input == manifest_path
    assert manifest["source_commit"] == "a" * 40
    assert (work_source / "code" / "module.py").read_bytes() == payload


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
