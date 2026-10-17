"""Small, explicit dataset loaders used by compression and evaluation."""

import hashlib
import urllib.request


WIKITEXT2_PROCESSED_BASE = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2"
)
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def load_wikitext2_raw_split(split):
    """Load pinned canonical raw WikiText; never substitute the word-level variant."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    split = "validation" if split == "valid" else split
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported WikiText-2 split: {split!r}")
    filename = f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    path = hf_hub_download(
        repo_id=WIKITEXT_REPO, repo_type="dataset", revision=WIKITEXT_REVISION,
        filename=filename,
    )
    from pathlib import Path
    source = Path(path).read_bytes()
    rows = pq.read_table(path, columns=["text"])["text"].to_pylist()
    text = "\n\n".join(rows)
    if not text.strip():
        raise ValueError(f"Empty canonical WikiText split: {split}")
    encoded = text.encode("utf-8")
    return text, {
        "dataset": "wikitext-2-raw-v1", "dataset_repo": WIKITEXT_REPO,
        "dataset_revision": WIKITEXT_REVISION, "split": split,
        "source_url": f"https://huggingface.co/datasets/{WIKITEXT_REPO}/resolve/{WIKITEXT_REVISION}/{filename}",
        "source_sha256": hashlib.sha256(source).hexdigest(), "source_bytes": len(source),
        "text_sha256": hashlib.sha256(encoded).hexdigest(), "text_bytes": len(encoded),
        "text_join": "double_newline", "rows": len(rows),
        "literal_unk_count": text.count("<unk>"), "fallback": False,
    }


def load_wikitext2_processed_split(split):
    """Reproduce the legacy PyTorch input, explicitly labeled as processed text."""
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported WikiText-2 split: {split!r}")

    url = f"{WIKITEXT2_PROCESSED_BASE}/{split}.txt"
    request = urllib.request.Request(url, headers={"User-Agent": "NoxResearch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()

    return raw.decode("utf-8"), {
        "dataset": "wikitext-2-processed-pytorch",
        "split": split,
        "source_url": url,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "literal_unk_count": raw.count(b"<unk>"),
    }
