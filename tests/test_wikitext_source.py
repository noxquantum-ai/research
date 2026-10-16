import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from nox.data import WIKITEXT_REVISION, load_wikitext2_raw_split


class WikiTextSourceTests(unittest.TestCase):
    def test_raw_split_uses_pinned_raw_parquet_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.parquet"
            pq.write_table(pa.table({"text": ["", "A rare word survives.", ""]}), path)
            with patch("huggingface_hub.hf_hub_download", return_value=str(path)) as download:
                text, metadata = load_wikitext2_raw_split("valid")
            self.assertEqual(text, "\n\nA rare word survives.\n\n")
            self.assertEqual(metadata["split"], "validation")
            self.assertEqual(metadata["dataset"], "wikitext-2-raw-v1")
            self.assertEqual(metadata["text_sha256"], hashlib.sha256(text.encode()).hexdigest())
            self.assertEqual(metadata["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(download.call_args.kwargs["revision"], WIKITEXT_REVISION)
            self.assertEqual(download.call_args.kwargs["filename"], "wikitext-2-raw-v1/validation-00000-of-00001.parquet")
            self.assertFalse(metadata["fallback"])

    def test_download_failure_does_not_silently_use_processed_data(self):
        with patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
            with patch("urllib.request.urlopen") as legacy_download:
                with self.assertRaisesRegex(OSError, "offline"):
                    load_wikitext2_raw_split("test")
                legacy_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
