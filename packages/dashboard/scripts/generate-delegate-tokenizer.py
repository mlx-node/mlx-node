# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = ["transformers==5.2.0", "tokenizers==0.22.2", "tiktoken==0.14.0", "blobfile==3.3.0"]
# ///
"""Regenerate the checked-in offline vocabulary: uv run this-file.py."""

import gzip
import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers.integrations.tiktoken import convert_tiktoken_to_fast

assets = Path(__file__).resolve().parents[1] / "assets"
with TemporaryDirectory() as directory:
    output = Path(directory)
    convert_tiktoken_to_fast("o200k_base", output)
    ranks = (output / "tiktoken/tokenizer.model").read_bytes()
    assert hashlib.sha256(ranks).hexdigest() == "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
    config = json.loads((output / "tokenizer.json").read_text())
    # Count literal text only. Text resembling a control token is ordinary input.
    config.update(added_tokens=[], post_processor=None, padding=None, truncation=None)
    raw = json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == "2246c67011479605d0ebef4a2fe06a1f5a2f3a1730d413ff5fc96445770fec2e"
    assets.mkdir(exist_ok=True)
    (assets / "o200k_base.json.gz").write_bytes(gzip.compress(raw, mtime=0))

package = distribution("tiktoken")
license_file = next(file for file in package.files if str(file).endswith("licenses/LICENSE"))
(assets / "tiktoken-LICENSE").write_bytes(package.locate_file(license_file).read_bytes())
