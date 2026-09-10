"""Make diagnostic config copies and inspect storage without loading a model.

Usage: python3 prepare.py [path/to/existing/native-gguf/cache]
Existing variants must match; this never overwrites a checkpoint or weights.
"""

import copy
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / ".cache/benchmarks/gemma4-path-audit-2026-09-10"
DATA.mkdir(parents=True, exist_ok=True)
(DATA / "raw").mkdir(exist_ok=True)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


previous = DATA / "provenance.json"
source = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else Path(json.loads(previous.read_text())["sourcePreparedCache"])
)
config_path = source / "config.json"
original = json.loads(config_path.read_text())
assert original.get("dtype") == "bfloat16"
assert "dtype" not in original["text_config"]
assert "torch_dtype" not in original["text_config"]

for variant in ("baseline", "hoisted"):
    target = DATA / variant
    target.mkdir(exist_ok=True)
    config = copy.deepcopy(original)
    if variant == "hoisted":
        config["text_config"]["dtype"] = "bfloat16"
    for asset in source.iterdir():
        dest = target / asset.name
        if asset.name == "config.json":
            if dest.exists():
                assert not dest.is_symlink()
                assert json.loads(dest.read_text()) == config
            else:
                dest.write_text(json.dumps(config, indent=2) + "\n")
        elif dest.is_symlink():
            assert dest.resolve() == asset.resolve()
        elif dest.exists():
            raise RuntimeError(f"Refusing to replace existing asset: {dest}")
        else:
            dest.symlink_to(asset, target_is_directory=asset.is_dir())

headers = {}
header_sources = {}
for shard in sorted(source.glob("*.safetensors")):
    with shard.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    for name, entry in header.items():
        if name != "__metadata__":
            assert name not in headers
            headers[name] = entry
            header_sources[name] = shard.name

counts = Counter()
sizes = Counter()
for name, entry in headers.items():
    category = "other"
    if header_sources[name] != "model.safetensors":
        category = "mediaCompanionArrays"
    elif ".layers." in name and entry["dtype"] == "U32":
        category = "q4ProjectionWeights"
    elif ".layers." in name and name.endswith(".scales") and entry["dtype"] == "F16":
        category = "q4ProjectionScales"
    elif "embed_tokens." in name:
        category = "packedEmbedding"
    elif entry["dtype"] == "BF16":
        category = "smallBf16Arrays"
    counts[category] += 1
    sizes[category] += entry["data_offsets"][1] - entry["data_offsets"][0]

scales = sizes["q4ProjectionScales"]
assert counts["q4ProjectionWeights"] == 328
assert counts["q4ProjectionScales"] == 328
assert counts["packedEmbedding"] == 3
provenance = {
    "sourcePreparedCache": str(source),
    "sourceConfigSha256": sha(config_path),
    "change": "hoisted clone only adds text_config.dtype=bfloat16; all other assets are symlinks to the same prepared cache",
    "nativeAddonSha256": sha(ROOT / "packages/core/mlx-core.darwin-arm64.node"),
    "variantConfigSha256": {
        v: sha(DATA / v / "config.json") for v in ("baseline", "hoisted")
    },
    "tensorInventory": {
        key: {"arrays": counts[key], "serializedBytes": sizes[key]}
        for key in sorted(counts)
    },
    "derivedAffineBiasBytesF16": scales,
    "sidecarBytesF16": scales * 2,
    "sidecarBytesF32": scales * 4,
    "hoistedSteadyStateExtraBytes": scales * 2,
    "logicalCastReadPlusWriteBytesPerFullForward": scales * 6,
    "textGeometry": {
        key: original["text_config"].get(key)
        for key in (
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "global_head_dim", "sliding_window", "layer_types",
        )
    },
}
previous.write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps(provenance["tensorInventory"], indent=2))
