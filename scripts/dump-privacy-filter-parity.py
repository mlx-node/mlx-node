#!/usr/bin/env python3
"""HuggingFace parity-fixture generator for openai/privacy-filter.

Run with:
  uv run --with 'transformers>=5.8' --with torch --with numpy \
    python scripts/dump-privacy-filter-parity.py \
    --model .cache/models/privacy-filter \
    --out packages/privacy/__test__/parity-fixtures.json

Decoding strategy: per-token argmax over raw logits.

  The HF checkpoint (`OpenAIPrivacyFilterForTokenClassification` in
  modeling_opf.py) is implemented as `GenericForTokenClassification` —
  it ships **only** a logits head, no Viterbi decoder, and no
  application of `viterbi_calibration.json` (the calibration file is
  consumed by *us*, not by HF). The canonical `pipeline(
  'token-classification', aggregation_strategy='simple')` path is a
  naive BIO aggregator that has no notion of BIOES (`E-`/`S-`) and will
  disagree with our Viterbi on multi-word entities in ways that are
  benign-but-noisy.

  The well-defined cross-implementation invariant is therefore the
  per-token argmax tag sequence over raw logits, which both our Rust
  forward pass and HF produce deterministically. Our public
  `PrivacyFilter.classify(text, { returnTokens: true })` exposes
  exactly that (the `tag` field on each `Token` is the BIOES tag of
  the argmax class id), so the parity test compares those tag
  sequences token-for-token.

  Determinism: we force CPU + fp32 to remove the bf16-on-Metal / MPS
  source of variance. This matches the precision we use for the
  diagnostic compare in scripts/diag_privacy_filter.py (which loads
  bf16 but converts captures to fp32 before saving).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# 5 fixture inputs covering all 8 PII labels at least once. Kept short
# (<150 chars) and obviously synthetic — names, emails, phone numbers,
# addresses, dates, URLs, account numbers, and secrets are all
# fictional or sentinel-style values.
FIXTURES: list[str] = [
    # private_person + private_email (canonical Alice Smith fixture so
    # this stays in sync with the existing forward.rs Rust test).
    "Hi I am Alice Smith, email alice@example.com",
    # private_phone
    "Call me at +1 555 123 4567 anytime",
    # private_address
    "Ship the package to 742 Evergreen Terrace, Springfield",
    # private_date + private_url
    "Born 1990-03-14, profile at https://example.org/profile/jdoe",
    # account_number + secret
    "Account 1234567890; api key sk-ZZZZZZZZZZZZZZZZZZZZZZZZZZZZ",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        required=True,
        help="Path to the openai/privacy-filter checkpoint directory.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON path (will be overwritten).",
    )
    args = ap.parse_args()

    # Force CPU + fp32 for byte-stable output across machines. The
    # diag_privacy_filter.py script uses bf16 because it's comparing
    # against our bf16 Rust path; here we only care about argmax
    # stability, so fp32/CPU is the safest choice.
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    model.to("cpu")

    id2label: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}

    records: list[dict] = []
    with torch.no_grad():
        for text in FIXTURES:
            # `add_special_tokens=False` matches the Rust path in
            # crates/mlx-core/src/models/privacy_filter/mod.rs which
            # tokenizes without CLS/SEP because the classification
            # head is fed raw BPE tokens.
            enc = tok(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"][0].tolist()
            token_strs = tok.convert_ids_to_tokens(input_ids[0])

            logits = model(input_ids=input_ids).logits  # [1, T, C]
            pred_ids: list[int] = logits.argmax(-1)[0].tolist()

            # Build a stable, JSON-roundtrippable record. We store
            # tag *names* rather than ids so the fixtures survive
            # any future id2label remapping in the checkpoint.
            tokens = []
            for i, tid in enumerate(pred_ids):
                start, end = offsets[i]
                tokens.append(
                    {
                        "text": token_strs[i],
                        "tag": id2label[tid],
                        "start": int(start),
                        "end": int(end),
                    }
                )

            records.append({"input": text, "tokens": tokens})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pretty-print with a trailing newline so `git diff` is readable.
    out_path.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {len(records)} fixtures to {out_path}")
    for r in records:
        non_o = sum(1 for t in r["tokens"] if t["tag"] != "O")
        print(f"  {r['input']!r}: {len(r['tokens'])} tokens, {non_o} non-O")
    return 0


if __name__ == "__main__":
    sys.exit(main())
