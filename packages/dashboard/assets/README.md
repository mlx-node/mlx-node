# Delegate tokenizer vocabulary

`o200k_base.json.gz` is the fixed vocabulary used by the dashboard's Hugging Face
[`tokenizers` Node binding](https://github.com/huggingface/tokenizers/tree/main/bindings/node).
It is bundled so measuring sessions never downloads a model or contacts a service.

The asset is converted from `tiktoken` 0.14.0's `o200k_base` encoding using
Hugging Face's documented
[`convert_tiktoken_to_fast`](https://huggingface.co/docs/transformers/main/en/tiktoken)
conversion. Added special tokens and post-processing are removed; padding and
truncation are disabled. Literal special-token strings count as ordinary text.
The vocabulary retains its original MIT license in `tiktoken-LICENSE`.

Regenerate with `uv run packages/dashboard/scripts/generate-delegate-tokenizer.py`.
The script pins its conversion dependencies and checks the vocabulary hashes.
Python and Transformers are regeneration tools, not runtime dependencies.

SHA-256:

- Source ranks: `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`
- Uncompressed JSON: `2246c67011479605d0ebef4a2fe06a1f5a2f3a1730d413ff5fc96445770fec2e`
- Gzip asset: `5e26b2591420efb1e123bdc74f30b160e07c2b1a38a15c590aa42ab80c940571`
