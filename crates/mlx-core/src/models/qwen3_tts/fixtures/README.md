Configuration fixtures from Qwen/Qwen3-TTS-12Hz-0.6B-Base, revision
`5d83992436eae1d760afd27aff78a71d676296fc` (Apache-2.0).

- `base.json`: root `config.json`
- `codec.json`: `speech_tokenizer/config.json`
- `generation.json`: `generation_config.json`

These exercise configuration compatibility without downloading weights. They are
test data, never runtime fallback configuration. The production parser reads the
loaded checkpoint and applies only the architecture defaults documented in
`config.rs`.
