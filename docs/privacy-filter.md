# Privacy Filter

An MLX-Node port of the [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) checkpoint — a gpt-oss-style MoE token classifier that detects and labels eight categories of personally identifiable information (PII). The forward pass uses a custom Metal kernel for bidirectional banded attention with attention sinks; everything runs on Apple Silicon through the existing Rust/NAPI bridge.

The eight label classes are:

```
account_number   private_address   private_date     private_email
private_person   private_phone     private_url      secret
```

## Install

1. Build the native addon: `yarn build:native`.
2. Acquire the checkpoint from Hugging Face (`openai/privacy-filter`). The loader expects a directory containing:
   - `config.json`
   - `model.safetensors`
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `viterbi_calibration.json` (optional — supplies default decoder biases)

   You can place this anywhere on disk; the API takes an absolute or relative path.

## High-level API (`@mlx-node/privacy`)

```typescript
import { PrivacyFilter } from '@mlx-node/privacy';

const pf = await PrivacyFilter.load('./models/privacy-filter');

const result = await pf.classify('Hi I am Alice Smith, email alice@example.com');
// result.entities → [
//   { label: 'private_person', start: 8,  end: 19, score: 0.98, text: 'Alice Smith' },
//   { label: 'private_email',  start: 27, end: 44, score: 0.97, text: 'alice@example.com' },
// ]

const { redacted, entities } = await pf.redact('Call me at +1 555 0100.');
// redacted → 'Call me at [private_phone].'
```

`start` and `end` are byte offsets into the original input string (Hugging Face `tokenizers` convention). `score` is the mean of per-token max-softmax probabilities across the span.

### `PrivacyFilter.load(modelPath)`

Static async factory. Returns a `PrivacyFilter` bound to the checkpoint at `modelPath`.

### `pf.classify(text, opts?) → { entities, tokens? }`

| Option         | Type                          | Default | Purpose                                                                |
| -------------- | ----------------------------- | ------- | ---------------------------------------------------------------------- |
| `threshold`    | `number`                      | `0.5`   | Minimum mean per-token probability for a span to be kept.              |
| `calibration`  | `Partial<ViterbiCalibration>` | —       | Per-call overrides on top of the checkpoint default (see Calibration). |
| `returnTokens` | `boolean`                     | `false` | When `true`, the result includes a `tokens` array.                     |

Each entity:

```typescript
interface Entity {
  label: PrivacyLabel; // one of the 8 classes above
  start: number; // byte offset
  end: number; // byte offset (exclusive)
  score: number; // mean per-token probability
  text: string; // text.slice(start, end)
}
```

When `returnTokens: true`, `tokens[i]` carries `{ text, tag, score, start, end }` where `tag` is the full BIOES tag (`'O'` or `'B-…'`/`'I-…'`/`'E-…'`/`'S-…'`) and `score` is the softmax probability of the argmax class at that token.

### `pf.redact(text, opts?) → { redacted, entities }`

Inherits every option from `classify`, plus:

| Option        | Type                                                  | Default   | Purpose                                                                                                 |
| ------------- | ----------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| `replacement` | `'label'` \| `string` \| `(entity: Entity) => string` | `'label'` | `'label'` produces `[<label>]`; any other string is inserted verbatim; a function is called per entity. |
| `labels`      | `PrivacyLabel[]`                                      | —         | Allowlist — only entities whose label is in this list are redacted (others stay verbatim).              |

`entities` in the return value is the post-filter set actually redacted, sorted by `start`.

## Native binding (`@mlx-node/core`)

The low-level NAPI class is exposed directly for callers that don't want the TS wrapper:

```typescript
import { PrivacyFilterModel } from '@mlx-node/core';

const m = PrivacyFilterModel.load('./models/privacy-filter');
const result = m.classify('Hi I am Alice Smith.', { threshold: 0.5 });
```

`PrivacyFilterModel.load` and `.classify` are **synchronous** at the binding level — no `await` needed. The `@mlx-node/privacy` wrapper exposes them as `async` so future implementations (e.g. off-main-thread offload) won't break the ABI.

## CLI: `mlx redact`

```bash
mlx redact --model <path> [options]
```

| Flag             | Default   | Purpose                                                                                                                             |
| ---------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `-m`, `--model`  | —         | Path to a privacy-filter model directory (required).                                                                                |
| `-i`, `--input`  | stdin     | Input text file.                                                                                                                    |
| `-o`, `--output` | stdout    | Output file for redacted text.                                                                                                      |
| `--replacement`  | `'label'` | Replacement string. `'label'` substitutes `[<label>]`; any other value is inserted verbatim.                                        |
| `--labels`       | —         | Comma-separated allowlist of labels (e.g. `private_email,private_person`).                                                          |
| `--threshold`    | `0.5`     | Minimum mean per-token probability for an entity to be kept.                                                                        |
| `--json`         | off       | Emit the entities sidecar as JSON. With `--output`, writes `<output>.entities.json`. Without `--output`, writes the JSON to stderr. |
| `-h`, `--help`   | —         | Show help.                                                                                                                          |

### Examples

```bash
# File in, file out.
mlx redact -m ./models/privacy-filter -i input.txt -o redacted.txt

# Pipe through stdin/stdout.
cat input.txt | mlx redact -m ./models/privacy-filter > redacted.txt

# Write redacted text + a sidecar JSON of detected entities.
mlx redact -m ./models/privacy-filter -i input.txt -o out.txt --json

# Only redact emails, leave everything else alone.
mlx redact -m ./models/privacy-filter -i input.txt --labels private_email

# Custom replacement string.
mlx redact -m ./models/privacy-filter -i input.txt --replacement '[REDACTED]'
```

## Calibration tuning

The checkpoint ships a `viterbi_calibration.json` that supplies default biases for the constrained BIOES Viterbi decoder. The decoder operates in log space — biases are added to the per-token log-softmax emission scores when scoring transitions. Six biases are exposed:

```typescript
interface ViterbiCalibration {
  transitionBiasBackgroundStay: number;
  transitionBiasBackgroundToStart: number;
  transitionBiasEndToBackground: number;
  transitionBiasEndToStart: number;
  transitionBiasInsideToContinue: number;
  transitionBiasInsideToEnd: number;
}
```

You can override any subset per call via `opts.calibration`. Omitted fields fall back to the checkpoint default.

```typescript
// Tighten the entry into a span — reduces false-positive spans.
await pf.classify(text, {
  calibration: { transitionBiasBackgroundToStart: -2.0 },
});

// Loosen it — accept weaker evidence to start a span (higher recall).
await pf.classify(text, {
  calibration: { transitionBiasBackgroundToStart: 1.0 },
});
```

## Quantization

The bf16 checkpoint can be re-quantized into one of four formats via `mlx convert`. The loader auto-detects the format from `config.json::quantization.mode` and instantiates the matching quantized matmul (4-bit affine `quantized_matmul` for affine; `gather_qmm` for the OCP/NVIDIA float-grouped formats).

| Mode     | Bits | Group | Storage layout                                                                              |
| -------- | ---- | ----- | ------------------------------------------------------------------------------------------- |
| `affine` | 4    | 64    | 4-bit packed weights + per-group scale + per-group bias. Only mode supporting `--q-recipe`. |
| `mxfp4`  | 4    | 32    | OCP MX FP4 + shared 8-bit exponent per group of 32. No biases stored.                       |
| `mxfp8`  | 8    | 32    | OCP MX FP8 (E4M3) + shared exponent per group of 32. No biases stored.                      |
| `nvfp4`  | 4    | 16    | NVIDIA FP4 + shared exponent per group of 16 (denser than mxfp4). No biases stored.         |

Producing a quantized checkpoint:

```bash
# 8-bit MX FP — highest quality of the float-grouped modes.
mlx convert -m privacy-filter -q --q-mode mxfp8 \
  -i .cache/models/privacy-filter \
  -o .cache/models/privacy-filter-mxfp8

# 4-bit affine — general-purpose, accepts mixed-bit recipes via --q-recipe.
mlx convert -m privacy-filter -q --q-mode affine \
  -i .cache/models/privacy-filter \
  -o .cache/models/privacy-filter-affine
```

The loader picks up the format automatically: `PrivacyFilter.load(path)` reads the `quantization.mode` field and selects the appropriate matmul kernel — no API change.

### NER parity

All four quantization modes preserve the named-entity tag positions on the canonical Alice/email single-input test (the same fixture as `forward_classify_alice_smith_email`). Across the broader 5-fixture parity sweep covering all eight PII label classes, every mode produces a small number of boundary-token flips on adjacent entity-position tokens — no NER-class confusions, but the precise span boundary shifts by one token in most cases:

- **mxfp8** and **mxfp4** miss the leading `I-account_number` token (the "123" in "Account 1234567890") on input #4. The remaining account-number tokens are still tagged correctly, so the span is just shorter by one token at the start. These divergences are stable run-to-run.
- **affine** drops the lone `E-private_address` token ("Springfield") on input #2. Stable run-to-run.
- **nvfp4** is **deterministic** but **argmax-fragile**: on inputs #3 (URL `https://example.org/profile/jdoe`) and #4 (secret `sk-ZZZZ…`), the raw per-token argmax flips on a small number of boundary tokens compared to bf16, occasionally dropping entire URL/secret span tags at the raw-tag level. The Viterbi BIOES decoder (`crates/mlx-core/src/models/privacy_filter/viterbi.rs`) recovers the correct entity spans via path constraints, so `pf.classify()` output stays stable run-to-run and matches the other modes at the entity level. Per-tensor reconstruction quality (cosine similarity vs the bf16 weights) is actually higher for nvfp4 (~0.9946) than mxfp4 (~0.9915) — the issue is noise _direction_, not magnitude: nvfp4's residual error happens to point across decision boundaries for a few boundary tokens, even though its overall reconstruction is more accurate.

These divergences are surfaced by the Rust test `parity_across_quantized_modes` in [`crates/mlx-core/src/models/privacy_filter/forward.rs`](../crates/mlx-core/src/models/privacy_filter/forward.rs); it gates on raw argmax tags rather than Viterbi-decoded entities, so it intentionally fails to keep the boundary-token flips visible. Treat the table above as a guide: **`mxfp8` is the safest quantized mode for production**. **`nvfp4` is safe at the entity level (`pf.classify()` / `pf.redact()`) but not recommended if downstream tooling consumes the raw per-token tag labels directly** — prefer `mxfp4` or `mxfp8` in that case.

For nvfp4 our convert path uses per-group scales only (no `global_scale`); tensor-scale nvfp4 is CPU/CUDA-only upstream. A related Metal gap — `QQMatmul`'s gemv path silently drops `global_scale_x` / `global_scale_w` instead of throwing NYI like the general path does — is tracked in [ml-explore/mlx#3550](https://github.com/ml-explore/mlx/issues/3550). It does not affect privacy-filter, which uses `gather_qmm` / `quantized_matmul` rather than `qqmm`.

### Performance

Measured on M3 Max, 36 GB unified memory, 5-mode median over 3 runs. Inputs synthesised by repeating a PII-dense seed sentence (`"Hi I am Alice Smith, email alice@example.com. Call +1 555 123 4567. Born 1990-03-14. "`) until the tokenized length lands inside the target band. `Peak footprint` is the maximum value observed by polling `vmmap -summary <pid>` at 1 Hz across the entire bench (load + warmup + short + long classify); `tok/s` is `tokens / wall_seconds` for the corresponding run.

| Mode     | Load (ms) | 512-tok (ms) | tok/s  | 2K-tok (ms) | tok/s  | Peak footprint |
| -------- | --------- | ------------ | ------ | ----------- | ------ | -------------- |
| `bf16`   | 234       | 25           | 21,219 | 112         | 18,768 | 4.30 GB        |
| `mxfp8`  | 234       | 26           | 20,805 | 117         | 17,888 | 3.10 GB        |
| `mxfp4`  | 234       | 26           | 20,807 | 114         | 18,358 | 1.40 GB        |
| `nvfp4`  | 240       | 27           | 19,993 | 114         | 18,382 | 1.40 GB        |
| `affine` | 236       | 26           | 20,701 | 111         | 18,874 | 1.40 GB        |

Throughput is essentially identical across modes — the privacy-filter forward pass is small enough (eight blocks, 640 hidden) that the matmul kernel choice is not the bottleneck on this hardware. **The win for quantization here is footprint, not speed**: the 4-bit modes (`mxfp4` / `nvfp4` / `affine`) shrink steady-state peak footprint to ~1.4 GB versus 4.30 GB for `bf16`, leaving room for larger batch sizes or co-residency with another model. `mxfp8` is in between at 3.10 GB.

### Quality caveat

Quality on long-form text was validated only on a 5-fixture short-input parity sweep. Production use on long documents (>~500 tokens) should be validated against the `bf16` baseline on a representative sample first. Combined with the existing recall ceiling at ~2000 tokens (see [Limitations](#limitations)), the recommended deployment for long inputs is `bf16` with chunking; the quantized modes are best suited to inference cost / footprint reduction on short PII scans.

## Limitations

- macOS / Apple Silicon (Metal) only. The banded-attention kernel is Metal-specific with no CUDA port, so the privacy filter is **not** covered by the experimental CUDA backend (which targets Qwen3.6 dense/MoE inference — see the main README's Platform Support).
- bf16 weights and forward by default. The Metal banded-attention kernel and the bf16 forward can produce small disagreements vs. Hugging Face's fp32 reference at low-confidence boundary tokens. See the parity test fixtures at [`packages/privacy/__test__/parity-fixtures.json`](../packages/privacy/__test__/parity-fixtures.json) for the tolerated budget.
- Attention is bidirectional banded with attention sinks; `sliding_window = 128` on alternating layers per the gpt-oss config (band ±128 → 257-token effective window).
- **Recall degrades sharply past ~2000 tokens of input.** The checkpoint is trained on short documents; long-context inputs (>~5000 chars) miss most entities, and >8000 chars typically returns nothing. When scanning long text, chunk the input — ~1500 chars (~500 tokens) per `classify` call is a reliable upper bound and stays well within the trained context window.

## Memory on Apple Silicon

`process.memoryUsage().rss` undercounts Metal buffer allocations because Apple's unified memory architecture charges GPU buffers to the process's **`phys_footprint`** (what Activity Monitor's "Memory" column displays) rather than the resident set. For accurate measurements use `vmmap -summary <pid> | grep "Physical footprint"` or the `footprint` CLI. Each `classify()` call clears the MLX buffer cache before returning, so steady-state footprint stays bounded; transient peaks between calls scale with input length.

## Internals

The architecture, kernel design, Viterbi decoder, and tokenizer integration are documented in the design spec at [`docs/superpowers/specs/2026-05-13-privacy-filter-design.md`](superpowers/specs/2026-05-13-privacy-filter-design.md). The Rust implementation lives at [`crates/mlx-core/src/models/privacy_filter/`](../crates/mlx-core/src/models/privacy_filter/).
