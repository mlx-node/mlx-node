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

## Limitations

- macOS only / Apple Silicon (Metal backend). No CUDA.
- bf16 weights and forward by default. The Metal banded-attention kernel and the bf16 forward can produce small disagreements vs. Hugging Face's fp32 reference at low-confidence boundary tokens. See the parity test fixtures at [`packages/privacy/__test__/parity-fixtures.json`](../packages/privacy/__test__/parity-fixtures.json) for the tolerated budget.
- Attention is bidirectional banded with attention sinks; `sliding_window = 128` on alternating layers per the gpt-oss config (band ±128 → 257-token effective window).

## Internals

The architecture, kernel design, Viterbi decoder, and tokenizer integration are documented in the design spec at [`docs/superpowers/specs/2026-05-13-privacy-filter-design.md`](superpowers/specs/2026-05-13-privacy-filter-design.md). The Rust implementation lives at [`crates/mlx-core/src/models/privacy_filter/`](../crates/mlx-core/src/models/privacy_filter/).
