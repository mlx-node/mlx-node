# Training

## GRPO (Group-based Relative Policy Optimization)

```typescript
import { loadModel } from '@mlx-node/lm';
import { GRPOTrainer, GRPOTrainerConfig } from '@mlx-node/trl';
```

Loss variants: **GRPO**, **DAPO**, **Dr.GRPO**, **BNPO** (selected via `loss_type` on `GRPOTrainerConfig`). Importance sampling can run at `token` or `sequence` granularity.

### Memory-budget knobs (large vocabularies)

For models with large vocabularies (Qwen3's 151,936 tokens), the lm-head and log-prob compute can dominate memory. Three chunking knobs trade compute for memory:

| Knob                 | Purpose                                            |
| -------------------- | -------------------------------------------------- |
| `forward_chunk_size` | Split prefill forward into chunks                  |
| `lm_head_chunk_size` | Stream lm-head logit compute in chunks             |
| `vocab_chunk_size`   | Chunked log-prob/entropy reduction over vocab axis |

## SFT (Supervised Fine-Tuning)

```typescript
import { SFTTrainer, SFTTrainerConfig } from '@mlx-node/trl';
```

Capabilities:

- Autograd (full automatic differentiation through the model)
- `gradient_accumulation_steps`
- Gradient clipping: `gradient_clip_norm` and `gradient_clip_value`
- `weight_decay`
- NaN detection: `max_nan_gradients`, `emergency_save_threshold`, `verbose_nan_detection`
- `label_smoothing` — cross-entropy refinement
- `compute_accuracy` — token-level accuracy metric (extra forward)
- `gradient_checkpointing` — reduces memory ~30% at the cost of a recompute pass
- Checkpoint resume

## Optimizers

`AdamW` (`crates/mlx-core/src/optimizers/`). With no optimizer configured, training applies plain SGD updates.

## Autograd

`crates/mlx-core/src/autograd.rs` exposes `value_and_grad` over MLX, integrated through the functional forward-pass architecture — stateless transformer components let MLX trace the computation graph from parameters to loss, so gradients for every trainable parameter (lm head, attention projections, MLP, embeddings, norms) are computed automatically.

## Datasets

`@mlx-node/trl` exports:

- `loadLocalGsm8kDataset` — preloads a JSONL gsm8k file
- `SFTDataset`, `createSFTDataset` — SFT-format dataset

## Training TUI

```bash
cargo build --release -p mlx-tui
```

Produces the `mlx-train` binary (Ratatui-based) — the standalone interactive training UI. `mlx-tui` is not part of the npm workspace; it talks to the same training engines through the Rust crates directly.

## Persistence

Training outputs flow through three layers:

- `crates/mlx-db` — SQLite persistence primitives
- `crates/mlx-core/src/output_store/` — training run / step / metric writes and reads
- `crates/mlx-core/src/response_store/` — model-response persistence (parallel store, used by the server)
