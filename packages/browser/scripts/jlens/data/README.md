# Evaluations

Six prompt distributions used to evaluate lens quality (§methods-comparison). Each `{slug}.json` is prompts only.

## Conventions

Unless a section says otherwise:

- **Lens readout** — at each (layer, token position) the Jacobian lens
  returns a ranked list of vocabulary tokens.
- **Workspace band** — the contiguous mid-network layer range where
  workspace content is read; experiments report over this band, not
  individual layers.
- **Hit** — a target token is a *hit* if it appears at lens rank 1 at any
  (layer, position) in the band over the scored span.
- **Swap** — clamping a lens coordinate replaces one token's direction with
  another's at every band layer at the specified positions, then samples
  the continuation.
- Prompts that span multiple turns are given as
  `[{"role": "user"|"assistant", "content": ...}]`.

## lens-eval-multihop

[`lens-eval-multihop.json`](lens-eval-multihop.json)

Lens-quality eval (§methods-comparison). `items[*]` has `prompt` and `intermediates`. `target` defines the readout position only and is not itself scored. Readout is at a single position — the token immediately preceding `target` — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-multilingual

[`lens-eval-multilingual.json`](lens-eval-multilingual.json)

Lens-quality eval (§methods-comparison). `items[*]` has `prompt` and `intermediates`. `target` defines the readout position only and is not itself scored. Readout is at a single position — the token immediately preceding `target` — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-poetry

[`lens-eval-poetry.json`](lens-eval-poetry.json)

Lens-quality eval (§methods-comparison). `items[*]` has `prompt` and `intermediates`. Readout is at a single position — the last newline token (end of line 1 of the couplet) — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-order-ops

[`lens-eval-order-ops.json`](lens-eval-order-ops.json)

Lens-quality eval (§methods-comparison). Each intermediate is a key expanded to a synonym set (numbers → digit and word forms; operations → symbol and word forms); rank is the min over single-token synonyms at each layer. `items[*]` has `prompt` and `intermediates`. `target` defines the readout position only and is not itself scored. Readout is at a single position — the token immediately preceding `target` — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-association

[`lens-eval-association.json`](lens-eval-association.json)

Lens-quality eval (§methods-comparison). Each item is a short vignette that evokes a single concept (grief, Einstein, noir, ...) without ever naming it; `intermediates` holds that one concept word. `items[*]` has `prompt` and `intermediates`. Readout is at a single position — the final prompt token — the closing period — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-typo

[`lens-eval-typo.json`](lens-eval-typo.json)

Lens-quality eval (§methods-comparison). Each prompt is a sentence ending in a common misspelling; `intermediates` holds the single correctly-spelled word. `items[*]` has `prompt` and `intermediates`. Readout is at a single position — the final prompt token, i.e. the last tokenizer fragment of the misspelling — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.
