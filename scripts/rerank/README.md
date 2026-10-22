# Native rerank validation

Run from the repository root on Apple Silicon with a built native addon and TypeScript packages. Checkpoints and reports belong in the ignored `.cache/rerank-validation/` directory. Do not run two GPU validation processes at the same time. Other active applications, thermal state, compiler work and filesystem cache affect timing; performance results are measurements, not an SLA.

## Pinned checkpoints and reference artifacts

`manifest.json` pins the official source repositories, community layout probes, reference environment and score thresholds. Download **those revisions**, including tokenizer, weights, projector, config and model card. For example:

```sh
hf download jinaai/jina-reranker-v3-mlx --revision 1d19fe38ae4e6658221479747c1152d6136dd6ab --local-dir .cache/rerank-validation/v3
hf download jinaai/jina-reranker-v3.5-mlx --revision 3dd4ac901ccdcac85abe3815df0a0aaaf44e4a21 --local-dir .cache/rerank-validation/v35
node scripts/rerank/convert.mjs
```

Conversion requires fresh output directories. The matrix is `v3`, `v35`, `v3-affine`, `v3-mxfp4`, `v35-affine`, `v35-mxfp4`. The two extra community probes in the manifest cover a merged quantized projector/embedding and a separate projector with dense embedding.

Python reference generation has been removed from the current tree. To regenerate the independent reference artifacts, use the [archived reference tools and procedure](https://github.com/Saviio/mlx-node/blob/370a4a35ec75ea2bb7e28a4de7d14bd631a4712f/scripts/rerank/README.md) from commit `370a4a35ec75ea2bb7e28a4de7d14bd631a4712f` in a separate checkout. The manifest and recorded results remain as historical provenance; removing the tools does not establish a new validation result.

The archived oracle runs the upstream full-sequence backbone, profile prompt, independent sequential block planner, and FP32 cosine/fusion. It uses the compatible v3.5 backbone with v3's all-global layout and pinned v3 prompt formatter for v3 references. Both reference and native runs must read the **same checkpoint**, including each quantized checkpoint. Do not use a BF16-to-4-bit comparison as evidence of quantized operator correctness.

With the matching `*-reference.json` artifacts prepared:

```sh
variant=v35-mxfp4
node scripts/rerank/compare.mjs ".cache/rerank-validation/$variant" \
  ".cache/rerank-validation/$variant-reference.json" \
  ".cache/rerank-validation/$variant-report.json"
MLX_RERANK_GOLDEN_MODEL="$PWD/.cache/rerank-validation/$variant" \
MLX_RERANK_GOLDEN_REFERENCE="$PWD/.cache/rerank-validation/$variant-reference.json" \
  cargo test --locked -p mlx-core --lib golden_checkpoint_features -- --ignored --test-threads=1
```

Cases cover multilingual input, reserved markers, duplicate/empty documents, sliding-window boundaries, explicit weighted multiblock fusion and 125/126 document boundaries. The native intermediate test checks exact token IDs, marker positions and projected-vector relative L2. The final score gate is frozen at maximum absolute error `1e-2`, mean absolute error `1e-3` **per case**, and zero inversions for reference pairs separated by more than `0.02`. Token totals and block counts must also match. An implementation failure must be diagnosed; do not loosen these tolerances to pass.

Hermetic native fixtures additionally exercise arbitrary marker IDs/dimensions, sharded and separate weights, missing tensors, shape/metadata conflicts, conversion round trips, zero-vector errors, full/chunked prefill (including chunk sizes on either side of a sliding window), truncation, candidate preservation, native cancellation and allocation release:

```sh
vp test __test__/rerank __test__/server/rerank.test.ts __test__/server/host/rerank-host.test.ts
```

## Performance and co-residency

```sh
node scripts/rerank/benchmark.mjs .cache/rerank-validation/v35-mxfp4 .cache/rerank-validation/v35-benchmark.json long
node_modules/.bin/oxnode scripts/rerank/coexistence.ts \
  .cache/models/qwen3-0.6b-mlx-bf16 .cache/rerank-validation/v35-mxfp4 \
  .cache/rerank-validation/coexistence.json
```

The benchmark records model load, 50/125/200 candidates, actual input/prompt tokens, block count, reset per-request peak memory, active/cached allocator memory, cancellation-to-stop time, repeated-request active allocations and deterministic disposal. `long` selects a longer passage length and crosses the default token block budget. Load times are fresh model loads with filesystem cache potentially warm.

The host test measures chat alone, chat with an idle resident reranker, and concurrent chat/rerank over HTTP; deterministic chat text must remain unchanged. This distinguishes unused capability overhead from GPU competition. It is a within-build control, not a statistical before/after performance comparison against `main`.

Run the repository's TS typecheck/lint, Rust fmt/clippy/tests and existing regression suite before merge. Save exact failures and distinguish missing fixture/environment failures from new regressions. The checked-in [report](../../docs/validation/rerank.md) summarizes this machine's results. Neither mock-only lifecycle tests nor a successful conversion replace same-checkpoint numerical validation.
