# BGE native validation

Run from the repository root on Apple Silicon. The production addon needs no
Python; this directory contains the native acceptance harness. The pinned model,
datasets, environment and thresholds live in `fixtures/rerank/bge/manifest.json`.
`fixtures/rerank/bge/reference-lock.txt` preserves the historical Python environment used for the report.
Never compare different quantized checkpoints to establish implementation parity.

Python preparation, reference inference, mixed-fixture generation, dataset selection,
and report aggregation have been removed from the current tree, together with their
installation requirements. Their [archived tools and procedure](https://github.com/Saviio/mlx-node/blob/370a4a35ec75ea2bb7e28a4de7d14bd631a4712f/scripts/rerank/bge/README.md)
remain available at commit `370a4a35ec75ea2bb7e28a4de7d14bd631a4712f`.
Use that revision in a separate checkout when regenerating reference JSON, mixed
weights, or `quality-inputs.json`; use the current native harness below to evaluate
the current implementation. Keep checkpoint and fixture identities identical.

With the pinned FP32 checkpoint prepared at `.cache/bge-validation/fp32`:

```sh
yarn build
node scripts/rerank/bge/convert.mjs .cache/bge-validation/fp32 .cache/bge-validation
```

Conversion deliberately refuses an existing nonempty destination. Use a new
root when changing conversion settings. Keep enough free disk and memory for
the FP32 reference and the converted checkpoints.

## Independent numerical references

The archived Torch chain executes the official Transformers architecture in FP32. The
archived functional Python MLX chain reads tensors directly and implements XLM-R without
importing the addon, native loader or native inference. Dense projections use
explicit matmul followed by bias; GELU uses standard compiled `mlx.nn.gelu`.
Packed projection boundaries restore the activation dtype. These conventions
fix low-precision evaluation order; thresholds remain frozen in the manifest.

After generating the matching reference artifacts with the archived tools:

```sh
MLX_BGE_GOLDEN_MODEL="$PWD/.cache/bge-validation/fp16" \
MLX_BGE_GOLDEN_REFERENCE="$PWD/.cache/bge-validation/fp16-reference.json" \
  cargo test -p mlx-core --lib golden_checkpoint_logits -- --ignored --nocapture --test-threads=1
```

Repeat for FP32/Torch and BF16, affine, MXFP4/MLX. For reference generation in the archived checkout, use `--cases
fixtures/rerank/bge/long-cases.json` and set `MLX_BGE_GOLDEN_CASES` to the same
file for 512/2048/8192-token cases. The harness compares exact token IDs,
positions, masks, usage and truncation; records per-layer CLS differences; then
enforces both internal-logit and public-sigmoid thresholds and ranking margins.
Internal logits exist only in these offline reference files and test output.

Generate mixed weights and a quantized classifier using the archived procedure,
then place the checkpoint at `.cache/bge-validation/mixed` alongside its matching
independent reference.

Golden-test environment paths must be absolute because Cargo runs tests from the
crate directory. Run the same MLX/reference test for `mixed`. To verify the compiler-disabled
execution mode, set `MLX_DISABLE_COMPILE=1` in **both** independent processes
and use a separate reference output. This does not alter production defaults.

## Fixed-subset quality

Prepare `.cache/bge-validation/quality-inputs.json` with the archived dataset-selection
procedure and the current pinned manifest before running:

```sh
node scripts/rerank/bge/quality.mjs .cache/bge-validation/fp32 \
  .cache/bge-validation/quality-inputs.json .cache/bge-validation/quality-fp32.json
node scripts/rerank/bge/quality.mjs .cache/bge-validation/affine \
  .cache/bge-validation/quality-inputs.json .cache/bge-validation/quality-affine.json \
  .cache/bge-validation/quality-fp32.json
```

Repeat the last command for MXFP4. Each dataset contributes 256 queries sorted
by SHA-256, with original positive/negative candidates and recorded skips. Both
quantized formats must meet the nDCG@10 drop gate separately on each dataset.
Results describe this fixed subset, not a full MTEB benchmark. Per-query progress
is persisted; the same command resumes only when input, model and addon hashes
match. The recovery key includes the dataset and original sample index, so
repeated query text with different candidates remains distinct. A changed artifact
requires a new output path.

Quality acceptance pins the official FP32 file hashes in
[`fp32-checksums.json`](../../../fixtures/rerank/bge/fp32-checksums.json). It checks
actual quantization format and policy, binds the FP32 baseline report, addon,
Metal library and execution environment, and recomputes metrics from all candidate
indices. Schema-3 journals cannot resume older evidence missing that identity.
`MLX_DISABLE_COMPILE` must be absent. `quality-summary.mjs ROOT` verifies evidence
without importing the addon and leaves missing,
partial, wrong-revision or inconsistent results unaccepted.

Run the CPU evidence-contract regression tests with:

```sh
node --test scripts/rerank/bge/quality-contract.test.mjs
```

## Performance and regression

```sh
node scripts/rerank/bge/benchmark.mjs .cache/bge-validation/fp16 .cache/bge-validation/performance-fp16.json
```

Run performance jobs alone, without builds or other inference jobs. Each case
warms up three times and measures ten times. The required 512-token cases run
first for both batch sizes; a failed dense FP16 throughput target stops before
the longer matrix and leaves the report incomplete. Passing that precheck does
not replace the full matrix or resource checks. Completed reports set `complete`
only after all scenarios and checks finish; reports contain all samples,
median/P95, padded workload, active/cache/peak memory and cancellation cycles.
Each completed scenario is saved atomically. Repeating the same command resumes
only when the model, tokenizer, addon, Metal library, script, manifest and host
identity and execution-environment fingerprint match. Performance acceptance
requires compilation enabled (`MLX_DISABLE_COMPILE` must be absent, including an
empty value). Interrupted cancellation cycles restart; completed timing samples
retain their timestamps. The optional final argument `512` runs only the batching
acceptance scenarios and needs a separate output path.
Cold-model figures include loading and the first request; integrity hashing warms
the filesystem cache, so they are not cold-storage measurements. Do not infer
speed from memory reduction. Keep baseline-addon comparisons and
unloaded/idle/concurrent measurements separate from numerical tests.

Freeze a copy of the baseline native addon **before** building this branch, with
its adjacent `mlx.metallib` and `paged_attn.metallib`. `MLX_BGE_ADDON_PATH`
selects the exact binary in the regression harness after the TypeScript loader
initializes. Do not export `NAPI_RS_NATIVE_LIBRARY_PATH` to `oxnode`: that generic
override would also replace the compiler’s own native addon. Also prepare an independent archive of the
frozen baseline commit, build its TypeScript packages, and write the full baseline
SHA to `source-revision.txt` in that archive. Its workspace dependency links must
resolve inside the archive; only third-party dependencies may be shared. Pass that
root through `MLX_BGE_SOURCE_ROOT`. Switching the addon alone does not measure SDK
or Server regressions. The sampler verifies package resolution and records source,
built-code, lockfile, checkpoint, tokenizer and execution-environment fingerprints.
Both runs use the same sampling script and compare actual chat output token counts.
Use the same checkpoint paths throughout:

```sh
MLX_BGE_SOURCE_ROOT=/absolute/baseline-source \
MLX_BGE_ADDON_PATH=/absolute/baseline/mlx-core.darwin-arm64.node \
  yarn exec oxnode scripts/rerank/bge/regression.ts CHAT_MODEL JINA_MODEL baseline.json
MLX_BGE_ADDON_PATH=/absolute/current/mlx-core.darwin-arm64.node \
  yarn exec oxnode scripts/rerank/bge/regression.ts CHAT_MODEL JINA_MODEL current.json BGE_MODEL
node scripts/rerank/bge/compare-regression.mjs baseline.json current.json regression-comparison-initial.json
yarn exec oxnode scripts/rerank/bge/surfaces.ts BGE_MODEL surfaces.json
```

`regression.ts` collects three rounds; only `compare-regression.mjs` applies the
frozen regression gate and returns failure when all three rounds exceed it.
Idle residency and concurrent inference measurements are reported separately.
`benchmark.mjs` also returns failure when either required dense FP16 batching
scenario misses the throughput target; other precisions report measurements with
that target explicitly marked not applicable. The surface test uses real weights through SDK, HTTP and
CLI and checks identical normalized results and BGE-only service startup.

The checked-in compact report is historical evidence bound to its recorded
checkpoint/tokenizer, reference and addon SHA-256 hashes. Its Python aggregation
tool is retained in the archived revision only; do not relabel the old report as
validation of a new build. Missing or partial gates remain pending. Raw per-query
quality rankings, timings and stage references remain in the validation folder.

The report at `docs/validation/bge-rerank.md` distinguishes completed gates,
baseline failures, resource contention and any remaining acceptance work.

## Cancellation during forward evaluation

```sh
node scripts/rerank/bge/cancellation.mjs .cache/bge-validation/fp16 \
  .cache/bge-validation/cancellation-fp16.json
```

Repeat for BF16, affine and MXFP4. The harness waits until active allocations
exceed already-materialized resident weights and the bounded input-array size.
This observes forward activation allocation; it does not claim that a Metal
kernel has already started. It then cancels and awaits the native result, runs
a recovery request, and requires active memory to return to the prior baseline.
Eight cycles and simultaneous active/queued shutdown test completion ownership,
including repeated disposal. Explicit unload must return to the process's initial
active-memory level; allocator cache is reported separately. This complements
the benchmark's timed cancellation, which may interrupt tokenization.

Keep failed control runs when investigating a regression. The archived evidence generator
reads `regression-comparison-initial.json` and `regression-comparison-repeat.json`
under its root, together with the corresponding `regression-current.json` and
`regression-current-repeat.json` coexistence reports. A later passing run does not
resolve an earlier failure automatically. The comparator also requires post-close
active memory to match the baseline; allocator cache is excluded from that check.
Run its CPU negative tests with `node --test scripts/rerank/bge/compare-regression.test.mjs`.
