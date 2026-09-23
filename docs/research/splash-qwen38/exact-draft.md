# Exact Splash Q4 draft in mlx-node

Historical report, September 22, 2026. The imported packed-Q4 loader and its
research importer/tests were removed in the subsequent cleanup. These results
and validation counts describe frozen candidate 16, not the current runtime.
The current loader supports dense native companions; BF16 remains the default.
See the [cleanup status](README.md) and [final cleanup report](cleanup-final.md).

This follow-up used the Q4 DFlash2 tensors from the same
local Splash package used in [the earlier comparison](final-performance.md).
The requested mixed GGUF target remains
`Qwen3.8-27B-UD-Q4_K_XL.gguf`.

## Results

The exact Splash Q4 draft worked in candidate 16, but did not provide a general
speedup. It was slower on the short and 6K fixtures and faster on the 32K
fixture. The original BF16 checkpoint remained the default. The historical
loader selected the imported format through checkpoint metadata; that load
path is no longer supported.

Values are median [minimum–maximum] of three fresh processes per cell.
Decode rates are tokens/second; differences in the last column compare medians.
These ranges are observations, not confidence intervals.

| Input tokens |   Native BF16 draft | Native Splash Q4 draft |              Splash | Q4 vs BF16 |
| -----------: | ------------------: | ---------------------: | ------------------: | ---------: |
|           87 | 30.54 [28.62–33.14] |    27.99 [27.85–31.47] | 57.29 [54.09–61.72] |      -8.4% |
|        6,219 | 35.82 [33.42–38.47] |    26.66 [24.54–26.82] | 73.76 [67.96–75.49] |     -25.6% |
|       32,488 | 20.59 [20.34–20.95] |    24.17 [22.85–24.66] | 50.34 [49.23–50.78] |     +17.4% |

Paired Q4/BF16 decode changes (each round compared with its own control):

| Input tokens | Median paired change |     Paired range |
| -----------: | -------------------: | ---------------: |
|           87 |                -5.1% |   -8.4% to -2.7% |
|        6,219 |               -26.6% | -30.3% to -25.6% |
|       32,488 |               +18.9% |  +9.1% to +19.8% |

Full request time, in seconds:

| Input tokens |      Native BF16 draft | Native Splash Q4 draft |              Splash |
| -----------: | ---------------------: | ---------------------: | ------------------: |
|           87 |    33.69 [31.05–35.94] |    36.73 [32.69–36.91] | 17.99 [16.72–19.04] |
|        6,219 |    39.61 [36.99–43.34] |    49.13 [48.69–53.06] | 24.08 [22.97–26.72] |
|       32,488 | 111.71 [111.48–112.45] | 105.93 [100.89–107.57] | 78.55 [78.28–85.76] |

Reported TTFT, in seconds (different emission boundaries; see below):

| Input tokens |   Native BF16 draft | Native Splash Q4 draft |              Splash |
| -----------: | ------------------: | ---------------------: | ------------------: |
|           87 |    0.19 [0.19–0.19] |       0.18 [0.18–0.19] |    0.26 [0.25–0.26] |
|        6,219 | 11.05 [10.39–12.72] |    10.98 [10.30–11.37] |  10.27 [9.47–11.73] |
|       32,488 | 62.13 [61.78–62.86] |    62.79 [59.38–63.59] | 58.31 [58.22–65.06] |

With the imported draft, native decode reaches **48.8%, 36.1%, 48.0%**
of Splash's reported rate on the three prompts. It does not match Splash.
The remaining differences include target quantization, shared embedding/head,
state representation and execution; this comparison does not isolate them.

Native acceptance and cycle counts are deterministic across repeats:

| Input tokens | BF16 accepted draft tokens/cycle | Q4 accepted draft tokens/cycle | BF16 cycles | Q4 cycles | Splash cycles |
| -----------: | -------------------------------: | -----------------------------: | ----------: | --------: | ------------: |
|           87 |                            2.734 |                          2.387 |         274 |       302 |           274 |
|        6,219 |                            3.736 |                          2.311 |         216 |       309 |           206 |
|       32,488 |                            2.177 |                          2.641 |         322 |       281 |           303 |

Q4 needs more verification cycles on short/6K and fewer on 32K. On 32K, it
needs fewer cycles than Splash and still decodes substantially more slowly:
matching the draft storage alone does not remove the remaining execution cost.
No per-kernel timing or completed-answer quality claim follows from this test.

All measured outputs reach the 1,024-token limit while still reasoning.
Hashes are stable within each arm and fixture; output text differs between
arms. The six unchanged BF16/old-runtime lifecycle cases and all fresh BF16
benchmark transcripts retain their earlier hashes.

Native peak process RSS is 21.01–21.09 GB with BF16
versus 18.40–18.50 GB with Q4. The imported file is
1.27 GB versus 3.85 GB for BF16. This is a useful memory reduction, but the
speed results do not justify replacing BF16 as a general default. Splash RSS
is not compared here because its GPU/mapped memory accounting differs.

The final set has 27 accepted observations and 832 resource samples.
5 interrupted group attempts were excluded solely for observed external compilation.
Minimum reported available system memory was 65%.
Raw rows, ranges, identities, source checksums and exclusions are in
`.cache/benchmarks/splash-qwen38-exact-draft/summary.json`.

## Measurement protocol

The three arms are the same native build with the original BF16 draft, that
build with the imported Splash Q4 draft, and the local Splash runtime/package.
Each arm runs three fresh processes per prompt: 87, 6,219 and 32,488 input
tokens, with exactly 1,024 generated tokens, greedy decoding, high reasoning,
fixed depth seven, a 16-token warmup, and zero cached prompt tokens. Native
caches are reset; Splash warms a distinct prompt. Model loading is excluded.
Splash's rendered input token IDs are checked against the native tokenizer.
Every arm occupies each execution-order position once for each prompt.

The resource guard samples every two seconds. Recognized external model jobs
and Rust/Clang/linker processes invalidate an entire three-arm group, which is
then repeated without selecting by speed. After repeated bursts of external
Rust compilation, a scheduling gate was added: if an interference discard
occurred within five minutes, wait for a full minute without recognized model
or native compiler activity before loading again. Original attempts and the
guard amendment are preserved. This waiting time is outside model timing.

Unrelated JavaScript/TypeScript builds, tests, and desktop CPU work remain in
the logs. GPU clocks are not controlled; a power check during the cohort shows
AC power. These are active-workstation observations, not an isolated hardware
ceiling. Fresh BF16 controls are used instead of attributing differences from
the older session to this loader change.

Decode metrics also have different boundaries. Native decode excludes one
initial token and includes final state settlement. Splash's reported stream
decode excludes its first emitted batch (eight, six or seven tokens in these
three fixtures); it can finish with a terminal anchor not materialized as a
KV row. Full request time is therefore reported alongside decode rate. The
native request is a Node API call; Splash is a local nonstreaming HTTP call.
Reported TTFT is internal, not client-observed streaming latency. Aggregate
time divided by cycle count is not an isolated GPU kernel measurement.

## What is shared

Splash's six draft binaries are repacked into an MLX safetensors container.
This is a lossless storage permutation: the Q4 integer codes and BF16 scales,
biases, norms, convolution kernels and selector codebooks retain their exact
bits. There is no dequantization or requantization. The fused QKV matrix is
split by output row. All 37 packed source sections pass an inverse-permutation
byte comparison, and all 175 output tensors have recorded payload checksums.
The 34 unquantized tensors also match their original BF16 checkpoint
counterparts byte-for-byte, including norms and selector codebooks.

The candidate-16 native loader accepted explicit affine Q4/group-64 checkpoint metadata and
validated the complete tensor inventory before materializing weights. It used
the stored triples directly, including the selector projection. Checkpoint
metadata selected the format; no new runtime switch was introduced. Existing
BF16 checkpoints retained their load path.

This does not make the two complete model packages identical. DFlash2 shares
its token embedding and output head with the target model. mlx-node still
uses its GGUF target's shared weights; Splash uses its packed target weights.
The target quantization, KV representation, kernels and accumulation behavior
also differ. Matching draft bytes does not imply matching intermediate states
or output text across engines.

## Archived import experiment

The removed importer was scoped to the audited Qwen3.8 package schema. It
checked manifest schema, architecture, file headers, source checksums, section
geometry and exact file sizes; it refused to overwrite an output directory.
Those fixed dimensions described the checkpoint format, not a particular GPU.
The importer and its tests are no longer runnable repository tools, and the
current benchmark is not an imported-checkpoint reproduction recipe.

The generated `imported-draft/`, `candidate-16/` addon, raw validation records,
runner/configuration records and payload manifest remain under
`.cache/benchmarks/splash-qwen38-exact-draft/`. The pre-cleanup benchmark runner
is also retained as
`.cache/benchmarks/splash-qwen38-phase8/benchmark-before.ts`. Reproducing the
historical load path requires the archived binary and its matching artifacts.

## Checkpoint identity

- Splash source: `7e3c67e8e3a9e9912ff6e02521457017cc4c65d0`.
- Package snapshot: `9d27070b71f7142c6b6025f03ac011d70a73cb48`.
- Draft upstream: `incoai/Qwen3.8-27B-DFlash2`, revision
  `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.
- Manifest SHA-256:
  `40121d0d933fb27a7206248a592451f9ebbd73671f19ce3162cf27a7db0968d5`.
- Imported safetensors: 1,265,644,576 bytes, 175 tensors, 47 Q4 projections.
- Imported SHA-256:
  `b52043b7621751e40aadf8e8221611f60cc416da6460055a507a5f515f5a080b`.

The imported directory contains `splash-import.json` with all source file and
output tensor checksums. Raw checks, frozen binaries, resource logs and
benchmark records are retained under
`.cache/benchmarks/splash-qwen38-exact-draft/`.

## Historical candidate-16 validation

- Canonical `vp run build:native`, release Clippy with warnings denied, Rust
  formatting, and the two changed TypeScript runners' type checks pass.
- Five importer regressions pass, including an independent scalar address
  oracle across multiple storage tiles/groups, every nibble position, BF16
  sidecars, QKV splitting, metadata drift, bad headers and truncated files.
- The actual converted safetensors file was independently read back: all 175
  payload hashes, shapes and dtypes match its audit manifest.
- Release native tests: **3,828 passed, 0 failed, 129 ignored**. Three unchanged
  tests requiring debug assertions were filtered from this release run; their
  prior debug checks are retained in the cleanup artifacts. All four new
  packed-checkpoint tests pass, including exact hand-packed projection output
  and selector precision under the legacy dense-load settings.
- Both BF16 and imported Q4 pass cold generation, cached continuation, cold
  conversation replay, streaming, sampled decoding, reset, and continuation
  beyond the draft's 2,048-token window. The final build matches all six
  deterministic scenarios from the preliminary build for each checkpoint.
  BF16 also matches the earlier runtime's golden transcripts.
- The BF16 cache fixture uses 321 generated tokens; Q4 uses 319 because its
  changed transcript ends in whitespace at 320/321. The chat template trims
  that whitespace. The positive full-prefix reuse assertion is retained,
  yielding 350 cached tokens for BF16 and 348 for Q4. Both long continuations
  reuse 6,251 tokens. These fixture-selection attempts and their failures are
  preserved separately and contribute no performance observations.
- Independent source review covered the importer, native loader and benchmark
  protocol. Manifest/RoPE validation, arm ordering, unique-repeat checks and
  compiler-overlap detection were corrected before timed measurements.

The final addon SHA-256 is
`32d00b023b337852cd36e976c407cebef90ad17eff65a75d4f32ab86cc581316`.
Both GPU libraries are byte-identical to the previous runtime; this change is
in checkpoint import/loading. All 37 retained native source identities match
the frozen build.
