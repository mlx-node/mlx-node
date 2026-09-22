# oMLX M3 Pro comparison, 2026-09-22

The reported mlx-node test-app result is 89.1 prefill tokens/s, 10.1 decode
tokens/s and 16.49 s TTFT, with approximately 1.5k new and 4.6k cached tokens.
It is a real cached coding-agent continuation, not a cold benchmark prompt.
GPU core count, current app revision, MTP acceptance and thermal state were
not supplied with this result.

## Verified community entries

Filtered the live oMLX site by `device=["M3","Pro"]`, 36 GB RAM and
`Qwen3.8-27B`. The `chip` filter only accepts a generation such as `M3`;
using `chip=M3 Pro` misleadingly produces no matches. The site hides
SpecPrefill by default but still includes MTP results. Selected entries:

| Checkpoint name | Context | PP tok/s | TG tok/s | Settings | Source |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B-4bit | 4096 | 99.6 | 7.7 | MTP off, ANE off, Code (Python), oMLX 0.6.2 | [oszqozon](https://omlx.ai/benchmarks/performance/oszqozon) |
| Qwen3.8-27B-oQ4e-fp16-mtp | 4096 | 108.2 | 8.2 | MTP off, ANE off, Code (Mixed), oMLX 0.6.3rc2 | [yeoz4lwo](https://omlx.ai/benchmarks/performance/yeoz4lwo) |
| Qwen3.8-27B-oQ4e-fp16-mtp | 4096 | 106.7 | 16.2 | Lightning MTP on, thinking off, Code (Python), oMLX 0.6.0 | [d4wsz7gd](https://omlx.ai/benchmarks/performance/d4wsz7gd) |

All three identify an 18-core GPU. The second reports 37.872 s TTFT,
17.6 GB peak MLX memory, nominal thermals and 99.9% average GPU load.
The third reports 38.373 s TTFT and 20.5 GB peak MLX memory.
These are separate community submissions, not a controlled MTP A/B test.
Their batching tables are aggregate multi-request throughput and are not
single-request decode speed.

The first page sorted by descending PP tops out at 108.2 tok/s. Filtering
by the exact substring `Qwen3.8-27B-UD-Q4_K_XL` on M3 Pro 36 GB returns
no matching entries. The community page therefore establishes useful
same-model-family targets, not an exact-checkpoint comparison.

89.1 is 10.5% below 99.6 and 17.7% below 108.2. Closing those raw gaps
would require 11.8% and 21.4% higher throughput, respectively. Differences
in quantization, prompt/cache geometry, sampling and GPU core count prevent
attributing those gaps entirely to the runtime. The 77 tok/s earlier report
and this 89.1 result are not paired measurements of test-app speedup.

## Implementation inspection

Local oMLX source inspected at `e467261edc786efd33b1e9023d5c4a827f8aa1c1`
(2026-09-03), not claimed to be today's upstream HEAD:

- `omlx/admin/benchmark.py` disables cache reuse for single-request benchmarks
  and pins prefill speed priority; its community PP rate is a cold-prompt metric.
- `omlx/patches/qwen35_fa256_attention.py` provides portable D256 attention.
  mlx-node test3 already includes its own bounded D256 SIMD implementation.
- `omlx/patches/qwen35_q4_mlp.py` uses shape-specific native affine QMM tiles.
  Those affine encodings differ from native GGUF K/IQ blocks; replacing the
  checkpoint with an affine quant is not a runtime-only optimization.
- `omlx/custom_kernels/qwen35_prefill/gdn.py` stages GDN inputs into shared
  memory and changes the contraction lane grouping. Its state contract is
  FP32; mlx-node's existing Qwen3.5 path must retain its own state and head
  mapping semantics when testing an optimization.

A local staged-input experiment retained mlx-node's existing 32-lane,
two-value-row recurrence, head mapping and output/state dtypes. It matched
both outputs and carried state exactly across 48 exploratory matrix cases
(Hk=16, Hv=32, Dk=Dv=128; the actual dense model has Hv=48), but did not
show a speedup on the available M5. It is not enabled in the runtime and
cannot establish how the M3 would perform.

The real UD-Q4_K_XL cache has Q3_K/Q4_K/Q5_K/Q6_K, IQ4_XS/IQ4_NL/IQ3_S,
and affine-8 projections. Test3's portable path covers Q4_K/Q5_K/Q6_K/IQ4_XS;
the other formats require separate measurement and qualification. No full
BF16 weight copy or lower-precision checkpoint conversion is introduced.

## Measured bottleneck and rejected changes

`omlx-followup-profile.json` records an instrumented run of the same 17.56 GB
GGUF: a 6,183-token cold prompt plus a cached continuation, 64 greedy tokens
per turn. The available M5 executed the pre-NAX kernels; it does not emulate
M3 performance. Evaluation was forced around stage and projection boundaries,
so the numbers include GPU execution rather than only graph construction.
Those extra barriers affect scheduling; these are diagnostic attribution
numbers, not ordinary uninstrumented throughput.

The measured layer stages on chunks of at least 64 tokens totalled 35,126.9 ms. Quantized projections account
for 32,323.5 ms (92.0%). The MLP stage accounts for 65.9% of stage time, with
another 23.9% in the GDN block (including its projections) and 10.3% in paged
attention (also including projections). Q3_K/IQ4_NL/IQ3_S together take only
3.4% of the layer-stage time. This did not support treating either the
recurrence or the rare quant formats as the main remaining bottleneck.

`diagnostic-prefill-profile.patch` contains the temporary measurement hooks.
It is deliberately not enabled in production source. Apply it to this commit,
rebuild the native addon, and set `MLX_PREFILL_PROFILE=1` only for diagnosis.
Its per-op barriers make it inappropriate for measuring normal throughput.

A six-tile sweep against test3's 64x64x32 QMM tile also failed to establish a
consistent gain on the local device. No new tile or staged-GDN default is
shipped from these experiments. The M3 throughput target remains unverified.

## Next measurement on the actual M3

The `m3-check/` kit runs with the already signed test3 app and bundled Node;
it needs neither developer tools nor remote access. It validates the exact
GGUF fingerprint, starts one model process at a time, retains native automatic
memory limits, and compares portable QMM on/off in AR before a separate MTP
run. It includes one excluded short warmup, a cold public code-review prompt,
and a cached follow-up of about 1.5k new tokens, with 64 greedy output tokens
and medium reasoning. It reports app/native hashes, RAM, GPU core count,
thermal/swap snapshots, effective memory budgets, cache hits, performance and
output hashes. The three single trials diagnose regressions; they are not a
statistically stable cross-library benchmark. No private conversation is read
and no results are uploaded automatically.

The complete kit was exercised using the packaged, signed test3 app on the
local M5: all three child processes exited successfully, every measured turn
produced 64 tokens, cold cache hits were zero and every continuation reused
6,244 tokens while consuming 1,443 new tokens. Four portable modes logged
activation in the enabled arms and none in the rollback arm. The M5 rollback
arm uses NAX while the forced portable arm uses SIMD, so its output hashes
are not expected to establish the M3's same-backend AR equality. Local timings
were highly variable and are not reported as improvements. The shell and
JavaScript syntax, fixture fingerprint, checkpoint fingerprint, and macOS
picker scripts were also validated.
