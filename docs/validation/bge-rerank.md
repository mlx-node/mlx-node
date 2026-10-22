# BGE-v2-M3 validation

This report accompanies the `feat/rerank` implementation based on `1fc3baee`.
The machine is an Apple M4 (Mac16,12), 32 GiB unified memory. No merge or release
is authorized by a successful build alone. Failed or incomplete gates below are
not accepted support claims.

The [manifest](../../fixtures/rerank/bge/manifest.json) freezes the checkpoint,
dataset revisions, reference packages and thresholds. The
[reproduction guide](../../scripts/rerank/bge/README.md) describes each command.
The [compact evidence file](bge-results.json) records model/tokenizer, reference and addon hashes;
full stage tensors, rankings and raw timing samples stay in `.cache/bge-validation`.

## Numerical correctness

| Storage / reference                                 | Short inputs | 512 / 2048 / 8192 tokens | Maximum logit error |
| --------------------------------------------------- | ------------ | ------------------------ | ------------------- |
| FP32 / official Transformers                        | Pass         | Pass                     | `4.04715538e-5`     |
| FP16 / same-checkpoint Python MLX                   | Pass         | Pass                     | `0`                 |
| BF16 / same-checkpoint Python MLX                   | Pass         | Pass                     | `0`                 |
| affine 4-bit, group 64 / same-checkpoint Python MLX | Pass         | Pass                     | `0`                 |
| MXFP4, group 32 / same-checkpoint Python MLX        | Pass         | Pass                     | `0`                 |

FP32's largest case mean logit error is `2.07424164e-5`; its largest public-score
error is `8.50104225e-6`. Every frozen maximum/mean threshold and margin-qualified
ranking assertion passed. Tokens, masks, position IDs, truncation and usage match
exactly. Coverage includes Chinese/English, empty candidates/documents, duplicates,
special-token text and literal pad tokens. A mixed checkpoint with FP32 affine
sidecars and a quantized classifier also passes; FP16 with compilation disabled
passes against the independently disabled reference.

The Python MLX implementation loads safetensors directly, without the native
loader or inference. It uses standard `mlx.nn.gelu`; dense projections use explicit
matmul then bias, matching the documented native projection rounding boundary.
The official Transformers FP32 chain independently checks the architecture.
An initially observed low-precision GELU difference was fixed with a model-owned
compiled exact GELU expression using the existing compiled-graph API. No shared
activation, chat transformer or Metal kernel changed. Thresholds were not widened.

Batch sizes 1 and 8 each match their same-policy reference exactly. Changing the
batch shape has ordinary low-precision rounding effects in **both** independent
implementations: maximum FP16/BF16 logit changes are `0.00390625` / `0.0625`, and
public-score changes are `4.4663e-6` / `1.8011e-4`. No margin-qualified inversion
occurred in the frozen batch fixture; reversing candidate order produced zero
logit change after restoring indices. This is not a claim of bitwise invariance
across batch policies.

All eight existing Jina fixed-checkpoint golden tests pass against their unchanged
references, including community affine/MXFP4 layouts. Jina remains cosine-scored.

## Quantization quality

The final schema-3 runs completed all 256 queries in each subset for FP32,
affine 4-bit and MXFP4. Offline verification checked checkpoint, addon, Metal,
execution-environment, input and baseline-report identity, and recomputed metrics
from every complete candidate ranking. Both 4-bit formats pass the frozen
maximum `0.02` nDCG@10 drop on **each** dataset.

| Subset      | Format       |  nDCG@10 |   MRR@10 | nDCG drop vs FP32 | MRR change vs FP32 | Changed rankings |
| ----------- | ------------ | -------: | -------: | ----------------: | -----------------: | ---------------: |
| T2Reranking | FP32         | 0.712752 | 0.748692 |                 — |                  — |                — |
| T2Reranking | affine 4-bit | 0.711401 | 0.747394 |          0.001351 |          −0.001297 |          253/256 |
| T2Reranking | MXFP4        | 0.718310 | 0.751376 |         −0.005558 |          +0.002685 |          254/256 |
| AskUbuntu   | FP32         | 0.707763 | 0.794361 |                 — |                  — |                — |
| AskUbuntu   | affine 4-bit | 0.692540 | 0.774141 |          0.015223 |          −0.020219 |          256/256 |
| AskUbuntu   | MXFP4        | 0.697801 | 0.778551 |          0.009962 |          −0.015809 |          256/256 |

A negative nDCG drop means a higher score on this fixed subset; it does not
establish a general improvement. Changed rankings count any difference in the
full candidate order. MRR changes are reported independently: passing the frozen
nDCG gate does not mean quantization preserves every metric or candidate order.

T2Reranking and AskUbuntu each use 256 queries ordered by query
SHA-256 with all original candidates retained. The selected subsets contain
4,131 and 5,120 candidates respectively. T2 selection records eight rejected
samples; AskUbuntu records none. Inputs and labels are hashed, and baseline
comparison rejects any input-identity mismatch before loading a model.

An earlier schema-2 FP32 run lacked complete compilation-environment identity
for its first 233 samples. It remains historical evidence only; no missing
history was inferred or reused in these final runs. These are fixed-subset
quality results, not full MTEB scores or a substitute for the performance gates.

## Lifecycle, interfaces and regressions

Completed checks include family dispatch and policy rejection, native-free imports,
sigmoid-only SDK/HTTP contracts, the host residency matrix for chat/ASR/TTS plus
Jina/BGE, and synthetic native loading/conversion/cancellation cases. The complete
pure-TS suite passed: 2,115 tests in 108 files, run serially after parallel timing
failures in existing supervisor/coordinator tests. TypeScript and clippy pass.

The final Rust core run passed 3,995 tests with zero failures and 161 ignored.
An expected conversion-registry list was updated for the new family. The existing
Qwen3.5 segmented-attention calibration, which had exceeded its 100 ms timing
bound under concurrent load, passed in the quiet final run without changing its
threshold. Native release build, Metal packaging checks and final clippy pass.
The remaining Rust integration binaries, other-crate unit tests and doc tests
also pass. The full addon lane initially passed 1,773 tests with two failures in
expected family/native-free path lists. Updating those lists for BGE and explicitly
tracking the rerank addon root made both affected files pass all 27 tests. This
was a test-contract correction; it did not change the production implementation.

Real FP16 SDK, HTTP and CLI responses are identical, expose only sigmoid scores,
and serve successfully with only a BGE model discovered. The chat endpoint rejects
that model with `400/model_capability_mismatch` before loading it as chat.
FP16/BF16/affine/MXFP4 each pass eight cancellation/recovery cycles after observing
forward activation allocation, active-plus-queued shutdown, and repeated disposal.
Every recovery returns active memory to its prior baseline; final disposal returns
to zero active bytes. Allocation observation does not prove a GPU kernel had
started; the test separately awaits the native task and disposal to completion.

Independent pair audit found and corrected incomplete-download revision provenance,
quality-baseline input identity and performance-script failure signaling. Partial
download markers invalidate older config/Hugging Face revision hints; unknown
provenance remains null. Explicit document policies exceeding effective context
are rejected. A later audit also closed unsupported quantization declarations on
dense modules and strengthened full-source baseline/input identity checks. The
AskUbuntu subset contains 256 rows but 255 unique query texts; recovery therefore
uses sample indices. Historical rows were backed up and checked before resuming an earlier run;
that legacy run is excluded from the final schema-3 acceptance evidence. A final evidence audit tightened FP32 baseline, actual quantization format,
required policy, dataset revision and complete-sample validation. Reports bind
checkpoint, addon, Metal, execution-environment and baseline-report hashes;
metrics are recomputed from complete candidate rankings. CPU negative tests
reject a quantized self-reference, a wrong format, missing policy, duplicate
datasets and changed execution identity. Audit is evidence of review, not a
guarantee of no defects.

## Performance and acceptance

**FP16 batching throughput gate failed on this M4.** All six 512-token precheck
cases completed with the frozen three warmups and ten measurements. The required
125/200-candidate results are:

| Candidates | Batch 1 median | Batch 8 median | Throughput change | Required |
| ---------: | -------------: | -------------: | ----------------: | -------: |
|        125 |   22,819.04 ms |   22,799.62 ms |           +0.085% |     ≥20% |
|        200 |   36,571.23 ms |   36,943.53 ms |           −1.008% |     ≥20% |

The precheck deliberately leaves `complete: false` and stops before the longer
performance matrix. Completed measurements, timestamps and artifact identity are
retained. Neither the threshold nor batch-1 behavior was changed to obtain a pass.
The 125/200-candidate default runs actually executed 16/25 batches, with zero
padding; the result is not caused by silently falling back to one document per batch.
An exploratory Python MLX layer profile did not establish a sufficient benefit
from QKV concatenation or whole-layer compilation; it is not an acceptance
benchmark or a substitute for measuring a changed native implementation.
An independent code review found no demonstrated implementation defect explaining
this result. Operator-shape and hardware-utilization limits remain hypotheses;
the profile does not include hardware counters or native full-model attribution.

Performance runs execute without simultaneous builds or other
inference jobs. The frozen matrix uses three warmups and ten measurements, with
median/P95, padding, active/cache/peak memory, cancellation and unload checks.
Required FP16 125/200-candidate, approximately 512-token cases must improve
throughput by at least 20% with default batching.

Separate full-source baseline/current runs use the same checkpoints and actual
chat output-token counts. The initial Jina comparison failed the three-round 5%
rule (+31.65%, +47.76%, +35.76%). Repeating both runs without changing production
code passed that rule: Jina changes were −1.04%, −0.08%, +4.46%. Chat passed both
attempts. Both reports are retained. **The first Jina regression was not stably
reproduced, and its cause remains undetermined**; machine-state variation is a
hypothesis, not a demonstrated explanation. This is an unresolved acceptance item.

BGE unloaded, idle resident and concurrent inference are measured separately.
In the initial coexistence run, chat median latency was 117.76 ms with idle BGE
and 426.30 ms during concurrent BGE inference; standalone BGE median was 1,193.12 ms
for the frozen 50-passage workload. These are contention measurements, not a
claim of zero overhead. Both original and repeat host shutdowns return to the
same 1,192,099,896 active bytes observed after baseline chat/Jina shutdown; the
existing chat allocation remains in both implementations, while BGE adds no
retained active allocation. Standalone BGE disposal returns to zero as above.

The batching target is unmet, the initial Jina performance failure is unresolved,
and the remaining matrix is incomplete. This branch is **not ready for merge**.
