# mlx.fast optimization for the Qwen dense agent path

## Scope and conclusion

This evaluation targets `qwen3.8-27b-mxfp4-mlx` through mlx-node's Qwen3.5
dense runtime. The workload is a recorded long coding-agent context followed
by resident continuations and a separate process restoring an SSD prefix.
The retained optimization submits the first four completed decode layers to
Metal asynchronously while the host constructs later layers. Across twelve
paired resident turns it improved geometric mean throughput from turn wall
time by **2.50%**: **2.33%** after full prefill and **2.67%** after a verified
SSD prefix restore. Eleven pairs improved; one regressed. Every generated
output and MTP acceptance count matched its control. Peak active allocation
was essentially unchanged.

This is a modest improvement to single-agent continuation, not a prefill or
SSD I/O speed claim. The model's math, PagedAttention layout, recurrent-sidecar
format and cache publication rules are unchanged. Other candidates either
failed numerical parity or did not establish a repeatable agent benefit and
were removed. Final native validation and the real CLI task are recorded below.

Measurements used baseline `0413d14d7d538774a5ed2f8a396eccf3c5fa8b38`, with MLX
at `6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb`. Before publication, the PR was
rebased onto `main` to exclude two unrelated desktop commits; the native
inference and agent source was unchanged by that rebase. Measurements use an M5 Max with
128 GiB unified memory and the external P4510 volume for models and isolated
SSD caches. Inference runs are serialized and do not overlap this work's
native compilation. Desktop background activity is not controlled, so
comparisons use adjacent controls and repeated runs.

## What transfers from mlx.fast

The current public challenge targets Gemma 4 26B A4B and eight concurrent
streams. Its score combines prefill and decode gains with exponents 0.25 and
0.75. The local iteration contract uses 1,024 seed tokens and 128 checked
decode steps per stream. These conditions differ substantially from one
Qwen agent with a growing 63K context. The leaderboard is evidence that its
specific workload improved; it is not an estimate for this checkpoint.
[Challenge and scoring](https://www.yukon.org/mlxfast),
[pinned benchmark contract](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine/blob/27c821c466c9799e87162e9436618863b7d0a0ba/README.md#L381-L434).

The supplied engine contains useful techniques beyond its active Gemma path.
In particular, its Qwen GatedDelta kernel maps a value head directly to a
compact key head with `hv / (Hv / Hk)`. Its pointers advance by the compact
key-head stride. mlx-node already has compact access for GGUF's tiled
ordering, while standard checkpoints expand each key head consecutively.
[mlx.fast GatedDelta implementation](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine/blob/27c821c466c9799e87162e9436618863b7d0a0ba/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/GatedDelta.swift#L25-L41).

The local vLLM checkout at
`8e1f97e70984192cc63c51a8af3e313cab8c5c73` independently uses the same grouped
head mapping in its FLA recurrent kernel. This is a compute-layout reference;
it does not require adopting a separate RAM cache tier on a unified-memory
machine. [vLLM FLA recurrent kernel](https://github.com/vllm-project/vllm/blob/8e1f97e70984192cc63c51a8af3e313cab8c5c73/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py#L63-L64).

| Technique                                         | Relevance to this agent                                                             | Evaluation                                                                                         |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Compact grouped GDN heads                         | Standard Qwen expands 16 Q/K heads to 48 value heads in each of 48 recurrent layers | Exact outputs; reverse-order agent comparison was neutral, removed                                 |
| Joined gate/up storage and projection             | Dense quantized FFNs perform separate projections                                   | Tested on the long replay; no repeatable agent benefit, removed                                    |
| Activation fusion with explicit BF16 rounding     | Reduces intermediate arrays in FFN and GDN gates                                    | Exact multiply-only fusion tested; no repeatable agent benefit, removed                            |
| Fold GDN scaling into stock RMSNorm               | Removes two pointwise operations per recurrent layer                                | Initially promising, reversed on repeat; removed                                                   |
| Dequantize to BF16 then GEMM                      | Could improve large prefill matrices                                                | Matrix-dependent results, extra temporary memory, some numerical differences; no production change |
| Expert sorting and routed-expert reductions       | Gemma's batch-eight MoE path                                                        | Outside the requested dense model                                                                  |
| Gemma-specific attention and cache specialization | Different head geometry, cache layout and batching                                  | Preserve the paged hybrid executor                                                                 |
| Increase prefill chunk size                       | Can amortize launches but greatly increases live intermediates                      | An 8192-token experiment failed; keep the production 2048-token setting                            |
| Matrix-based chunked GDN                          | Existing CUDA ops implementation also executes on Metal                             | About 15–17 ms versus 3–5 ms per 2048-token call; no Metal routing change                          |

The retained early-submission technique overlaps eager paged decode graph
construction with GPU execution. Its twelve paired agent continuations,
including SSD restart, are detailed below.

The projection experiment was motivated by
[Gemma's joined dense gate/up storage](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine/blob/27c821c466c9799e87162e9436618863b7d0a0ba/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift#L7954-L7983).
The fusion experiments followed its explicit intermediate-rounding discipline,
without transplanting Gemma's fixed-width shader geometry.
[Gemma prefill fusion notes](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine/blob/27c821c466c9799e87162e9436618863b7d0a0ba/Vendor/mlx-swift-lm/Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4PrefillGlueV1.swift#L29-L46).

## Checkpoint identity and recovery

The model is present. The old name
`qwen3.8-27b-unsloth-mxfp4-mlx` is a directory symlink to
`qwen3.8-27b-mxfp4-mlx`. Agent discovery checks directory entries with
`isDirectory()`, which skips that symlink. Use the canonical model name; no
reconversion or duplicate weight copy is needed.
[Agent model discovery](../../../packages/agent/src/provider/models.ts).

Local validation found all five shards and their index, containing 1,600
tensors and 23,277,610,464 logical tensor bytes. The mixed recipe includes
168 MXFP4 modules, 233 MXFP8 modules, BF16 vision tensors and the intact BF16
inline MTP layer. The benchmark uses these existing weights unchanged.
The published checkpoint is
[Brooooooklyn/Qwen3.8-27B-MXFP4-mlx](https://huggingface.co/Brooooooklyn/Qwen3.8-27B-MXFP4-mlx).

If rebuilding is ever necessary, the current converter takes the BF16 source
directory with `--model-type qwen3_5 --dtype bfloat16 --quantize --q-recipe
unsloth --q-mxfp`, plus explicit input and output directories. Omit `--q-mtp`
to preserve the inline MTP layer in BF16. The present source directory
`qwen3.8-27b` also exists on the external model volume.
[Conversion options](../../../packages/cli/src/commands/convert.ts).

The dense target has 64 layers: 16 full-attention layers and 48 GDN layers.
Hidden width is 5,120 and intermediate FFN width is 17,408. Full attention
uses 24 query heads and four K/V heads of width 256. GDN uses 16 key heads
and 48 value heads, each of width 128. This 3:1 GDN ratio is the opportunity
addressed by the compact-head change.

## Compact-head experiment and architecture boundaries

The following describes the tested, subsequently removed compact-head
candidate. It is retained here to explain the experiment and its correctness
requirements, rather than to imply a production speedup.

For standard checkpoints, value heads 0, 1 and 2 read key head 0; heads 3, 4
and 5 read key head 1. For tiled GGUF ordering, value head `hv` continues to
read `hv % Hk`. The kernel specializes this choice at compilation. The
existing FFI entrypoint retains tiled semantics, and a grouped entrypoint
shares its dispatch and implementation.

Q/K retain shape `[B, T, 16, 128]` instead of being copied to
`[B, T, 48, 128]`. At a 2048-token BF16 chunk, the two expanded outputs total
48 MiB per layer; the compact inputs total 16 MiB. Eliminating the repeat
operators avoids producing those 48 MiB of expanded outputs. This is
intermediate traffic, not a promise that process peak memory falls by the
sum over all layers. MLX evaluates lazily and reuses allocator storage.

The arithmetic and recurrent state shape remain unchanged. In particular,
the recurrent state still has one matrix per value head. Ops fallbacks and
the optional chunked implementation expand heads in their required order.
The differentiable training and non-Metal paths keep their existing behavior.
[GDN dispatch](../../../crates/mlx-core/src/models/qwen3_5/gated_delta.rs),
[Metal dispatch](../../../crates/mlx-sys/src/mlx_gated_delta.cpp).

MTP verification records the compact tensors and their head ordering. Tape
batching, owner-row selection and accepted-prefix replay retain that ordering.
Replay still rounds the recurrent state through BF16 after every accepted
step, matching autoregressive state evolution rather than a single window's
FP32 accumulation. No draft-cache seed or speculative scheduling rule changes.
[MTP layer tape](../../../crates/mlx-core/src/models/qwen3_5/gated_delta_net.rs).

PagedAttention's K/V slot layout, block hash identity and scheduler ownership
remain unchanged. GDN SSD sidecars still serialize convolution and recurrent
state at a validated token boundary. Q/K projection intermediates are not
part of those sidecars. A restored K/V prefix is usable only when the matching
recurrent checkpoint can be installed; a K/V hit counter alone does not prove
a valid hybrid-model restore.
[GDN sidecar implementation](../../../crates/mlx-core/src/models/qwen3_5/gdn_sidecar.rs).

The SSD tier retains its bounded capture policy: by default 128 blocks per
turn and a 250 ms capture budget. A few turns from a 63K context therefore
need not persist the full history. Performance comparisons report the actual
restored token count and installed sidecars. They do not substitute a RAM
restoration test or increase the writer budget to manufacture full coverage.
[Cold-tier capture policy](../../../crates/mlx-core/src/cold_tier.rs).

## Representative workload and measurement controls

The principal replay selects the input to the final long assistant turn in
the local September 3 coding-agent session. The recorded turn used 65,555
input tokens and produced 2,812 output tokens. Pi resolves its parent chain
and compaction into 133 historical messages, including source reads and tool
results. The original system prompt was not recorded; current Pi prompt and
tool definitions reconstruct it, yielding 62,892 measured input tokens.

The context SHA-256 is
`6900ee940d7f347acb12aad71af5c533af1007b66122a5ac378ed4542af0a3a1`.
Each run records the actual loaded native addon SHA, checkpoint configuration
SHA, runtime architecture and relevant environment variables. Temperature is
zero, thinking is high, and each turn generates up to 256 tokens. The full
original 2,812-token answer is not reproduced by this bounded replay.

The replay calls the production `MlxModelHost` and streaming adapter with
paged cache required, persistent SSD caching enabled, 2048-token prefill
chunks and the normal inline MTP path. After the first response, three new
follow-ups append the generated assistant messages. Resident continuation
therefore reuses the model's live cache. Historical commands are data only;
new tool requests receive an explicit offline replay result.
[Replay harness](../../../scripts/benchmark-qwen35-agent-session.ts).

Capture arms start from separate empty SSD roots. Restore arms run in fresh
processes from copies of the same persisted snapshot. A restore arm fails
unless the recurrent-sidecar installation counter increases. Every run also
checks generation success and completion of the final cold-writer drain.
Raw session text and model outputs stay in ignored local artifacts; shareable
results contain hashes, timings and counters only.

Native time to first token, decode throughput and complete warm-turn time
are reported separately. Queue/model-loading time is recorded but excluded
from native inference comparisons: observed loading delays ranged from a
few seconds to tens of seconds. A host restart occurred after the failed
large-chunk experiment. New compact-head comparisons use only post-restart
runs from the same binary; earlier and later absolute speeds are not treated
as optimization gains.

## Results

### Retained change: early decode submission

`maybe_submit_paged_decode_layer` queues completed layers 0–3 with
`MxArray::async_eval_arrays` in the eager paged AR step, MTP Step A and MTP
verification forward. Metal availability gates the calls. Prefill retains
its existing chunking and materialization barriers; training/traced graphs
are untouched. The submitted residual tensor carries the layer's attention
and cache-write dependencies. Completion of the full turn still controls
publication of its cache state.
[Implementation](../../../crates/mlx-core/src/models/qwen3_5/paged_forward.rs).

`MLX_QWEN35_DECODE_EARLY_EVAL_LAYERS=0` disables early submission for a
same-binary control. An unset value defaults to four; the value is cached at
first use. The comparison used addon SHA-256
`653b265c4e6dab6b715c88b78cf3a66ad9dc44169aa52009280137611045924b` in every arm,
with stock 50 MiB / 50-operation Metal command-buffer limits.

Both processes remained resident, with one active inference workload at a
time and a 2 GiB allocator-cache cap per process. Each 256-token continuation
used identical input and output in its pair. Native producer completion and
the SSD writer drain preceded the next process's release. Both arms stayed
resident until the final pair finished. Setup prefills are excluded from the
paired speed calculation because the second load runs with the first model
already resident.

| Turn                      | Full-prefill control | Early submission | SSD-restart control | Early submission |
| ------------------------- | -------------------: | ---------------: | ------------------: | ---------------: |
| 1                         |             13.250 s |         12.873 s |            12.403 s |         12.027 s |
| 2                         |             13.669 s |         13.588 s |            12.901 s |         12.837 s |
| 3                         |             13.535 s |         13.835 s |            13.289 s |         12.370 s |
| 4                         |             14.015 s |         13.535 s |            12.747 s |         12.597 s |
| 5                         |             14.601 s |         13.649 s |            13.905 s |         13.593 s |
| 6                         |             13.135 s |         12.836 s |            12.563 s |         12.365 s |
| Arithmetic mean wall time |             13.701 s |         13.386 s |            12.968 s |         12.631 s |

The first experiment loaded control first and alternated control/candidate,
then candidate/control for the next pair. The SSD experiment reversed both
loading and turn order. Geometric means of `control wall / candidate wall`
were 1.02330 and 1.02668; the combined twelve-pair ratio was 1.02499. This
supports a small improvement on this workload, with individual-turn noise.
It does not establish the same gain on other hardware, models, or concurrent
agent batches.

All 28 generated outputs across the four seven-turn processes matched their
paired controls, as did MTP cycle counts and acceptance statistics. Every
writer drain completed. Cold-cache corruption and write-error counters were
zero. The SSD arms started from byte-verified copies of the same snapshot;
each restored 3,920 tokens, installed one recurrent sidecar, recorded 245 KV
hits and read 259,241,115 bytes. The preserved bounded capture policy therefore
provided partial prefix recovery, not full-history restoration.

Full-prefill peak active allocation was 34,204,061,256 bytes in both arms;
SSD-restart prefill peak was 28,230,471,224 bytes in both. Warm-turn peaks
were within 16 KiB of each other in every pair. Although the
paired method bounds allocator retention during setup, warm-turn retained
cache was about 0.23–0.25 GB in both capped and earlier uncapped controls.

### GPU submission profile

A successful five-second Metal System Trace of resident agent continuation
recorded 16,155 command-buffer submissions for the target Node process.
Its compute intervals covered a 5.629-second span, with 5.289 seconds in the
union of active compute intervals. Recorded encoder intervals summed to
730.849 ms. These are instrumented event durations, not uninstrumented
throughput or CPU execution time. The shader-profiler table contained no
samples, so this trace does not establish individual shader costs.

MLX's Metal command encoder counts each distinct referenced input allocation
against its byte threshold, including persistent weight arrays. On a Max
GPU the default threshold is 50 MiB; a large quantized projection can exceed
it by itself. This explains a plausible source of frequent submissions.
[MLX command encoder accounting](https://github.com/ml-explore/mlx/blob/6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb/mlx/backend/metal/device.cpp#L342-L349),
[commit thresholds](https://github.com/ml-explore/mlx/blob/6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb/mlx/backend/metal/device.cpp#L513-L515).

mlx.fast has a startup policy for this case, gated at 96 GiB physical memory.
Its MTP startup hook sets both the referenced-byte budget and operation budget
to 512, while the full-profile policy object reports 512 MiB and 50 operations.
Some surrounding comments describe the latter pair. Both concrete configurations
were compared against 50/50 controls before choosing a policy.
[mlx.fast startup policy](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine/blob/27c821c466c9799e87162e9436618863b7d0a0ba/Sources/MLXFastModel/RuntimeStartupMemoryPolicy.swift#L48-L112).

| Configuration, in execution order | Native first-token time | Warm turn 1 | Warm turn 2 | Warm turn 3 |
| --------------------------------- | ----------------------: | ----------: | ----------: | ----------: |
| 50 MiB / 50 ops, control 1        |               123.174 s |    14.146 s |    14.585 s |    14.328 s |
| 512 MiB / 50 ops                  |               138.787 s |    14.409 s |    14.764 s |    14.349 s |
| 512 MiB / 512 ops                 |               114.587 s |    12.412 s |    12.939 s |    12.393 s |
| 50 MiB / 50 ops, control 2        |               102.329 s |    12.148 s |    12.620 s |    12.399 s |

The apparent 13% warm-turn improvement of 512/512 against the first control
vanished against the following control. All outputs and MTP counts matched.
Larger buffers raised peak prefill allocation from 34,204,061,256 to
37,474,668,104 bytes. These process-level runs do not establish a benefit.
The follow-up kept both processes resident, alternated individual warm turns
in reversed order, bounded each allocator cache to 2 GiB and released a turn
only after the other process's native producer and SSD writer were idle.
Initial prefills in this arrangement were setup, not comparable timing samples.
The machine was on AC power in high-performance mode, with zero swap usage.

| Resident turn | 50/50 control | 512/512 candidate | Throughput gain from wall time |
| ------------- | ------------: | ----------------: | -----------------------------: |
| 1             |      12.989 s |          12.675 s |                         +2.48% |
| 2             |      13.327 s |          13.275 s |                         +0.39% |
| 3             |      13.036 s |          13.143 s |                         -0.81% |
| 4             |      12.912 s |          12.838 s |                         +0.58% |
| 5             |      13.874 s |          13.878 s |                         -0.02% |
| 6             |      13.068 s |          13.026 s |                         +0.32% |

Geometric mean paired gain was 0.49%; mean wall time was 13.201 s versus
13.139 s. Outputs and MTP counts matched on all seven turns, every writer
drain completed, and corruption/write-error counters stayed zero. The mixed,
small latency difference does not justify the additional prefill memory as
an agent default. No command-buffer startup policy was added.

The retained change follows the engine's early completed-layer submission
while preserving mlx-node's paged executor and cache dependencies.
[mlx.fast early graph submission](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine/blob/27c821c466c9799e87162e9436618863b7d0a0ba/Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/Gemma4Text.swift#L31-L100).

### Compact-head comparison

The compact-head experiment used one native addon with SHA-256
`868fb295b8204c0605827bb471b1c07944dbf2c76439989cd0facc1f39a048f2` and a
fresh-process diagnostic switch. The run order was control, compact, compact,
control. Each arm used an empty isolated SSD root, the same context hash and
four 256-token responses. All output hashes and MTP acceptance counts matched.

| Run, in execution order | Native first-token time | Warm turn 1 | Warm turn 2 | Warm turn 3 |
| ----------------------- | ----------------------: | ----------: | ----------: | ----------: |
| Control 1               |               120.632 s |    14.445 s |    15.199 s |    13.939 s |
| Compact 1               |               119.818 s |    14.303 s |    14.431 s |    13.841 s |
| Compact 2               |               127.527 s |    14.190 s |    14.723 s |    14.238 s |
| Control 2               |               120.386 s |    13.929 s |    14.178 s |    13.931 s |

Across both runs, mean warm-turn time was 14.270 s for control and 14.288 s
for compact heads: effectively unchanged, with the candidate 0.12% slower.
Mean native first-token time was 120.509 s versus 123.673 s. The initially
favorable warm-turn comparison did not survive the reversed-order repeat.
There is no supported speed claim for this candidate.

Full-prefill peak active allocation was 34,204,061,256 bytes in both arms.
Warm-turn peaks fell slightly, by roughly 1–3 MiB, without establishing a
latency benefit. Cold-cache corruption and write-error counters remained zero;
bounded queue drops varied with writer activity. All final drains succeeded.
These are capture/resident results; no compact-specific SSD speedup is claimed.

The compact kernel passed exact BF16 comparison against explicit expansion
for grouped and tiled ordering, scalar and vector gates, active masked rows,
and sequence lengths through 2048. MTP owner batching, row selection and
every accepted prefix of a five-token tape also matched autoregressive state
replay. All three per-step Metal variants passed. Correctness alone did not
justify retaining a performance change with neutral agent results.

[Sanitized measurements and output hashes](results.json) include these four
runs and the earlier negative comparisons. Missing metadata in older runs
is explicitly null, not inferred. Raw local session text is not included.

## Negative results and limits

The packed gate/up and multiply-fusion candidate produced identical output
on all four long-replay turns, but did not improve the agent. One adjacent
pair had full-prefill TTFT of 133.833 s for control and 132.964 s for the
candidate, while warm turns changed from 15.339/16.007/15.696 s to
15.649/16.271/15.967 s. The change was removed.

Folding GDN scaling into stock RMSNorm initially reduced TTFT from 99.389 s
to 95.527 s with identical output. A repeat on a reduced binary reversed the
result: 102.165 s for control versus 118.714 s with the change; all three warm
turns also slowed. This does not establish a repeatable benefit. That code
was removed rather than selecting only the favorable measurement.

The prior SSD comparison restored 3,920 tokens, installed one recurrent
sidecar and read 259,241,115 bytes in each arm, with zero corruption or write
errors. All four outputs matched. The normalization candidate had no SSD
advantage: first-turn TTFT was 99.384 s versus 98.126 s for control. This is
partial prefix recovery, approximately 6.2% of the input, not a full 63K
history restore.

Fully fusing sigmoid failed exact numerical parity for one of 65,280 finite
BF16 inputs. At -6.84375 the fused sigmoid yielded `0x3a8c` where the existing
operation yielded `0x3a8b`. That candidate was rejected before agent timing.
Multiply-only fusion preserved the intermediate rounding, but its agent
performance did not justify retaining it.

Dequantize-then-GEMM screening used actual target-model matrices. On the
first MXFP4 gate projection, 128-token chunks took 2.738 ms versus 1.108 ms
for direct quantized matmul; at 2048 tokens both were about 6.6 ms. Some
MXFP8 projections improved modestly at 2048 tokens, while others were mixed.
The MXFP4 down projection differed in 1,341 of 10,485,760 BF16 output words
in one test despite matching top-one indices. This supports further
matrix-specific investigation, not a blanket dequantization policy or a
claimed agent gain.

An 8192-token prefill chunk experiment stalled and was terminated; the host
subsequently restarted. It produced no successful inference measurement.
The remaining measurements and implementation keep 2048-token chunks.

The matrix-based chunked GDN screen used BF16 tensors at 48 value heads and
128-dimensional keys/values. At 2048 tokens, repeated matrix-based calls took
15.352–17.005 ms, while the existing per-step calls took 2.602–5.416 ms.
Maximum absolute output and final-state differences were 0.000488 and
0.000977. This was a candidate rejection screen, not an end-to-end agent
benchmark. The production Metal selector was never changed to use it.

A separate CLI fixture attempts a read/edit/test task with independent
acceptance tests. Its first run with thinking disabled repeated incorrect
cancellation edits and was stopped. The final run with high thinking and
the retained native default hit the 600-second limit: it executed three read
tools, then continued generating reasoning without making an edit. Independent
acceptance failed. The actual loaded addon matched the measured candidate.
This establishes neither completed coding-task performance nor a general
agent-quality result; the supported speed claim is the fixed-work long-session
comparison above. The failed fixture is retained as a validation limit.
[Sanitized CLI outcome](task-validation.json),
[CLI task harness](../../../scripts/benchmark-qwen35-agent-task.ts).

## Validation

The native wrapper build passed and validated both colocated metallibs. The
TypeScript build, focused benchmark-script lint, Rust formatting and diff
checks passed. The final targeted Qwen dense native run passed 42 tests,
including scheduled MTP owner replay, scalar/batched replay, recurrent cache
policy and speculative lookahead exhaustion. All four paged dispatch stress
tests passed: V1 at 1,000 iterations, V2 at 100, and both ten-iteration smokes.
These compare against an explicitly synchronized reference and reject a
no-write baseline, so consistent stale reads cannot pass as determinism.

This is not a claim that the entire repository test suite is green. An earlier
dense fixture run had two failures constructing unsupported 16-dimensional
KV heads; a broader final filter passed 72 tests and hit the same fixture
problem in a MoE construction test. The targeted final run excluded those
unrelated fixtures and a debug-assertion test that cannot run meaningfully in
the release profile. They fail before executing the changed decode helper.

## Reproduction

Build with native ARM64 Node using `yarn build:native` and `yarn build:ts`.
The native wrapper validates and colocates both required metallibs. On this
host `/opt/homebrew/bin/node` is native ARM64; the alternative Node installation
runs under Rosetta and must not be used for these comparisons.

```sh
MLX_COLD_CACHE_DIR=/an/isolated/empty/cache \
MLX_PAGED_PREFILL_CHUNK_SIZE=2048 \
yarn oxnode scripts/benchmark-qwen35-agent-session.ts \
  MODEL SESSION.jsonl OUTPUT.json capture ASSISTANT_ENTRY_ID
```

The removed compact-head experiment used `MLX_DISABLE_GDN_COMPACT_GQA=1`
for its same-binary control and an unset switch for compact heads. That switch
belongs to the archived experiment, not the unchanged production source.
For future candidates, repeat in reversed order with one inference workload
at a time. Use fresh cache roots for
capture and identical snapshot copies for restore; replace `capture` with
`restore` for the latter. Never compare two arms that used different input
hashes or generated different fixed-work outputs as if they were equivalent.

Run the CLI fixture with
`yarn oxnode scripts/benchmark-qwen35-agent-task.ts MODEL RESULT_DIRECTORY`.
It uses an isolated workspace, high thinking and the production paging,
SSD and MTP defaults. Its acceptance result matters more than raw elapsed
time; a shorter failed task is not an optimization.

For alternating resident comparisons, the reusable controller accepts a JSON
configuration and refuses existing result/cache directories:

```json
{
  "model": "/path/to/qwen3.8-27b-mxfp4-mlx",
  "session": "/path/to/recorded-agent-session.jsonl",
  "entryId": "86bd44d6",
  "outputDir": "/tmp/qwen-early-pair-01",
  "coldCacheDir": "/Volumes/P4510/.cache/mlx-agent-fast-bench/repro-early-pair-01",
  "common": {
    "MLX_MAX_MB_PER_BUFFER": "50",
    "MLX_MAX_OPS_PER_BUFFER": "50"
  },
  "control": { "MLX_QWEN35_DECODE_EARLY_EVAL_LAYERS": "0" },
  "candidate": {},
  "first": "control",
  "warmTurns": 6
}
```

Run `python3 scripts/benchmark-qwen35-agent-pair.py EXPERIMENT.json` with
native ARM64 Node first on `PATH`. The candidate uses the shipped default.
For SSD restart, add `"snapshot": "/path/to/saved/mlx-paged-v1"` and set
`"first": "candidate"`. The controller copies and hashes every snapshot file,
keeps both models resident until the last pair completes, verifies matched
work/output, and terminates only its own child process groups on failure.
It fixes chunk size at 2048 and each allocator-cache cap at 2 GiB. That cap
is an experimental control for two loaded instances, not an agent default.
[Pair controller](../../../scripts/benchmark-qwen35-agent-pair.py).
