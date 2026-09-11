# Muse Glimmer Q4_K_XL: mlx-node versus llama.cpp

September 11, 2026. Completed **36 accepted measurements**: three real review histories × two runtimes × DFlash off/on × three repetitions. The opening tables and chart describe the original implementation. The subsequent [adaptive decode study](#decode-optimization-resource-bounded-global-attention) records 12 fresh samples: +14.3% over compact at 60k and about 1.2% behind llama.cpp in that comparison. The later [compact-read A/B](#decode-optimization-compact-sliding-reads) adds eight samples and improves measured AR decode by 15.7% at 36k and 17.8% at 60k context.

**Observed medians:** mlx-node processed prompts faster in all six matched comparisons, and mlx-node with DFlash had the lowest request time at all three lengths. llama.cpp had faster AR decode at every length. DFlash slowed decode on the longest continuation in both runtimes. The ranges below capture substantial variability on this active desktop; these are not peak-performance claims.

![Muse Glimmer Q4_K_XL performance](muse-glimmer-q4-k-xl-benchmark.svg)

## Decode, tokens/second

Each cell is **median [minimum–maximum]** across three fresh processes.

| Input tokens |                MLX, off |          llama.cpp, off |             MLX, DFlash |       llama.cpp, DFlash |
| -----------: | ----------------------: | ----------------------: | ----------------------: | ----------------------: |
|        7,254 | **22.96** [22.36–22.98] | **23.59** [15.71–24.96] | **24.06** [18.24–25.70] | **18.22** [13.84–19.11] |
|       36,325 | **17.32** [13.98–17.89] | **19.18** [15.67–21.48] | **25.61** [19.78–26.66] | **27.34** [20.85–28.27] |
|       60,547 | **13.61** [12.20–14.92] | **20.00** [17.18–20.32] | **10.39** [10.28–10.52] |    **8.31** [6.53–8.46] |

## Prefill, tokens/second

Each cell is **median [minimum–maximum]** across three fresh processes.

| Input tokens |                MLX, off |          llama.cpp, off |             MLX, DFlash |       llama.cpp, DFlash |
| -----------: | ----------------------: | ----------------------: | ----------------------: | ----------------------: |
|        7,254 | **677.2** [656.7–709.6] | **547.4** [394.4–612.8] | **716.9** [510.1–787.7] | **595.2** [439.4–624.4] |
|       36,325 | **501.0** [415.8–511.7] | **463.6** [377.1–465.3] | **662.3** [502.6–693.1] | **498.2** [367.0–511.4] |
|       60,547 | **428.3** [348.8–452.9] | **404.2** [391.9–414.8] | **639.4** [616.3–657.6] | **439.5** [422.8–447.0] |

## Request time, seconds

Each cell is **median [minimum–maximum]** across three fresh processes.

| Input tokens |                   MLX, off |             llama.cpp, off |                MLX, DFlash |          llama.cpp, DFlash |
| -----------: | -------------------------: | -------------------------: | -------------------------: | -------------------------: |
|        7,254 |    **14.85** [14.37–15.30] |    **17.28** [15.65–24.44] |    **14.08** [12.92–19.44] |    **17.40** [16.59–23.38] |
|       36,325 |    **77.85** [76.52–94.19] |   **83.31** [82.49–102.38] |    **58.59** [56.00–77.11] |   **76.40** [74.40–103.53] |
|       60,547 | **148.38** [140.11–181.42] | **154.56** [150.64–160.04] | **103.78** [101.37–107.45] | **149.22** [146.70–157.78] |

## DFlash effect

| Input tokens | MLX decode change | llama.cpp decode change | MLX request change | llama.cpp request change |
| -----------: | ----------------: | ----------------------: | -----------------: | -----------------------: |
|        7,254 |             +4.8% |                  -22.8% |              -5.2% |                    +0.7% |
|       36,325 |            +47.9% |                  +42.6% |             -24.7% |                    -8.3% |
|       60,547 |            -23.7% |                  -58.4% |             -30.1% |                    -3.5% |

Positive decode change means faster generation; negative request change means less total time. These ratios compare medians, not paired statistical estimates. Host variability is substantial; do not use this run to set fixed CI speed thresholds.

All 36 requests generated exactly 96 tokens, had zero cached prompt tokens, and passed the thermal/performance-warning checks. Every speculative sample recorded actual drafting and accepted drafts. These checks do not establish thermal stability or a quiet machine.

Output stability across the three repeats and both modes:

| History      | MLX identical text across all six runs | llama.cpp identical text across all six runs |
| ------------ | -------------------------------------- | -------------------------------------------- |
| diff-review  | no                                     | yes                                          |
| test-review  | yes                                    | yes                                          |
| final-review | no                                     | yes                                          |

Within each of the 12 runtime/mode/context combinations, all three repeats produced identical text. Timing variation within a cell therefore does not come from changed output text. The table above additionally compares across speculation modes.

Median accepted/drafted token ratio, from each runtime's native counters:

| Input tokens | MLX DFlash | llama.cpp DFlash |
| -----------: | ---------: | ---------------: |
|        7,254 |      21.8% |            23.2% |
|       36,325 |      38.6% |            48.0% |
|       60,547 |      13.7% |            11.0% |

## Workload and protocol

Measured September 11, 2026 on an Apple M5 Max, 40 GPU cores, 128 GB, macOS 26.6.2, AC power. This is a single active desktop, with two background model downloads and desktop/monitoring activity. No competing inference or compilation was observed. Record the observed ranges; these are not idle-machine ceilings or confidence intervals.

The target is `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf` (19,653,960,832 bytes). Its 418 packed target tensors comprise 51 Q4_K, 130 Q5_K, and 237 Q6_K tensors; the preset name does not mean every weight is four-bit. The shared companion is `dflash-kquant.gguf` (1,631,205,312 bytes): a five-layer DFlash model, not a DSpark checkpoint. Both engines use this same companion. DFlash's 16-position block contains an anchor plus 15 proposed tokens; the benchmark fixes both engines to a maximum of 15 proposals, disables MLX adaptive depth/fallback, and leaves llama.cpp's CPU thread count automatic. Final blocks may shrink to respect the remaining output budget. This is a matched fixed-width comparison, not a search for an optimal draft width.

The unchanged public [`gemma4-oxc-review-v1`](../../scripts/fixtures/gemma4-oxc-review-v1.json) histories come from real mlx-node session `01a067cf-b782-7b58-855e-8158dcb283ab`, reviewing Oxc-node PR #745. Its original system prompt was absent; the previous Gemma study reconstructed and froze a Pi 0.84.4 wrapper. This benchmark reuses those exact frozen messages/tools and renders them with Muse's template/tokenizer. The resulting input lengths are **7,254 / 36,325 / 60,547**. All three llama.cpp tokenizations are checked against the exact Muse input IDs. Recorded tool calls remain data and are never executed.

Each of the 12 runtime/mode/context combinations has three measured fresh-process repetitions. Every process warms up for 32 tokens on the shortest history, resets prompt state, then measures exactly 96 generated tokens. A 256-token pilot naturally ended at 115 tokens on the shortest history, so a 96-token prefix gives equal output counts without padding inputs or suppressing EOS. Pilots and failed harness attempts are excluded. No measured request may contain prefix-cache hits, truncated input, or fewer than 96 output tokens. Runtimes/modes alternate in forward/reverse order with 20-second cooldowns; all inference runs serially.

MLX uses the production LM `ChatSession` with owner-scoped cache lifecycle. A raw-core, ownerless AR pilot failed sliding-cache admission before measurement; switching the harness to the production session path resolved it without changing the model runtime. MLX AR uses its paged cache; DFlash uses the current default flat speculative path. MLX preserves BF16 target/draft activations and KV, with original FP16 quantization sidecars retained. llama.cpp uses Metal, all GPU layers, flash attention, BF16 target/draft KV, a 61,440-token context, one slot, logical batch 2,048, physical batch 512, and no context shifting. MLX's physical prefill chunk is also 512. Sampling is greedy, high thinking, with repetition/frequency/presence penalties disabled.

One original long-context llama.cpp AR sample overlapped a one-second process-stack diagnostic after an unusually slow model load. Its JSON/logs are retained under `diagnostics/` and excluded from the tables. That cell was repeated after the matrix with unchanged settings and no diagnostic sampling; the reported 36 samples include the replacement. Exclusion was based on observer overlap, not its speed or output.

Loading and warmup are excluded. Prefill measures through the first generated token; decode covers the subsequent **95** tokens in both runtimes. Request wall time is also recorded, including the respective LM wrapper or local HTTP overhead. This measures a capped continuation, not a completed review, executed tool round trip, or end-to-end agent task. Greedy outputs may differ because of numerical and execution-path differences; throughput does not establish quality parity or lossless speculation.

## Decode optimization: compact sliding reads

The original AR gap grows with context: **43.55 vs 42.39 ms/token** at 7,254 input tokens and **73.46 vs 49.99 ms/token** at 60,547. Source inspection found unnecessary attention work in the production scheduler path, separate from weight quantization.

Muse has 39 sliding-window layers (2,048 tokens) and 13 global layers. Owner-scoped AR uses `run_paged_decode_step_batched` → `forward_paged_batched` → `gather_kv_for_decode_graph_batched`. The old `build_batched_attention_inputs` discarded the compact metadata from `decode_attention_inputs` and rebuilt the full logical block table and absolute sequence length, including for a batch of one. The generic Metal kernel loaded keys and computed QK before masking expired positions.

The adapter now preserves compact reads for singleton decode and builds compact metadata for multiple owners and ragged queries. Each owner's first query determines the oldest required key; later verifier queries cannot shorten that first query's window. Only read metadata is rebased. Cache writes, allocation, retention, RoPE positions, and rollback retain their absolute positions. The policy follows the admitted window, block size, and query span, with no chip-specific constants or tuning override.

At the first 60,547-token continuation step, each sliding layer needs **5 rather than 119** generic 512-token partitions. Across 52 layers, first-stage workgroups fall from 198,016 to 55,744 (**71.8%**). This is a dispatch-count reduction, not a token-latency or DRAM-bandwidth prediction: retired pages can alias one null block, and projections/global attention still run.

The references implement the same bounded-work principle:

- llama.cpp `a14dba686aaafba3a2d6b5eb8820b0df5c5d2d92`: `src/llama-kv-cache-iswa.cpp` sizes a separate SWA cache from the window plus physical batch; `src/llama-graph.cpp` selects it for Muse sliding layers.
- vLLM `a9271c750f6eb07ad7e4c52b9dffa65ff2ee336a`: `compute_tile_loop_bounds` in `vllm/v1/attention/ops/triton_attention_helpers.py` bounds the allowed KV tiles before `triton_unified_attention.py` enters the loop. Retention for speculative rollback remains a separate concern. Its [hybrid cache design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) also distinguishes sliding and full-attention requirements. This is an algorithmic reference, not an Apple GPU performance comparison.

A separate eight-sample A/B experiment uses the unchanged real fixture and benchmark worker, 32 warmup tokens, 96 measured output tokens, zero prompt-cache hits, serial fresh processes, and 20-second cooldowns. Long-context order is baseline/compact/compact/baseline; shorter contexts have one pair each. No build or profiler overlaps the measured jobs.

| Input tokens | Samples per variant | Baseline decode tok/s | Compact decode tok/s | Change | Same greedy output? |
| -----------: | ------------------: | --------------------: | -------------------: | -----: | :------------------ |
|        7,254 |                   1 |                 22.81 |                22.82 |    ~0% | No                  |
|       36,325 |                   1 |                 17.31 |                20.02 | +15.7% | Yes                 |
|       60,547 |                   2 |                 13.84 |                16.30 | +17.8% | No                  |

Long-context ranges are 13.79–13.89 baseline and 15.41–17.18 compact. Both long-context repeats reproduce their own variant's text. Changed partition boundaries alter floating-point reduction order; exact output parity is not claimed. The independent numerical regression checks BF16 output against FP64 softmax with nonuniform Q/K/V, head size 128, 32 Q/2 KV heads, 513/2,048-token windows, partial pages, and one/17 query rows (absolute error below 0.003). It passes; the throughput samples still do not establish model-quality equivalence. The original llama.cpp long-context median was 20.00 tok/s in the earlier matrix, so this does not close the measured gap or demonstrate a hardware ceiling.

Prefill medians for these pairs are 687.9/647.8, 491.9/505.4, and 395.9/373.0 tok/s (baseline/compact). This change targets decode metadata; the small, active-desktop experiment is insufficient to assign the prefill variation to it. Short and middle contexts have only one pair and provide no repeatability estimate.

Focused release validation passed **479 tests**, with 16 explicitly ignored: 155 adapter tests, 225 Muse tests, 48 scheduler tests, the new BF16 numerical regression, and 50 Gemma attention/cache/speculation regressions. The optimized native build succeeds. Full repository tests and remote CI were not run.

Local evidence: `.cache/benchmarks/muse-optimization-2026-09-11/` contains isolated baseline/compact addons, the measured source patch, protocol and input identities, eight raw samples, summary, build/test logs, and resource records. Baseline addon SHA-256 is `7b272a44d4d44f6b08c61059fcde5ecf1b087452185737588f10a3ef27d84b93`; compact addon is `4d2a773b97663e92dca1e46ec89ef64edc7ccfe3b9d908cf97a3df7352d04bd5`. Its `environment.json` identifies the parent benchmark; `protocol.json` identifies the actual A/B binaries. The bootstrap scopes the native-library override to mlx-core so it cannot redirect oxnode's parser addon. The installed addon and original 36-sample results remain unchanged.

Resource controls stop only the owned job on excessive RSS, low available memory, time, or log size. Accepted inference jobs peak at 20.70 GB RSS. These controls replace the earlier failed broad Metal trace: `xctrace` reached 263.44 GB during post-processing, its unusable trace was discarded, and the concurrent diagnostic build was cancelled. There is still no usable per-kernel GPU timing from that attempt. Do not repeat that tracing method.

## Quantization and dtype audit

Scope: the actual Q4_K_XL target and `dflash-kquant.gguf`, native GGUF preparation, owner-scoped paged AR, flat DFlash proposal/verification, and shared sampling. This is a source/control-flow audit checked against the prepared tensor headers, not an instrumented count of GPU conversion dispatches.

Prepared target storage has 418 packed matrices: U32 codes, 181 U8 and 237 I8 sub-scale arrays, and 418 FP16 super-scale arrays. The remaining 209 tensors are BF16 vectors (2,782,208 bytes). The draft has 36 packed matrices and 22 BF16 vectors (162,304 bytes). Neither file contains a dense floating-point matrix. Header hashes and dtype/byte counts are saved in `dtype-audit-headers.json`.

| Stage                                         | Actual conversion behavior                                                                                                                                                                                                                               | Assessment                                                                                                                                                                                                                             |
| :-------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native GGUF preparation                       | `prepare_native_gguf_inner` sets `import_k_quants=true`, `quantize=false`; `load_kquant_repack` rearranges packed codes and preserves original integer/FP16 sidecars. Only ordinary floating weights are converted to BF16.                              | One-time preparation, no dense-weight dequantize/requantize cycle.                                                                                                                                                                     |
| Embedding lookup                              | `Embedding::forward` gathers requested packed weight/scale rows and dequantizes only those rows to BF16. Target and draft share the packed target embedding.                                                                                             | Required activation materialization; no full-table expansion. One decode token produces 6,656 BF16 values (13,312 bytes). Three gathers plus row dequantization are a possible small fusion opportunity, not a whole-model round trip. |
| Target/draft linear layers and LM head        | `build_projection` resolves native K modes. `QuantizedLinear::forward_qmm` calls MLX `quantized_matmul`; its K branch passes activation, packed codes, integer scales, and FP16 super-scales unchanged, with output dtype equal to the activation dtype. | No activation or sidecar promotion at this boundary for this checkpoint. The Rust conditional cast back to activation dtype does not execute.                                                                                          |
| Decode Metal QMV                              | `kquant_qmv_impl` / `kquant_qmv_fast_impl` unpack values and widen activations/scales in registers, accumulate FP32, and store BF16.                                                                                                                     | Normal fused quantized arithmetic; no full dequantized weight buffer or subsequent requantization.                                                                                                                                     |
| Prefill / verifier QMM                        | `QuantizedBlockLoader` reconstructs small weight tiles in registers and BF16 threadgroup memory for matrix multiplication; accumulation is FP32.                                                                                                         | On-chip tiled dequantization inside the matmul, not a separate full-matrix conversion pass. Small verifier batches can use QMV-wide instead.                                                                                           |
| Norms, RoPE, gates, residuals, output softcap | GPU fast RMSNorm/RoPE preserve BF16 tensors; reductions use internal wider arithmetic. Scalar wrappers create scalars directly in the activation dtype.                                                                                                  | No BF16→FP32→BF16 tensor cycle from these scalar operations. RMSNorm's explicit FP32 fallback is the CPU route, not the Metal path.                                                                                                    |
| Attention and KV                              | Muse allocates BF16 KV; native writes and attention validate matching BF16 query/cache types. Dense/rotating caches concatenate or slice the existing type. Paged attention returns BF16.                                                                | No quantized-KV round trip in this configuration. The `.astype(x.dtype())` attention calls reach MLX's same-dtype early return and create no GPU cast node.                                                                            |
| Greedy sampling and DFlash acceptance         | Shared scheduler and verifier use direct argmax on logits; proposal token conversion is integer typing. Stochastic probabilities/residual correction use FP32 where required.                                                                            | No weight conversion. The measured greedy path bypasses the FP32 probability work.                                                                                                                                                     |

Relevant source: `crates/mlx-core/src/{utils/gguf.rs,nn/embedding.rs,models/muse_glimmer/{persistence,attention,dflash,dflash_decode,model}.rs,models/gemma4/quantized_linear.rs,engine/{batch_sampling,dspark_turn}.rs}`, plus `crates/mlx-sys/mlx/mlx/{ops.cpp,fast.cpp,backend/metal/{quantized.cpp,kernels/kquant.h}}`.

Two other-format paths must not be confused with this result. Affine Q4_0 with FP16 sidecars can promote BF16 activations/output to FP32 in generic QMM and narrow the output afterward; the mixed affine decode QMV avoids that for eligible shapes, while generic prefill remains a separate optimization opportunity. Plain E4M3 weights have an explicit one-time BF16 reconstruction fallback. Neither path is selected by these target/draft K-quant tensors. Do not remove precision guards globally based on the K-quant result.

The second optimization pass below addresses the previously missing grouped D128 route and early submission. Per-token graph construction, projection kernels, and cache-write settlement still contribute to latency; their individual costs have not been isolated. [MLX compilation can fuse operations](https://ml-explore.github.io/mlx/build/html/usage/compile.html), but that capability does not automatically optimize every model graph. Equal hardware does not imply equal performance between two different kernel and execution stacks.

## Decode optimization: resource-bounded global attention

The [Qwen baseline in #142](https://github.com/mlx-node/mlx-node/issues/142) does not establish a backend-wide speed guarantee: it uses another checkpoint and token history, 512 generated tokens, a 2,048-token prefill chunk, and F16 llama.cpp KV. This Muse study uses 96 generated tokens, 512-token chunks, and BF16 KV in both engines. Existing grouped fast paths were guarded for other head sizes, so Muse could not enter them.

After compacting sliding reads, the 13 global layers still used generic attention: at context 60,548, each dispatched 119 partitions × 32 query heads = 3,808 first-stage workgroups. Instantiate the existing grouped BF16 template for head size 128, grouping the 16 query heads belonging to each KV head in one threadgroup. The implementation uses direct reads with GPU-cache reuse; it does **not** stage one shared KV copy in threadgroup memory. Partition count remains a runtime decision. At 256 partitions the first stage has 512 workgroups, with a different thread layout; this ratio is not a predicted speedup or measured DRAM-traffic reduction.

A bounded 157 MB capture of the first global layer's actual 60,548-token attention inputs isolates this opportunity. Alternating generic/grouped/grouped/generic replay gives **1.196 ms versus 0.531 ms** per completed call at 256 partitions, a 55.6% reduction. This includes dispatch and evaluation synchronization, not isolated GPU timestamps. Too little parallelism is worse: four partitions take 2.767 ms. These forced settings establish the performance curve only; none becomes a production machine preset.

The local vLLM reference groups queries by KV head and dispatches split softmax according to request geometry and available intermediate buffers (`triton_unified_attention.py:1040–1090`, commit recorded above). The inspected llama.cpp vector path instead retains query-head groups with 32 partitions and adapts SIMD groups; it also reuses eligible graphs (`ggml-metal-ops.cpp:3350–3450`, `llama-context.cpp:1334–1376`). Thus fewer repeated reads alone cannot explain the whole-engine difference. The new Muse path also submits completed layer prefixes while the CPU builds the remaining graph, retaining the existing cache-write and retirement barriers.

### Fresh comparison after adaptive tuning

Twelve accepted fresh-process samples use the unchanged pinned history, exact token IDs, 32 short-history warmup tokens, 96 measured generated tokens, zero cached prompt tokens, greedy sampling, BF16 KV, and 512-token physical prefill chunks. The loaded model retains any tuning observations from normal warmup. No forced plan, capture, profiler, build, or competing inference overlaps these samples. A few plan-selection events are logged for audit; calibration remains inside measured latency. Order is compact/adaptive/llama/llama/adaptive/compact at 60k, adaptive/llama/compact at 36k, and compact/llama/adaptive at 7k, with 20-second cooldowns.

**AR decode, tokens/s.** Two-sample cells show median [minimum–maximum]; the shorter rows are single samples.

| Input tokens | Samples per engine |         Compact MLX |        Adaptive MLX |           llama.cpp | Adaptive vs compact |
| -----------: | -----------------: | ------------------: | ------------------: | ------------------: | ------------------: |
|        7,254 |                  1 |               22.57 |               23.43 |               23.67 |               +3.8% |
|       36,325 |                  1 |               20.04 |               20.74 |               20.65 |               +3.5% |
|       60,547 |                  2 | 17.08 [16.33–17.82] | 19.51 [18.97–20.06] | 19.76 [19.59–19.92] |              +14.3% |

Adaptive/llama.cpp decode ratios are 0.990×, 1.004×, and 0.988×. The 60k gain over compact is present in both adjacent MLX pairs (+16.2% and +12.5%). The small remaining llama.cpp differences and overlapping long-context ranges do not establish a stable winner or statistical equivalence. Short/middle rows provide no repeatability estimate.

**Prefill, tokens/s**, through the first token:

| Input tokens | Compact MLX | Adaptive MLX | llama.cpp |
| -----------: | ----------: | -----------: | --------: |
|        7,254 |       620.8 |        676.3 |     544.4 |
|       36,325 |       496.1 |        516.8 |     463.5 |
|       60,547 |       417.7 |        419.8 |     406.8 |

At 60k, prefill ranges are 395.6–439.9 compact, 400.3–439.3 adaptive, and 405.6–408.1 llama.cpp. The code change targets decode; prefill differences on this active desktop are not established effects. Request-time medians (compact/adaptive/llama.cpp seconds) are 15.90/14.79/17.34, 77.99/74.93/82.97, and 150.97/149.46/153.63. These are prompt-dominated, capped continuations, not completed agent tasks.

The adaptive/compact greedy text matches at 7k and 36k. At 60k, compact and llama.cpp each reproduce their own text, while the two adaptive runs differ from each other and from compact. Both adaptive runs end at 256 partitions and two early layers, but their earlier calibration choices differ. The 7k and 36k runs finish at 64 partitions, with eight and two early layers respectively. These observations illustrate runtime selection, not settings to copy to other devices. DFlash was not re-benchmarked: its flat path is unchanged.

All accepted jobs report no thermal/performance warning; this does not prove stable clocks or an idle desktop. Peak observed process-tree RSS is 21.24 GB. Raw results, plan choices, binary/fixture identities, resource logs, and the exact serial order are retained in the fresh-comparison directory.

### Device and workload policy

No chip names, assumed GPU-core counts, or saved winning constants enter the policy. Metal pipeline limits gate support; the D128 stage requires 512 threads and its reducer 1,024. [Apple documents these limits as pipeline-specific](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes), including resource usage. Unsupported devices and other query geometries retain generic attention.

For the supported 32-Q/2-KV/head-128, BF16, block-16 singleton route, the maximum partition count is the largest supported power of two satisfying all of:

- The validated kernel/ABI limit: 1,024.
- Available work: `ceil(context_tokens / 16)`.
- Temporary-memory headroom: `available_bytes / global_layer_count / 8448`, where 8,448 bytes accounts for FP32 sum/max and BF16 partial outputs across 32 heads. `available_bytes` is the positive difference between `min(MLX memory limit, Metal recommended working set)` and MLX active memory; it is an allocation budget, not a claim about free system RAM.
- Metal's maximum buffer length divided by the per-partition output size, 8,192 bytes.

Within that bound, reuse Gemma's completed-token tuner for generic/grouped attention and early-submission depth. The depth candidates derive from the model's layer count. Each candidate's first observation is discarded; three completed-token observations, alternating sweep order, and a median/MAD noise margin determine selection. Recheck neighboring attention choices after selecting submission depth. There are no synthetic prompts, extra forward passes, or added GPU synchronization calls for measurement. Forced or failed samples cannot qualify a selection.

Decisions belong to the loaded model and context bucket, with an eight-entry bound. A changed partition budget triggers a separate calibration; decisions are never exported as another machine's defaults. Device specifications constrain legality and memory usage, while completed-token timings capture effects that specifications alone cannot predict, including occupancy, cache behavior, bandwidth, and CPU/GPU overlap. This selects among tested candidates, not a proof of a global optimum. Calibration runs on real tokens and its cost is included in the request benchmark.

`MLX_MUSE_DECODE_TUNING=0` disables automatic selection. `MLX_MUSE_GROUPED_STRIPES` and `MLX_MUSE_DECODE_EARLY_EVAL_LAYERS` are process-local diagnostic overrides; use them with automatic tuning disabled. Production benchmarking uses neither override. Sliding layers, multiple-owner batches, prefill, and DFlash keep their existing routes.

### Interpreting the remaining limit

The prepared tensor headers contain 19.152 GB outside the row-gathered input embedding. A one-read estimate adds 0.888 GB for global plus sliding KV at this context. Dividing that approximately 20.04 GB by [Apple's published 614 GB/s bandwidth for this 40-core configuration](https://www.apple.com/macbook-pro/specs/) gives **about 30.6 tokens/s**. This is an optimistic storage-only reference: it assumes sustained peak bandwidth, reads each tensor once, and makes arithmetic, intermediate traffic, dispatch, and synchronization free. It is neither measured DRAM traffic/utilization nor a demonstrated attainable rate. No bandwidth constant enters the runtime policy. Near-parity with llama.cpp therefore does not establish the theoretical limit; projection/dequantization throughput and graph execution remain separate optimization targets. Header accounting is retained in `bandwidth-bound.json`.

### Correctness and evidence

All nine power-of-two choices from 4 through 1,024 passed captured-input replay. Against an independent FP64 softmax reference for four actual query heads, both generic and grouped maximum absolute error were 0.005624. Across all 4,096 output components, generic/grouped maximum difference was 0.015625 and RMS difference was 0.000709–0.001125. The separate nonuniform-input regression covers full attention, 513/2,048-token windows, partial pages, empty trailing partitions, and one/17 query rows, with absolute error below 0.003. Changing partition reductions can change BF16 rounding and subsequent greedy tokens; numerical checks do not establish model-quality or exact-output equivalence.

Local validation passed 495 focused Rust tests (18 ignored), including resource-limit changes and different device timing curves, plus the nine captured-input replay configurations. The native release build, repository typecheck, Clippy with warnings denied, Rust formatting, and whitespace checks pass. JavaScript lint passes with warnings in unchanged files. The repository-wide formatter reports 29 unchanged files; those unrelated files were not reformatted.

Evidence is split by purpose: `.cache/benchmarks/muse-attention-2026-09-11/` holds the bounded capture, replay logs, numerical references, exploratory runs, builds, and validation; `.cache/benchmarks/muse-adaptive-2026-09-11/` holds the fresh serial comparison. The measured adaptive addon is `2a63aa9b2f6deac6ac348dbc7c0007d641f99d7ffd0cb1ea2e624bbcd04fcce8`, with paged-attention metallib `88153586471b2c16031885ed1bb5cbf2737240137aa1e555bbe3614eff4e3b10`. Original inputs and worker are unchanged. Exploratory forced-route and profiling/capture runs are excluded from the fresh comparison.

## Further source audit: fusion, quantized layout, and calibration

The SwiGLU change below is a **numerically tested optimization candidate; its performance is unvalidated**. Five pilot requests were excluded after unrelated builds/tests and heavy host activity overlapped the comparison. Further timing was deferred at the user's request. The preceding accepted benchmark tables remain the latest performance evidence.

Source inspection found another model-path difference: Muse's MLP evaluated sigmoid, gate multiplication, and up multiplication as separate primitives. Reuse the existing shape-independent compiled SwiGLU helper for `sigmoid(gate) * gate * up`. MLX's compiler explicitly admits these primitives to fusion, replacing the three activation nodes with one compiled node per layer. Gate, up, and down projections keep their individual native quantization formats; the change adds no hardware parameters. It applies to the shared Muse MLP, including prefill, paged/flat target decode, and the DFlash draft.

The current **quantized Qwen3.8** `MLPVariant::Quantized` also calls the unfused `Activations::swiglu`; the shared compiled helper comes from other existing paths. Do not attribute Qwen's benchmark advantage to MLP fusion.

This candidate follows the inspected references: vLLM's `MuseGlimmerMLP` uses `SiluAndMul` (`vllm/model_executor/models/muse_glimmer.py:1090–1117`); llama.cpp's Muse `LLM_FFN_SILU`/`LLM_FFN_PAR` graph selects `ggml_swiglu_split`, dispatched to `kernel_swiglu` in its Metal backend. Reference commits are recorded above. [MLX compilation documentation](https://ml-explore.github.io/mlx/build/html/usage/compile.html) describes explicit graph fusion and first-call compilation; use of MLX alone does not apply that transformation to every Rust-created graph.

The inspected checkpoint configurations and runtime routes differ materially:

| Checkpoint       | Attention layout                           | Full-attention Q/KV heads and head dimension | Relevant implementation difference                                                                                         |
| :--------------- | :----------------------------------------- | :------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| Qwen3.8 27B      | 48 recurrent linear + 16 full layers       | 24 / 4, D256                                 | GatedDeltaNet state updates and existing D256 paged specialization; dense MLP does not mean all layers use full attention. |
| Gemma4 12B       | 40 sliding (1,024 window) + 8 full layers  | 16 / 1, D512                                 | Earlier gains included fixing affine Q4_0 metadata/activation promotion and bounded sliding reads.                         |
| Muse Glimmer 30B | 39 sliding (2,048 window) + 13 full layers | 32 / 2, D128                                 | Native K-quants avoid that affine cast path; its D128 grouped specialization was added in this PR.                         |

These differences select different work and kernels; they do not predict either engine's relative speed without measurement. llama.cpp also has Apple Metal kernels, including format-specific Q4_K/Q5_K/Q6_K matvecs. A backend's name is not an efficiency guarantee.

Two additional source/header checks narrow the remaining search:

- **No projection shape fallback:** all 417 target matmul matrices, including the LM head, satisfy MLX QMV-fast's `N % 8 == 0 && K % 512 == 0` condition for singleton decode. The 418th matrix is the input embedding, which gathers rows instead. Native K weights already enter the fast vector route; changing a generic edge-path threshold is not supported by this evidence.
- **Small packed-metadata overhead exists:** Q4_K/Q5_K preparation expands packed six-bit scale/min fields into byte sidecars, adding four bytes per 256-weight block. The target has 291,599,360 extra packed-storage bytes versus its GGUF blocks, of which 270,586,368 belong to matrices read in decode rather than the row-gathered embedding. This is approximately 1.4% of the 19.152 GB projection-weight footprint. It is integer metadata expansion, not BF16 weight expansion. Keeping metadata bit-packed might trade bandwidth for more unpacking instructions; storage arithmetic alone cannot establish a speedup or explain the observed latency gap.

Remaining candidates, in order of increasing scope: fuse Muse's separate attention sigmoid/multiply; investigate reuse of the per-token graph (`run_paged_decode_step_batched` rebuilds the layer graph, whereas llama.cpp reuses eligible graphs in `llama-context.cpp`); then compare the K-quant projection kernels on captured real activations. Preserve owner-specific cache writes, absolute positions, rollback, and settlement barriers when changing graph execution. No chip presets or tensor-format conversions are justified by this audit. Individual time shares remain unmeasured.

The existing first-use protocol includes most calibration inside its 95 measured decode steps. At 60k, the ten attention candidates need 40 observations, eight submission candidates need 32, and typical three-way refinement needs 12: 84 steps total. If generic attention wins initially, refinement can require another 40 instead. A short-history warmup does not calibrate the separate long-context bucket. This cost matters for short responses and must remain visible.

The runner now supports `--warmup-case measured --warmup-tokens 128` to investigate performance after calibration using the same real history in each engine. Prompt state is reset, measured cache hits must still be zero, and warmup's actual generated count and timing are recorded with the measurement start timestamp. Natural EOS remains enabled: the requested warmup cap does not guarantee calibration completed. Verify the final selection event precedes measurement before calling a result calibrated. The original default remains 32 tokens on the shortest history; results from the two protocols must be labeled separately.

Local validation for the fusion candidate: native release build; **228 focused Rust tests passed, 12 ignored**, including an independent FP64 activation reference for BF16/FP16/FP32, changing decode/verifier/prefill shapes, non-contiguous inputs, and finite extremes; all-target Clippy with warnings denied; repository typecheck; JavaScript lint with warnings in unchanged files; Rust formatting and whitespace checks. These establish functional/numerical coverage, not model-quality equivalence or performance. DFlash's shared MLP changes, but no accepted new DFlash timing exists.

Local evidence: `.cache/benchmarks/muse-fusion-2026-09-11/` retains `architecture-audit.json`, `projection-layout-audit.json`, the candidate source patch, isolated addon, validation logs, all five excluded pilot samples with exclusion reasons, and bounded activity records. Candidate addon SHA-256: `e2a52a6fac576f594f6d83f1dd427ab77f9de5c5da0063810c5a086ddf3c1cfd`. The installed addon is unchanged. Next measurement: compare first-use and same-history-warmed protocols separately, run old/new/llama in forward and reverse order without other builds or inference, verify zero prompt-cache hits and exact token IDs, and confirm calibration completion timestamps for the warmed cohort.

## Reproduce

Build the current native addon and LM package before running. Fetch the already public fixture, prepare model-specific token IDs and provenance, then run the serial matrix:

```sh
oxnode scripts/benchmark-fixture.ts fetch --fixture gemma4
oxnode scripts/benchmark-muse-gguf.ts prepare --llama-server /path/to/llama-server --output .cache/benchmarks/muse-local
oxnode scripts/benchmark-muse-gguf.ts run --llama-server /path/to/llama-server --output .cache/benchmarks/muse-local
```

To measure after a same-history warmup, add `--warmup-case measured --warmup-tokens 128` to **both** prepare and run and use a separate output directory.

The default model path is `~/.mlx-node/models/muse-glimmer-30b-gguf/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf`; use `--model` to relocate the same checkpoint and companion. Use a new output directory for a different protocol or build. The runner records hashes and rejects changed inputs or stale resumed samples. Hardware-specific thread/draft presets are not baked in.

Full local evidence is under `.cache/benchmarks/muse-gguf-2026-09-11/`: `environment.json`, `llama-libraries.json`, the measured source patch, retokenized `inputs.json`, 36 raw JSON results and worker/server logs, and `publish/summary.json` plus `samples.csv`. The runner is [`scripts/benchmark-muse-gguf.ts`](../../scripts/benchmark-muse-gguf.ts). No new fixture upload is needed: the pinned public input already contains the real histories.

## Provenance

Original 36-sample matrix: mlx-node base commit `66344b00900cda5bb49524434e6e0dddd5574f3e` plus the uncommitted native Muse GGUF support patch; no inference implementation changed during that matrix. The later compact-read A/B has separate provenance above. Measured source patch SHA-256: `58a489dc158f238b558d2401a691a9c5b95e38de79373528155af22cc49d56b8`.

llama.cpp: `a14dba686aaafba3a2d6b5eb8820b0df5c5d2d92`, build 10610, Metal/Accelerate. MLX submodule: `6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb`. Local binary/shared-library identities are retained with the raw evidence.

| Artifact                    | SHA-256                                                            |
| --------------------------- | ------------------------------------------------------------------ |
| Target GGUF                 | `ac7023d6a4c704eb9af54ab53e476a66b7f5b6c0ef2fc4a8dde5253c291a6c38` |
| Draft GGUF                  | `27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc` |
| Muse tokenizer              | `c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d` |
| Native addon                | `7b272a44d4d44f6b08c61059fcde5ecf1b087452185737588f10a3ef27d84b93` |
| Measured runner             | `03ca19fd7de78e7c38dafb3612bdec5cb1796f35ee9fd13f567e9ab9fa775b42` |
| Muse-rendered input payload | `e2e5918410c78e37e2b7836557817793760181f244668bbaf3c8a57b6ef367e1` |
| Original pinned fixture     | `812bd4adb5b10688a2c7ced9ccabf0cc43159d3b80541eb2009960ced347b697` |

Input token-array SHA-256 values:

- diff-review (7,254 tokens): `37e0ea613674337e3be5a3ba75848943f554b8e5aeebfe91c877507956b9dfb3`
- test-review (36,325 tokens): `a2df6ef3c8ce3e590f057da5d02cd7bd0d57ddd17295fc22f382d0c8eb56f047`
- final-review (60,547 tokens): `d1596ec6be60d02e13767c5d41351966825a9d414fe3277cba84a6bd98495c91`
