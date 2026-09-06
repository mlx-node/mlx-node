# Inference optimization follow-up

Date: 2026-09-06. Baseline: PR #138 at `bbc3157c`. Host: Apple M5 Max,
128 GiB unified memory, macOS 26.6.2. Local validation is complete for the
implementation below; PR checks are recorded separately.

This continues the [transfer audit](transfer-audit.md). Its CPU draft sampler
and exclusive speculative lane describe the earlier revision. The follow-up
replaces those paths where the model and cache contracts support the change.
The [measurement ledger](followup-measurements.json) records alternating arms,
native binding hashes and output hashes. Timings below are local measurements;
they do not establish performance on other models or hardware.

## Implemented changes

- Dense draft draws stay on the GPU, with explicit per-draw keys. Gemma's
  Markov and assistant draft chains consume device token arrays directly;
  Muse's parallel mask rows materialize their sampled IDs together. The host
  reads compact IDs rather than scanning or copying the vocabulary.
- DSpark and MTP prepare target distributions against each known draft prefix
  together. Acceptance coins, rejection sampling and the full-accept bonus
  retain their sequential order. Greedy verification groups argmax evaluation.
- Native MTP keeps its draft token-to-embedding chain in the graph. Its verifier
  accepts authoritative host token slices and uploads the embedding input once,
  removing repeated array-to-vector-to-array conversions in dense Qwen, MoE
  Qwen and Nemotron.
- Qwen3 uses direct paged attention for suffixes of at most 16 query tokens.
  Larger chunks use graph-native K/V gather plus SDPA, selected from the measured
  crossover rather than assuming paged attention always wins.
- Fixed-depth Gemma DSpark and Muse DFlash have request-owned draft contexts and
  packed target verification. A shared trait provides the default recording,
  rejection, rollback and settlement transaction. Each model implements target
  math and draft state; it does not duplicate the transaction driver.
- One-token batch slot mappings, block tables, context lengths and query offsets
  are retained across layers in one cache group. Reuse checks the exact row order,
  table incarnation, physical revision, token frontier and pool generation.
  Arrays remain immutable while previous GPU graphs reference them.
- SSD restore uses a bounded reader queue, borrowed serialization payloads,
  retained staging and one upload submission for multiple blocks and layers.
  Decode polls upload one chunk at a time. Validated completion proofs still
  gate publication, and memory admission charges the staging reservation once.
- Greedy whole-turn DSpark can compare confidence-weighted accepted prefixes
  against verifier costs measured on this host. Stochastic proposals retain
  the existing policy: truncating a Markov proposal based on later sampled
  confidence can change its conditional distribution.

The sampling review also fixed correctness defects. An exact-zero uniform
draw now receives finite Gumbel noise, preventing a tie with masked logits.
`sample_and_logprobs` acquires its random key outside a C++ compiled graph;
previously the compiled graph reused its captured key. Tests exercise the zero
endpoint, masked support, filtered and unfiltered draws, returned log probabilities
and one global key consumed per ordinary draw. Speculative residual and bonus
draws now use the request-owned RNG, as does scheduled speculative prefill. A
sampled multi-owner cancellation test exposed the former global-key coupling;
it passes after the change. Invalid categorical axes now return an error across
the C ABI instead of allowing a C++ exception to escape. The final recurrent
state review also tightened missing-tape validation: paged full-attention shells
may omit tapes, but GDN layers must provide them before replay can succeed.

## Measurements and decisions

| Workload                                                      | Baseline           | Follow-up          | Interpretation                                                      |
| ------------------------------------------------------------- | ------------------ | ------------------ | ------------------------------------------------------------------- |
| Qwen3.5 4B NVFP4 MTP, 200 greedy tokens                       | 1554.1 / 1557.9 ms | 1535.1 / 1534.8 ms | About 1.4% lower latency; identical output and 68 cycles            |
| Qwen3.6 35B A3B MXFP8 MTP, 200 greedy tokens                  | 2199.4 / 2200.7 ms | 2174.7 / 2175.7 ms | About 1.1% lower latency; identical output and 82 cycles            |
| Gemma4 12B DSpark, four concurrent requests, 128 tokens each  | 10.33 / 10.71 s    | 6.09 / 6.12 s      | About 42% lower aggregate latency with occupancy four               |
| Qwen3 0.6B BF16 AR, four concurrent requests, 128 tokens each | 579.0 / 581.1 ms   | 570.9 / 571.8 ms   | About 1.5% lower latency from one-token metadata reuse              |
| SSD upload stage, 8 blocks / 14 MiB                           | 1.666 ms           | 0.454 ms           | Batched submissions; excludes filesystem read/checksum latency      |
| SSD upload stage, 32 blocks / 56 MiB                          | 6.467 ms           | 1.364 ms           | Bounded retained staging; excludes filesystem read/checksum latency |

Each slash separates medians from two alternating arms. Builds, other GPU tests
and trace exports were stopped during these measurements. Warmups are excluded.
The final Gemma comparison uses the completed native binding, SHA-256
`20e526e4a6c8aed82d1ffaac86f134e493dbe9560584fd5e20332287feac897b`,
in baseline/candidate/candidate/baseline order, with one warmup and two measured
waves per occupancy per arm. Combined four-request medians were 10.569 and
6.111 seconds, or 1.73 times the aggregate throughput. Singleton medians were
2.070 and 1.979 seconds; singleton output hashes matched throughout. Packed
four-request output hashes repeat across candidate arms but differ from the
baseline, as discussed below. This is not universal output parity.

The paged attention microbenchmark covers contexts of 512, 4096 and 32768 tokens
and query widths 1, 4, 8, 16, 64 and 157. At context 32768, width 16 falls from
5304 to 2191 microseconds; width 157 rises from 4943 to 18700 microseconds.
Maximum numerical error in this BF16 test was 0.0001221. This supports the short
suffix dispatch boundary and rejects an unconditional paged-prefill switch.

GPU sampling has a crossover too. A single 32768-way draw took 176.8
microseconds on device versus 45.6 for the shared CPU scan. Seven grouped device
draws took 37.9 microseconds per draw. At 262144 tokens, one device draw took
220.3 versus 364.1 microseconds for the CPU scan; seven grouped draws took 73.2
per draw. These measurements include the endpoint correction. They measure the
sampler operation, not model throughput.

Residual sampling now selects its invalid/zero-mass fallback on the GPU. The
previous path evaluated the residual sum, read one float on the CPU, then
submitted its draw. Alternating operation measurements at vocabularies 32768
and 262144 reduced median latency from 349–394 to 250–267 microseconds, removing
one completion round trip per rejection. The fallback still selects the target
argmax. Each rejection now consumes one request key, including degenerate mass;
tests pin valid-input parity and zero/NaN/infinite-mass handling.

One-token metadata reuse reduced Qwen AR latency about 1–2% across occupancies
1, 2, 4 and 8. Long output hashes matched across arms except for one owner at
occupancy four; the timing experiment does not establish invariant scheduling
or universally identical greedy output. Fixed-wave tests cover array reuse,
write alias rejection, shared-prefix reads, COW, rollback and owner recycling.
The same reuse applied to multi-token Muse verification regressed both candidate
arms at occupancy four (20.21/18.96 seconds versus surrounding controls at
17.89/17.72 seconds), so multi-token metadata preparation was restored.

Gemma's packed verification changes some long greedy outputs relative to its
single-owner baseline. Real-model teacher-forced verification retained the same
argmax in 52 inspected rows; the eight-query owner's logits were bit-identical,
while the five-query owner had small relative differences from the legacy path.
Changing peer content and row order at fixed total width left the inspected
owner's logits bit-identical. Tiny-model tests additionally cover rejection,
sliding boundaries, shared K/V, cancellation and owner recycling. This supports
owner isolation; it does not establish universal bit-for-bit batch invariance.
MLX selects different GEMV/GEMM reductions for different row counts (see the
pinned backend and [MLX's M5 GEMV change](https://github.com/ml-explore/mlx/pull/3888)).

The first Muse comparison is inconclusive as a general performance improvement.
Four-request latency improved about 4%, but baseline singleton latency drifted
26.5% between arms. The candidate also needed 41 rather than 38 cycles on one
prompt and differed from the flat baseline already at occupancy one. Candidate
output remained identical across its two arms and occupancies 1, 2 and 4. The
multi-token metadata comparison also regressed, as recorded above. Scheduled
Muse DFlash therefore requires `MLX_SCHEDULED_DFLASH=1`; the default retains its
existing flat speculative route. Its packed implementation remains available
for controlled measurement and further optimization.
The final binding's default Muse smoke retained the baseline's 128-token output
hash (4.396 versus 4.309 seconds, one measured turn each; insufficient for a
performance claim). Its opt-in four-request smoke completed all 512 requested
tokens with occupancy four. Three sampled Gemma turns at temperature 0.7 each
completed 200 tokens. These are execution checks, not seeded sampling parity.

The [validation record](validation.md#optimization-follow-up) contains suite
counts, strict Metal coverage, fixture corrections and ignored/manual cases.

Ordinary Qwen AR asynchronous completion did not yield a reliable improvement
in alternating synchronous and callback-streaming runs. The Qwen paged override
therefore retains synchronous completion. A separate lifetime guard remains:
forced-token work is drained before an error can release its cache owner.

The valid Metal capture attached to the actual inference worker after model
loading. It contains 21,732 compute intervals. Their union occupies about 83.1%
of the 4.826-second interval from the first to last compute event. Gaps include
prefill and transitions between four turns; they cannot all be attributed to
avoidable CPU encoding. The trace is evidence about control flow, not a clean
performance baseline.

## Boundaries still requiring separate work

Recurrent MTP remains on the exclusive lane. Source review established that a
cycle-history proposal can pair the final normalized prompt hidden with the
pending anchor; subsequent cycles must retain `verify_hidden[accepted]`. GDN
already accepts equal-width batch rows, but its tape replay takes one accepted
length for the whole batch. Concurrent verification therefore still requires
owner-specific tape splitting, replay after partial and full acceptance, and
a multi-token paged attention route. These changes are not implemented or
claimed as a completed optimization. Interleaving serial complete cycles alone
does not demonstrate a throughput win.
Qwen DFlash2 remains flat-only and requires its existing cache provenance checks.

Adaptive scheduled speculation remains gated while its batch cost policy is
validated. Equal total query counts can have different latency when per-owner
projection shapes differ. A safe cost policy must measure the actual allocation
vector, preserve temporary zero-draft owners, and distinguish those calibration
waves from permanent AR fallback. Whole-turn greedy adaptive runs preserved
output but did not establish a strong end-to-end gain.

The SSD tier remains secondary storage; no permanent second RAM cache was
introduced. Unified memory removes a separate CPU/GPU address-space transfer
for shared allocations, but not private-buffer staging, layout conversions,
owned byte copies or completion hazards. MLX already manages Metal events,
resource residency and concurrent encoders. [Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading)
would require an allocator/storage integration compatible with checksum and
publication boundaries; simply selecting Metal IO or Metal 4 is not an
established optimization for this pipeline.
