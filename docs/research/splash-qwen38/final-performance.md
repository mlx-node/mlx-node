# Final runtime versus local Splash

The [exact-draft follow-up](exact-draft.md) adds a lossless import of Splash's
Q4 draft and a fresh three-arm comparison. This page retains the earlier BF16
baseline and its original measurements.

September 22, 2026. This compares the cleaned candidate-14 runtime from
[the cleanup](cleanup.md) with the local Splash checkout on the same Apple M5
Max (40 GPU cores, 128 GiB). These are measurements of the two available model
packages, not an isolation of runtime efficiency or model quality.

## Results

Our final runtime reaches **56.3%, 49.7% and 43.6% of Splash's reported decode
rate** on the short, 6K and 32K prompts. Splash is 1.78×, 2.01× and 2.29× faster
on that metric. We have not matched its performance.

Values are median [minimum–maximum] over three fresh processes per cell.
Decode is tokens/second; times are seconds. Each request generates 1,024 tokens.

| Input tokens |     mlx-node decode |       Splash decode | Ours / Splash |
| -----------: | ------------------: | ------------------: | ------------: |
|           87 | 35.99 [35.02–40.47] | 63.92 [63.76–64.44] |         56.3% |
|        6,219 | 41.23 [36.27–42.76] | 82.96 [71.38–83.42] |         49.7% |
|       32,488 | 22.46 [21.91–22.47] | 51.46 [51.42–56.78] |         43.6% |

| Input tokens | mlx-node full request | Splash full request | Splash speedup |
| -----------: | --------------------: | ------------------: | -------------: |
|           87 |   28.59 [25.45–29.38] | 16.15 [16.01–16.18] |          1.77× |
|        6,219 |   34.73 [33.15–38.41] | 21.38 [21.22–25.14] |          1.62× |
|       32,488 |  98.50 [95.64–101.91] | 76.35 [64.13–76.96] |          1.29× |

| Input tokens | mlx-node reported TTFT |   Splash reported TTFT |
| -----------: | ---------------------: | ---------------------: |
|           87 |    0.169 [0.168–0.171] |    0.246 [0.245–0.254] |
|        6,219 |   9.907 [9.219–10.196] |   9.096 [9.010–10.868] |
|       32,488 | 51.788 [50.071–56.365] | 56.557 [46.189–57.144] |

The gap is largest during decoding. Full request time improves less at long
context because prompt processing dominates. First-emission and terminal-state
boundaries differ between engines, as detailed below; the TTFT columns are not
a precision comparison of identical first-token work.

## Workload and controls

Both engines receive the same messages and exactly the same rendered input
token IDs: 87, 6,219 and 32,488 tokens. Each request generates 1,024 tokens,
uses greedy decoding, high reasoning, seven draft proposals, one request at a
time, and zero cached prompt tokens. Every measured output reaches the token
limit while still reasoning; this does not measure completed-answer quality.

There are three fresh processes per engine and context, with alternating
engine/context order. Each process warms up for 16 tokens before measurement;
native caches are reset and Splash warms a distinct prompt. Model loading is
excluded. The native runner uses the final defaults, with inherited `MLX_*`
environment controls removed. No runtime code changes were made for this
comparison.

The initial cohort was excluded after external compilation overlapped the
engines unevenly. The accepted cohort records resource activity every two
seconds and requires no observed active compilation or recognized external
model work. Packaged-app validation and GPU-backed `mlx delegate github` jobs
bypassed the first detector; five affected pairs were excluded in full and
repeated, including an earlier app activation found by the final retrospective
scan. The detector now recognizes those workloads. The last replacement pair
runs at the end, preserving its native/Splash order. Original results and all
exclusion manifests remain in the artifact directory; no selection was based
on measured speed.

Desktop activity, unrelated CPU tests and a background Node process remain.
GPU clocks were not controlled, so this is not an isolated throughput ceiling.
System thermal-warning checks are retained for each job. Ranges are the minimum
and maximum of three observations, not confidence intervals.

## Package and timing differences

|             | mlx-node                                      | Splash              |
| ----------- | --------------------------------------------- | ------------------- |
| Target      | Requested mixed `Qwen3.8-27B-UD-Q4_K_XL.gguf` | Packed Q4, group 64 |
| Draft       | Supplied BF16 `qwen3.8-27b-dflash2`           | Package Q4 DFlash2  |
| KV          | Native flat DFlash lane                       | Q8 paged KV         |
| Entry point | Native Node API                               | Local HTTP server   |
| Source      | `377e78f3` plus retained working changes      | Clean `7e3c67e8`    |

The native decode rate counts 1,023 tokens after its first token through final
state settlement. Splash's reported stream rate excludes its entire first
speculative emission, which can contain up to eight tokens, and ends at its
Done event. Native also materializes the terminal token for continuation;
Splash can finish with a terminal anchor whose target KV row is not present.
First-token latency therefore has a different first-emission boundary. TTFT
is internal timing here: both measured calls use nonstreaming completion.
Full request time is included as the most directly comparable user-facing
duration, although the API/server overhead and terminal-state work still differ.

Different weight quantization, draft acceptance and KV/state representations
mean these results cannot assign the gap to CPU/GPU transfers, SIMD use, or
another individual implementation choice. Output hashes establish repeatability
within each engine; cross-engine output equality is checked separately.

## Verification

All 18 accepted requests passed the input-token audit, greedy configuration
checks, zero-cache checks and 1,024-token length checks. Output text hashes,
cycle counts and acceptance statistics repeat exactly within each engine.
Outputs differ across engines on all three prompts. Native/Splash cycle counts
are 274/274, 216/206 and 322/303; different draft acceptance and output
trajectories are part of the package comparison.

The accepted resource logs contain no observed known GPU-work or compiler
overlap, including a retrospective check for packaged-app activity and the
identified model-delegation PIDs. Splash Metal/capacity failure counters did
not increase. Final verification passed 51 source/binary/fixture identity checks:
the current native sources and packaged addon/metallibs match candidate-14,
and Splash's checkout is clean. Independent review checked the metric formulas,
configuration gates, selected plan and activity-based exclusions.

The comparison changes the research runner and documentation only. It does not
introduce runtime flags or alter the final implementation. The prior native
correctness/build/test results remain documented in [the cleanup](cleanup.md).

## Evidence and reproduction

The fixed fixtures are
`.cache/benchmarks/fixtures-public-2026-09-11/qwen38-inputs.json`, SHA-256
`c6c1373328614919617deca668fe67301fe0df853f7780143d9690464148bf5b`.
The short prompt is the identical TypeScript LRU-cache request in both runners.

Native addon SHA-256:
`5680fd5658f053207f5dd1bdb82e8493a01619756d81d0558427ab8f71628b96`.
Splash binary SHA-256:
`606bdeb039263cc81ba35b718a9fc7973c0c933ca3c58dbc66b28f422c286943`.
The Splash package is `incoai/Qwen3.8-27B-Splash`, snapshot
`9d27070b71f7142c6b6025f03ac011d70a73cb48`.

Evidence is in `.cache/benchmarks/splash-qwen38-final-comparison/quiet/`:
`plan.json`, `identity.json`, `run.mjs`, `continue.mjs`, `guard.mjs`,
`summarize.mjs`, `summary.json`, `final-identity-check.json`, raw responses, prompt-token audits, resource logs and per-job status. The parent
directory retains the excluded cohort and its exclusion reason. Local cache
artifacts are prerequisites, not files downloaded by these runners.

The retained coordinators run each engine under the resource guard,
sequentially from the repository root. Their engine commands are shown below;
use a new output label and directory to preserve the original evidence:

```sh
oxnode docs/research/splash-qwen38/benchmark.ts \
  .cache/benchmarks/splash-qwen38-phase7/candidate-14/mlx-core.darwin-arm64.node \
  <output.json> dflash short 1 1024
node docs/research/splash-qwen38/splash-local.mjs \
  /Users/brooklyn/workspace/github/splash <artifact-dir> <label> short
```

Replace `short` with `6k` or `32k` for the other inputs. The Splash runner
asserts the current packaged tokenizer's input IDs against Splash before
timing, and the summary checks that its addon hash is candidate-14. Splash uses
port 18938, a 40 GiB memory budget and a 40,960-token context capacity.
