# Splash on the same M5 Max

Measured September 22, 2026 on this Mac: Apple M5 Max, 40 GPU cores,
128 GiB unified memory, macOS 27. Splash was built from the clean local
checkout at `7e3c67e8e3a9e9912ff6e02521457017cc4c65d0` with Xcode 27.
Its source was not modified.

The local comparison confirms a substantial decode gap. Across three fresh
repeats per engine and workload, Splash's median reported decode rate was
64.9 tokens/s for the short prompt and 78.9 tokens/s for the 6K prompt.
The frozen validated mlx-node addon measured 35.6 and 41.4 tokens/s.
These are package-level measurements, not an isolated same-weights kernel test.

## Results

All requests generated exactly 1,024 tokens, with greedy sampling, high/xhigh
reasoning, and zero cached prompt tokens. Input token IDs matched exactly.

| Input        | Splash decode tokens/s, median [range] | mlx-node decode tokens/s, median [range] | mlx-node share of Splash rate |
| ------------ | -------------------------------------- | ---------------------------------------- | ----------------------------- |
| 87 tokens    | 64.854 [61.577–70.396]                 | 35.560 [34.229–41.472]                   | 54.8%                         |
| 6,219 tokens | 78.857 [76.642–81.885]                 | 41.417 [40.504–44.736]                   | 52.5%                         |

The engines use slightly different decode timing boundaries. Splash excludes
the entire first speculative emission, eight tokens for the short prompt and
six for 6K; mlx-node excludes its first prefill-sampled token. Splash's native
executor-counter rate, recorded separately, was 65.555 and 79.382 tokens/s
at the median. Do not interpret the decode ratios as an exact kernel speedup.

Full request time includes prompt processing and all 1,024 generated tokens:

| Input        | Splash median time to first token | mlx-node median time to first token | Splash median full request | mlx-node median full request | Full-request speed ratio |
| ------------ | --------------------------------- | ----------------------------------- | -------------------------- | ---------------------------- | ------------------------ |
| 87 tokens    | 248.5 ms                          | 176.0 ms                            | 15.918 s                   | 28.945 s                     | 1.82×                    |
| 6,219 tokens | 9.320 s                           | 9.796 s                             | 22.238 s                   | 34.553 s                     | 1.55×                    |

Splash's full request measurement includes localhost HTTP overhead; mlx-node's
is the native addon's caller wall time. Model loading is excluded from both.
The 6K prefill/first-token times are similar; the main gap is sustained decode.
On the short workload both engines used 274 speculative cycles, so Splash's
advantage there cannot be explained by needing fewer cycles. For 6K, Splash
used 206 cycles versus mlx-node's 216. This supports investigating the cost
of each verification cycle, without attributing the difference to a single
kernel or host-side optimization.

## Models and configuration

Splash used the official `incoai/Qwen3.8-27B-Splash` snapshot
`9d27070b71f7142c6b6025f03ac011d70a73cb48`, downloaded to
`/Volumes/P4510/.cache/splash-hf-hub` and fully checksum-verified by its installer.
Its 78 manifest artifacts total 17,382,658,769 bytes. The package uses packed
affine Q4 target/draft weights and Q8 attention KV. The server had a 40 GiB
Metal allocation budget and a 40K context limit. Native initialization completed
its prefill, decode, speculative verification, and state-restore warmups.

mlx-node used the requested `Qwen3.8-27B-UD-Q4_K_XL.gguf` and supplied BF16
DFlash2 companion, fixed depth seven, with adaptive depth disabled. Its addon
was `splash-qwen38-phase2/final-validated/mlx-core.darwin-arm64.node`, SHA-256
`9f1e274b9178a410bb8cc3d967d4c54b80e672bd4297d1be2c2e2bec064469f4`.
This is the same retained implementation validated in the preceding work;
this comparison did not change its production code.

The checkpoints and quantization differ. Outputs were stable across all three
repeats within each engine but differed between engines. All samples remained
in reasoning at the 1,024-token cap, so these results measure fixed-length
generation rather than completed-answer latency or answer quality.

## Verification and reproducibility

Artifacts are in
[`splash-local-20260922`](../../../.cache/benchmarks/splash-local-20260922/summary.json).
`summary.json` contains every accepted row, medians, ranges, and ratios;
`summarize.mjs` regenerates it and rejects nonzero job exits, cache hits,
wrong output lengths, runtime failures, prompt mismatches, or unstable outputs.
Each sample retains full results, input/output hashes, server counters,
resource monitoring, and its process exit record.

The fixture translation preserves reasoning and tool-call history. A separate
tokenizer-only audit confirmed exact 6K rendered-text and token-ID equality;
the native template copy explicitly uses the session's `preserve_thinking=true`
flag. The source checkpoint remains untouched. The 6K token-ID SHA-256 is
`c301418ba1449c6fc0d252798fd5c780ef700c1b4bdbaa01b7ec570481e123ce`.
Two initial prompt-audit attempts aborted before measured inference and are
excluded. All twelve accepted requests passed the cold-cache and length checks.

Splash has no HTTP cache reset, so [splash-local.mjs](splash-local.mjs) launches
a fresh server per measured prompt, checks PID/model ownership, and shuts it
down afterward. A distinct 16-token warmup precedes each measured request.
mlx-node resets its caches after its own warmup and before each sample.
Both jobs run under the same resource guard; GPU workloads are sequential.

The run order interleaved the engines, but desktop activity and power/thermal
conditions were not controlled. An unrelated process consumed about one CPU
core throughout and was left running. The observed ranges are a reason to use
the repeated cohort rather than select either engine's fastest sample.

The Splash build, benchmark syntax/lint checks, prompt audit, all twelve
sample checks, and independent harness review passed. Review prompted moving
result publication after validation and checking the warmup result. No Splash
server remained listening after the experiment; its checkout remains clean.

Example reproduction from the mlx-node repository root:

```sh
node .cache/benchmarks/splash-local-20260922/guard.mjs splash-new-short \
  node docs/research/splash-qwen38/splash-local.mjs \
  /Users/brooklyn/workspace/github/splash \
  .cache/benchmarks/splash-local-20260922 splash-new-short short

env MLX_DFLASH2_DRAFT_QUANT=off MLX_DFLASH2_DRAFT_REUSE_TARGET_HEAD=0 \
  node .cache/benchmarks/splash-local-20260922/guard.mjs mlx-new \
  oxnode docs/research/splash-qwen38/benchmark.ts \
  .cache/benchmarks/splash-qwen38-phase2/final-validated/mlx-core.darwin-arm64.node \
  .cache/benchmarks/splash-local-20260922/mlx-new.json dflash short,6k 1 1024
```

This is the command as run against that frozen addon, which read both variables.
The current runtime reads neither: the draft always loads as affine Q4/group64
and reuses the target head. See [Draft precision](README.md#draft-precision-current).
