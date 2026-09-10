# Gemma 4 QAT GGUF: recorded agent replay versus llama.cpp

Measured September 10, 2026 on an Apple M5 Max with a 40-core GPU and
128 GiB unified memory, macOS 26.6.2. With these settings, llama.cpp had
**1.55–2.32× higher median generation throughput** and **27–35% lower
median request latency** than mlx-node. All results use recorded agent
history; synthetic-input measurements are excluded.

## Results

Medians of three measured samples per runtime and boundary. Every sample
generated exactly 256 tokens from an empty prompt cache. Request time is
the capped continuation, with loading and warmup excluded.

| Real session boundary   | Input tokens | Prompt tok/s, MLX / llama.cpp | Decode tok/s, MLX / llama.cpp | Request seconds, MLX / llama.cpp |
| ----------------------- | -----------: | ----------------------------: | ----------------------------: | -------------------------------: |
| After reading the diff  |        7,733 |             940.74 / 1,108.40 |                 28.21 / 43.74 |                    17.27 / 12.52 |
| After running tests     |       40,528 |               479.43 / 702.36 |                 16.15 / 35.61 |                   100.35 / 64.87 |
| Before the final review |       66,904 |               380.35 / 497.48 |                 12.22 / 28.40 |                  196.81 / 143.47 |

Median prompt throughput was 1.18×, 1.46×, and 1.31× higher in llama.cpp,
respectively. These are descriptive measurements on an active desktop;
three samples do not establish statistical confidence. In particular,
short-context prompt throughput overlaps across runtimes. All measured
samples are retained; the observed ranges are:

| Input tokens | Runtime   | Prompt tok/s range | Decode tok/s range | Request seconds range |
| ------------ | --------- | -----------------: | -----------------: | --------------------: |
| 7,733        | MLX       |      900.20–948.77 |        27.02–29.38 |           17.26–17.59 |
| 7,733        | llama.cpp |    820.78–1,156.05 |        37.80–48.67 |           12.22–16.17 |
| 40,528       | MLX       |      453.24–553.95 |        15.73–17.34 |          87.88–105.65 |
| 40,528       | llama.cpp |      614.78–734.17 |        32.15–36.23 |           62.25–73.86 |
| 66,904       | MLX       |      372.53–397.86 |        11.18–12.26 |         188.99–202.43 |
| 66,904       | llama.cpp |      495.28–515.46 |        27.92–28.84 |         138.65–144.23 |

All 18 samples passed the expected prompt length, input-token identity,
empty-cache, output-length, and timing checks. No prompt was truncated.
Native prefill plus decode accounted for request time within 30.53 ms.
The native binary, tracked source patch, and recorded source session were
unchanged across the run. macOS reported no thermal or performance warning
before, during the sampled checks, or after the benchmark.

For each boundary, all three output hashes matched within each runtime.
Outputs differed across runtimes in all nine pairs; this benchmark does
not establish numerical parity or answer quality. Every sample hit the
generation limit, and some ended during reasoning before a visible answer.

[results.json](./results.json) contains all measurements, ranges, output
hashes, environment details, and fixture provenance. Raw prompts and
generated text remain in the ignored local data directory described below.

## Fixture

The source is a real mlx-node agent session recorded September 3, 2026:
`01a067cf-b782-7b58-855e-8158dcb283ab`. Its user request was
`Deepreview https://github.com/oxc-project/oxc-node/pull/745`.
The session contains 134 messages and 71 tool results covering source reads,
the PR diff, JSON module resolution, tsconfig aliases, and test output.

| Boundary                | Historical messages | Tool results | Gemma input tokens | Next recorded assistant entry |
| ----------------------- | ------------------: | -----------: | -----------------: | ----------------------------- |
| After reading the diff  |                   8 |            4 |              7,733 | `3de6f891`                    |
| After running tests     |                  56 |           30 |             40,528 | `48096389`                    |
| Before the final review |                 133 |           71 |             66,904 | `86bd44d6`                    |

Each input ends at a complete recorded turn boundary. Pi's
`buildSessionContext` follows the recorded parent chain, `convertToLlm` converts
the session, and mlx-node's production `contextToChatMessages` constructs
the native messages. The current Gemma tokenizer renders the chat template
with tool definitions and high thinking. Nothing is padded or repeated, and
no invented user prompt or tool result is inserted. Historical tools and any
new generated calls are data; this benchmark never executes them.

The JSONL does not contain the original system prompt. The wrapper is
reconstructed using Pi 0.84.4's `buildSystemPrompt` and `createCodingTools`,
with the recorded working directory. This follows the fixture method used
by the earlier [Qwen agent measurements](../qwen35-agent-mlxfast/README.md).
The original session used Qwen; this benchmark replays its history through
Gemma's template and generates new Gemma continuations.

Source JSONL SHA-256:
`cdf4938c2ce1f37acbe87659b38782bc48dd95cabb98723c7328819d8cbc2988`.
Exact messages, token IDs, rendered prompts, raw outputs, and server logs stay
in the ignored local directory `.cache/benchmarks/gemma4-agent-2026-09-10/`.
The report publishes only provenance, hashes, and measurements.

## Protocol

- Same source weights: `gemma-4-12b-it-qat-q4_0.gguf`, SHA-256
  `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`.
  The main GGUF contains 328 Q4_0 tensors, one Q6_K embedding, and 338 F32
  tensors. mlx-node uses its prepared native GGUF cache, preserving the
  quantized weights; llama.cpp loads the source GGUF.
- mlx-node base `185bc1b0a19e70e3a522452d4cbe69f2f24b47a9` plus this task's
  Gemma native GGUF loading changes; MLX library
  `6d45ab90cfec5e7fe0cfe25bc635a23ac8a351bb`. Runtime patch and addon hashes
  are saved with the measurements.
- Local llama.cpp `a14dba686aaafba3a2d6b5eb8820b0df5c5d2d92`, build 10610,
  Release, Metal and Accelerate enabled. This is the installed checkout,
  not a claim about the newest upstream revision.
- Three measured repeats per runtime and boundary, one inference process
  at a time. Runtime order alternates by repeat; a 20-second pause separates
  samples. Each sample starts in a fresh process and warms up with the first
  real fixture for 32 generated tokens.
- Cold prompt cache for every measurement: mlx-node calls `resetCaches()`
  after warmup, and llama.cpp receives `cache_prompt: false`. Every result
  must report zero cached tokens and the exact expected prompt length.
  The MLX session API requires `reuseCache: true`; resetting before the
  measured request prevents reuse. Persistent paged caching is disabled.
- Exact input token IDs are passed to llama.cpp. Its tokenizer must reproduce
  those IDs from the rendered Gemma prompt before every measured request.
  No input is truncated; llama.cpp has a 73,728-token context allocation.
- Greedy generation, high thinking, no speculation, no sampling penalties,
  maximum 256 new tokens. Natural stops remain enabled, and actual output
  counts and stop reasons are retained.
- BF16 KV in both runtimes. MLX uses its paged Gemma cache and 512-token
  prefill chunks. llama.cpp uses full GPU offload, flash attention, a logical
  batch of 2,048, physical batch of 512, six CPU threads, and one slot.
- Loading and warmup are excluded. The source GGUF and prepared MLX cache
  are already available on disk; this does not time first-load conversion.
  MLX also loads the matching media companion, whereas llama.cpp is text-only.
  The workstation remains an active desktop, with normal applications open.

## Metrics and limits

Prompt throughput is input tokens divided by native time to first token;
the latter includes prompt preparation, prefill, and first-token sampling.
Decode throughput counts the remaining `generatedTokens - 1` tokens. These
are matching timing conventions in the two runtimes. Request time measures
the complete native session call in MLX and the local completion request in
llama.cpp, including response handling. The harness checks that native
prefill plus decode agrees with request time within 250 ms.

These are cold-cache, single-request continuations from recorded agent
contexts. They do not measure an entire live review, tool execution, warm
prefix reuse, SSD restoration, concurrent requests, multimodal performance,
or review quality. High thinking can consume the generation budget before
the model reaches a visible answer. Output equality and actual lengths are
reported separately from speed.

## Reproduce

The source session and model are local fixtures. [benchmark.ts](./benchmark.ts)
contains their paths and the llama.cpp binary path. Run from the repository
root after building the native addon:

```sh
oxnode docs/research/gemma4-agent-benchmark-2026-09-10/benchmark.ts prepare
oxnode docs/research/gemma4-agent-benchmark-2026-09-10/benchmark.ts run
python3 docs/research/gemma4-agent-benchmark-2026-09-10/summarize.py
```

`prepare` captures the environment and produces hashed inputs. Reuse those
saved inputs to reproduce the exact prompt; regenerating the current system
wrapper later may change its date or tool descriptions. `run` writes the
18 measured samples. `summarize.py` requires every sample, checks workload
and timing invariants, excludes pilots, and writes `results.json` without
including session content or generated text.
