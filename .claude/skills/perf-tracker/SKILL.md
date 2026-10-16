---
name: perf-tracker
description: Rerun the mlx-node vs llama.cpp benchmarks tracked in GitHub issue #142 (Qwen3.8 27B, Gemma 4 12B, Muse-Glimmer 30B DFlash off/on) with the pinned protocol, compare against the previous entry, and draft the dated issue comment. Use when asked to rerun or refresh the #142 benchmark, to check a prefill/decode regression against llama.cpp, or to measure a commit before a release.
---

# mlx-node vs llama.cpp performance tracker

Issue [#142](https://github.com/mlx-node/mlx-node/issues/142) is the index. Each suite replays a real coding-agent
history through both runtimes with identical token IDs and reports prefill and decode tokens/second.

```
worktree at the commit under test ─► build ─► preflight ─► prewarm ─► matrix ─► report
                                                                                  │
                     compare.md + chart.svg + samples.csv ◄───────────────────────┘
                                   │
                                   ▼
                     draft the #142 comment ─► ask the user ─► post
```

| Suite    | Entry                                    | Workload                                | Samples | Time with an idle GPU |
| -------- | ---------------------------------------- | --------------------------------------- | ------: | --------------------- |
| `qwen38` | `scripts/perf-tracker/qwen38/run.ts all` | 6,219 / 18,754 / 32,488 tokens, 512 out |      18 | about 1 h             |
| `gemma4` | `scripts/perf-tracker/gemma4/run.ts all` | 7,733 / 40,528 / 66,904 tokens, 256 out |      18 | about 35 min          |
| `muse`   | `scripts/perf-tracker/muse/run.ts all`   | 7,254 / 36,325 / 60,547 tokens, 96 out  |      36 | about 1.5 h           |

The first run of each suite after a native GGUF cache format change also converts the checkpoint (`prewarm`, not timed).

## Rules

- Measure a clean, dedicated worktree. Never measure in the shared main checkout. Preflight refuses a dirty tree;
  `--allow-dirty` exists only to develop the scripts and marks the run `dirty: true`.
- Run one suite at a time. Do not run builds, tests or other inference at the same time. The Qwen suite waits until
  the GPU is at most 5% busy before its 6k and 16k rows; the other suites record the load instead of waiting.
- Do not change the protocol: fixtures, llama.cpp arguments, token counts, order, warmups and cooldowns. A changed
  protocol is a new fixture version and a new tracker row, not a rerun.
- Do not export `MLX_*`, `NAPI_RS_*`, `MUSE_*`, `INT8_QMV_*`, `GGML_*`, `LLAMA_ARG_*`, `MTL_*` or `METAL_*`. The runners
  refuse them and set the protocol variables for their own workers.
- Do not change the checkout, the addon or the llama.cpp build during a run. Each suite hashes them before and after.
- Never post to GitHub or push an asset branch without asking the user first.

## One-time setup

1. Build the pinned llama.cpp (commit `a14dba686aaafba3a2d6b5eb8820b0df5c5d2d92`, build 10610):

   ```bash
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
   git checkout a14dba686aaafba3a2d6b5eb8820b0df5c5d2d92
   cmake -B build-bench -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
     -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=ON
   cmake --build build-bench --target llama-server -j
   export LLAMA_SERVER=$PWD/build-bench/bin/llama-server
   ```

   The runners refuse another llama.cpp commit. `PERF_TRACKER_ANY_LLAMA=1` measures another build on purpose; such a
   run is not comparable with earlier entries.

2. Put the checkpoints under `<repo>/.cache/models/` (or point `MODELS_DIR` at another folder):

   | Folder                         | File(s)                                             | Source repo                           | SHA-256 starts |
   | ------------------------------ | --------------------------------------------------- | ------------------------------------- | -------------- |
   | `qwen3.8-27b-gguf`             | `Qwen3.8-27B-UD-Q4_K_XL.gguf`                       | `unsloth/Qwen3.8-27B-GGUF`            | `3f227079`     |
   | `gemma-4-12b-it-qat-q4_0-gguf` | `gemma-4-12b-it-qat-q4_0.gguf` + its `mmproj-…gguf` | `google/gemma-4-12b-it-qat-q4_0-gguf` | `93567e57`     |
   | `muse-glimmer-30b-gguf`        | `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf`      | `meta-models/Muse-Glimmer-30B-GGUF`   | `ac7023d6`     |
   | `muse-glimmer-30b-gguf`        | `dflash-kquant.gguf` (draft)                        | `meta-models/Muse-Glimmer-30B-GGUF`   | `27d9a805`     |

   ```bash
   mlx download -m <source repo> -g <file> -o .cache/models/<folder>
   ```

   Preflight checks every hash. It also checks the sidecar files next to each GGUF (`tokenizer.json`,
   `tokenizer_config.json`, `config.json`, `chat_template.jinja`): Gemma against the hashes in its fixture manifest,
   Qwen and Muse by rendering the fixture messages and comparing the token IDs. Sidecars that render other IDs make
   the run incomparable, so fix them before measuring.

3. Fetch the public fixtures (Muse re-renders the Gemma fixture with its own template):

   ```bash
   oxnode scripts/benchmark-fixture.ts fetch --fixture gemma4
   oxnode scripts/benchmark-fixture.ts fetch --fixture qwen38
   ```

## Prepare the checkout under test

```bash
git worktree add --detach .worktrees/perf-<date> <commit>
cd .worktrees/perf-<date>
git submodule update --init crates/mlx-sys/mlx
vp install
MACOSX_DEPLOYMENT_TARGET=26.2 PATH=/usr/bin:$PATH SDKROOT="$(xcrun --show-sdk-path)" yarn build
ln -s <folder that holds models/ and benchmarks/> .cache     # .cache is gitignored
git status --short                                            # must print nothing
```

- `26.2` is the deployment floor; a lower value stops the native build.
- The scripts import mlx-node from the checkout that contains them. To measure a commit older than these scripts,
  make a measurement branch at that commit with only `scripts/perf-tracker/` added, and commit it.

## Run

```bash
export LLAMA_SERVER=/path/to/llama.cpp/build-bench/bin/llama-server
oxnode scripts/perf-tracker/gemma4/run.ts all
oxnode scripts/perf-tracker/qwen38/run.ts all
oxnode scripts/perf-tracker/muse/run.ts all
```

- Each `all` makes `.cache/benchmarks/perf-tracker/<suite>-<UTC stamp>/` and keeps the Mac awake with `caffeinate`.
- Single steps take `--out <run dir>`. After a failure, run the failed step again with the same `--out`; finished
  samples are kept.
- `report --out <run dir>` rebuilds the outputs of a finished run.
- Qwen only: `check-ids --out <run dir>` (about 2 min, untimed) proves that the native session admits exactly the
  pinned prompt IDs. Run the steps one by one to include it: `preflight`, `prewarm`, `check-ids`, `matrix`, `report`.
- Start a long suite in the background and watch its log for the end line and for errors. Do not poll it in a loop.

## Outputs and checks

| File              | Content                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| `provenance.json` | OS, chip, commits, addon/metallib/llama/model hashes, host load snapshots |
| `samples.csv`     | one row per measured request, same columns as the #142 CSV for that suite |
| `summary.json`    | medians, min/max and mlx-node/llama.cpp ratios per cell                   |
| `chart.svg`       | prefill and decode medians with min–max whiskers                          |
| `compare.md`      | new medians against `scripts/perf-tracker/<suite>/baseline.csv`           |

Before you trust the numbers:

1. Confirm that `report` finished. It refuses runs with a wrong token count, cached prompt tokens or a changed
   identity hash.
2. Read the host load in `provenance.json` and the per-sample logs. Name any busy process in the write-up.
3. Read `compare.md` for both runtimes. The llama.cpp build is the control: if llama.cpp moved too, suspect the host
   (OS, load, heat) before mlx-node.
4. Look at a few generated texts. They must read as a code review. Greedy outputs can differ from earlier runs.

## Publish to issue #142 (ask the user first)

1. Push the chart and data to a new asset branch, for example `perf-tracker/<suite>-<date>` with `chart.svg`,
   `samples.csv` and `summary.json`. Link the chart as
   `https://raw.githubusercontent.com/mlx-node/mlx-node/<asset commit>/chart.svg`, pinned to the commit.
2. Write one dated comment per suite, in the shape of the earlier entries:
   - a headline with the ratios, the chart, and the median table with ratios;
   - request time (Gemma, Muse) and the DFlash effect (Muse);
   - conditions: machine, OS, power, load, workload, matched settings, repetitions, timing;
   - revisions: mlx-node commit, MLX submodule, llama.cpp commit, checkpoint hash;
   - the change against the previous entry (`compare.md`) and what changed in between;
   - the samples CSV and the fingerprints in `<details>` blocks.
3. Update the model's row in the issue body (latest measurement and link). Keep every older entry.
4. Show the draft to the user. Run `gh issue comment`, `gh issue edit` or `git push` only after they approve.

## Updating the baselines

After a new entry is posted, copy its `samples.csv` to `scripts/perf-tracker/<suite>/baseline.csv` in a separate
commit, so the next `compare.md` compares with the latest entry.
