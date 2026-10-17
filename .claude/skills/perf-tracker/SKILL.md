---
name: perf-tracker
description: Run the mlx-node vs llama.cpp benchmarks tracked in GitHub issue #142 (Qwen3.8 27B, Gemma 4 12B, Qwen3.8 Flash Next UD-Q4_K_XL), always comparing the latest mlx-node main with the latest llama.cpp prebuilt release, then compare against the previous entry and draft the dated issue comment. Use when asked to rerun or refresh the #142 benchmark, to check prefill/decode against llama.cpp, or to measure a commit before a release.
---

# mlx-node vs llama.cpp performance tracker

Issue [#142](https://github.com/mlx-node/mlx-node/issues/142) is the index. Each suite replays a real coding-agent
history through both runtimes with identical token IDs and reports prefill and decode tokens/second.

**Always compare the latest of both:** mlx-node at `origin/main` HEAD, and llama.cpp at its newest `bNNNNN` GitHub
release, as the prebuilt macOS arm64 binary. Preflight enforces both.

```
worktree at origin/main ─► build ─► preflight ─► prewarm ─► matrix ─► report
                                       │                                 │
             HEAD contains origin/main ┤                                 ▼
  newest llama.cpp release downloaded ─┘          compare.md + chart.svg + samples.csv
                                                                 │
                                                                 ▼
                                       draft the #142 comment ─► ask the user ─► post
```

| Suite        | Entry                                        | Workload                                | Samples | Time                         |
| ------------ | -------------------------------------------- | --------------------------------------- | ------: | ---------------------------- |
| `qwen38`     | `scripts/perf-tracker/qwen38/run.ts all`     | 6,219 / 18,754 / 32,488 tokens, 512 out |      18 | about 1 h                    |
| `gemma4`     | `scripts/perf-tracker/gemma4/run.ts all`     | 7,733 / 40,528 / 66,904 tokens, 256 out |      18 | 35 min                       |
| `flash-next` | `scripts/perf-tracker/flash-next/run.ts all` | 6,219 / 18,754 / 32,488 tokens, 256 out |      18 | about 1.5–2 h, not yet timed |

The first run of each suite after a native GGUF cache format change also converts the checkpoint (`prewarm`, not
timed). Flash Next also fills its packed weight cache on the first load after every copy of its splits: mlx-node keys
that cache on the file path, inode, ctime and device.

## Rules

- Measure the latest code on both sides. Never build llama.cpp yourself and never pin an older llama.cpp build:
  preflight downloads the newest release. Preflight also refuses a HEAD that does not contain `origin/main`.
- Measure a clean, dedicated worktree. Never measure in the shared main checkout. Preflight refuses a dirty tree;
  `--allow-dirty` exists only to develop the scripts and marks the run `dirty: true`.
- Run one suite at a time. Do not run builds, tests or other inference at the same time. The two Qwen3.8 suites wait
  until the GPU is at most 5% busy before their 6k and 16k rows; Gemma records the load instead of waiting. If the
  desktop never reaches GPU idle and the user accepts it, `run.ts all --desktop-load` (either Qwen3.8 suite) admits
  every row like the 32k row (no other local LLM). The run records this; say so in the write-up.
- Do not change the protocol: fixtures, llama.cpp arguments, token counts, order, warmups and cooldowns. A changed
  protocol is a new fixture version and a new tracker row, not a rerun.
- Every suite starts llama-server with `-ctxcp 0 -cram 0` on top of the fixture's arguments. Each measured request is
  a single cold request, and context checkpoints and the RAM prompt cache only help a later request reuse the prompt;
  the checkpoints still copy state twice in each timed prefill. Provenance records the final arguments, and
  `compare.md` says so when the baseline ran without them.
- mlx-node's reuse features that have a switch are off. Its small remaining reuse work has no switch and counts
  against mlx-node: one extra GDN checkpoint chunk in the Qwen3.8 27B prefill, one extra forward of the last token
  when Gemma stops on length, and about 1 ms of block hashing per request.
- Do not export `MLX_*`, `NAPI_RS_*`, `INT8_QMV_*`, `GGML_*`, `LLAMA_ARG_*`, `MTL_*`, `METAL_*` or `DYLD_*`. The
  runners refuse them and set the protocol variables for their own workers.
- Do not change the checkout, the addon or the llama.cpp install during a run. Each suite hashes them before and after.
- Never post to GitHub or push an asset branch without asking the user first.

## Flash Next rules

| Topic         | Rule                                                                                                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Disk          | The four splits must be real files on the internal SSD. Preflight refuses a USB disk or a link, and records the volume.                                                                     |
| Output tokens | 256 in every row. mlx-node caps this model's context at 32,768 tokens, so 32,488 + 512 cannot run.                                                                                          |
| Weight budget | mlx-node's automatic budget: no `MLX_QWEN4_*` variable is set. It is sized from free memory at each load, so it can differ per block. `compare.md` lists it per load.                       |
| Leftovers     | Before each load the runner waits up to 60 s for any other `llama-server` or mlx worker to exit, then stops. A leftover model process would shrink the budget.                              |
| Prompt IDs    | `prewarm` traces the IDs the native session admits and compares them with the pinned IDs. So `all` includes the proof that `qwen38` runs as `check-ids`; `matrix` refuses a run without it. |
| Fixture       | The Qwen3.8 fixture as is: same messages, same token IDs.                                                                                                                                   |
| Agreement     | `compare.md` shows how many characters both runtimes generated identically. Information only, not a gate.                                                                                   |
| Baseline      | The first entry has no `baseline.csv`. `compare.md` says so instead of a before/after table.                                                                                                |

## One-time setup

1. llama.cpp needs no setup. Preflight reads the GitHub releases of `ggml-org/llama.cpp`, takes the newest `bNNNNN`
   release that has `llama-bNNNNN-bin-macos-arm64.tar.gz`, checks its SHA-256 against the release digest, and
   extracts it to `<repo>/.cache/llama.cpp/bNNNNN/`. Set `GITHUB_TOKEN` if the API rate limit stops it. Later steps
   use the newest installed release, and provenance records its build number, commit and file hashes.

   A new llama.cpp release can change CLI flags, defaults or `/completion` fields. If llama-server fails to start or
   `report` refuses its samples, read the llama.cpp changes before you touch the protocol.

2. Put the checkpoints under `<repo>/.cache/models/` (or point `MODELS_DIR` at another folder):

   | Folder                               | File(s)                                                     | Source repo                                    | SHA-256 starts                                 |
   | ------------------------------------ | ----------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
   | `qwen3.8-27b-gguf`                   | `Qwen3.8-27B-UD-Q4_K_XL.gguf`                               | `unsloth/Qwen3.8-27B-GGUF`                     | `3f227079`                                     |
   | `gemma-4-12b-it-qat-q4_0-gguf`       | `gemma-4-12b-it-qat-q4_0.gguf` + its `mmproj-…gguf`         | `google/gemma-4-12b-it-qat-q4_0-gguf`          | `93567e57`                                     |
   | `qwen3.8-flash-next-gguf/UD-Q4_K_XL` | `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1,2,3,4}-of-00004.gguf` | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `766911a6` | `44481862`, `3f342f1c`, `56758f40`, `753bda48` |

   ```bash
   mlx download -m <source repo> -g <file> -o .cache/models/<folder>
   ```

   Flash Next is 111 GB in four splits. Copy them into one folder on the internal SSD and pass it with
   `--model-dir`. The default `<repo>/.cache/models/…` is refused when `.cache` points at a USB disk.

   Preflight checks every model hash. The sidecar files next to each GGUF (`tokenizer.json`, `tokenizer_config.json`,
   `config.json`, `chat_template.jinja`) decide the prompt token IDs on the mlx-node side. Before the matrix, each
   suite renders the fixture messages with them and compares the token IDs with the pinned IDs (Qwen and Gemma in
   `preflight`). Flash Next has no sidecars: mlx-node extracts these files from the GGUF at the first load, so its
   `prewarm` renders with the extracted files, and every later load must extract the same files. Files that render
   other IDs make the run incomparable, so fix them before measuring.

3. Fetch the public fixtures (Flash Next uses the Qwen fixture):

   ```bash
   oxnode scripts/benchmark-fixture.ts fetch --fixture gemma4
   oxnode scripts/benchmark-fixture.ts fetch --fixture qwen38
   ```

## Prepare the checkout under test

```bash
git fetch origin main
git worktree add --detach .worktrees/perf-<date> origin/main
cd .worktrees/perf-<date>
git submodule update --init crates/mlx-sys/mlx
vp install
MACOSX_DEPLOYMENT_TARGET=26.2 PATH=/usr/bin:$PATH SDKROOT="$(xcrun --show-sdk-path)" yarn build
ln -s <folder that holds models/ and benchmarks/> .cache     # .cache is gitignored
git status --short                                            # must print nothing
```

- `26.2` is the deployment floor; a lower value stops the native build.
- To measure a branch or PR, rebase it onto the latest `origin/main` first and commit everything.
- If `origin/main` moves during a long session, the next preflight refuses; rebase and rebuild, then start again.
- Rebuild after every checkout change (`yarn build` above), before preflight. See check 2 below.

## Run

```bash
oxnode scripts/perf-tracker/gemma4/run.ts all
oxnode scripts/perf-tracker/qwen38/run.ts all
oxnode scripts/perf-tracker/flash-next/run.ts all --model-dir <internal SSD folder with the four splits>
```

- Each `all` makes `.cache/benchmarks/perf-tracker/<suite>-<UTC stamp>/` and keeps the Mac awake with `caffeinate`.
- Single steps take `--out <run dir>`. After a failure, run the failed step again with the same `--out`. Flash Next
  reads `--model-dir` only in `preflight`; later steps read it from `provenance.json`.
- Resume keeps finished cells. A cell is one runtime on one case: one process, its warmup and all repetitions. A cell
  that stopped partway moves to `discarded/<stamp>/` and reruns in full, and `report` refuses a cell whose samples
  come from more than one process. Gemma runs every sample in its own process, so it keeps each finished sample.
- `report --out <run dir>` rebuilds the outputs of a finished run.
- Qwen3.8 27B only: `check-ids --out <run dir>` (about 2 min, untimed) proves that the native session admits exactly
  the pinned prompt IDs. Run the steps one by one to include it: `preflight`, `prewarm`, `check-ids`, `matrix`,
  `report`. Flash Next runs the same proof inside `prewarm`.
- Start a long suite in the background and watch its log for the end line and for errors. Do not poll it in a loop.

## Outputs and checks

| File              | Content                                                                          |
| ----------------- | -------------------------------------------------------------------------------- |
| `provenance.json` | OS, chip, commits, `origin/main`, addon/llama/model hashes, llama.cpp args, load |
| `samples.csv`     | one row per measured request, same columns as the #142 CSV for that suite        |
| `summary.json`    | medians, min/max and mlx-node/llama.cpp ratios per cell                          |
| `chart.svg`       | prefill and decode medians with min–max whiskers                                 |
| `compare.md`      | new medians against `scripts/perf-tracker/<suite>/baseline.csv` (previous entry) |

Flash Next adds `prewarm.json` (load, ID proof, llama.cpp log lines) and `check-ids.json`; its `summary.json` and
`compare.md` also hold the weight budget per load and the output agreement.

Before you trust the numbers:

1. Confirm that `report` finished. It refuses runs with a wrong token count, cached prompt tokens or a changed
   identity hash.
2. The addon hash proves the addon did not change during the run. It does not prove which commit built it, so
   rebuild after every checkout change.
3. Read the host load in `provenance.json` and the per-sample logs. Name any busy process in the write-up.
4. Both runtimes are new versions in each entry, so both can move. The main result is the mlx-node/llama.cpp ratio
   per cell. `compare.md` shows how each runtime moved since the previous entry; when both move the same way, suspect
   the host (OS, load, heat) before either runtime.
5. Look at a few generated texts. They must read as a code review. Greedy outputs can differ from earlier runs.

## Publish to issue #142 (ask the user first)

1. Push the chart and data to a new asset branch, for example `perf-tracker/<suite>-<date>` with `chart.svg`,
   `samples.csv` and `summary.json`. Link the chart as
   `https://raw.githubusercontent.com/mlx-node/mlx-node/<asset commit>/chart.svg`, pinned to the commit.
2. Write one dated comment per suite, in the shape of the earlier entries:
   - a headline with the ratios, the chart, and the median table with ratios;
   - request time (Gemma); the weight budget per load and the output agreement (Flash Next);
   - conditions: machine, OS, power, load, workload, matched settings, repetitions, timing;
   - revisions: mlx-node commit, MLX submodule, llama.cpp release tag and commit, checkpoint hash;
   - the change against the previous entry (`compare.md`) and what changed in between;
   - the samples CSV and the fingerprints in `<details>` blocks.
3. Update the model's row in the issue body (latest measurement and link). Keep every older entry.
4. Show the draft to the user. Run `gh issue comment`, `gh issue edit` or `git push` only after they approve.

## Updating the baselines

After a new entry is posted, copy its `samples.csv` to `scripts/perf-tracker/<suite>/baseline.csv` in a separate
commit, so the next `compare.md` compares with the latest entry. Flash Next has no baseline until its first entry.
