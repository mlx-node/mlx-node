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
  the checkpoints still copy state twice in each timed prefill. Provenance records the final arguments.
- Speculative decoding is on in both runtimes. Never turn it off to make a row easier: people run these models with
  their draft, so the tracker measures that. mlx-node runs a fixed depth (`mtpAdaptiveDepth: false`).

  | Suite        | mlx-node                                                             | llama.cpp                                                                       |
  | ------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
  | `qwen38`     | DFlash2 draft, 7 drafts per step (block 8), flat BF16 K/V            | DFlash2 Q4_K_M GGUF: `--spec-type draft-dflash --spec-draft-n-max 7 -ngld 99`   |
  | `gemma4`     | DSpark draft, block 7 (`mtpDepth: 7`), block-paged cache             | assistant MTP Q4_0 GGUF: `--spec-type draft-mtp --spec-draft-n-max 3 -ngld all` |
  | `flash-next` | native MTP, `mtpDepth: 2`, BF16 `mtp.*` block (`auxiliaryModelPath`) | unsloth Q8_0 MTP GGUF: `--spec-type draft-mtp --spec-draft-n-max 2 -ngld all`   |

  Gemma uses two methods on purpose: llama.cpp cannot run the DSpark draft (its converter refuses the draft's
  `final_logit_softcapping`, and its DFlash graph adds the Markov bias to logits it never softcaps, where DSpark adds it
  after the softcap), and mlx-node does not run the assistant MTP head. The DSpark draft was trained against the
  non-QAT Gemma 4 12B, not the QAT Q4_0 target measured here. Say both in the write-up.

  Both runtimes fall back to plain decoding without an error when a draft is missing or fails to start, so every
  sample is gated: mlx-node needs `hasMtpWeights()` after load, `mtpCycles > 0`, and the suite's depth `d` in every
  cycle but the length tail, so `d × cycles − d(d − 1)/2 ≤ drafted ≤ d × cycles` (`d` = 7, 7, 2; the Qwen3.8 27B GGUF
  also carries a one-token MTP head); llama.cpp needs `timings.draft_n > 0` and its type in
  `generation_settings["speculative.types"]`. Every counter the report uses must be present, finite and a possible
  count: accepted ≤ drafted, and mlx-node's committed tokens per cycle equal its accepted drafts + 1. The draft files
  are part of the identity: preflight checks their hashes, and provenance records them.

  Speculation changes prefill as well as decode. mlx-node's DFlash2 turn runs on its flat cache in 2,048-token
  chunks, DSpark keeps target hidden states for the draft, and both Qwen MTP and llama.cpp's DFlash run the draft over
  the prompt inside prefill; only llama.cpp's Gemma assistant reads the target K/V with no prefill pass. So the
  `-v2` fixtures are a new protocol, and their first entry has no before value: entries measured before 2026-10-18
  ran without speculation.

- mlx-node's reuse features that have a switch are off. Its small remaining reuse work has no switch and counts
  against mlx-node: one extra forward of the last token when Gemma stops on length (AR and DSpark alike), and about
  1 ms of block hashing per request on a block-paged cache. The Qwen3.8 27B DFlash2 turn runs on the flat cache, so
  it has no GDN checkpoint chunk.
- Do not export `MLX_*`, `NAPI_RS_*`, `INT8_QMV_*`, `GGML_*`, `LLAMA_ARG_*`, `MTL_*`, `METAL_*` or `DYLD_*`. The
  runners refuse them and set the protocol variables for their own workers.
- Do not change the checkout, the addon or the llama.cpp install during a run. Each suite hashes them before and after.
- Never post to GitHub or push an asset branch without asking the user first.

## Flash Next rules

| Topic         | Rule                                                                                                                                                                                                           |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Disk          | The four splits, the MTP head and the MTP folder must be real files on the internal SSD. Preflight refuses a USB disk or a link, and records the volume.                                                       |
| Output tokens | 256 in every row. mlx-node caps this model's context at 32,768 tokens, so 32,488 + 512 cannot run.                                                                                                             |
| Weight budget | mlx-node's default policy: an automatic budget sized from free memory at each load, with routed experts streamed from the internal SSD. Full residency is not the protocol. Each load's budget is information. |
| Free memory   | A normal desktop: do not close apps or free memory for the run. A smaller budget streams more from SSD; name heavy apps in the write-up.                                                                       |
| MTP           | mlx-node reads the `mtp.*` block of `Qwen/Qwen3.8-Flash-Next` through `auxiliaryModelPath` (`mtp-aux.ts` builds the folder); llama.cpp runs unsloth's Q8_0 MTP GGUF.                                           |
| Owner         | Each mlx-node request sends its own `cacheOwnerId`. Builds that route an ownerless request to the exclusive lane decode MTP as plain AR (`mtpCycles` absent); no earlier mlx-node sample here ran MTP.         |
| Leftovers     | Before each load the runner waits up to 60 s for any other `llama-server` or mlx worker to exit, then stops. A leftover model process would shrink the budget.                                                 |
| Prompt IDs    | `prewarm` traces the IDs the native session admits and compares them with the pinned IDs. So `all` includes the proof that `qwen38` runs as `check-ids`; `matrix` refuses a run without it.                    |
| Fixture       | The Qwen3.8 fixture as is: same messages, same token IDs.                                                                                                                                                      |
| Agreement     | `compare.md` shows how many characters both runtimes generated identically. Information only, not a gate.                                                                                                      |
| Baseline      | The first entry has no `baseline.csv`. `compare.md` says so instead of a before/after table.                                                                                                                   |

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

   The drafts are pinned to a revision, so fetch them with `hf download <repo> <files> --revision <revision>
   --local-dir <folder>`:

   | Suite        | Folder                                   | File(s)                                               | Source repo @ revision                              | SHA-256 starts         |
   | ------------ | ---------------------------------------- | ----------------------------------------------------- | --------------------------------------------------- | ---------------------- |
   | `qwen38`     | `qwen3.8-27b-dflash2`                    | `config.json`, `model.safetensors`                    | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c`            | `873e3556`, `67fc76d6` |
   | `qwen38`     | `qwen3.8-27b-dflash2-gguf`               | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`                     | `z-lab/Qwen3.8-27B-DFlash2-GGUF` @ `2d9571f8`       | `1a25c568`             |
   | `gemma4`     | `dspark_gemma4_12b_block7`               | `config.json`, `model.safetensors`                    | `deepseek-ai/dspark_gemma4_12b_block7` @ `2fa72e76` | `312e893c`, `864d974e` |
   | `gemma4`     | `gemma-4-12b-it-mtp-gguf`                | `mtp-gemma-4-12B-it-Q4_0.gguf`                        | `ggml-org/gemma-4-12B-it-GGUF` @ `e3e68173`         | `b894e614`             |
   | `flash-next` | `<ssd>/qwen3.8-flash-next-gguf/MTP-Q8_0` | `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (repo path `MTP/`) | `unsloth/Qwen3.8-Flash-Next-GGUF` @ `766911a6`      | `58a47d6a`             |
   | `flash-next` | `<ssd>/qwen3.8-flash-next-mtp-aux`       | `config.json`, `model.safetensors` (built)            | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4d`              | `d77ccad8`, `00474214` |

   ```bash
   hf download z-lab/Qwen3.8-27B-DFlash2 config.json model.safetensors \
     --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 --local-dir .cache/models/qwen3.8-27b-dflash2
   hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
     --revision 2d9571f8ce46e151f61c6499c99dee6079e1d610 --local-dir .cache/models/qwen3.8-27b-dflash2-gguf
   hf download deepseek-ai/dspark_gemma4_12b_block7 config.json model.safetensors \
     --revision 2fa72e765eec2965fc4d86a8663ce6769eba6218 --local-dir .cache/models/dspark_gemma4_12b_block7
   hf download ggml-org/gemma-4-12B-it-GGUF mtp-gemma-4-12B-it-Q4_0.gguf \
     --revision e3e681731089efaa3f0917336944ac64752db8ba --local-dir .cache/models/gemma-4-12b-it-mtp-gguf
   ```

   Flash Next's two MTP inputs go on the internal SSD next to the splits (`<ssd>` below). The mlx-node folder is built
   from the 28 of 131 source shards that hold `mtp.*` (55 GB); `mtp-aux.ts` copies the 31 tensors byte for byte,
   drops `vision_config` from `config.json`, and refuses output that does not match both pins:

   ```bash
   hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
     --revision 766911a6b7369840a91dbcd95f9f997acaab6cd6 --local-dir <ssd>/download
   mkdir -p <ssd>/qwen3.8-flash-next-gguf/MTP-Q8_0
   mv <ssd>/download/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf <ssd>/qwen3.8-flash-next-gguf/MTP-Q8_0/
   REV=de4b8e4d43b917e7706784d8bb445c9af86a3540 SRC=<ssd>/qwen3.8-flash-next
   hf download Qwen/Qwen3.8-Flash-Next config.json model.safetensors.index.json --revision $REV --local-dir $SRC
   hf download Qwen/Qwen3.8-Flash-Next --revision $REV --local-dir $SRC \
     $(jq -r '.weight_map | to_entries[] | select(.key | startswith("mtp.")) | .value' $SRC/model.safetensors.index.json | sort -u)
   oxnode scripts/perf-tracker/flash-next/mtp-aux.ts --source $SRC --out <ssd>/qwen3.8-flash-next-mtp-aux
   ```

   The source shards are not read again after `mtp-aux.ts`; delete them if you need the space.

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
oxnode scripts/perf-tracker/flash-next/run.ts all --model-dir <ssd>/qwen3.8-flash-next-gguf/UD-Q4_K_XL \
  --mtp-head <ssd>/qwen3.8-flash-next-gguf/MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --mtp-aux-dir <ssd>/qwen3.8-flash-next-mtp-aux
```

- Each `all` makes `.cache/benchmarks/perf-tracker/<suite>-<UTC stamp>/` and keeps the Mac awake with `caffeinate`.
- Single steps take `--out <run dir>`. After a failure, run the failed step again with the same `--out`. Flash Next
  reads `--model-dir`, `--mtp-head` and `--mtp-aux-dir` only in `preflight`; later steps read them from
  `provenance.json`.
- Resume keeps finished cells. A cell is one runtime on one case: one process, its warmup and all repetitions. A cell
  that stopped partway moves to `discarded/<stamp>/` and reruns in full, and `report` refuses a cell whose samples
  come from more than one process. Gemma runs every sample in its own process, so it keeps each finished sample.
- `report --out <run dir>` rebuilds the outputs of a finished run.
- Qwen3.8 27B only: `check-ids --out <run dir>` (about 2 min, untimed) proves that the native session admits exactly
  the pinned prompt IDs. Run the steps one by one to include it: `preflight`, `prewarm`, `check-ids`, `matrix`,
  `report`. Flash Next runs the same proof inside `prewarm`.
- Start a long suite in the background and watch its log for the end line and for errors. Do not poll it in a loop.

## Outputs and checks

| File              | Content                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| `provenance.json` | OS, chip, commits, `origin/main`, addon/llama/model hashes, llama.cpp args, load                            |
| `samples.csv`     | one row per measured request: the #142 columns plus draft steps, drafted and accepted tokens                |
| `summary.json`    | medians, min/max and mlx-node/llama.cpp ratios per cell, and the acceptance per cell                        |
| `chart.svg`       | prefill and decode medians with min–max whiskers; the note names the speculation                            |
| `compare.md`      | new medians against `scripts/perf-tracker/<suite>/baseline.csv` (previous entry), and the speculation table |

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
6. Read the speculation table. `report` refuses a sample that did not speculate, but a low acceptance rate still
   slows decode. llama.cpp reports no verify-step count, so only mlx-node has tokens per verify step.

## Publish to issue #142 (ask the user first)

1. Push the chart and data to a new asset branch, for example `perf-tracker/<suite>-<date>` with `chart.svg`,
   `samples.csv` and `summary.json`. Link the chart as
   `https://raw.githubusercontent.com/mlx-node/mlx-node/<asset commit>/chart.svg`, pinned to the commit.
2. Write one dated comment per suite, in the shape of the earlier entries:
   - a headline with the ratios, the chart, and the median table with ratios;
   - request time (Gemma); the weight budget per load and the output agreement (Flash Next);
   - the speculation per runtime and its acceptance; for Gemma, that the methods differ and why;
   - conditions: machine, OS, power, load, workload, matched settings, repetitions, timing;
   - revisions: mlx-node commit, MLX submodule, llama.cpp release tag and commit, checkpoint hash;
   - the change against the previous entry (`compare.md`) and what changed in between;
   - the samples CSV and the fingerprints in `<details>` blocks.
3. Update the model's row in the issue body (latest measurement and link). Keep every older entry.
4. Show the draft to the user. Run `gh issue comment`, `gh issue edit` or `git push` only after they approve.

## Updating the baselines

After a new entry is posted, copy its `samples.csv` to `scripts/perf-tracker/<suite>/baseline.csv` in a separate
commit, so the next `compare.md` compares with the latest entry. A baseline belongs to one fixture version: the
`-v1` baselines ran without speculation and were removed with the move to `-v2`, so no suite has a baseline until
its first `-v2` entry.
