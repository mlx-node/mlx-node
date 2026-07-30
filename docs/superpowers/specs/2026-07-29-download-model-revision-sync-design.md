# `mlx download model`: revision-aware update sync

**Date:** 2026-07-29
**Status:** Approved (approach + design approved in session)

## Problem

`yarn mlx download model -m <repo>` refuses to download when the output directory
already looks complete. The gate (`isModelAlreadyDownloaded`,
`packages/cli/src/commands/download-model.ts:332-368`, called at `:628`) is
local-only and revision-blind: it checks that `config.json` plus the weight
file(s) exist on disk and exits before any network call. When the HuggingFace
repo is updated upstream, the only way to get the new files is `rm -rf` of the
whole model directory. There is no `--force` flag (strict `parseArgs` rejects
it) and the CLI writes no revision metadata at download time.

The dashboard (`packages/dashboard/src/download.ts`) already solves this with a
pinned commit sha and a `.mlx-download-complete.json` marker; this design ports
that idea to the CLI.

## Goals

- `mlx download model` detects upstream changes and syncs only changed files.
- A repo re-upload with identical file sizes (re-trained weights) is detected.
- Existing model dirs downloaded before this change heal automatically on the
  next run — no `rm -rf` needed.
- The dashboard stops classifying CLI-downloaded dirs as foreign
  (`packages/dashboard/src/catalog.ts:102-106`) because both now write the same
  marker.
- Offline / API-failure behavior degrades to today's behavior, never worse.

## Non-goals

- No automatic download of untracked nested checkpoints such as
  `original/*.safetensors`. The remote truth listing is recursive so previously
  marked nested sidecars can be verified, while fresh default selection stays
  root-only.
- No byte-range resume, no staging directory, no dashboard UI changes.
- No change to `mlx download dataset`.

## Design

### New flow of `run()` (packages/cli/src/commands/download-model.ts)

```
resolve token, outputDir                       (unchanged)
remoteSha = resolveRemoteRevision(repo)        NEW — modelInfo, 1 API call, retried
        │ null on failure → legacy path: existing local-only gates + warn
        ▼
outputDir exists?
 ├─ no  → fresh download, pinned to remoteSha
 └─ yes
     completion = readCompletion(outputDir)          NEW
     ├─ completion && completion.repo == repo
     │   && completion.revision == remoteSha
     │   && local completeness predicate passes
     │   && !force
     │        → "Already up to date (rev <sha7>)" — exit 0
     └─ else → SYNC pass pinned to remoteSha:
            for each selected remote file:
              not on disk / size differs → download
              size matches               → hash local vs remote oid
                                            equal → skip, differ → download
            prune: files listed in OLD completion.files that are no longer
                   in the recursive remote tree → delete (full runs only)
            write new marker
```

### Marker

- Filename: `.mlx-download-complete.json` in the model output dir — identical
  to the dashboard's `DOWNLOAD_COMPLETE_MARKER`
  (`packages/dashboard/src/models.ts:71`).
- Shape:
  `{ repo: string, revision: string, files: string[], scope?: "full" | "partial", completedAt: string }`
  — identical to the dashboard's `DownloadCompletion`
  (`packages/dashboard/src/models.ts:74-83`).
- New CLI module `packages/cli/src/commands/download-marker.ts` exporting the
  constant, the type, `readCompletion(dir)` (returns `null` on
  missing/malformed, never throws), and `writeCompletion(dir, completion)`
  (write to `<marker>.tmp`, then rename — no torn marker).
- The CLI must NOT import from `packages/dashboard` (no cli↔dashboard package
  dependency). Instead a contract test pins the two implementations together
  (see Testing).
- The marker is written on every success branch: default safetensors path,
  GGUF branch (`download-model.ts:780-788`), and glob branch (`:789-797`),
  after the branch's existing verification passes.
- Glob results and in-progress syncs use `scope: "partial"`; CLI/dashboard
  completion gates reject them while ownership checks continue to accept them.
- A marker naming another repo refuses the output directory. A marker-less
  legacy full sync removes only superseded standard top-level SafeTensors
  artifacts before publishing the new revision.

### Revision resolution

- `resolveRemoteRevision(repo, accessToken)` calls `modelInfo({ name,
additionalFields: ['sha'] })` from `@huggingface/hub`, wrapped in the
  existing `withRetries`. Result must match `/^[0-9a-f]{40}$/i`; otherwise
  return `null`.
- `null` (offline, 401 on gated repo, API change) → print a warning
  ("could not check for updates") and run the existing local-only gates
  exactly as today. Local-complete still exits 0; local-incomplete still
  downloads from mutable `main` as today.

### Pinning

- When `remoteSha` is known, both `listFiles` (`download-model.ts:282`) and
  `downloadFileToCacheDir` (`:765`) receive `revision: remoteSha` so a push
  mid-download cannot mix two revisions. The hub cache is keyed by sha, so
  this is free.

### Per-file sync decision

- New pure helper `fileUpToDate(localPath, remoteFile)`:
  - local file missing → `false`
  - local size ≠ remote size → `false`
  - sizes equal → hash the local file and compare with the remote oid:
    LFS files → sha256 vs `lfs.oid`; non-LFS files → git blob sha1
    (`sha1("blob <size>\0" + bytes)`) vs `oid`. Port of the dashboard's
    verify logic (`packages/dashboard/src/download.ts:162-178`).
- Hashing runs only inside a sync pass (revision mismatch / no marker /
  `--force`) and only for size-matching files, so steady-state runs never
  read weight bytes.
- If the recursive tree listing does not carry oid/lfs info for a file, fetch it
  via `pathsInfo({ paths, expand: true })` for the size-matching candidates
  only (implementation detail; verify what `listFiles` returns).
- The existing size-only `isLocalCopyComplete` fast-skip (`:501-509`, used at
  `:757`) is replaced by `fileUpToDate` in the sync pass; the fresh-download
  path may keep the size-only skip (it exists to resume interrupted fresh
  downloads, where a same-size wrong-content file is not a realistic state —
  but see Testing: the sync path must NOT use size-only).

### Prune

- After all downloads succeed, a full run deletes files that appear in the OLD
  `completion.files` but not in the recursive remote tree. Glob runs never
  prune.
- With no marker, only superseded standard top-level SafeTensors artifacts are
  eligible, and only when the remote has a replacement SafeTensors layout.
- Paths are resolved inside `outputDir` and must not escape it
  (reject entries containing `..` or absolute paths).

### `--force`

- New boolean flag in `parseArgs` (`download-model.ts:543-573`), plus help
  text (`:203-240`), `docs/cli.md`, and `packages/cli/README.md`.
- Semantics: skip the up-to-date short-circuit and run a full sync pass.
  It does NOT blind-redownload: unchanged files still skip via hash.

## Error handling

- `modelInfo` failure → warning + legacy behavior (see above).
- Marker unreadable/malformed → treated as absent (sync pass runs, marker is
  rewritten). Never throws.
- A failed sync (download error after retries) → no marker write, no prune;
  the old marker (if any) stays. Next run re-enters the sync pass.
- Marker write is temp+rename so a crash cannot leave a truncated marker.

## Testing

New/updated tests in `__test__/cli/` (same style as
`__test__/cli/download-model.test.ts` — pure helpers only; `run()` stays
untested, a pre-existing gap this change does not widen). Each test is named
for the mutation it catches:

1. `readCompletion`: missing file → null; malformed JSON → null; wrong field
   types → null; roundtrip with `writeCompletion` → equal.
2. Sync decision (`needsSync(completion, remoteSha, force)` or equivalent):
   marker matching sha → no sync; sha mismatch → sync; no marker → sync;
   `force` → sync even when matching; `repo` mismatch → sync.
3. `fileUpToDate`: same-size DIFFERENT-content file → `false` (this is the
   mutation a size-only compare would miss — the test must fail if hashing is
   removed); sha256 path for LFS-shaped remote entries; git-blob-sha1 path for
   small files.
4. Prune scoping: a local file NOT listed in the old marker is never deleted;
   a marker entry containing `..` is rejected.
5. Contract test vs dashboard: import `packages/dashboard/src/models.ts`
   directly (test-only import, not a package dependency) and assert (a) the
   marker filename constants are identical, (b) a marker produced by the CLI's
   `writeCompletion` is accepted by the dashboard's `readCompletion`.

## Docs

- `docs/cli.md` download section: document update detection, `--force`, and
  the marker file (also add the previously undocumented `--cache-dir`).
- Help text in `download-model.ts`.
