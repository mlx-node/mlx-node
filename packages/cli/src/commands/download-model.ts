import { existsSync, readFileSync, statSync } from 'node:fs';
import { readdir, copyFile, rm } from 'node:fs/promises';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { listFiles, downloadFileToCacheDir, modelInfo, type ListFileEntry } from '@huggingface/hub';
// Leaf subpath on purpose: `@mlx-node/server/host` would dlopen the native
// addon, and downloading a model must work before any of that is needed.
import { isGgufCompanionName } from '@mlx-node/lm/model-discovery';
import { resolveModelsDir } from '@mlx-node/server/host/paths';

import { ensureDir, formatBytes } from '../utils.js';
import { markCompletionPartial, readCompletion, writeCompletion, type DownloadCompletion } from './download-marker.js';
import {
  assertCompletionRepoCompatible,
  buildMarkerFiles,
  canShortCircuitFullRun,
  computeLegacyWeightPruneList,
  computePruneList,
  fileUpToDate,
  isCompletionCurrent,
  markerRevisionToClaim,
  sameRepoCompletion,
} from './download-sync.js';
import { resolveHuggingFaceToken, setToken } from './hf-token.js';

const DEFAULT_CACHE_DIR = join(homedir(), '.cache', 'huggingface');

/** Attempts per file before a download gives up. */
const MAX_FETCH_ATTEMPTS = 4;

/** Base backoff; attempt N waits `BASE << (N - 1)` ms (1s, 2s, 4s). */
const RETRY_BASE_MS = 1_000;

/** Node errno values no retry can fix: the LOCAL filesystem said no. */
const PERMANENT_FS_CODES = new Set([
  'ENOSPC', // disk full
  'EDQUOT', // over quota
  'EACCES', // not permitted
  'EPERM',
  'EROFS', // read-only filesystem
  'EISDIR',
  'ENOTDIR',
  'ENAMETOOLONG',
  'EFBIG',
  'EXDEV',
]);

/**
 * Hub-client PROTOCOL refusals: the bytes arrived and the parser rejected
 * them. `XetBlob` throws these from inside the chunk reader
 * (`XetBlob.ts:387,395`) after part of the shard has already streamed, and
 * they carry no status and no errno — so without this they land on the
 * default-retry branch and cost three more full multi-GB transfers to fail
 * identically. Deterministic in the payload, so a retry cannot change them.
 *
 * Deliberately NOT here: `Failed to fetch all data for term …`
 * (`XetBlob.ts:473`), which means the transfer was TRUNCATED. That is the
 * transient case, and it must keep retrying.
 */
const PERMANENT_PROTOCOL_MESSAGES = ['Unsupported chunk version', 'Unsupported compression scheme'];

/** The human-readable text of a failure, whatever shape it arrived in. */
function failureText(error: unknown): string {
  if (typeof error === 'string') return error;
  if (error instanceof Error) return error.message;
  return String(error);
}

/**
 * A hub error flattened to text; see {@link isRetriableFetchError}.
 *
 * Only matches when the failing response had a NON-JSON body. `createApiError`
 * (`@huggingface/hub@2.13.2`, `src/error.ts:9-26`) builds this exact prefix and
 * then REPLACES the whole message with `json.error || json.message` whenever
 * the response is `application/json`, keeping only a `. URL: …` trailer. The
 * status is not recoverable from that text — it survives only on the error
 * OBJECT, which the blob-stream path throws away. So a JSON-bodied failure on
 * the content GET reaches the default-retry branch regardless of its status.
 * That is a known looseness, not an oversight: it errs toward retrying, which
 * is the safe direction here (see the harm asymmetry below).
 */
const FLATTENED_STATUS = /^Api error with status (\d{3})\b/;

/**
 * 408 is transient by definition (RFC 9110 §15.5.9 — the server did not get a
 * complete request in time, and "the client MAY repeat that request"), so it
 * belongs with 429 and 5xx rather than with the settled 4xx answers.
 *
 * Included as hardening, not as a fix for an observed failure: nothing in
 * `@huggingface/hub@2.13.2` filters, branches on, or internally retries 408
 * (its only repeat-a-request logic is a one-shot 403 token refresh in
 * `XetBlob.ts:307`), so if a CDN ever emits one it lands here and aborts a
 * multi-GB download for free. Costs one comparison.
 *
 * 425 Too Early is deliberately NOT here: it requires TLS 0-RTT early data,
 * which this client never sends. Adding it would be speculation, and the
 * default-retry branch already covers anything unmodelled.
 */
function isRetriableStatus(status: number): boolean {
  return status >= 500 || status === 429 || status === 408;
}

/**
 * Whether a failed download step is worth repeating.
 *
 * Three shapes reach here, because the hub client does not normalize them:
 *
 *  - A `HubApiError` with a numeric `statusCode`. This is what CI hit (a 500
 *    from the Xet CDN), thrown by `createApiError` and carried out through the
 *    stream's async-iterator `pull`, so the object survives intact.
 *  - A bare STRING. `WebBlob.stream()` and `XetBlob.stream()` abort their
 *    writable with `error.message` rather than the error, so a failure on the
 *    content GET — the multi-GB shard itself — arrives with no type at all
 *    (verified: `typeof e === 'string'`, `e instanceof Error === false`). A
 *    predicate that only understands `Error` refuses to retry the single most
 *    important case, so the status is read back out of the text.
 *  - A plain `Error` with a Node errno and no status. This is BOTH a transport
 *    failure and a local filesystem failure, which is why the errno matters:
 *    `downloadFileToCacheDir` mkdirs, streams to `<blob>.incomplete`, renames,
 *    then symlinks, and wraps none of it.
 *
 * Unknown status-less failures default to RETRY, deliberately. The transport
 * failure space is open-ended and library-version-dependent — `fetch` rejects
 * `TypeError: fetch failed` on connect but `TypeError: terminated` on a
 * mid-body reset, and a timeout is a `DOMException` with a NUMERIC code — so an
 * allowlist of known network errors turns every unmodelled one into a hard
 * abort of a 30 GB download, which is the failure this retry exists to prevent.
 * The harm is asymmetric, but NOT as cheaply as "7 s of backoff": because each
 * attempt truncates `<blob>.incomplete` and re-GETs without a Range header, a
 * wrongly-retried failure also re-transfers the shard up to three more times.
 * That is why the two categories which are deterministic AND status-less —
 * {@link PERMANENT_FS_CODES} and {@link PERMANENT_PROTOCOL_MESSAGES} — are
 * named explicitly instead of being left to the default. Refusing to retry
 * something transient still costs the whole download, so everything else
 * unmodelled keeps defaulting to retry.
 *
 * Never retried: 4xx other than 429 and 408 (401/403 is no token or no access,
 * 404 is the wrong repo or revision) and the filesystem refusals above. Those
 * are settled answers, and repeating them only buries the message the user
 * needs.
 *
 * That last paragraph holds for the ERROR-OBJECT shape, which keeps its
 * `statusCode`. It does NOT hold for the flattened-string shape when the
 * response body was JSON: the hub client overwrites the message with the
 * server's own text and the status is gone, so such a failure takes the
 * default-retry branch whatever it was. See {@link FLATTENED_STATUS}. Closing
 * that would mean pattern-matching human-readable server prose, which is the
 * allowlist this function exists to avoid.
 */
export function isRetriableFetchError(error: unknown): boolean {
  if (typeof error !== 'string' && !(error instanceof Error)) return false;

  // Checked on BOTH shapes: the chunk reader's refusals reach us as a bare
  // string through the blob stream, but as an Error when awaited directly.
  const text = failureText(error);
  if (PERMANENT_PROTOCOL_MESSAGES.some((m) => text.includes(m))) return false;

  if (typeof error === 'string') {
    const status = FLATTENED_STATUS.exec(error);
    return status ? isRetriableStatus(Number(status[1])) : true;
  }

  const status = (error as { statusCode?: unknown }).statusCode;
  if (typeof status === 'number') return isRetriableStatus(status);

  const code = (error as NodeJS.ErrnoException).code;
  // EMFILE/ENFILE ("too many open files") are deliberately absent: those do
  // clear on their own, so they stay retriable.
  if (typeof code === 'string' && PERMANENT_FS_CODES.has(code)) return false;

  return true;
}

/**
 * Run `attempt`, repeating it while the failure looks transient.
 *
 * A checkpoint is tens of files and tens of gigabytes pulled from a CDN, so a
 * single 5xx somewhere in the set is ordinary — and without this ONE of them
 * aborted the whole multi-GB download, discarding every completed file. This
 * is what CI hit:
 *
 *   HubApiError: Api error with status 500
 *     data: { message: 'Key service error: Timeout occurred while creating a new object' }
 *
 * Safe to repeat, but NOT free: `downloadFileToCacheDir` short-circuits on a
 * blob that is already COMPLETE, so finished files are never re-fetched — a
 * PARTIAL one is not resumed. It reopens `<blob>.incomplete` with `'w'`
 * (truncating it) and re-issues the GET without a Range header, so an
 * interrupted shard restarts from byte 0. That is what bounds the attempts at
 * {@link MAX_FETCH_ATTEMPTS} rather than retrying indefinitely, and it is why
 * a local disk-full failure must not be retried at all.
 */
export async function withRetries<T>(what: string, attempt: () => Promise<T>): Promise<T> {
  for (let n = 1; ; n++) {
    try {
      return await attempt();
    } catch (error) {
      if (n >= MAX_FETCH_ATTEMPTS || !isRetriableFetchError(error)) throw error;
      const waitMs = RETRY_BASE_MS << (n - 1);
      // The status when there is one, else the failure's text — never the error
      // object itself, which stringifies to `[object Object]` and would make the
      // one line explaining the pause useless. `failureText` rather than
      // `.message` because the content-GET failures this retry exists for
      // arrive as bare STRINGS, on which `.message` is `undefined` — printing
      // `failed (undefined)` for exactly the case that matters most.
      const status = (error as { statusCode?: unknown }).statusCode;
      const reason = typeof status === 'number' ? `HTTP ${status}` : failureText(error);
      console.warn(`    ${what} failed (${reason}); retry ${n}/${MAX_FETCH_ATTEMPTS - 1} in ${waitMs}ms`);
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }
}

const DEFAULT_MODEL = 'Qwen/Qwen3-0.6B';

function printHelp(): void {
  console.log(`
Download a model from HuggingFace

Usage:
  mlx download model [options]

Options:
  -m, --model <name>      HuggingFace model name (default: ${DEFAULT_MODEL})
  -o, --output <dir>      Output directory (default: ~/.mlx-node/models/<model-slug>;
                          honors MLX_MODELS_DIR env and ~/.mlx-node/config.json modelsDir)
  -g, --glob <pattern>    Filter files by glob pattern (can be repeated)
  --assets-repo <repo>    Base-model repo to fetch tokenizer/config sidecar
                          files from after the download. GGUF quantization
                          repos ship weights only; without an official
                          tokenizer the runtime extracts the embedded one,
                          which strips the tool-call wrapper on decode.
  --complete              The selection IS the complete prescribed model (a
                          catalog install): write the completion marker as a
                          full-model record. Without it a --glob run is
                          recorded as partial, which the dashboard never
                          treats as installed and never offers updates for.
  --force                 Re-sync against upstream even if the local copy
                          looks up to date (changed files re-download,
                          unchanged files are skipped by content hash)
  --cache-dir <dir>       HuggingFace cache directory (default: ~/.cache/huggingface)
  -h, --help              Show this help message
  --set-token             Set HuggingFace token

Glob Filtering:
  Use --glob to download only specific files from a repo. This is especially
  useful for GGUF repos that contain many quantization variants. Patterns use
  simple wildcard matching (* matches any characters).

  Multiple --glob flags can be combined; a file is included if it matches ANY
  of the patterns.

Examples:
  mlx download model
  mlx download model --model Qwen/Qwen3-1.7B --output ~/.mlx-node/models/qwen3-1.7b

  # Download only the BF16 GGUF variant
  mlx download model -m unsloth/Qwen3.5-9B-GGUF -g "*BF16*"

  # Download only Q4_K_M and Q8_0 variants
  mlx download model -m unsloth/Qwen3.5-9B-GGUF -g "*Q4_K_M*" -g "*Q8_0*"

  # Download all .gguf files (skip everything else)
  mlx download model -m unsloth/Qwen3.5-9B-GGUF -g "*.gguf"

  # One Unsloth Dynamic variant plus the base-model tokenizer sidecars that
  # every GGUF quantization repo lacks (required for correct tool calling):
  mlx download model -m unsloth/Qwen3.8-27B-GGUF -g "*UD-Q4_K_XL*" \\
    --assets-repo Qwen/Qwen3.8-27B
`);
}

const CORE_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
];

/**
 * Sidecar files a GGUF quantization repo typically lacks, fetched from
 * `--assets-repo` (the base model's repo). Names absent from that repo are
 * skipped, so one fixed list serves every model family.
 *
 * Mandatory for correct tool calling, not a nicety: when no sidecar
 * `tokenizer.json` sits next to the `.gguf`, the native runtime extracts the
 * embedded tokenizer, and that extraction marks `<tool_call>`/`</tool_call>`
 * `special: true` (the official files mark them false) — every decode path
 * then skips special tokens, the tool-call wrapper is stripped, and the model
 * silently "answers" instead of calling tools.
 */
const ASSET_SIDECAR_CANDIDATES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'chat_template.jinja',
  'generation_config.json',
  'preprocessor_config.json',
  'video_preprocessor_config.json',
  'processor_config.json',
];

/** Convert a simple glob pattern (with * wildcards) to a RegExp */
function globToRegex(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
  return new RegExp(`^${escaped}$`, 'i');
}

/** Check if a filename matches any of the glob patterns */
function matchesAnyGlob(filename: string, patterns: RegExp[]): boolean {
  return patterns.some((re) => re.test(filename));
}

/**
 * List the repo's files, retrying a transient listing failure.
 *
 * Retried as a WHOLE call: every accumulator below is local to one invocation,
 * so a second attempt starts from an empty set rather than appending to a
 * half-walked one. `listFiles` is paginated, so a 5xx can land part-way through
 * a large repo's walk.
 */
async function getModelFiles(
  modelName: string,
  accessToken?: string,
  globPatterns?: string[],
  revision?: string,
  previouslyTrackedPaths?: string[],
) {
  return withRetries(`listing ${modelName}`, () =>
    listModelFilesOnce(modelName, accessToken, globPatterns, revision, previouslyTrackedPaths),
  );
}

async function listModelFilesOnce(
  modelName: string,
  accessToken?: string,
  globPatterns?: string[],
  revision?: string,
  previouslyTrackedPaths?: string[],
) {
  let totalSize = 0;
  const filesToDownload: ListFileEntry[] = [];
  const allFiles: ListFileEntry[] = [];

  // Compile glob patterns if provided
  const globs = globPatterns?.map(globToRegex);
  const tracked = new Set(previouslyTrackedPaths);

  for await (const file of listFiles({
    repo: { type: 'model', name: modelName },
    accessToken,
    revision,
    recursive: true,
  })) {
    if (file.type === 'directory') continue;
    allFiles.push(file);

    if (globs) {
      // When glob patterns are active, include files that match the pattern
      // OR are essential metadata files (config, tokenizer)
      const basename = file.path.split('/').pop() || file.path;
      if (matchesAnyGlob(basename, globs) || matchesAnyGlob(file.path, globs)) {
        filesToDownload.push(file);
        if (file.size) totalSize += file.size;
      } else if (CORE_FILES.includes(file.path)) {
        // Always include core config/tokenizer files
        filesToDownload.push(file);
        if (file.size) totalSize += file.size;
      }
    } else {
      // The recursive listing supplies complete remote truth for nested marker
      // entries, but a fresh full run retains the historical root-only
      // selection (some repos carry another full checkpoint under original/).
      // Nested files already tracked by a dashboard/CLI marker are included so
      // a full sync verifies them before advancing the revision.
      if (isDefaultModelDownloadPath(file.path, tracked)) {
        filesToDownload.push(file);
        if (file.size) {
          totalSize += file.size;
        }
      }
    }
  }

  return { totalSize, filesToDownload, allFiles };
}

/** Root-default selection plus nested files already claimed by a prior marker. */
export function isDefaultModelDownloadPath(path: string, previouslyTracked: ReadonlySet<string>): boolean {
  if (previouslyTracked.has(path)) return true;
  if (path.includes('/')) return false;
  return (
    CORE_FILES.includes(path) ||
    path.endsWith('.safetensors') ||
    path.endsWith('.json') ||
    path.endsWith('.pdiparams') ||
    path.endsWith('.yml') ||
    path.endsWith('.gguf') ||
    path.endsWith('.jinja')
  );
}

/**
 * The current commit sha of the repo's `main`, or `null` when it cannot be
 * resolved (offline, missing auth on a gated repo, API change). `null` makes
 * the caller fall back to the legacy local-only behavior — the update check
 * must never make the command less capable than it was without it.
 */
async function resolveRemoteRevision(modelName: string, accessToken?: string): Promise<string | null> {
  try {
    const info = await withRetries(`resolving latest revision of ${modelName}`, () =>
      modelInfo({ name: modelName, additionalFields: ['sha'], accessToken }),
    );
    const sha: unknown = (info as { sha?: unknown }).sha;
    return typeof sha === 'string' && /^[0-9a-f]{40}$/i.test(sha) ? sha : null;
  } catch {
    return null;
  }
}

/**
 * Pre-flight check: does `outputDir` already hold a complete model download?
 *
 * Returns true ONLY when we can declare "already downloaded" with confidence.
 * For sharded models we additionally parse `model.safetensors.index.json`
 * and verify every shard listed under `weight_map` is present on disk —
 * otherwise an interrupted prior run that landed the index but not all
 * shards would silently exit as "already downloaded" and leave the user
 * with a broken local copy.
 *
 * Pure function: takes the directory + its file list and only reads the
 * index file when sharded-model checks need it. No network, no other I/O.
 */
export function isModelAlreadyDownloaded(outputDir: string, files: string[]): boolean {
  const fileSet = new Set(files);
  const hasConfig = fileSet.has('config.json');
  if (!hasConfig) return false;

  const hasSingleModel = fileSet.has('model.safetensors');
  const hasPaddleModel = fileSet.has('inference.pdiparams');
  if (hasSingleModel || hasPaddleModel) return true;

  const hasShardedModel = fileSet.has('model.safetensors.index.json');
  if (!hasShardedModel) return false;

  // Sharded: verify every shard the index references actually exists on disk.
  let parsed: unknown;
  try {
    const raw = readFileSync(join(outputDir, 'model.safetensors.index.json'), 'utf8');
    parsed = JSON.parse(raw);
  } catch {
    return false;
  }
  const weightMap =
    parsed && typeof parsed === 'object' && 'weight_map' in parsed
      ? (parsed as { weight_map?: unknown }).weight_map
      : undefined;
  if (!weightMap || typeof weightMap !== 'object') return false;

  const shardFilenames = new Set<string>();
  for (const value of Object.values(weightMap as Record<string, unknown>)) {
    if (typeof value === 'string') shardFilenames.add(value);
  }
  if (shardFilenames.size === 0) return false;

  for (const shard of shardFilenames) {
    if (!existsSync(join(outputDir, shard))) return false;
  }
  return true;
}

/**
 * Pre-flight check: do any files in `outputDir` actually match the user's
 * glob patterns?
 *
 * Returns true ONLY when at least one existing file matches a glob —
 * proving the requested variant is already on disk. CORE_FILES (config,
 * tokenizer, etc.) are deliberately excluded: they're auxiliary metadata
 * laid down by ANY prior download, so counting them would falsely report
 * a Q4 variant as "already downloaded" the moment a Q8 variant had been
 * fetched (which already drops config.json + tokenizer.json into the
 * same directory).
 *
 * Pure function: no I/O, no network. Caller passes the file list and
 * patterns; helper returns boolean.
 */
export function isGlobVariantPresent(files: string[], globPatterns: string[]): boolean {
  if (globPatterns.length === 0) return false;
  const globs = globPatterns.map(globToRegex);
  return files.some((f) => matchesAnyGlob(f, globs));
}

/**
 * Pre-flight check: is the user's glob-filtered download complete?
 *
 * Returns true ONLY when EVERY remote file whose basename matches any of
 * the supplied glob patterns is also present locally. Returns false on
 * empty intersection (no remote file matches any glob — the downstream
 * "no files matched" path handles that case) and false on empty manifest
 * (likely upstream error).
 *
 * Why this exists: `isGlobVariantPresent` only checks for AT LEAST ONE
 * local hit. If a prior `--glob "*Q4*"` run was interrupted after fetching
 * one Q4 shard but before the others, rerunning the same command would
 * exit as "Matched files already downloaded" while silently leaving the
 * local copy incomplete. This helper closes that gap by verifying the
 * full glob-matched set against the remote manifest — symmetric to
 * `isGgufRepoComplete` for the no-glob branch.
 *
 * Per-file basenames are compared on both sides (the remote manifest may
 * publish under a sub-directory like `models/foo.gguf` while the local
 * `readdir(outputDir)` is flat) so nested-prefix repos don't false-negative.
 *
 * Pure function: no I/O, no network. Caller fetches the manifest via
 * `getModelFiles` (or equivalent) and hands the basenames in.
 */
export function isGlobMatchedSetComplete(localFiles: string[], remoteFiles: string[], globPatterns: string[]): boolean {
  if (globPatterns.length === 0) return false;
  if (remoteFiles.length === 0) return false;
  const globs = globPatterns.map(globToRegex);
  const localSet = new Set(localFiles);
  let matched = 0;
  for (const remote of remoteFiles) {
    const basename = remote.split('/').pop() ?? remote;
    if (!matchesAnyGlob(basename, globs) && !matchesAnyGlob(remote, globs)) continue;
    matched++;
    if (!localSet.has(basename)) return false;
  }
  // Empty intersection: no remote file matches any glob. The caller's
  // downstream "no files matched the given criteria" path handles this
  // case after listing available variants; declaring "complete" here
  // would be wrong (nothing was supposed to be downloaded but nothing
  // was — that's not the same as "we already have what was requested").
  if (matched === 0) return false;
  return true;
}

/**
 * Pre-flight check: is a no-glob GGUF download complete?
 *
 * Returns true ONLY when EVERY `.gguf` file in the remote repo manifest
 * is also present locally. Returns false otherwise — including the
 * "this is not a GGUF repo" case (the remote has no `.gguf` files), so
 * the caller knows to fall through to the normal sharded/single-file
 * checks rather than mis-routing through the GGUF early-return.
 *
 * Why this exists: the previous early-return was `files.some((f) =>
 * f.endsWith('.gguf'))`, so as soon as ANY `.gguf` was on disk the
 * command silently exited. For multi-variant GGUF repos that publish
 * Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 as separate files, an
 * interrupted prior download that left only Q2_K on disk would short-
 * circuit a re-run without `--glob` and never fetch the rest. The user
 * received zero warning that their local copy was incomplete.
 *
 * The fix is manifest-aware: compare the local file list against the
 * remote `.gguf` filenames and only declare "already downloaded" when
 * every advertised variant is present. Per-file basenames are compared
 * (the remote manifest uses paths like `models/Q4_K_M.gguf` while local
 * `readdir` is flat) so nested-prefix repos don't false-negative.
 *
 * Pure function: no I/O, no network. Caller fetches the manifest via
 * `getModelFiles` (or equivalent) and hands the basenames in.
 */
export function isGgufRepoComplete(localFiles: string[], remoteFiles: string[]): boolean {
  // Empty manifest is never complete — and is almost certainly an upstream
  // error rather than a legitimate empty repo. Falling through to the
  // download loop will surface the real failure (404 / auth rejection /
  // network) instead of masking it as "already downloaded".
  if (remoteFiles.length === 0) return false;
  const remoteGguf = remoteFiles.filter((f) => f.endsWith('.gguf'));
  // Not a GGUF repo — caller should route through the normal
  // `isModelAlreadyDownloaded` (sharded / single-file) path instead.
  if (remoteGguf.length === 0) return false;
  const localSet = new Set(localFiles);
  for (const remote of remoteGguf) {
    // Remote manifest paths can include prefixes (`models/foo.gguf`);
    // local `readdir(outputDir)` is flat. Compare basenames so a repo
    // that publishes under a sub-directory still resolves cleanly.
    const basename = remote.split('/').pop() ?? remote;
    if (!localSet.has(basename)) return false;
  }
  return true;
}

/**
 * Per-file check inside the download loop: is `destPath` already a complete
 * copy of the remote `file`?
 *
 * Returns true when the local file exists AND its byte size matches the
 * remote manifest's `file.size`. Truncated/interrupted prior copies fail
 * the size check and re-copy. When the manifest has no size (`<= 0`), we
 * fall back to existence-only — losing partial-recovery for that one
 * file but matching the pre-d139679 reconciler's `existsSync` check.
 *
 * Why this exists: `downloadFileToCacheDir` is content-addressed (no
 * network re-fetch when the cache already holds the blob), but the
 * subsequent `copyFile` always copies bytes regardless. For a sharded
 * model with the early-return short-circuited (e.g. an interrupted run
 * left the safetensors index but not all shards), the per-file copy
 * would otherwise re-write every already-complete shard to disk on
 * resume — gigabytes of pointless I/O.
 */
export function isLocalCopyComplete(destPath: string, expectedSize: number): boolean {
  if (!existsSync(destPath)) return false;
  if (expectedSize <= 0) return true;
  try {
    return statSync(destPath).size === expectedSize;
  } catch {
    return false;
  }
}

async function verifyDownload(outputDir: string, weightFiles: string[]): Promise<boolean> {
  console.log('\nVerifying download...');

  let allPresent = true;

  const configPath = join(outputDir, 'config.json');
  if (!existsSync(configPath)) {
    console.error('  ✗ Missing required file: config.json');
    allPresent = false;
  } else {
    console.log('  ✓ config.json');
  }

  if (weightFiles.length === 0) {
    console.error('  ✗ No weight files found');
    allPresent = false;
  }

  for (const file of weightFiles) {
    const path = join(outputDir, file);
    if (!existsSync(path)) {
      console.error(`  ✗ Missing weight file: ${file}`);
      allPresent = false;
    } else {
      console.log(`  ✓ ${file}`);
    }
  }

  return allPresent;
}

/** Root-level sidecar entries found in an assetsRepo listing. */
export function pickAssetSidecars(allFiles: ListFileEntry[]): ListFileEntry[] {
  // Candidate names are all root-level; a nested same-named file is not the
  // base model's tokenizer/config. Order follows ASSET_SIDECAR_CANDIDATES so
  // the log and the download order are deterministic.
  const byName = new Map(allFiles.filter((file) => !file.path.includes('/')).map((file) => [file.path, file]));
  return ASSET_SIDECAR_CANDIDATES.flatMap((name) => {
    const file = byName.get(name);
    return file === undefined ? [] : [file];
  });
}

/** Does `file` name a weight artifact a loader can open (companions/projectors do not count)? */
function isLoadableWeightFile(file: string): boolean {
  return (
    file.endsWith('.safetensors') ||
    file.endsWith('.pdiparams') ||
    (file.endsWith('.gguf') && !isGgufCompanionName(basename(file)))
  );
}

/** The assets-repo sidecar selection, resolved before anything is written. */
interface AssetSidecarPlan {
  /** Root-level candidates the primary selection does not already ship. */
  candidates: ListFileEntry[];
  /** The commit the candidates were listed at; `null` only when unresolvable AND optional (repair runs). */
  revision: string | null;
}

/**
 * Work out which tokenizer/config sidecars `--assets-repo` would supply —
 * WITHOUT touching `outputDir`. The result feeds the preflight guards: the
 * prune/certify computation needs the sidecar set before the download loop
 * runs, because both the loop and {@link fetchAssetSidecars} write IN PLACE
 * and any refusal that claims "the installed directory is unchanged" must
 * precede the first byte written.
 *
 * The primary repo's own manifest (`primaryPaths`) is authoritative for
 * everything it ships: a path it lists is never fetched from the assets repo,
 * regardless of content differences — the same precedence the dashboard
 * downloader applies. Without that rule a `--force` run would overwrite a
 * GGUF repo's customized config or template with the base model's copy.
 */
async function planAssetSidecars(opts: {
  assetsRepo: string;
  accessToken: string | undefined;
  primaryPaths: ReadonlySet<string>;
  /** Fail instead of installing against mutable `main` when the revision cannot be pinned. */
  requireRevision: boolean;
}): Promise<AssetSidecarPlan> {
  const resolved = await resolveRemoteRevision(opts.assetsRepo, opts.accessToken);
  if (resolved === null && opts.requireRevision) {
    // Installing against mutable `main` would publish sidecars with no
    // provenance at all: the dashboard would report the model installed and
    // could never surface a later tokenizer or template fix. A clear,
    // retryable failure beats a permanently untracked install.
    throw new Error(
      `Could not resolve the latest revision of "${opts.assetsRepo}"; refusing to install sidecar files that could never receive updates. ` +
        `Re-run when the network is reachable.`,
    );
  }
  const revision = resolved ?? undefined;
  const { allFiles } = await getModelFiles(opts.assetsRepo, opts.accessToken, undefined, revision);
  const candidates = pickAssetSidecars(allFiles).filter((file) => !opts.primaryPaths.has(file.path));
  if (candidates.length === 0) {
    console.warn(`  No tokenizer/config sidecars found in ${opts.assetsRepo}\n`);
  }
  return { candidates, revision: revision ?? null };
}

/**
 * Top up the tokenizer/config sidecars a GGUF repo lacks, applying a
 * {@link planAssetSidecars} plan.
 *
 * Runs after the primary download, is idempotent (a sidecar already on disk
 * with the manifest size — content hash when the run is in verify mode — is
 * skipped), and never overwrites a file the PRIMARY repo itself shipped (the
 * plan already excluded those): `config.json` from a GGUF repo describes its
 * own quantization and wins over the base repo's. Files the primary repo did
 * not ship (tokenizer.json, tokenizer_config.json, chat_template.jinja, …)
 * are written from the assets repo at the plan's resolved revision.
 *
 * `ensured` is ALWAYS the full plan — every candidate is either verified on
 * disk or (re)written — which is what lets the preflight treat the planned
 * names as the installed sidecar set. A fetch that throws never reaches the
 * marker write, same as a failed primary download.
 *
 * Deliberately NOT recorded in the completion marker: marker entries are
 * judged for pruning against the primary repo's remote tree only
 * (`computePruneList`), so listing sidecars there would delete them on a
 * later full sync. The top-up re-verifies them on every downloading run
 * instead. (The dashboard records them because its prune judges the staging
 * manifest, where an unlisted sidecar would be quarantined instead.)
 */
async function fetchAssetSidecars(opts: {
  assetsRepo: string;
  outputDir: string;
  cacheDir: string;
  accessToken: string | undefined;
  plan: AssetSidecarPlan;
}): Promise<{ ensured: string[]; repo: string; revision: string | null }> {
  const { candidates } = opts.plan;
  const revision = opts.plan.revision ?? undefined;
  if (candidates.length === 0) {
    return { ensured: [], repo: opts.assetsRepo, revision: opts.plan.revision };
  }

  console.log(`Fetching ${candidates.length} tokenizer/config sidecar(s) from ${opts.assetsRepo}...\n`);
  // Every candidate that ends up on disk — downloaded or already current — is
  // part of this install and must reach the completion marker.
  const fetched: string[] = [];
  for (const file of candidates) {
    const destPath = join(opts.outputDir, file.path);
    // ALWAYS content-verify a present sidecar, independent of the primary
    // repo's marker: the assets repo evolves on its own revision, so a
    // tokenizer/template fix that keeps the same byte length is invisible to
    // a size check and would otherwise stay stale forever.
    if (existsSync(destPath) && (await fileUpToDate(destPath, file))) {
      console.log(`  ${file.path} — already present and verified, skipping`);
      fetched.push(file.path);
      continue;
    }
    console.log(`  ${file.path} (${formatBytes(file.size)})...`);
    const snapshotPath = await withRetries(`sidecar ${file.path}`, () =>
      downloadFileToCacheDir({
        repo: { type: 'model', name: opts.assetsRepo },
        path: file.path,
        cacheDir: opts.cacheDir,
        accessToken: opts.accessToken,
        revision,
      }),
    );
    await copyFile(snapshotPath, destPath);
    fetched.push(file.path);
  }
  console.log('');
  return { ensured: fetched, repo: opts.assetsRepo, revision: opts.plan.revision };
}

export async function run(argv: string[]) {
  const { values: args } = parseArgs({
    args: argv,
    options: {
      model: {
        type: 'string',
        short: 'm',
        default: DEFAULT_MODEL,
      },
      output: {
        type: 'string',
        short: 'o',
      },
      glob: {
        type: 'string',
        short: 'g',
        multiple: true,
      },
      'assets-repo': {
        type: 'string',
      },
      complete: {
        type: 'boolean',
      },
      force: {
        type: 'boolean',
        default: false,
      },
      help: {
        type: 'boolean',
        short: 'h',
        default: false,
      },
      'set-token': {
        type: 'boolean',
        default: false,
      },
      'cache-dir': {
        type: 'string',
      },
    },
  });

  if (args.help) {
    printHelp();
    return;
  }

  if (args['set-token']) {
    await setToken();
    return;
  }

  const modelName = args.model!;
  const globPatterns = args.glob;
  const assetsRepo = args['assets-repo'];
  const modelSlug = modelName.split('/').pop()!.toLowerCase();
  const outputDir = resolve(args.output ?? join(resolveModelsDir(), modelSlug));

  const HUGGINGFACE_TOKEN = await resolveHuggingFaceToken();

  if (!HUGGINGFACE_TOKEN) {
    console.warn('No HuggingFace token found, the model will download with anonymous access');
  }

  const title = `${modelName} Model Download from HuggingFace`;
  const boxWidth = Math.max(title.length + 6, 58);
  const padding = Math.floor((boxWidth - title.length - 2) / 2);
  const rightPadding = boxWidth - title.length - padding;
  console.log('╔' + '═'.repeat(boxWidth) + '╗');
  console.log('║' + ' '.repeat(padding) + title + ' '.repeat(rightPadding) + '║');
  console.log('╚' + '═'.repeat(boxWidth) + '╝\n');

  console.log(`Model: ${modelName}`);
  if (globPatterns?.length) {
    console.log(`Filter: ${globPatterns.join(', ')}`);
  }
  if (assetsRepo !== undefined) {
    console.log(`Assets: ${assetsRepo}`);
  }
  console.log(`Output: ${outputDir}\n`);

  const force = args.force ?? false;
  const remoteSha = await resolveRemoteRevision(modelName, HUGGINGFACE_TOKEN);
  if (remoteSha === null) {
    console.warn('Could not resolve the latest revision from HuggingFace; update check disabled for this run.\n');
  }

  let cachedManifest: { totalSize: number; filesToDownload: ListFileEntry[]; allFiles: ListFileEntry[] } | null = null;
  let completion: DownloadCompletion | null = null;
  let existingTopLevelFiles: string[] = [];
  // Content-hash verification is needed exactly when local files might be
  // STALE: the dir predates this run and either no current marker proves it
  // matches `remoteSha`, or the user forced a re-verify. A fresh dir or a
  // current-marker dir only ever needs the cheap size check.
  let verifyContent = false;
  /** Resolved sidecar source, recorded in the completion marker when present. */
  let sidecarSource: { repo: string; revision: string } | null = null;
  /** Sidecar paths installed from that source, listed in the completion marker. */
  let sidecarPaths: string[] = [];
  const cacheDir = args['cache-dir'] ? resolve(args['cache-dir']) : DEFAULT_CACHE_DIR;

  /**
   * Run the requested assetsRepo sidecar pass before an early success return.
   *
   * The completion short-circuits below return without ever reaching the
   * download loop, so without this a run whose only remaining work is the
   * tokenizer sidecars — a retry after a sidecar fetch failed, or a
   * directory installed before `--assets-repo` existed — would print success
   * and keep the broken tool-calling state the flag exists to fix.
   * Idempotent: present files are skipped by size (by hash under `--force`).
   */
  const repairSidecarsBeforeReturn = async (): Promise<void> => {
    if (assetsRepo === undefined) return;
    cachedManifest ??= await getModelFiles(modelName, HUGGINGFACE_TOKEN, globPatterns);
    // Only the paths this run's selection ACTUALLY stages count as primary
    // supplied: a file the repo lists but the globs/CORE_FILES do not select
    // is not on disk, so excluding it would lose the sidecar entirely.
    const primaryNames = new Set(cachedManifest.filesToDownload.map((file) => file.path));
    // Repairing an EXISTING install: a transient resolution failure must not
    // fail the run — the marker keeps the provenance it already recorded.
    const plan = await planAssetSidecars({
      assetsRepo,
      accessToken: HUGGINGFACE_TOKEN,
      primaryPaths: primaryNames,
      requireRevision: false,
    });
    // But best-effort still requires an immutable revision. Without one,
    // `fetchAssetSidecars` would download from MUTABLE `main`, overwrite live
    // install files, and the marker write below would keep the OLD pinned
    // `assetsRevision` — provenance that lies about where the bytes came
    // from, and a branch move mid-run would mix revisions. The repair is
    // skipped instead; the install keeps the sidecars it already has and a
    // later run with a resolvable revision repairs it then. This also covers
    // the `planned.length === 0 && revision === null` case: nothing to do.
    if (plan.revision === null) {
      console.warn(`Could not resolve the latest revision of "${assetsRepo}"; skipping sidecar repair for this run.\n`);
      return;
    }
    const planned = plan.candidates.map((file) => file.path);
    // The refusal below claims the install is unchanged, so it must run on
    // the plan BEFORE any sidecar is fetched — `ensured` is always the full
    // plan, making the planned names exactly the set the fetch would leave
    // on disk (a failed fetch throws before the marker write anyway).
    let stale: string[] = [];
    let surviving: string[] = [];
    if (completion !== null && (planned.length > 0 || plan.revision !== null)) {
      const plannedSet = new Set(planned);
      // Obsolete candidates go too: initializing from every old entry would
      // keep a file the assets repo REMOVED on disk and listed while the new
      // revision is recorded — discovery reports current and the runtime keeps
      // consuming it.
      stale = completion.files.filter(
        (file) => ASSET_SIDECAR_CANDIDATES.includes(file) && !primaryNames.has(file) && !plannedSet.has(file),
      );
      const listed = new Set(completion.files);
      // Planned candidates count as present without an existsSync gate: the
      // fetch below ensures every one of them, or the run throws first.
      for (const path of planned) listed.add(path);
      for (const file of stale) listed.delete(file);
      surviving = [...listed].sort();
      if (stale.length > 0 && (!surviving.includes('config.json') || !surviving.some(isLoadableWeightFile))) {
        throw new Error(
          `Refusing to apply "${assetsRepo}" sidecar removals: they would leave no loadable checkpoint. ` +
            `The installed directory is unchanged.`,
        );
      }
    }
    const topUp = await fetchAssetSidecars({
      assetsRepo,
      outputDir,
      cacheDir,
      accessToken: HUGGINGFACE_TOKEN,
      plan,
    });
    sidecarSource = topUp.revision !== null ? { repo: topUp.repo, revision: topUp.revision } : null;
    sidecarPaths = topUp.ensured;
    // The repair just wrote (or verified) files inside an install whose marker
    // the caller is about to accept as current: without re-finalizing, those
    // files stay UNLISTED — a deletion would not invalidate the marker — and an
    // advanced assets revision stays unpinned, which is the update badge that
    // can never clear. Only ever touches an existing marker (this is a repair,
    // not a first install). `ensured` === the plan, so this gate is the same
    // predicate the preflight above computed `stale`/`surviving` under.
    if (completion !== null && (sidecarPaths.length > 0 || sidecarSource !== null)) {
      // Delete BEFORE publishing the updated marker, same rule as the
      // dashboard: a failure must leave the marker able to derive the same set
      // again.
      for (const file of stale) await rm(join(outputDir, file), { force: true });
      await writeCompletion(outputDir, {
        ...completion,
        files: surviving,
        ...(sidecarSource !== null ? { assetsRepo: sidecarSource.repo, assetsRevision: sidecarSource.revision } : {}),
      });
    }
  };

  if (existsSync(outputDir)) {
    existingTopLevelFiles = await readdir(outputDir);
    const hasGguf = existingTopLevelFiles.some((f) => f.endsWith('.gguf'));
    completion = await readCompletion(outputDir);
    assertCompletionRepoCompatible(completion, modelName, outputDir);
    const markerCurrent = remoteSha !== null && isCompletionCurrent(completion, modelName, remoteSha);
    if (remoteSha === null) {
      // Legacy local-only behavior, byte-for-byte: without a resolvable
      // upstream revision there is nothing to compare against.
      if (isModelAlreadyDownloaded(outputDir, existingTopLevelFiles)) {
        await repairSidecarsBeforeReturn();
        console.log('Model already downloaded!\n');
        console.log('To re-download, delete the output directory first:');
        console.log(`   rm -rf ${outputDir}\n`);
        return;
      }
      if (hasGguf && !globPatterns?.length) {
        // Manifest-aware completeness: a multi-variant GGUF repo where a
        // prior interrupted run left only one variant on disk must not exit
        // as "already downloaded" — only short-circuit when EVERY remote
        // `.gguf` is present locally. The extra listing round-trip on the
        // hot path is intentional; correctness wins over the ~200ms saved.
        console.log('Fetching file list from HuggingFace...\n');
        cachedManifest = await getModelFiles(modelName, HUGGINGFACE_TOKEN, globPatterns);
        const remoteBasenames = cachedManifest.allFiles.map((f) => f.path.split('/').pop() ?? f.path);
        if (isGgufRepoComplete(existingTopLevelFiles, remoteBasenames)) {
          await repairSidecarsBeforeReturn();
          console.log('GGUF file(s) already downloaded!\n');
          console.log('To re-download, delete the output directory first:');
          console.log(`   rm -rf ${outputDir}\n`);
          return;
        }
        const missing = cachedManifest.allFiles.filter(
          (f) => f.path.endsWith('.gguf') && !existingTopLevelFiles.includes(f.path.split('/').pop() ?? f.path),
        );
        if (missing.length > 0) {
          console.log(`Detected ${missing.length} missing GGUF file(s); resuming download...`);
          for (const f of missing) {
            console.log(`  ${f.path}${f.size ? ` (${formatBytes(f.size)})` : ''}`);
          }
          console.log('');
        }
      }
      // Glob runs are manifest-aware for the same reason: "at least one
      // local hit" (`isGlobVariantPresent`) does not prove the whole
      // glob-matched set is present, so completeness is checked against
      // the remote manifest before declaring "already downloaded".
      if (hasGguf && globPatterns?.length && isGlobVariantPresent(existingTopLevelFiles, globPatterns)) {
        if (cachedManifest === null) {
          console.log('Fetching file list from HuggingFace...\n');
          cachedManifest = await getModelFiles(modelName, HUGGINGFACE_TOKEN, globPatterns);
        }
        const remoteBasenames = cachedManifest.allFiles.map((f) => f.path.split('/').pop() ?? f.path);
        if (isGlobMatchedSetComplete(existingTopLevelFiles, remoteBasenames, globPatterns)) {
          await repairSidecarsBeforeReturn();
          console.log('Matched files already downloaded!\n');
          console.log('To re-download, delete the output directory first:');
          console.log(`   rm -rf ${outputDir}\n`);
          return;
        }
        const missing = cachedManifest.filesToDownload.filter((f) => {
          const basename = f.path.split('/').pop() ?? f.path;
          return !existingTopLevelFiles.includes(basename);
        });
        if (missing.length > 0) {
          console.log(`Detected ${missing.length} missing file(s); resuming download...`);
          for (const f of missing) {
            console.log(`  ${f.path}${f.size ? ` (${formatBytes(f.size)})` : ''}`);
          }
          console.log('');
        }
      }
    } else if (markerCurrent && !force) {
      // Same revision as the last successful sync. For a whole-model run with
      // every marker file still on disk AND a locally complete model shape
      // there is nothing to do. The shape predicate is load-bearing: a
      // selection-only marker from a `--glob` run must not satisfy a full
      // run's short-circuit. A glob run (selection may not be satisfied by
      // the marker), missing files, or an incomplete shape fall through — at
      // an unchanged revision the size-only per-file skip is trustworthy, so
      // the loop stays cheap.
      if (
        !globPatterns?.length &&
        canShortCircuitFullRun(completion!, outputDir, isModelAlreadyDownloaded(outputDir, existingTopLevelFiles))
      ) {
        await repairSidecarsBeforeReturn();
        console.log(`Model already up to date (revision ${remoteSha.slice(0, 7)}).\n`);
        console.log('Use --force to re-verify every file against upstream.\n');
        return;
      }
    } else {
      verifyContent = true;
      if (force) {
        console.log('Re-verifying every file against upstream (--force)...\n');
      } else if (completion === null) {
        console.log('No download marker found; verifying local files against upstream...\n');
      } else if (completion.scope === 'partial') {
        console.log('Partial or interrupted download marker found; verifying local files against upstream...\n');
      } else {
        console.log(
          `Upstream revision changed (${completion.revision.slice(0, 7)} → ${remoteSha.slice(0, 7)}); syncing...\n`,
        );
      }
      // Keep downloader ownership if this sync fails, while preventing the
      // old marker from satisfying CLI/dashboard completion gates during the
      // in-place update. The final marker write below replaces this atomically.
      // A foreign marker was refused above, before any mutation.
      if (completion !== null && completion.repo === modelName) {
        await writeCompletion(outputDir, markCompletionPartial(completion));
      }
    }
  }

  // Foreign markers were refused before mutation. This still narrows null to
  // genuinely marker-less/invalid legacy directories for finalization.
  const previousCompletion = sameRepoCompletion(completion, modelName);

  // The marker was downgraded to `partial` above so a mid-sync CRASH cannot
  // satisfy completion gates — but a deterministic refusal or a selection
  // with no weights is not a crash: the install it found is still valid, and
  // leaving `partial` would strand it as present-but-uncertified with no
  // update affordance. A run that certifies nothing restores the found
  // marker instead: installed at its previous revision, out of date, and
  // repairable through the normal update path. Best-effort — a failed
  // restore must not mask the error the caller is reporting (the dir then
  // keeps the partial marker, same as an interrupted run).
  const restoreCompletionMarker = async (): Promise<void> => {
    if (previousCompletion === null) return;
    try {
      await writeCompletion(outputDir, previousCompletion);
    } catch {
      // Best-effort only — see above.
    }
  };

  await ensureDir(outputDir);

  // Reuse the manifest fetched during the GGUF completeness check if we
  // already have it, otherwise fetch fresh. Either way the same shape
  // is destructured below.
  if (cachedManifest === null) {
    console.log('Fetching file list from HuggingFace...\n');
  }
  let manifest: { totalSize: number; filesToDownload: ListFileEntry[]; allFiles: ListFileEntry[] };
  try {
    manifest =
      cachedManifest ??
      (await getModelFiles(
        modelName,
        HUGGINGFACE_TOKEN,
        globPatterns,
        remoteSha ?? undefined,
        previousCompletion?.files,
      ));
  } catch (error) {
    // A listing failure certifies nothing and — preflight below having moved
    // every guard input ahead of the writes — cannot have written a byte:
    // restore the found marker rather than stranding the install partial.
    await restoreCompletionMarker();
    throw error;
  }
  const { totalSize, filesToDownload, allFiles } = manifest;

  if (filesToDownload.length === 0) {
    console.error('No files matched the given criteria.\n');
    if (globPatterns?.length) {
      const ggufFiles = allFiles.filter((f) => f.path.endsWith('.gguf'));
      if (ggufFiles.length > 0) {
        console.log('Available GGUF files in this repo:');
        for (const f of ggufFiles) {
          console.log(`  ${f.path} (${formatBytes(f.size)})`);
        }
        console.log(`\nTry: mlx download model -m ${modelName} -g "<pattern>"`);
      }
    }
    // Nothing was downloaded, so the install this run found is still valid —
    // don't leave its marker downgraded (see restoreCompletionMarker).
    await restoreCompletionMarker();
    process.exit(1);
  }

  // ── Preflight: resolve every prune/certify guard input BEFORE the first
  // write. The download loop and the sidecar fetch write IN PLACE into
  // outputDir, so a refusal that claims "the installed directory is
  // unchanged" must fire here — afterwards it would be a lie, and the
  // restored marker would label a mixed-revision directory with the old
  // snapshot's name. Every input is manifest-derived (the only disk reads
  // are the pre-run `existingTopLevelFiles` listing and previous-marker
  // entries, which the loop can only rewrite via paths already inside the
  // selection), so `pruneList`/`certified` are identical to what finalizeSync
  // used to compute post-download.
  let sidecarPlan: AssetSidecarPlan | null = null;
  if (assetsRepo !== undefined) {
    try {
      sidecarPlan = await planAssetSidecars({
        assetsRepo,
        accessToken: HUGGINGFACE_TOKEN,
        primaryPaths: new Set(filesToDownload.map((file) => file.path)),
        requireRevision: true,
      });
    } catch (error) {
      // Same rule as the manifest fetch above: nothing has been written, so a
      // resolution/listing failure must not strand the found marker partial.
      await restoreCompletionMarker();
      throw error;
    }
  }
  // `ensured` is always the full plan (see fetchAssetSidecars), so the planned
  // names ARE the post-fetch sidecar set on every path that reaches
  // finalization — the post-download `existsSync` filter this replaces was
  // already a no-op there.
  const plannedSidecars = sidecarPlan?.candidates.map((file) => file.path) ?? [];

  const isGlobRun = Boolean(globPatterns?.length);
  // A --complete run's selection IS the prescribed model, so it carries
  // FULL-run semantics: the new revision is claimed (a glob run deliberately
  // under-claims, which for a catalog install left the dashboard offering an
  // update the CLI had already applied, forever) and the directory is pruned
  // to the prescription like the dashboard's own publish swap.
  const fullSemantics = !isGlobRun || args.complete === true;
  // Legacy runs (no resolvable revision) never prune or certify — finalizeSync
  // returns early on them — so their guard inputs stay empty and no refusal
  // below can fire.
  let pruneList: string[] = [];
  let certified: string[] = [];
  if (remoteSha !== null) {
    // Files the old marker recorded that the remote repo no longer has are
    // stale garbage (e.g. shards of a superseded sharding layout) — a loader
    // globbing the dir would read them. Only ever deletes old-marker entries
    // of THIS repo's marker.
    const remotePaths = allFiles.map((f) => f.path);
    // Files the marker attributes to the assetsRepo are exempt: the primary
    // tree cannot prove anything about them (they were never in it), so
    // pruning them here would delete the tool-calling sidecars this very run
    // just verified — and an install that HAS such provenance must keep its
    // candidate names exempt even when this run passed no --assets-repo.
    const exemptFromPrune = new Set<string>(plannedSidecars);
    if (assetsRepo === undefined && previousCompletion?.assetsRepo !== undefined) {
      // No assets manifest was consulted this run, so the sidecars' state is
      // unknown: every candidate name stays exempt (and recorded). When the
      // manifest WAS consulted, only the names it still supplies are exempt —
      // a candidate it dropped is pruned and dropped from the marker, instead
      // of lingering while the revision advances past it.
      for (const name of ASSET_SIDECAR_CANDIDATES) exemptFromPrune.add(name);
    }
    // A --complete run prunes and carries against the PRESCRIPTION, not the
    // whole remote tree: `allFiles` would let an unselected variant (another
    // quant the user once downloaded here) survive as "proven on remote" while
    // the marker claims the new revision — an unverified file riding along that
    // discovery can then expose as a loadable model.
    const scopedPaths =
      args.complete === true ? [...filesToDownload.map((f) => f.path), ...plannedSidecars] : remotePaths;
    pruneList =
      previousCompletion !== null
        ? computePruneList(previousCompletion.files, scopedPaths, outputDir, !fullSemantics, exemptFromPrune)
        : computeLegacyWeightPruneList(existingTopLevelFiles, scopedPaths, outputDir, !fullSemantics);
    // The exact file list this run will certify — computed BEFORE the prune so
    // the guard and the marker agree on one set.
    certified = buildMarkerFiles(
      previousCompletion,
      scopedPaths,
      [...filesToDownload.map((f) => f.path), ...plannedSidecars],
      outputDir,
      !fullSemantics,
      exemptFromPrune,
    );
    // A FRESH install that claims a loadable model — a catalog prescription, or
    // one that pulled sidecars — must certify a config: isModelInstalled
    // requires one, so a marker without it can never read as installed while
    // the wizard calls the run a success. Updates are covered by the prune
    // guard below, whose message names the removals that would break the
    // install; this gate exists for the case with nothing to prune.
    if (previousCompletion === null && (args.complete === true || assetsRepo !== undefined)) {
      if (!certified.includes('config.json') || !certified.some(isLoadableWeightFile)) {
        // The refusal cannot leave what it downloaded: a markerless directory
        // of loadable weights reads as a PRESENT but foreign install — the
        // dashboard renders a disabled card it is not allowed to repair. When
        // nothing predates this run (the directory is new, or was empty), the
        // honest "nothing was published" is removing it; a pre-existing
        // markerless directory keeps its files, which are the user's to mix.
        await restoreCompletionMarker();
        if (existingTopLevelFiles.length === 0) {
          await rm(outputDir, { recursive: true, force: true });
        }
        throw new Error(
          `Refusing to certify "${modelName}": the selection has no ` +
            `${certified.includes('config.json') ? 'model weights' : 'config.json'}` +
            `${assetsRepo !== undefined ? ` (the repository provides none and "${assetsRepo}" provides none)` : ''}. ` +
            `Nothing was published.`,
        );
      }
    }
    if (pruneList.length > 0) {
      // Pruning must never destroy the install: the surviving manifest has to
      // keep a config and a weight. Upstream dropping the LAST config.json (a
      // weight-only GGUF repo relies on the assets repo for it) is exactly
      // that case — fail with everything unchanged, which running here makes
      // literally true.
      const remaining = certified.filter((file) => !pruneList.includes(file));
      if (!remaining.includes('config.json') || !remaining.some(isLoadableWeightFile)) {
        // "Unchanged" is only honest if the marker goes back too: this run
        // already downgraded it to `partial` before downloading, so refusing
        // here would strand the install as present-but-uncertified with no
        // update affordance (see restoreCompletionMarker).
        await restoreCompletionMarker();
        throw new Error(
          `Refusing to sync "${modelName}": removing ${pruneList.join(', ')} would leave no loadable checkpoint ` +
            `(missing ${remaining.includes('config.json') ? 'model weights' : 'config.json'}). ` +
            `The installed directory is unchanged.`,
        );
      }
    }
  }

  // Show what will be downloaded
  if (globPatterns?.length) {
    console.log(`Matched ${filesToDownload.length} file(s):`);
    for (const f of filesToDownload) {
      console.log(`  ${f.path} (${formatBytes(f.size)})`);
    }
    console.log('');
  }

  const sizeStr = formatBytes(totalSize);
  console.log(`Downloading ${filesToDownload.length} file(s) (~${sizeStr})...\n`);

  const weightFiles: string[] = [];

  const total = filesToDownload.length;
  for (let i = 0; i < total; i++) {
    const file = filesToDownload[i];
    const fileSizeStr = file.size ? formatBytes(file.size) : '';
    const destPath = join(outputDir, file.path);
    // Cheap size gate normally; full content hash when local files might be
    // stale (marker missing/stale or --force) — a re-uploaded repo can keep
    // identical file sizes, which only the hash catches.
    const alreadyPresent = verifyContent
      ? await fileUpToDate(destPath, file)
      : isLocalCopyComplete(destPath, file.size);
    if (alreadyPresent) {
      console.log(
        `  [${i + 1}/${total}] ${file.path}${fileSizeStr ? ` (${fileSizeStr})` : ''} — already present, skipping copy`,
      );
    } else {
      console.log(`  [${i + 1}/${total}] ${file.path}${fileSizeStr ? ` (${fileSizeStr})` : ''}...`);
      const snapshotPath = await withRetries(file.path, () =>
        downloadFileToCacheDir({
          repo: { type: 'model', name: modelName },
          path: file.path,
          cacheDir,
          accessToken: HUGGINGFACE_TOKEN,
          revision: remoteSha ?? undefined,
        }),
      );
      await ensureDir(dirname(destPath));
      await copyFile(snapshotPath, destPath);
    }
    if (file.path.endsWith('.safetensors') || file.path.endsWith('.pdiparams')) {
      weightFiles.push(file.path);
    } else if (file.path.endsWith('.gguf') && !isGgufCompanionName(basename(file.path))) {
      // A companion (mmproj/imatrix/dflash/draft) is never the payload: a
      // selection of only a projector must not finalize a marker — the same
      // rule the dashboard's publish gate applies.
      weightFiles.push(file.path);
    }
  }

  // Sidecars land before the prune/marker step so the certified directory is
  // exactly what the loader will open: GGUF weights plus the base-model
  // tokenizer files the runtime needs (see ASSET_SIDECAR_CANDIDATES). The
  // selection was already resolved in the preflight — this only applies it.
  if (assetsRepo !== undefined && sidecarPlan !== null) {
    const topUp = await fetchAssetSidecars({
      assetsRepo,
      outputDir,
      cacheDir,
      accessToken: HUGGINGFACE_TOKEN,
      plan: sidecarPlan,
    });
    // Record the pair only when the revision resolved: an unresolved source is
    // unknown provenance, and an empty string would compare unequal forever.
    // (`topUp.ensured` needs no variable here: it is always the full plan —
    // `plannedSidecars` — and finalization consumed that set in the preflight.)
    sidecarSource = topUp.revision !== null ? { repo: topUp.repo, revision: topUp.revision } : null;
  }

  // Prune + marker write, invoked ONLY from a SUCCESS path. Pruning any
  // earlier could destroy the only working copy: if upstream restructures
  // the recursive download/verification must finish before old artifacts are
  // removed.
  const finalizeSync = async (): Promise<void> => {
    if (remoteSha === null) return; // nothing trustworthy to pin — legacy run
    // `pruneList`/`certified` were computed in the preflight, BEFORE the
    // download loop and the sidecar fetch wrote anything — so the two
    // refusals below are FAIL-CLOSED backstops over the same values. The
    // preflight already refused every case that could fire; these are kept
    // so a future change that reorders a write still cannot certify a
    // broken directory.
    if (previousCompletion === null && (args.complete === true || assetsRepo !== undefined)) {
      if (!certified.includes('config.json') || !certified.some(isLoadableWeightFile)) {
        await restoreCompletionMarker();
        if (existingTopLevelFiles.length === 0) {
          await rm(outputDir, { recursive: true, force: true });
        }
        throw new Error(
          `Refusing to certify "${modelName}": the selection has no ` +
            `${certified.includes('config.json') ? 'model weights' : 'config.json'}` +
            `${assetsRepo !== undefined ? ` (the repository provides none and "${assetsRepo}" provides none)` : ''}. ` +
            `Nothing was published.`,
        );
      }
    }
    if (pruneList.length > 0) {
      const remaining = certified.filter((file) => !pruneList.includes(file));
      if (!remaining.includes('config.json') || !remaining.some(isLoadableWeightFile)) {
        await restoreCompletionMarker();
        throw new Error(
          `Refusing to sync "${modelName}": removing ${pruneList.join(', ')} would leave no loadable checkpoint ` +
            `(missing ${remaining.includes('config.json') ? 'model weights' : 'config.json'}). ` +
            `The installed directory is unchanged.`,
        );
      }
    }
    for (const rel of pruneList) {
      console.log(`  Removing ${rel} (no longer in the upstream repo)`);
      // A stale standard weight can take precedence over the newly downloaded
      // layout. If removal fails, abort before certifying this revision.
      await rm(join(outputDir, rel), { force: true });
    }
    await writeCompletion(outputDir, {
      repo: modelName,
      revision: markerRevisionToClaim(previousCompletion, remoteSha, !fullSemantics),
      // The set the guard above validated (primary selection AND sidecars: a
      // mandatory tokenizer that vanishes later must invalidate the marker).
      files: certified,
      scope: isGlobRun && !args.complete ? 'partial' : 'full',
      // Provenance for update discovery, same as the dashboard's marker: a
      // tokenizer fix in the base repo moves nothing in the primary repo.
      // This run's source wins; otherwise CARRY the previous marker's — a plain
      // re-sync that passed no --assets-repo touched nothing there, and erasing
      // the pair would silently disable update discovery for the install.
      ...(sidecarSource !== null
        ? { assetsRepo: sidecarSource.repo, assetsRevision: sidecarSource.revision }
        : previousCompletion?.assetsRepo !== undefined
          ? { assetsRepo: previousCompletion.assetsRepo, assetsRevision: previousCompletion.assetsRevision }
          : {}),
      completedAt: new Date().toISOString(),
    });
  };

  // For GGUF downloads, skip strict verification (no config.json required in GGUF repos)
  const hasGgufFiles = weightFiles.some((f) => f.endsWith('.gguf'));
  if (hasGgufFiles) {
    await finalizeSync();
    console.log(`\nDownload complete! ${weightFiles.length} file(s) saved to ${outputDir}\n`);
    console.log('To convert GGUF to MLX SafeTensors format:');
    for (const wf of weightFiles) {
      const ggufPath = join(outputDir, wf);
      console.log(`  mlx convert -i ${ggufPath} -o ${outputDir}-mlx`);
    }
    console.log('');
  } else if (weightFiles.length === 0 && globPatterns?.length) {
    if (filesToDownload.length === 0) {
      console.error(`\nNo files matched the glob pattern(s): ${globPatterns.join(', ')}`);
      console.error('Check the pattern and available files in the repository.');
      process.exit(1);
    }
    // Glob filter matched non-weight files (e.g. imatrix, calibration data, or
    // a companion-only projection when the real target is missing upstream).
    // Skip model verification — and leave no completion marker: a marker is
    // what makes a directory an INSTALL, and a selection with no model weights
    // is not one. Certifying it would let the dashboard present a directory
    // nothing can load. For an EXISTING install the found marker is restored
    // instead: this run certified nothing, so the directory keeps its last
    // valid certification rather than a downgrade that reads as uninstalled.
    await restoreCompletionMarker();
    console.warn('  No model weights in the selection — nothing was certified as installed.');
    console.log(`\nDownload complete! ${filesToDownload.length} non-weight file(s) saved to ${outputDir}\n`);
  } else {
    console.log(`Format: Base model (needs MLX conversion)`);
    console.log('Note: After download, convert to MLX format:');
    console.log(`    mlx convert --input ${outputDir} --output ${outputDir}-mlx-bf16\n`);

    const success = await verifyDownload(outputDir, weightFiles);
    if (success) {
      await finalizeSync();
      console.log('\nModel downloaded successfully!\n');
    } else {
      console.error('\nDownload incomplete. Please try again.\n');
      process.exit(1);
    }
  }
}
