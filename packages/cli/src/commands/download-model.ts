import { existsSync, readFileSync, statSync } from 'node:fs';
import { readdir, copyFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { parseArgs } from 'node:util';

import { listFiles, downloadFileToCacheDir, type ListFileEntry } from '@huggingface/hub';
// Leaf subpath on purpose: `@mlx-node/server/host` would dlopen the native
// addon, and downloading a model must work before any of that is needed.
import { resolveModelsDir } from '@mlx-node/server/host/paths';

import { ensureDir, formatBytes } from '../utils.js';
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
async function getModelFiles(modelName: string, accessToken?: string, globPatterns?: string[]) {
  return withRetries(`listing ${modelName}`, () => listModelFilesOnce(modelName, accessToken, globPatterns));
}

async function listModelFilesOnce(modelName: string, accessToken?: string, globPatterns?: string[]) {
  let totalSize = 0;
  const filesToDownload: ListFileEntry[] = [];
  const allFiles: ListFileEntry[] = [];

  // Compile glob patterns if provided
  const globs = globPatterns?.map(globToRegex);

  for await (const file of listFiles({ repo: { type: 'model', name: modelName }, accessToken })) {
    allFiles.push(file);

    if (globs) {
      // When glob patterns are active, include files that match the pattern
      // OR are essential metadata files (config, tokenizer)
      const basename = file.path.split('/').pop() || file.path;
      if (matchesAnyGlob(basename, globs) || matchesAnyGlob(file.path, globs)) {
        filesToDownload.push(file);
        if (file.size) totalSize += file.size;
      } else if (CORE_FILES.includes(basename)) {
        // Always include core config/tokenizer files
        filesToDownload.push(file);
        if (file.size) totalSize += file.size;
      }
    } else {
      // Default behavior: download model files
      if (
        CORE_FILES.includes(file.path) ||
        file.path.endsWith('.safetensors') ||
        file.path.endsWith('.json') ||
        file.path.endsWith('.pdiparams') ||
        file.path.endsWith('.yml') ||
        file.path.endsWith('.gguf') ||
        file.path.endsWith('.jinja')
      ) {
        filesToDownload.push(file);
        if (file.size) {
          totalSize += file.size;
        }
      }
    }
  }

  return { totalSize, filesToDownload, allFiles };
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
  console.log(`Output: ${outputDir}\n`);

  // Check if already downloaded.
  //
  // For non-GGUF repos the local-only `isModelAlreadyDownloaded` check
  // is sufficient (single-file safetensors → 1 weight file; sharded →
  // we parse the index and verify every shard is present). For GGUF
  // repos the local-only path is wrong: a multi-variant repo (Q2_K,
  // Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 as separate files) where a
  // prior interrupted run left only Q2_K on disk would silently exit
  // as "already downloaded" without ever fetching the rest. So for the
  // GGUF early-return we hoist the manifest fetch above the check and
  // require EVERY remote `.gguf` to be present locally before
  // declaring complete. The trade-off — one extra network round-trip
  // on the hot "already downloaded" path — is intentional; correctness
  // wins over the ~200ms saved.
  let cachedManifest: { totalSize: number; filesToDownload: ListFileEntry[]; allFiles: ListFileEntry[] } | null = null;
  if (existsSync(outputDir)) {
    const files = await readdir(outputDir);
    const hasGguf = files.some((f) => f.endsWith('.gguf'));
    if (isModelAlreadyDownloaded(outputDir, files)) {
      console.log('Model already downloaded!\n');
      console.log('To re-download, delete the output directory first:');
      console.log(`   rm -rf ${outputDir}\n`);
      return;
    }
    if (hasGguf && !globPatterns?.length) {
      // Manifest-aware completeness: only short-circuit when every
      // remote `.gguf` is present locally. Falling through to the
      // download loop is safe — `downloadFileToCacheDir` is content-
      // addressed and the per-file loop's `isLocalCopyComplete` check
      // skips the `copyFile` for files already on disk at the right
      // size, so resume only does I/O for the genuinely missing shards.
      console.log('Fetching file list from HuggingFace...\n');
      cachedManifest = await getModelFiles(modelName, HUGGINGFACE_TOKEN, globPatterns);
      const remoteBasenames = cachedManifest.allFiles.map((f) => f.path.split('/').pop() ?? f.path);
      if (isGgufRepoComplete(files, remoteBasenames)) {
        console.log('GGUF file(s) already downloaded!\n');
        console.log('To re-download, delete the output directory first:');
        console.log(`   rm -rf ${outputDir}\n`);
        return;
      }
      // Incomplete: fall through to the download loop. Report which
      // variants are missing so the user knows what's about to fetch.
      const missing = cachedManifest.allFiles.filter(
        (f) => f.path.endsWith('.gguf') && !files.includes(f.path.split('/').pop() ?? f.path),
      );
      if (missing.length > 0) {
        console.log(`Detected ${missing.length} missing GGUF file(s); resuming download...`);
        for (const f of missing) {
          console.log(`  ${f.path}${f.size ? ` (${formatBytes(f.size)})` : ''}`);
        }
        console.log('');
      }
    }
    // For glob downloads, only declare "already downloaded" when EVERY
    // remote file matching the user's glob is also present locally. The
    // previous "at least one local hit" check (`isGlobVariantPresent`)
    // would short-circuit on a partial download — e.g. an interrupted
    // prior `--glob "*Q4*"` run that fetched one Q4 shard but not the
    // others would silently exit as "Matched files already downloaded"
    // and leave the local copy incomplete. CORE_FILES (config.json,
    // tokenizer.json, …) are still implicitly excluded because they
    // never match a quantization-variant glob.
    //
    // This is manifest-aware, so we pay one extra HuggingFace round-trip
    // on the hot "already downloaded" path. Correctness wins over the
    // ~200ms saved — symmetric to the non-glob GGUF completeness check
    // a few lines above. We cache the manifest into `cachedManifest` so
    // the post-`ensureDir` block reuses it instead of fetching twice.
    if (hasGguf && globPatterns?.length && isGlobVariantPresent(files, globPatterns)) {
      if (cachedManifest === null) {
        console.log('Fetching file list from HuggingFace...\n');
        cachedManifest = await getModelFiles(modelName, HUGGINGFACE_TOKEN, globPatterns);
      }
      const remoteBasenames = cachedManifest.allFiles.map((f) => f.path.split('/').pop() ?? f.path);
      if (isGlobMatchedSetComplete(files, remoteBasenames, globPatterns)) {
        console.log('Matched files already downloaded!\n');
        console.log('To re-download, delete the output directory first:');
        console.log(`   rm -rf ${outputDir}\n`);
        return;
      }
      // Incomplete: fall through to the download loop. The per-file
      // loop's `isLocalCopyComplete` check skips already-present files,
      // so resume only fetches/copies the genuinely missing shards.
      const missing = cachedManifest.filesToDownload.filter((f) => {
        const basename = f.path.split('/').pop() ?? f.path;
        return !files.includes(basename);
      });
      if (missing.length > 0) {
        console.log(`Detected ${missing.length} missing file(s); resuming download...`);
        for (const f of missing) {
          console.log(`  ${f.path}${f.size ? ` (${formatBytes(f.size)})` : ''}`);
        }
        console.log('');
      }
    }
  }

  await ensureDir(outputDir);

  // Reuse the manifest fetched during the GGUF completeness check if we
  // already have it, otherwise fetch fresh. Either way the same shape
  // is destructured below.
  if (cachedManifest === null) {
    console.log('Fetching file list from HuggingFace...\n');
  }
  const { totalSize, filesToDownload, allFiles } =
    cachedManifest ?? (await getModelFiles(modelName, HUGGINGFACE_TOKEN, globPatterns));

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
    process.exit(1);
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

  const cacheDir = args['cache-dir'] ? resolve(args['cache-dir']) : DEFAULT_CACHE_DIR;
  const weightFiles: string[] = [];

  const total = filesToDownload.length;
  for (let i = 0; i < total; i++) {
    const file = filesToDownload[i];
    const fileSizeStr = file.size ? formatBytes(file.size) : '';
    const destPath = join(outputDir, file.path);
    // Skip files already on disk with matching size — `downloadFileToCacheDir`
    // is content-addressed (no network re-fetch) but `copyFile` always copies
    // bytes, so without this check a resumed sharded download re-writes every
    // already-complete shard from cache to outputDir on every invocation.
    if (isLocalCopyComplete(destPath, file.size)) {
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
        }),
      );
      await ensureDir(dirname(destPath));
      await copyFile(snapshotPath, destPath);
    }
    if (file.path.endsWith('.safetensors') || file.path.endsWith('.pdiparams') || file.path.endsWith('.gguf')) {
      weightFiles.push(file.path);
    }
  }

  // For GGUF downloads, skip strict verification (no config.json required in GGUF repos)
  const hasGgufFiles = weightFiles.some((f) => f.endsWith('.gguf'));
  if (hasGgufFiles) {
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
    // Glob filter matched non-weight files (e.g. imatrix, calibration data).
    // Skip model verification — user is downloading auxiliary files.
    console.log(`\nDownload complete! ${filesToDownload.length} non-weight file(s) saved to ${outputDir}\n`);
  } else {
    console.log(`Format: Base model (needs MLX conversion)`);
    console.log('Note: After download, convert to MLX format:');
    console.log(`    mlx convert --input ${outputDir} --output ${outputDir}-mlx-bf16\n`);

    const success = await verifyDownload(outputDir, weightFiles);
    if (success) {
      console.log('\nModel downloaded successfully!\n');
    } else {
      console.error('\nDownload incomplete. Please try again.\n');
      process.exit(1);
    }
  }
}
