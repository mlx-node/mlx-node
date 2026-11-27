/**
 * Client-side mirrors of the dashboard API response shapes (server: C5
 * `packages/dashboard/src/api.ts`, data modules `models.ts` / `catalog.ts` /
 * `cache.ts` / `download.ts`). Kept in one place so the Overview and Models
 * pages consume identical types; only fields the UI reads are modelled.
 */

import type { UserAttachment } from '../../../src/chat/attachment-rule.js';

export interface LocalModel {
  name: string;
  path: string;
  modelType: string;
  quant: string | null;
  contextWindow: number | null;
  sizeBytes: number;
  fileCount: number;
  /**
   * Names this checkpoint is served and traced under (a GGUF folder serves one per variant). Match with
   * `servedIds.includes(key) || (servedIds.length === 0 && name === key)`.
   */
  servedIds: string[];
  /** Set when Locate made the folder from model files the user already had. */
  located: (LocatedFiles & { filesStay: boolean }) | null;
  /** The entry in the models folder is a link: Delete removes only the link. */
  link: boolean;
}

/** Where the model files of a folder that Locate made are. */
export interface LocatedFiles {
  /** The real path of the folder the user located. */
  from: string;
  /** `shared`: the models folder shares the files on the same disk. `linked`: it links to them on another disk. */
  mode: 'shared' | 'linked';
  /** The files in `from` can be reached; false while their disk is not connected. */
  filesAvailable: boolean;
}

export interface ModelsResponse {
  models: LocalModel[];
  /** Draft-only disk inventory, excluded from model counts and the model table. */
  companions?: Array<Pick<LocalModel, 'name' | 'path' | 'sizeBytes' | 'fileCount'>>;
  warnings: string[];
  /** Absolute path of the models directory these checkpoints were discovered in. */
  dir: string;
}

/** `GET /api/host`: this machine's memory, free space on the models disk, and OS platform. */
export interface HostResponse {
  totalMemBytes: number;
  /** `null` when the models folder cannot be read. */
  diskFreeBytes: number | null;
  platform?: string;
}

export type RuntimeCapability = 'chat' | 'audio' | 'rerank' | 'decision';

/** One build of a catalog model with the state of its install folder. */
export interface CatalogBuildItem {
  /** The build's name on every surface: `UD-Q4_K_XL`, `Original`. */
  id: string;
  hfRepo: string;
  sizeGb: number;
  /** Who made the files and in which format: `Unsloth GGUF`, `Thinking Machines`. */
  source: string;
  /** One short fact after the source: `1.1 GB smaller`. */
  note?: string;
  /** What comes with the weights. */
  included?: string;
  /**
   * Base-model repo the build CURRENTLY sources its tokenizer sidecars from — compared against `localAssetsRepo` so
   * a switched source is itself an update (the marker's repo is what the probe map lacks).
   */
  assetsRepo?: string;
  /** The folder under the models directory this build installs to; the key of its download. */
  slug: string;
  /** A dashboard-owned (completion-marker) install of this build is in its folder. */
  installed: boolean;
  /**
   * A loadable checkpoint of this build is on disk regardless of the dashboard marker — true for {@link installed},
   * and also for a build installed via the `mlx download` CLI / wizard. Gate the Install action on this (an unowned
   * present model can't be overwritten).
   */
  present: boolean;
  /**
   * The build the folder holds when it holds another build of the same repository (`UD-Q4_K_M`), or `another model`
   * for a loadable model that names no build. The server refuses an install into it. `null` otherwise.
   */
  heldBy: string | null;
  /**
   * The folder is occupied by something the downloader does not own and holds no loadable checkpoint (an
   * interrupted `mlx download`, a hand-made folder, a symlink). Install is refused server-side in this state, so the
   * card names the blockage rather than offering a button that always errors.
   */
  blockedByForeignDir: boolean;
  /** What occupies the folder; non-null exactly when `blockedByForeignDir` is true. */
  blocker: CatalogBlocker | null;
  /**
   * Commit sha the local bytes were pinned to, or `null` when no completion marker exists to read (hand copy,
   * pre-marker CLI install, renamed dir). `null` means staleness is unknowable, never "up to date".
   */
  localRevision: string | null;
  /**
   * Base-model repo the install's tokenizer sidecars came from, and the commit they were pinned to — or `null` when
   * the install used none or predates the field. Update discovery compares this pair too: a sidecar-only upstream
   * change moves nothing in the primary repo, so without it no badge appears and the Installed button (disabled)
   * keeps the repair job unreachable.
   */
  localAssetsRepo: string | null;
  localAssetsRevision: string | null;
  /**
   * The install lacks a file the catalog names for the build (a glob with no `*`), in its marker and on disk, and an
   * Update deletes no weight of another build, in the marker or in the folder; absent otherwise.
   */
  lacksNamedFile?: true;
  /** The folder was made by Locate from model files elsewhere on this Mac; `null` otherwise. */
  located: LocatedFiles | null;
  /** Local models lists the folder at `slug`. */
  listed: boolean;
}

export interface CatalogItem {
  label: string;
  description: string;
  category?: 'chat' | 'systemone';
  isDefault?: boolean;
  hidden?: boolean;
  /** Platforms where this entry is supported (absent means every platform). */
  platforms?: readonly string[];
  /** Physical RAM in GiB the model's load needs at least; the page leaves the entry out below it. */
  minMemoryGb?: number;
  /** In catalog order; `builds[0]` is the build every surface chooses first. */
  builds: CatalogBuildItem[];
  /** Separate companion draft weights, never a standalone chat-model entry. */
  draft?: CatalogDraftItem;
}

export type CatalogDraftItem = Pick<
  CatalogBuildItem,
  'hfRepo' | 'sizeGb' | 'slug' | 'installed' | 'present' | 'blockedByForeignDir' | 'blocker' | 'localRevision'
> & { label: string; autoInstall?: boolean };

/**
 * An entry at a catalog slug that the app did not create, measured without following a link or opening a file. The
 * cleanup confirmation shows these numbers, and its request echoes them back.
 */
export interface CatalogBlocker {
  kind: 'directory' | 'symlink' | 'file';
  /** directory: files below it. file: 1. symlink: 0. */
  fileCount: number;
  sizeBytes: number;
  /** The walk stopped early: both numbers are lower bounds. */
  truncated: boolean;
  /** Relative paths, only for 3 files or fewer. */
  names: string[];
  /** Local models lists this entry. */
  listed: boolean;
  /** A `config.json` is at its root. */
  modelLike: boolean;
  /** It holds model files (a checkpoint or a GGUF); for a link, the folder it points to does. */
  holdsModel: boolean;
  /** The link text, cut to 200 characters and `…`. */
  linkTarget: string | null;
  dangling: boolean;
  /** A link whose target is there and is not a folder. */
  linkToFile: boolean;
  /** A real folder the app has no write permission on: no cleanup and no Update can change it. */
  locked: boolean;
}

export interface CatalogResponse {
  items: CatalogItem[];
}

/** One row of `/catalog/updates` — the upstream half of the staleness check. */
export interface CatalogUpdateItem {
  hfRepo: string;
  /** Upstream commit sha, or `null` when Hugging Face could not be reached. */
  remoteRevision: string | null;
}

export interface CatalogUpdatesResponse {
  items: CatalogUpdateItem[];
}

/** What a job does: download into a free folder, Update a folder in place, clean up an entry first, or Locate. */
export type DownloadJobKind = 'install' | 'update' | 'cleanup' | 'locate';

export interface DownloadJob {
  id: string;
  /** The folder the job installs to; the key of a download. */
  slug: string;
  repo: string;
  kind: DownloadJobKind;
  state: 'running' | 'committing' | 'done' | 'error' | 'cancelled';
  receivedBytes: number;
  totalBytes: number;
  /** A done job: where old files were kept. */
  keptDir?: string;
  /** A done cleanup: how the old entry went. */
  removed?: 'link' | 'trash';
  /** A cleanup: the entry it replaces. */
  entry?: Pick<CatalogBlocker, 'kind' | 'linkToFile'>;
}

export interface DownloadsResponse {
  jobs: DownloadJob[];
}

export interface DownloadStartResponse {
  id: string;
  slug: string;
  repo: string;
  kind: DownloadJobKind;
}

/**
 * Result of `POST /api/downloads/adopt-plan`: what Update would check, download and leave in a folder the app did not
 * create. `token` is posted back as `adopt` to start that exact plan.
 */
export interface AdoptPlanResponse {
  repo: string;
  slug: string;
  token: string;
  revision: string;
  assetsRevision: string | null;
  check: { count: number; bytes: number };
  download: { count: number; bytes: number };
  /** Files that stay in the folder; `names` holds at most 3, sorted. */
  other: { count: number; bytes: number; names: string[] };
}

/** Result of `POST /api/downloads/locate-plan`: what Locate would use from a folder, before anything changes. */
export interface LocatePlanResponse {
  slug: string;
  repo: string;
  token: string;
  /** The real path of the folder whose files are used: the picked folder or one of its child folders. */
  folder: string;
  mode: 'shared' | 'linked';
  /** The folder has this app's install record: the job checks that the files are there and reads none of them. */
  byRecord: boolean;
  revision: string;
  /** The files of the folder the app uses. */
  use: { count: number; bytes: number };
  /** The weight files the job reads once to check their content; zeros when `byRecord`. */
  read: { count: number; bytes: number };
  /** The files the job downloads into the models folder. */
  download: { count: number; bytes: number };
}

/** Result of `DELETE /api/downloads/:id` (cancel an in-flight/queued download). */
export interface CancelDownloadResponse {
  cancelled: boolean;
  id: string;
}

export interface DeleteModelResponse {
  deleted: boolean;
  name: string;
}

export interface ColdCacheDiskInfo {
  root: string;
  exists: boolean;
  /** KV blocks only; state sidecars are counted by `sidecarCount`. */
  entryCount: number;
  sidecarCount: number;
  /** Blocks + sidecars, i.e. everything the cold-tier quota covers. */
  totalBytes: number;
  sidecarBytes: number;
  quotaBytes: number;
  oldestMtime: number | null;
  newestMtime: number | null;
  ageHistogram: Array<{ label: string; count: number; bytes: number }>;
}

export interface CacheTrendRow {
  /** Local day start, epoch ms. */
  t: number;
  hits: number;
  misses: number;
  bytesWritten: number;
  bytesRestored: number;
}

/** Lookups the trend's root filter left out, so the page can say so out loud. */
export interface CacheExcludedBucket {
  turns: number;
  hits: number;
  misses: number;
}

/** What `trend` is (and is not) counting. */
export interface CacheScope {
  /** Canonical cache root every `trend` row is filtered to. */
  root: string;
  /** Trace retention, i.e. the real width of the trend window. */
  trendWindowDays: number;
  /** Turns recorded before cache attribution existed — excluded, not lost. */
  legacy: CacheExcludedBucket;
  /** Turns attributed to a DIFFERENT cache directory. */
  otherRoots: CacheExcludedBucket;
  /**
   * Turns whose tier was ON but which recorded no root. The writer cannot
   * produce these any more; the bucket keeps the partition EXHAUSTIVE so such a
   * row's lookups are shown rather than silently dropped.
   */
  unattributed: CacheExcludedBucket;
  /** Turns that ran with the cold tier switched off (their deltas are all 0). */
  disabledTurns: number;
  /** Turns that ran with the cold tier switched on, whatever their root. */
  enabledTurns: number;
  /**
   * Sidecar captures recorded by turns carrying no root — the one counter the
   * buckets above cannot report, because a tier-less turn has no hits or
   * misses to put in one. `record_capture_reached()` is the first statement of
   * every family's capture, above its `adapter.cold_tier()` guard, so a hybrid
   * turn under `--no-persist-cache` or one whose tier failed to open still
   * counts the capture. Kept out of {@link ColdTierHealth} because these turns
   * name no cache root and `health` promises one.
   */
  unrootedSidecarCaptures: number;
}

/**
 * Cold-tier counters beyond hit/miss. The `*Total` fields are CUMULATIVE
 * process totals reduced with MAX (not summed deltas), so a fault during a
 * turn that aborted before it could record a delta still shows.
 */
export interface ColdTierHealth {
  enqueued: number;
  queueDrops: number;
  evictions: number;
  corruptions: number;
  corruptionsTotal: number;
  queueDropsTotal: number;
  /**
   * Writes the queue accepted that never reached disk. The native writer is
   * fail-open and reports its error to nobody, so this is the only number that
   * separates a cache storing nothing from a cache with nothing to store.
   */
  writeErrors: number;
  writeErrorsTotal: number;
  /**
   * Restores the walk refused. Neither a hit nor a miss — without this, a
   * refused restore renders as "no lookups recorded", which reads as "nothing
   * ran" rather than "reuse was refused". SUMMED, not MAX'd: a decline is
   * normal on any prompt's first turn, so a latch would carry no information.
   */
  restoreDeclines: number;
  /**
   * Turns that entered a family's sidecar capture. Every other `sidecar*`
   * counter is a sub-count of this one. Only the hybrid families
   * (gemma4 / qwen3_5 / qwen3_5_moe) ever enter it, so `0` beside real block
   * traffic is the normal reading for a dense model, not a fault.
   */
  sidecarCaptureReached: number;
  /** Captures that found no whole persisted block to anchor state under. */
  sidecarChainEmpty: number;
  /** Captures whose chain covered blocks but had no retained checkpoint at or below its reach. */
  sidecarBoundarySkips: number;
  /**
   * Captures that selected a boundary already on disk — the steady state of a
   * repeated prompt. It is what separates a healthy run (`sidecarEnqueued = 0`
   * because the first turn already wrote the chain) from one whose ladder
   * never reaches a persistable rung.
   */
  sidecarAlreadyPersisted: number;
  /**
   * Sidecars handed to the writer queue. The sidecar SHARE of `enqueued`, not
   * a second population — one admission bumps both, so never add them.
   */
  sidecarEnqueued: number;
  /** Sidecars the queue refused. Likewise the sidecar share of `queueDrops`. */
  sidecarQueueDrops: number;
  /**
   * Restored sidecars a family INSTALLED as live state. The one counter here
   * that cannot be inferred: every `install_*_cold_sidecar` early-return falls
   * through to a full O(prefix) replay producing CORRECT state, so a
   * regression from "restored and used" to "restored and re-derived" leaves
   * the hit rate, the trend and the corruption count untouched.
   */
  sidecarInstalled: number;
}

export interface CacheResponse {
  disk: ColdCacheDiskInfo;
  trend: CacheTrendRow[];
  scope: CacheScope;
  health: ColdTierHealth;
  /** Families whose prefixes may be restored — served, never bundled. */
  restoreFamilies: string[];
}

/** Result of `DELETE /api/cache` (clear all or evict older-than). */
export interface CacheMutationResult {
  removed: number;
  freedBytes: number;
}

/** The model caches under `prefix-cache/` (server: `src/prefix-cache.ts`). */
export type PrefixLockState = 'held' | 'free' | 'unknown';

export interface PrefixCacheNamespace {
  /** The directory's name: 32 lowercase hex digits. */
  namespace: string;
  /** The model a running engine reports for this cache; null when no running engine reports it. */
  model: string | null;
  /** Regular files but `lock`, by size. A lower bound while `complete` is false. */
  bytes: number;
  files: number;
  /** Newest mtime of the files and of `lock`: the last open, write or restore. */
  lastUsedMs: number | null;
  /** The lock file's time: when an engine last opened this cache. A delete sends it back. */
  openedMs: number;
  /** False until a first walk of this directory reached its end. */
  complete: boolean;
  /** When the walk behind `bytes` and `files` ended. */
  measuredAtMs: number;
  lock: PrefixLockState;
}

export interface PrefixCacheEngine {
  pid: number;
  host: HostKind;
  /** One open persistent state tier's limit; null when the engine's figure is not one tier's or it does not say. */
  limitBytes: number | null;
}

export interface PrefixCacheRoot {
  root: string;
  exists: boolean;
  /** Why the folder could not be read (an error code); null otherwise. */
  readError: string | null;
  isDefault: boolean;
  namespaces: PrefixCacheNamespace[];
  truncated: boolean;
  volume: { availableBytes: number; sizeBytes: number } | null;
  engines: PrefixCacheEngine[];
}

export interface PrefixCacheResponse {
  roots: PrefixCacheRoot[];
  /** A first walk is unfinished: one more GET continues it. */
  counting: boolean;
  canLock: boolean;
}

/** One step of `DELETE /api/cache/prefix/:namespace`. */
export interface PrefixCacheDeleteResult {
  removed: number;
  freedBytes: number;
  done: boolean;
  left: number;
  directoryRemoved: boolean;
}

/**
 * Sessions API (server: `src/agent-sessions/types.ts`), read from grok-build
 * (`mlx agent`) session folders. Timestamps in `SessionRow` are ISO strings as
 * stored in `summary.json`; transcript and turn `ts` are epoch milliseconds.
 */
export interface SessionTokens {
  input: number;
  output: number;
  cached: number;
}

/** Who started a session: a terminal `mlx agent`, or the app in Chat or Work mode. */
export type SessionOrigin = 'agent' | 'chat' | 'work';

export interface SessionRow {
  id: string;
  origin: SessionOrigin;
  /** The agent's title, else the title of the first typed prompt; '' only for a session with no prompt. */
  title: string;
  cwd: string;
  model: string | null;
  branch: string | null;
  createdAt: string;
  lastActiveAt: string;
  messageCount: number;
  live: boolean;
  kind: string | null;
  /** Fork source or subagent parent; may name a session that no longer exists. */
  parentId: string | null;
  subagentCount: number;
  /** `null` when no completed turn carried usage. */
  tokens: SessionTokens | null;
}

export interface SubagentRef {
  id: string;
  title: string;
  type: string;
  description: string;
  status: string;
  model: string | null;
  /** False when the child folder is gone or ambiguous; the UI must not link it. */
  exists: boolean;
}

export interface SessionDetail extends SessionRow {
  gitRoot: string | null;
  headCommit: string | null;
  remotes: string[];
  agentName: string | null;
  reasoningEffort: string | null;
  lastTurnSummary: string | null;
  /** `title: null` means the parent folder is missing. */
  parent: { id: string; title: string | null } | null;
  subagents: SubagentRef[];
}

/** What woke the agent for a turn the user did not type, read from the prompt id the agent recorded. */
export type WakeCause = 'workflow' | 'subagent' | 'task' | 'notifications' | 'goal' | 'message' | 'unknown';

/** Why the agent sent a model request again, read from its `retry_state` / `auto_recovery_*` lines. */
export type RetryCause = 'stalled' | 'connection' | 'server' | 'busy' | 'loop' | 'recovery' | 'other';

/** One picture the agent made smaller (`image_compressed`): bytes and pixel sides before and after. */
export interface PictureSize {
  fromBytes: number;
  toBytes: number;
  from: [number, number];
  to: [number, number];
}

export type TranscriptEntry = (
  | {
      kind: 'user';
      ts: number;
      text: string;
      promptIndex: number | null;
      interjection?: boolean;
      /** The `continue` prompt right after a length end: drawn as a divider, not a message. */
      continued?: true;
      /** The files and pictures sent with the message: names and sizes only, never their content. */
      attachments?: UserAttachment[];
      /** The user's note on a denied call: part of the running answer, never a new turn. */
      note?: true;
      /**
       * A host turn (`_meta.hostTurn`) typed as `/workflow pause|resume|stop <name>`: the verb and the run the name
       * picks by the agent's rule (a prefix of its run id or name, an exact one first). None when no run or two match.
       */
      runControl?: { verb: 'pause' | 'resume' | 'stop'; runId: string };
    }
  /** A hidden prompt the agent wrote to itself when a background event woke it. Never a message of the user. */
  | {
      kind: 'wake';
      ts: number;
      cause: WakeCause;
      /**
       * cause 'workflow': the run and the revision its prompt id names (`workflow-completed-<run>-<revision>`), and that
       * revision's stored update (null when the transcript holds none). A later update never changes it: a resume
       * empties the run's report, and a run that wakes twice has two rows. `report` is cut at 20 KB.
       */
      workflow?: {
        runId: string;
        revision: number;
        name: string;
        snapshot: {
          status: string;
          report?: string;
          truncated?: true;
          pauseMessage?: string;
          agentsUsed: number;
          agentBudget: number | null;
          /** Roster rows that say `failed`. */
          agentsFailed: number;
          elapsedMs: number;
        } | null;
      };
      /** cause 'subagent': the child whose finish woke the agent. */
      subagent?: { childId: string; description: string };
      /** cause 'task': the command or monitor whose end woke the agent. */
      task?: { id: string; kind: 'command' | 'monitor'; label: string };
      /** cause 'notifications': what ended with no wake of its own since the turn before, at most 8. */
      tasks?: Array<{ id: string; kind: 'command' | 'monitor'; label: string }>;
      /** Cut at 16 KB. */
      text: string;
      truncated?: true;
    }
  | { kind: 'assistant'; ts: number; text: string; endTs?: number }
  | {
      kind: 'thinking';
      ts: number;
      text: string;
      endTs?: number;
      /** Made from live lines: `ts` and `endTs` are its first and last chunk. A thought read from disk has no time. */
      timed?: true;
    }
  | {
      kind: 'tool';
      ts: number;
      id: string;
      name: string;
      title: string;
      status: string;
      input?: string;
      output?: string;
      truncated?: boolean;
      endTs?: number;
      /** A shell command's non-zero exit code. */
      exitCode?: number;
      /** A shell command stopped by the agent or the system (`timeout`, `oom`, …). */
      signal?: string;
      timedOut?: true;
      /**
       * The background task this call started: a command the agent moved to the background, or a monitor. `label` is
       * the call's description, else the first line of its command. `endTs`: the agent said it ended (it exited, was
       * killed or hit its time limit, or a new agent process no longer has it); unset while it runs. `killed`: its end
       * line says it was stopped (the user's Stop, the model's kill, a rewind), not that it ended by itself.
       */
      task?: { id: string; kind: 'command' | 'monitor'; label: string; endTs?: number; killed?: true };
      /**
       * The schedule this call made, with the agent's last words on it: its prompt, how it repeats (`every 5 minutes`)
       * and the next run it named (epoch ms; null when it named none). `endTs`: the agent removed it (deleted, expired,
       * or a one-time run done). The agent process stopping is no removal: the schedule comes back when it loads again.
       */
      schedule?: { id: string; prompt: string; repeat: string | null; nextTs: number | null; endTs?: number };
      /**
       * The change this call asks for (before its result) or made (status `completed`), per file, as the agent's diff
       * blocks say. The page gets hunks only, never the whole old or new file.
       */
      edits?: Array<{
        path: string;
        /** The result had no old text: a new file. */
        create?: true;
        added: number;
        removed: number;
        /**
         * `start`: old and new number of the hunk's first line (old 0 when the old side is empty); null when the agent
         * named no line. `body`: one line per row, first character ' ', '-' or '+'.
         */
        hunks: Array<{ start: [number, number] | null; body: string }>;
        /** Changed lines left out by the caps. */
        cut?: number;
        /** The texts differ only in line endings or the last newline. */
        endsOnly?: true;
      }>;
      /**
       * `ask_user_question`: the questions as the model wrote them (option labels only) and what the stored result
       * sentence says. `result` null: no result yet (or the call failed). `answered`: per question, the picked labels
       * and the user's own text, `raw` when that answer cannot be read exactly (shown as written, a pick is never
       * guessed), null when it was not answered. `none`: skipped, stopped or timed out (one sentence on disk).
       * `discuss` / `plan-now`: the plan-mode replies. `text`: a sentence the reader does not know.
       */
      ask?: {
        questions: Array<{ text: string; multi: boolean; options: string[] }>;
        result:
          | null
          | {
              kind: 'answered';
              answers: Array<{ picks: string[]; other: string | null } | { raw: string } | null>;
            }
          | { kind: 'none' }
          | { kind: 'discuss' }
          | { kind: 'plan-now' }
          | { kind: 'text'; text: string };
      };
      /**
       * A hook denied this call: the agent's words on it (`hook_annotation` of kind `tool_outcome`, paired with the
       * call's "Hook denied: " failure), cut at 600 characters.
       */
      blocked?: string;
      /**
       * Pictures the call read (`read_file` of a picture, or of a PDF as one picture per page), sizes only: the page
       * asks the session reader for one picture's bytes when the row opens. `n`: the picture's place among the call's
       * image blocks, from 0; `bytes`: its decoded size; `page`: the PDF page it shows, from 1.
       */
      pictures?: Array<{ n: number; mediaType: string; bytes: number; page?: number }>;
      /** A PDF read: how many pages the file has. */
      pdfPages?: number;
      /**
       * A `workflow` call, by its `rawInput.source.type`: `launch` (a name, script or script path), `check`
       * (`validate_only`), `pause`, `stop`, `resume`. `id`: the run it started or names; null when none came back and
       * the run it names is not known here (a refused launch, a check).
       */
      run?: { id: string | null; kind: 'launch' | 'check' | 'pause' | 'stop' | 'resume' };
    }
  | { kind: 'plan'; ts: number; entries: { content: string; status: string; priority: string }[] }
  | {
      kind: 'subagent';
      ts: number;
      childId: string;
      type: string;
      description: string;
      status?: string;
      durationMs?: number;
      /** `tokens_used`: the context size at the end of this attempt. */
      tokens?: number;
      /** `total_tokens_used`: input plus output tokens this attempt billed, nested subagents included; absent at 0. */
      tokensUsed?: number;
      /** `subagent_finished`: the helper's tool calls and turns. */
      toolCalls?: number;
      turns?: number;
      error?: string;
      /** Set by the detail route; only `true` may be linked. */
      exists?: boolean;
      /** Time of the `subagent_finished` line. */
      endTs?: number;
      /** Made by a finish line with no spawn line: `ts` is the finish time. */
      noSpawn?: true;
      /** `workflow_run_id`: the run that started it, whether or not the run's `workflow` entry is in the transcript. */
      run?: string;
      /** `resumed_from`: the child session it resumed, by a contract retry, a script `resume_from` or a Task `resume_from`. */
      resumedFrom?: string;
      /** `retry_of`: the attempt this contract retry follows. */
      retryOf?: string;
      /** `subagent_spawned.model`: the model it runs on. */
      model?: string;
    }
  | { kind: 'model'; ts: number; modelId: string }
  /**
   * The agent's session mode moved (`current_mode_update`): `plan` writes a plan and changes no files. A repeat of the
   * mode makes none. Like a notice, it never starts a turn and is placed right after the prompt it came with.
   */
  | { kind: 'plan-mode'; ts: number; mode: 'plan' | 'default' | 'ask' }
  /** The agent summarized older messages to free context (`auto_compact_*`): done, failed, or stopped by Stop. */
  | {
      kind: 'compaction';
      ts: number;
      outcome: 'done' | 'failed' | 'stopped';
      tokensBefore?: number;
      tokensAfter?: number;
      /** A failure the agent tries again at its next step. */
      retry?: true;
    }
  /**
   * The agent's "where things stand" (`session_recap`), written when the user asked for one. `text` is the model's,
   * cut at 2,000 characters. Like a notice, it never starts a turn and never counts as work.
   */
  | { kind: 'recap'; ts: number; text: string }
  /**
   * One goal of the goal loop (`/goal`), from its `goal_updated` lines: made at the first line of its id, then each
   * stored line sets it again (live-only ticks too, but a tick sends the page nothing unless a shown field moved).
   * `status` and `phase` are the agent's words (`active`, `user_paused`, `back_off_paused`, `no_progress_paused`,
   * `infra_paused`, `blocked`, `budget_limited`, `complete`; `cleared` once `/goal clear` ran). `asOf` is the time of
   * the line `elapsedMs` and `tokensUsed` were read from. `pauseMessage` is the agent's, cut at 300 characters.
   * `verifying` / `planning`: the agent checks the result or writes the plan now. Like a notice, it never starts a
   * turn and never counts as work.
   */
  | {
      kind: 'goal';
      ts: number;
      goalId: string;
      objective: string;
      status: string;
      phase: string;
      tokensUsed: number;
      tokenBudget?: number;
      elapsedMs: number;
      pauseMessage?: string;
      verifying?: true;
      planning?: true;
      asOf: number;
    }
  /**
   * One workflow run, from its `workflow_updated` lines: made at the first, replaced by each with a higher `revision`.
   * `status`, `phases[].state` and `agents[].state` are the agent's words. The Sessions reader also reads the run's
   * `state.json`: a newer revision there replaces the fields, and it adds the steps' `detail` and the `history`.
   */
  | {
      kind: 'workflow';
      /** The first update. */
      ts: number;
      /** The update applied last (for a newer `state.json`, the file's time). */
      updatedTs: number;
      runId: string;
      revision: number;
      name: string;
      status: string;
      objective?: string;
      /** `elapsed_ms`: run time with paused time left out, as of `updatedTs`. */
      elapsedMs: number;
      /** `agent_usage_incomplete`: `agentsUsed` may be low. */
      usageIncomplete?: true;
      /** `detail`: the script's words for the step, from `state.json` only. */
      phases: { title: string; state: string; detail?: string }[];
      currentPhase: string | null;
      agentBudget: number | null;
      agentsUsed: number;
      agents: {
        agentId: string;
        label: string;
        phase: string | null;
        /** Only when the script named one. */
        model?: string;
        state: string;
        /** The agent's total use over all its attempts, not its context. */
        tokens: number;
        durationMs: number;
      }[];
      /** Each `phase_entered` an update or `state.json` named: the step and when it began, oldest first. */
      phasesEntered: { title: string; ts: number }[];
      /** `state.json`'s history (steps, `log` lines, pauses, the end), oldest first; absent without the file. */
      history?: { event: string; detail: string | null; ts: number }[];
      pauseMessage?: string;
      /** `result_summary`, cut at 20 KB: the agent cuts it at 16 KB, so its own cut is the one shown. */
      report?: string;
      truncated?: true;
    }
  | {
      kind: 'turn-end';
      ts: number;
      stopReason: string;
      elapsedMs?: number;
      error?: string;
      /** The turn ended on the model's length limit (the agent stores it as an error). */
      lengthLimit?: true;
      /** A hook blocked the turn's prompt: the stored end is cancelled with the category `HookDenied`. */
      hookStopped?: true;
    }
  /**
   * A stored fact of the agent's that is no step of the answer. It never starts a turn and never counts as work; one
   * made while no turn is open is placed right after the next turn's prompt.
   */
  | ({ kind: 'notice'; ts: number } & (
      | {
          /**
           * One retry episode: the agent sent the model request again (`attempts`, `max`: its last try and budget).
           * `endTs` is set when output came back (`recovered`), the agent gave up (`gave-up`), or the turn ended (no
           * outcome); none while it retries. `detail`: the agent's words, cut at 300 characters.
           */
          code: 'retry';
          cause: RetryCause;
          attempts: number;
          max: number;
          endTs?: number;
          outcome?: 'recovered' | 'gave-up';
          detail?: string;
        }
      | {
          /**
           * Pictures the agent left out of a request (`image_dropped`) or made smaller (`image_compressed`). `count`: the
           * pictures made smaller, or a dropped line's notes (one note can speak for several pictures); `notes`: the
           * agent's words, each cut at 300 characters, at most 6; `sizes`: each shrunk picture.
           */
          code: 'picture';
          what: 'dropped' | 'shrunk';
          count: number;
          notes: string[];
          sizes?: PictureSize[];
        }
      | {
          /**
           * A hook run that failed (a hook's block is no failure), once per hook, event, tool and error in the session:
           * `times` counts them, `lastTs` is the newest. `hook`: the name as the agent sent it without its
           * `:{event}[i].hooks[j]` tail; `event` and `tool` as the agent named them; `detail`: the error, cut at 600
           * characters.
           */
          code: 'hook';
          event: string;
          tool?: string;
          hook: string;
          detail: string;
          times: number;
          lastTs: number;
        }
      | {
          /** A hook's own message, or the agent's words on a hook's verdict (`hook_annotation`), cut at 600 characters. */
          code: 'hook-note';
          text: string;
        }
    ))
) & {
  /**
   * Made from a line another session wrote (its `_meta.eventId` does not start with this session's id and `-`): a
   * retried or resumed child starts with a copy of the attempt it continues, which keeps that attempt's times. Set on a
   * child session's own transcript only.
   */
  copied?: true;
};

export interface TurnMetric {
  promptId: string;
  ts: number;
  /** The `turn_completed` line is a copy from another session, as {@link TranscriptEntry} `copied` says. */
  copied?: true;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  reasoningTokens: number;
  elapsedMs: number | null;
  /** Time to first token of the turn's first model call. */
  ttftMs: number | null;
  /** Summed measured prefill windows; set with `prefillTokens`, never alone. */
  prefillMs: number | null;
  prefillTokens: number | null;
  /** Model calls in the prefill sums. */
  prefillCalls: number | null;
  /** Model calls of the turn. */
  modelCalls: number | null;
  decodeTps: number | null;
  model: string | null;
}

export interface SessionListResponse {
  sessions: SessionRow[];
  total: number;
  cwds: string[];
}

export interface SessionDetailResponse {
  session: SessionDetail;
  transcript: TranscriptEntry[];
  turns: TurnMetric[];
  transcriptError?: string;
  /** The last `_meta.totalTokens` in `updates.jsonl`: the conversation's current context in tokens. */
  contextUsed?: number;
  /** Null for a Chat conversation: a terminal cannot open the Chat agent's home. */
  resumeCommand: string | null;
}

/** `PATCH /api/sessions/:id` body `{title}`; `title` is the sanitized title as stored. */
export interface SessionRenameResponse {
  id: string;
  title: string;
}

/** `DELETE /api/sessions/:id`: the session first, then its subagent children. */
export interface SessionDeleteResponse {
  deleted: string[];
}

/** `GET /api/sessions/:id/pictures/:call/:n`: the n-th picture (from 0) a tool call read, as the agent stored it. */
export interface SessionPictureResponse {
  /** png, jpeg, webp or gif. */
  mediaType: string;
  /** base64. */
  data: string;
}

export type TimelineLaneStatus = 'completed' | 'failed' | 'cancelled' | 'running' | 'unknown';
export type TimelineNoSteps =
  | 'missing'
  | 'empty'
  | 'too-large'
  | 'unreadable'
  /** The file has events, none inside this run. */
  | 'out-of-range'
  /** A finish line with no spawn line. */
  | 'no-spawn'
  | 'folded';

export interface TimelineLane {
  /** 'main', the child session id, or 'folded'. */
  id: string;
  role: 'main' | 'subagent' | 'folded';
  name: string;
  /** Subagent type; null on the main lane of a top-level session and on the folded lane. */
  type: string | null;
  model: string | null;
  /** 0..7 for a subagent lane; null = neutral. */
  slot: number | null;
  /** The lane's own session when its folder exists; only then the UI links it. */
  sessionId: string | null;
  status: TimelineLaneStatus;
  /** The lane's first run was spawned before this window. */
  carriedIn: boolean;
  /** The window is finished and the lane's last run returns after it. */
  carriedOut: boolean;
  /** Set when the lane has one `run` span instead of steps. */
  noSteps: TimelineNoSteps | null;
  /** Size of the file that was refused, for 'too-large'. */
  fileBytes: number | null;
  /** A step crossed a handoff time and was cut there. Events are never moved. */
  clamped: boolean;
  /** Subagents this lane spawned itself (not drawn). On the folded lane: how many subagents it stands for. */
  subagents: number;
  /** Usage reports inside the window, ascending. */
  tokenReports: Array<{ atMs: number; tokens: number }>;
}

interface TimelineSpanBase {
  /** `${lane}:${index}`, stable while the files do not change. */
  id: string;
  lane: string;
  /** ms from range.startMs. */
  startMs: number;
  /** null only with status 'running'. */
  endMs: number | null;
  status: 'ok' | 'failed' | 'running' | 'unfinished';
  inferredStart: boolean;
  inferredEnd: boolean;
  /** The turn's window cut the span on this side: it began before the window or went on after it. */
  cutStart: boolean;
  cutEnd: boolean;
  /** A model request retry at or right before this span. Only on `wait` / `model` and `generate` spans. */
  retry: { attempt: number; max: number } | null;
}

export type TimelineSpan =
  | (TimelineSpanBase & { kind: 'generate'; phase: 'reasoning' | 'answer' | 'tool-call' })
  | (TimelineSpanBase & { kind: 'tool'; tools: string[]; calls: number; failedCalls: number })
  /** until: what ended the wait. 'open' = still waiting (running) or cut by the lane's end. */
  | (TimelineSpanBase & { kind: 'wait'; reason: 'model'; until: 'text' | 'tool' | 'open' })
  /** calls: spawn tool calls merged into it; 0 = built from spawn lines alone. */
  | (TimelineSpanBase & { kind: 'wait'; reason: 'subagent'; tools: string[]; calls: number; failedCalls: number })
  | (TimelineSpanBase & { kind: 'run'; noSteps: TimelineNoSteps })
  /** The main lane before a woken turn: no turn was open. At most one, the main lane's first span. */
  | (TimelineSpanBase & { kind: 'idle' });

export interface TimelineHandoff {
  /** `${lane}:${kind}:${run index}`. */
  id: string;
  /** The child lane. The other end is always 'main'. */
  lane: string;
  kind: 'spawn' | 'return';
  atMs: number;
  failed: boolean;
}

/** `GET /api/sessions/:id/timeline?turn=<n>`: one turn as lanes of spans. Metadata only, never text. */
export interface SessionTimelineResponse {
  turn: number;
  /** Finished turns in the transcript; an open turn is `turns + 1`. */
  turns: number;
  open: boolean;
  /** The session's live verdict when this body was built (the rule of SessionRow.live). */
  live: boolean;
  stopReason: string | null;
  /** Worker clock when this body was built, epoch ms. */
  asOfMs: number;
  /**
   * Epoch ms. null = the turn has no usable time; lanes, spans and handoffs are then empty.
   * endMs >= turnStartMs >= startMs always. leadMs: how far the prompt lies before the start the agent measured (0 = not
   * at all).
   */
  range: {
    /** Axis origin. Earlier than turnStartMs only when a background event started the turn and subagents ran before it. */
    startMs: number;
    /** Where the turn itself starts: the prompt or the start the agent measured, whichever is earlier. */
    turnStartMs: number;
    endMs: number | null;
    lastEventMs: number;
    /** Refers to turnStartMs. */
    startExact: boolean;
    leadMs: number;
  } | null;
  lanes: TimelineLane[];
  /** Ordered by lane order, then start. */
  spans: TimelineSpan[];
  handoffs: TimelineHandoff[];
  notes: { untimedEvents: number; foldedSubagents: number; droppedSpans: number; unplacedRetries: number };
  /** Same text as SessionDetailResponse.transcriptError; range is then null. */
  transcriptError?: string;
}

export type MetricsBucket = '5m' | 'hour' | 'day';
export type TurnOutcome = 'ok' | 'error' | 'cancelled';

/** A median below this many samples is drawn as low-confidence. */
export const LOW_SAMPLE_N = 5;
/** A p05 / p95 below this many samples is drawn as low-confidence. */
export const LOW_SAMPLE_TAIL_N = 20;

export interface Quantiles {
  n: number;
  p05: number | null;
  p50: number | null;
  p95: number | null;
}

export interface MetricsRange {
  from: number | null;
  to: number | null;
}

// GET /api/metrics/models  — no params. Whole retention window.
// Order: firstTs ascending, then model. A model's index here is its color index on every page.
export interface MetricsModels {
  models: Array<{
    model: string;
    modelType: string | null;
    firstTs: number;
    lastTs: number;
    turns: number;
  }>;
}

export interface MetricsTotals {
  turns: number;
  ok: number;
  error: number;
  cancelled: number;
  /** Ok turns that carry a measured prefill window. */
  prefillMeasured: number;
  /** Net of cache. */
  inputTokens: number;
  cachedTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  /** Wall-time sums in ms, for the time-breakdown bar. */
  queueWallMs: number;
  loadWaitWallMs: number;
  resolveWallMs: number;
  /** The model was busy with other requests, before the time to first token. */
  waitWallMs: number;
  /** No lane or memory to start, inside the time to first token. */
  slotWallMs: number;
  /** Prompt preparation (render, prefix match, cache restore), inside the time to first token. */
  prepareWallMs: number;
  /** Time to first token of turns without a prefill window: setup and prefill together. */
  firstTokenWallMs: number;
  /** Time to first token outside the prefill window, measured turns only. */
  setupWallMs: number;
  /** Sum of measured prefill windows, each capped at its time to first token. */
  prefillWallMs: number;
  decodeWallMs: number;
  /** total − every phase above, clamped at 0. */
  otherWallMs: number;
  totalWallMs: number;
  /** Ok turns with a decode rate: output tokens after the first, and their decode ms. */
  decodeRatedTokens: number;
  decodeRatedMs: number;
  /** Ok turns with a measured prefill of at least `PREFILL_MIN_TOKENS`: prefill tokens and prefill ms. */
  prefillRatedTokens: number;
  prefillRatedMs: number;
}

/** Σ tokens and Σ ms of one rated phase: their quotient is the token-weighted speed. */
export interface RatedSum {
  tokens: number;
  ms: number;
}

export interface MetricsLatency {
  ttftMs: Quantiles;
  /** Time to first token outside the prefill window, measured turns only. */
  setupMs: Quantiles;
  decodeTps: Quantiles;
  prefillTps: Quantiles;
  /** preDispatch + duration. */
  totalMs: Quantiles;
  queueMs: Quantiles;
  /** Percent of the context window in use at turn end. */
  ctxFillPct: Quantiles;
}

// GET /api/metrics/summary?from&to&model
export interface MetricsSummary {
  range: MetricsRange;
  model: string | null;
  totals: MetricsTotals;
  latency: MetricsLatency;
  /** The window of equal length just before `from`; null unless both bounds were given. */
  previous: { totals: MetricsTotals; latency: MetricsLatency } | null;
  byModel: Array<{
    model: string;
    modelType: string | null;
    ctxWindow: number | null;
    totals: MetricsTotals;
    latency: MetricsLatency;
    mtpTurns: number;
    lastTs: number;
  }>;
  finishReasons: Array<{ reason: string; turns: number }>;
  failures: Array<{
    outcome: 'error' | 'cancelled';
    errorKind: 'finish_error' | 'exception' | null;
    failPhase: 'queued' | 'slot' | 'prepare' | 'prefill' | 'decode' | null;
    turns: number;
  }>;
  endpoints: Array<{ endpoint: string; stream: boolean; turns: number }>;
  dispatches: Array<{ dispatch: string; turns: number; promptTokens: number; cachedTokens: number }>;
}

// GET /api/metrics/timeseries?from&to&bucket&model&groupBy
export interface MetricsTimeseries {
  range: MetricsRange;
  bucket: MetricsBucket;
  groupBy: 'model' | 'none';
  points: Array<{
    /** Bucket start, epoch ms. */
    t: number;
    model: string;
    totals: MetricsTotals;
    ttftMs: Quantiles;
    decodeTps: Quantiles;
    prefillTps: Quantiles;
    totalMs: Quantiles;
    /** Cycle-weighted committed tokens per cycle; null when no MTP turn. */
    mtpMeanAccepted: number | null;
    mtpTurns: number;
    coldHits: number;
    coldMisses: number;
    concurrencyPeak: number | null;
  }>;
}

export interface Histogram {
  /** Samples counted. */
  n: number;
  /** One count per bin; `counts.length === edges.length`. */
  counts: number[];
}

// GET /api/metrics/distributions?from&to&model&groupBy
export interface MetricsDistributions {
  range: MetricsRange;
  groupBy: 'model' | 'none';
  /** Lower bounds. Bin i is [edges[i], edges[i+1]); the last bin is open-ended. */
  edges: {
    ttftMs: number[];
    decodeTps: number[];
    prefillTps: number[];
    totalMs: number[];
    ctxFillPct: number[];
    promptTokens: number[];
    outputTokens: number[];
  };
  series: Array<{
    model: string;
    ttftMs: Histogram;
    decodeTps: Histogram;
    prefillTps: Histogram;
    totalMs: Histogram;
    ctxFillPct: Histogram;
    promptTokens: Histogram;
    outputTokens: Histogram;
  }>;
}

// GET /api/metrics/speculative?from&to&model&bucket
export interface MetricsSpeculative {
  range: MetricsRange;
  models: Array<{
    model: string;
    turns: number;
    mtpTurns: number;
    cycles: number;
    /** Cycle-weighted. */
    meanAccepted: number | null;
    meanDrafts: number | null;
    meanDepth: number | null;
    /** `pos` is 0-based: pos 0 = the first draft position. */
    acceptByPos: Array<{ pos: number; rate: number; cycles: number }>;
    decodeTpsWithMtp: Quantiles;
    decodeTpsWithoutMtp: Quantiles;
    /** Decode sums of ok turns that ran speculation, and of those that did not. */
    decodeWith: RatedSum;
    decodeWithout: RatedSum;
  }>;
  trend: Array<{ t: number; model: string; meanAccepted: number | null; mtpTurns: number }>;
}

export interface TurnRow {
  traceId: string;
  /** End of turn, epoch ms. */
  ts: number;
  model: string;
  /** Alias the client used, when different from `model`. */
  alias: string | null;
  modelType: string | null;
  outcome: TurnOutcome;
  errorKind: string | null;
  failPhase: string | null;
  finishReason: string | null;
  endpoint: string | null;
  stream: boolean | null;
  dispatch: string | null;
  promptTokens: number | null;
  cachedTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  ctxWindow: number | null;
  ctxFillPct: number | null;
  /** Span segments in ms; `t0 = ts − durationMs − (preDispatchMs ?? 0)`. */
  preDispatchMs: number | null;
  queueMs: number | null;
  loadWaitMs: number | null;
  resolveMs: number | null;
  /** The model was busy with other requests (native creation → pickup); null from a record that predates it. */
  waitMs: number | null;
  /** No lane or memory to start the turn. */
  slotMs: number | null;
  /** Prompt preparation before the prefill. */
  prepareMs: number | null;
  ttftMs: number | null;
  /** Measured prefill window; null when the turn has none. */
  prefillMs: number | null;
  prefillTokens: number | null;
  /** `ttftMs − prefillMs`, measured turns only. */
  setupMs: number | null;
  decodeMs: number | null;
  durationMs: number | null;
  /** Rated turns only: measured, at least `PREFILL_MIN_TOKENS`. */
  prefillTps: number | null;
  /** Null when native measured no rate (0) or the turn wrote one token or none. */
  decodeTps: number | null;
  mtpCycles: number | null;
  mtpMeanAccepted: number | null;
  coldHits: number | null;
  coldMisses: number | null;
  toolCount: number | null;
  toolCallCount: number | null;
  imageCount: number | null;
  concurrency: number | null;
  concurrencyPeak: number | null;
  memActiveBytes: number | null;
  /** GPU pipelines created while the turn ran; null while the counter is off. */
  pipelinesCreated: number | null;
  pipelineCreateMs: number | null;
  pid: number | null;
}

// GET /api/metrics/turns?from&to&model&outcome&limit&beforeTs&beforeTraceId
// Newest first. limit default 200, max 2000.
// Cursor: WHERE (ts, trace_id) < (beforeTs, beforeTraceId).
// traceId=<id>: exact match on trace_id; every other param is ignored; `rows` holds 0 or 1 row, `next` is null,
// `total` is rows.length. Used by the turn drawer to open a turn that is not on the loaded page.
export interface MetricsTurns {
  range: MetricsRange;
  rows: TurnRow[];
  next: { beforeTs: number; beforeTraceId: string } | null;
  /** Rows matching the filter, ignoring the cursor and limit. */
  total: number;
}

export type HostKind = 'serve' | 'launch' | 'desktop' | 'embedded';
export type EngineStatus =
  | 'running'
  | 'loading'
  | 'starting'
  | 'restarting'
  | 'failed'
  | 'untrusted'
  | 'unresponsive'
  | 'stopped';

/** Lane-ms per state; `loading` is wall ms with at least one model load in flight. */
export interface WorkMs {
  loading: number;
  /** Dispatched, not started: the model thread is busy or no lane is free. */
  waiting: number;
  prepare: number;
  prefill: number;
  decode: number;
  /** Lanes without native progress (non-streamed turns, System One decisions). */
  running: number;
}

export interface RuntimeProcessLive {
  status: 'running' | 'unresponsive' | 'dead';
  /** now − ts. */
  ageMs: number;
  pid: number;
  startedAt: number;
  ts: number;
  host: HostKind;
  version: string;
  port: number;
  /** The switch the host resolved for the paged KV disk cache, not proof of a write; null from an older engine. */
  persistPagedCache: boolean | null;
  /** The per-turn capture limits; null when the cache is off or the engine is older. */
  coldCapture: { blocks: number; ms: number } | null;
  health: 'ok' | 'loading' | 'degraded' | 'error';
  mem: { active: number; peak: number; cache: number; wiredLimit: number; rss: number };
  /** Lane-ms per state since the process started; cumulative, never decreasing. */
  work: { ms: WorkMs };
  /**
   * Over the last `windowMs`: output tokens per second while at least one lane decodes (all lanes), and prefill tokens
   * per second of lane prefill time (prefills of 256+ tokens, cached prefix not counted). Null when the window holds no
   * such time, never 0.
   */
  rates: { windowMs: number; outputTps: number | null; prefillTps: number | null };
  /** Loads in flight, oldest first; empty when idle. The step counts are null when native reports none. */
  loading: Array<{
    model: string;
    startedAt: number;
    phase: 'weights' | 'kernels';
    stepsDone: number | null;
    steps: number | null;
  }>;
  counters: { ok: number; error: number; cancelled: number; outputTokens: number; loads: number; rejected: number };
  models: Array<{
    name: string;
    modelType: string;
    ctxWindow: number;
    paged: boolean;
    maxConcurrent: number;
    loadedAt: number | null;
    loadMs: number | null;
    inFlight: number;
    queued: number;
    queueLimit: number | null;
    pool: {
      at: number;
      blockCapacity: number;
      allocatedBlocks: number;
      freeBlocks: number;
      reclaimableBlocks: number;
    } | null;
    /** Routed experts per layer held in memory; null when the model reports none or the engine is older. */
    experts: { inMemory: number; total: number; streaming: boolean } | null;
  }>;
  turns: Array<{
    traceId: string;
    /** Alias the client used. Display and color use `modelId ?? model`. */
    model: string;
    modelId: string | null;
    ctxWindow: number | null;
    startedAt: number;
    stream: boolean;
    dispatch: 'start' | 'continue' | 'continue_tool';
    /** `queued`: the model is busy with other requests. `slot`: no lane or memory yet. `prepare`: before its prefill. */
    phase: 'queued' | 'slot' | 'prepare' | 'prefill' | 'decode' | null;
    waitMs: number | null;
    slotMs: number | null;
    prepareMs: number | null;
    /** Prefill start to the first token (or to now); null before the prefill. */
    prefillElapsedMs: number | null;
    /** Time since the first token; null before it. */
    decodeElapsedMs: number | null;
    promptTokens: number | null;
    cachedTokens: number | null;
    prefillTotal: number | null;
    prefillDone: number | null;
    generatedTokens: number | null;
    /** The session the client named on the request; null when it named none or the engine is older. */
    clientSession: string | null;
  }>;
}

export interface SupervisorLive {
  state: 'stopped' | 'starting' | 'running' | 'restarting' | 'failed' | 'lying';
  pid: number | null;
  generation: number;
  streak: number;
  shortLivedFailures: number;
  lastExit: { verdict: 'clean' | 'crash'; code: number | null; signal: string | null; atMs: number } | null;
  healthStatus: 'ok' | 'loading' | 'degraded' | 'error' | null;
  nativeErrorCount: number;
  /** The count is at the supervisor's cap of 20; print "20+". */
  nativeErrorsCapped: boolean;
  lastNativeErrorContext: string | null;
  lastNativeErrorAt: number | null;
  /** Why the last start failed, one line; set only while `restarting` or `failed`. */
  failureReason: string | null;
  writtenAt: number;
}

// GET /api/runtime/live  — no params. Also the payload of the `runtime` live event.
export interface RuntimeLive {
  now: number;
  status: EngineStatus;
  /** Desktop-managed sidecar; null when not run by the desktop app. */
  supervisor: SupervisorLive | null;
  /** Running first, then unresponsive, then dead; newest `startedAt` first within each. */
  processes: RuntimeProcessLive[];
}

export interface SchedSnapshot {
  ts: number;
  blockCapacity: number;
  allocatedBlocks: number;
  freeBlocks: number;
  reclaimableBlocks: number;
  watermarkBlocks: number;
  reservedBlocks: number;
  memoryCapacityBytes: number;
  /** Largest decode batch since the model was loaded. A high-water mark, not a counter. */
  maxBatchOccupancy: number;
  /** Turns sent to the model since its load, all lanes; null when the engine did not write the count. */
  turns: number | null;
}

// GET /api/runtime/series?from&to&bucketMs
// bucketMs ∈ {5000, 15000, 60000, 300000, 900000, 3600000}; (to − from) / bucketMs ≤ 2000. Both bounds required.
/** Lane-ms per engine state. `loading` is wall ms with a load in flight; `waiting` is model busy plus lane wait. */
export interface RuntimePhaseMs {
  loading: number;
  waiting: number;
  prepare: number;
  prefill: number;
  decode: number;
  running: number;
}

export interface RuntimeSeries {
  range: { from: number; to: number };
  bucketMs: number;
  points: Array<{
    t: number;
    samples: number;
    memActiveAvg: number | null;
    memActiveMax: number | null;
    memPeakMax: number | null;
    memCacheAvg: number | null;
    rssMax: number | null;
    inFlightAvg: number | null;
    inFlightMax: number | null;
    queuedMax: number | null;
    /** Sum of the samples' measured `busyMs`. */
    busyMs: number;
    okTurns: number;
    errorTurns: number;
    cancelledTurns: number;
    outTokens: number;
    rejected: number;
    phaseMs: RuntimePhaseMs;
    /** Output tokens after each lane's first, and the ms with at least one lane decoding. */
    rateOutTokens: number;
    rateOutMs: number;
    /** Tokens of prefills of 256+ tokens and their lane prefill ms. */
    ratePrefillTokens: number;
    ratePrefillMs: number;
  }>;
  procs: Array<{
    procKey: string;
    pid: number;
    startedAt: number;
    host: string | null;
    version: string | null;
    port: number | null;
    chip: string | null;
    wiredLimitBytes: number | null;
    totalMemBytes: number | null;
    firstTs: number;
    lastTs: number;
    exitTs: number | null;
  }>;
  loads: Array<{
    ts: number;
    procKey: string;
    model: string;
    modelType: string | null;
    capability: string | null;
    ms: number | null;
    ok: boolean;
  }>;
  /** One entry per (process, model) with at least one snapshot in or before the range. */
  sched: Array<{
    procKey: string;
    model: string;
    latest: SchedSnapshot;
    /** Pool at rest over time, snapshots inside the range. */
    snapshots: SchedSnapshot[];
    /** Growth of monotonic counters inside the range. */
    delta: {
      globalSteps: number;
      admitted: number;
      completed: number;
      preemptions: number;
      preemptionsRecompute: number;
      preemptionsSsd: number;
      admissionDeferredBlocks: number;
      admissionDeferredState: number;
      rowsAllocEvicted: number;
      ssdRestoreBytes: number;
      ssdRestoreWaitMs: number;
      /** Turns sent to the model, all lanes. 0 when the engine did not write the count. */
      turns: number;
      /** Decode steps per batch size. */
      occupancy: Array<{ occupancy: number; steps: number }>;
    };
  }>;
}

// Push events
export type LiveEvent =
  | { type: 'runtime'; snapshot: RuntimeLive }
  | { type: 'data'; version: number; traces: boolean; runtime: boolean };

/** `/api/health`: the directories this Control Panel reads. */
export interface HealthResponse {
  status: string;
  modelsDir: string;
  tracesDir: string;
  runtimeDir: string;
  /** The user's home folder: paths under it are shown as `~/…`. */
  homeDir: string;
}

/** One installed System One model (Clairic or CLEF). */
export interface SystemOneModelInfo {
  id: string;
  name: string;
  modelType: 'clairic' | 'clef';
  format: string;
  /** The build discovery names the model by (`Q8_0`), the catalog's build id; `null` for a model only the running endpoint lists. */
  variant: string | null;
  supportsImages: boolean;
  contextWindow: number;
  resident: boolean;
  path: string;
}

/** One completed, failed, or cancelled System One run stored in history. */
export interface SystemOneHistoryEntry {
  id: string;
  model: string;
  mode: 'extract' | 'decide';
  createdAt: number;
  elapsedMs: number;
  status: 'done' | 'error' | 'cancelled';
  fieldCount: number;
  inputTokens: number;
  outputTokens: number;
  request: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
}

/** `GET /api/systemone/state` */
export interface SystemOneState {
  sidecarRunning: boolean;
  models: SystemOneModelInfo[];
  activeJobId: string | null;
  history: SystemOneHistoryEntry[];
}

/** `GET /api/systemone/jobs/:id` and push event payload for `subscribeSystemOne`. */
export interface SystemOneJobSnapshot {
  id: string;
  status: 'running' | 'done' | 'error' | 'cancelled';
  model: string;
  mode: 'extract' | 'decide';
  startedAt: number;
  elapsedMs: number;
  result?: Record<string, unknown>;
  error?: string;
}
