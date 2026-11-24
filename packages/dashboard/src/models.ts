/**
 * Local model store for the dashboard: pure `fs` + JSON, never the native
 * addon (the dashboard is a viewer process that must start without Metal).
 *
 * Model metadata (family label, quantization, context window) is parsed from
 * each checkpoint's `config.json`; the family label runs the registry's own
 * `matchFamily`, reached through the native-free `@mlx-node/agent/catalog`
 * subpath. A folder whose weights are GGUF takes its quantization from the
 * GGUF file name instead, by the engine's name parser
 * (`@mlx-node/lm/model-variant`, native-free). A GGUF folder with no `config.json`
 * takes its family from the GGUF header (`@mlx-node/lm/gguf-metadata`, native-free).
 */

import {
  accessSync,
  closeSync,
  constants,
  type Dir,
  type Dirent,
  existsSync,
  fstatSync,
  lstatSync,
  opendirSync,
  openSync,
  readdirSync,
  readFileSync,
  readlinkSync,
  realpathSync,
  rmSync,
  statSync,
} from 'node:fs';
import { homedir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';

import { matchFamily } from '@mlx-node/agent/catalog';
import { readGgufArchitectureSync } from '@mlx-node/lm/gguf-metadata';
import { bareGgufFamily, ggufSplitPartNames, isGgufCompanionName } from '@mlx-node/lm/model-discovery';
import { ggufNames, ggufVariant } from '@mlx-node/lm/model-variant';

export interface LocalModel {
  /** Checkpoint directory name under `modelsDir` (the model's local id). */
  name: string;
  /** Absolute path to the checkpoint directory. */
  path: string;
  /** Family label from `config.json` (`qwen3`, `qwen3_5`, `gemma4`, …), or from the GGUF header without one. */
  modelType: string;
  /**
   * Quantization label. For GGUF weights ({@link ggufBuildLabel}) the build the file name states (`UD-Q4_K_XL`,
   * `Q4_K_M`, `BF16`), or `GGUF` when the folder holds several builds or the name states none. Otherwise from the
   * `config.json` quant block (`mxfp4`, `affine-4bit`, …), or `null` for full-precision.
   */
  quant: string | null;
  /** Trained context window from `config.json` ({@link detectContextWindow}), or `null` if absent. */
  contextWindow: number | null;
  /** Total size on disk of the checkpoint directory, recursively. */
  sizeBytes: number;
  /** Number of files in the checkpoint directory, recursively. */
  fileCount: number;
  /**
   * Names the inference host serves from this directory (one per GGUF variant),
   * sorted. The join key to traces (`model_key`) and live models. Empty when the
   * directory serves nothing, or when discovery could not run.
   */
  servedIds: string[];
  /** Set when Locate made the folder from files the user already had ({@link locatedState}). */
  located: LocatedState | null;
  /** The entry at {@link path} is a link: Delete removes only the link ({@link deleteLocalModel}). */
  link: boolean;
}

/** Draft-only storage, managed on disk but excluded from standalone model lists. */
export type LocalCompanion = Pick<LocalModel, 'name' | 'path' | 'sizeBytes' | 'fileCount'>;

/** `/api/models`: the inventory of the models folder. This Mac's memory and free disk are `/api/host`. */
export interface ModelsResponse {
  models: LocalModel[];
  companions: LocalCompanion[];
  warnings: string[];
  dir: string;
}

/**
 * Filename of the atomic-publish completion marker the download runner writes as
 * the LAST step of a download (inside the staging dir, before it is renamed onto
 * the final path). Its presence — not bare directory existence — is what marks a
 * checkpoint "installed".
 */
export const DOWNLOAD_COMPLETE_MARKER = '.mlx-download-complete.json';

/** Contents of {@link DOWNLOAD_COMPLETE_MARKER}: the pinned snapshot a dir holds. */
export interface DownloadCompletion {
  /** HuggingFace repo the checkpoint came from. */
  repo: string;
  /** The exact commit sha the whole download was pinned to (one snapshot). */
  revision: string;
  /** Repo-relative paths of every file published into the final dir. */
  files: string[];
  /**
   * Whether the marker describes a full-model run. Missing means `full` for
   * compatibility with markers written before selection scope was recorded.
   */
  scope?: 'full' | 'partial';
  /**
   * Base-model repo that supplied this install's tokenizer/config sidecars,
   * and the exact commit they were pinned to — the provenance update
   * discovery compares in addition to `repo`/`revision`.
   *
   * Sidecar files carry no other source identity (they are folded into
   * `files` like primary files), so without these two a base-repo tokenizer
   * fix is invisible: the primary revision matches, no badge is raised, and
   * no job exists to run the repair. Absent on installs that used no
   * assetsRepo and on markers written before this field existed — unknown
   * provenance compares as "no update known", never as "up to date".
   */
  assetsRepo?: string;
  assetsRevision?: string;
  /** ISO timestamp of the atomic publish. */
  completedAt: string;
  /**
   * Written only by Update, which adds a marker to a folder the app did not create: the `lstat` size and
   * `mtimeNs` (decimal digits) of every listed file as Update proved it. Any other file in that folder is the
   * user's. A later owned update, backup recovery and the sidecar refresh read it: they delete a file of such a
   * folder only when this record lists it and its size and `mtimeNs` still match, and keep everything else. A
   * record in any other shape reads as `{}`, which proves nothing, so nothing is deleted. The mlx CLI carries it
   * when it rewrites the marker.
   */
  adopted?: Record<string, { size: number; mtimeNs: string }>;
  /**
   * Written only by Locate: the weight-like files of this folder ({@link isLinkableFile}) are second names of files
   * in `from` (`shared`, same disk) or links to them (`linked`, another disk). Every other file is the app's own.
   * The mlx CLI carries it when it rewrites the marker.
   */
  located?: LocatedRecord;
}

/** The `located` record of a marker written by Locate. */
export interface LocatedRecord {
  /** The real path of the folder the user located. */
  from: string;
  mode: 'shared' | 'linked';
}

/** The `located` record of a marker, or `undefined` when it is absent or in any other shape. */
function readLocated(value: unknown): LocatedRecord | undefined {
  const record = asObject(value);
  if (record === undefined) return undefined;
  const { from, mode } = record;
  if (typeof from !== 'string' || !from.startsWith('/') || from.includes('\0')) return undefined;
  if (mode !== 'shared' && mode !== 'linked') return undefined;
  return { from, mode };
}

/**
 * A file Locate gives a second name or a link instead of a copy: its path ends in `.gguf`, `.safetensors` or
 * `.pdiparams`, in any case. Split parts, projectors, drafts and heads included.
 */
export function isLinkableFile(path: string): boolean {
  const lower = path.toLowerCase();
  return lower.endsWith('.gguf') || lower.endsWith('.safetensors') || lower.endsWith('.pdiparams');
}

/** The `adopted` record of a marker: `undefined` when absent, `{}` when present in any shape but the one Update writes. */
function readAdopted(value: unknown): DownloadCompletion['adopted'] {
  if (value === undefined) return undefined;
  const record = asObject(value);
  if (record === undefined) return {};
  const entries: [string, { size: number; mtimeNs: string }][] = [];
  for (const [path, entry] of Object.entries(record)) {
    const fields = asObject(entry);
    if (fields === undefined) return {};
    const { size, mtimeNs } = fields;
    if (typeof size !== 'number' || !Number.isSafeInteger(size) || size < 0) return {};
    if (typeof mtimeNs !== 'string' || !isDecimalDigits(mtimeNs)) return {};
    entries.push([path, { size, mtimeNs }]);
  }
  // `fromEntries` defines own properties, so a path named `__proto__` stays a key and never sets the prototype.
  return Object.fromEntries(entries);
}

function isDecimalDigits(text: string): boolean {
  if (text.length === 0) return false;
  for (const ch of text) {
    if (ch < '0' || ch > '9') return false;
  }
  return true;
}

/**
 * A file that carries a model's weights: safetensors, GGUF, or PaddlePaddle
 * params. The single source of truth for "is this a weight payload" — shared
 * with the download runner (`download.ts` imports it) so the publish-time
 * payload gate and the install check agree. A companion GGUF does not count:
 * a Gemma-style manifest carrying only `mmproj-BF16.gguf` (partial upstream
 * upload, renamed target) must fail the payload gate rather than publish.
 */
export function isWeightFile(path: string): boolean {
  if (path.endsWith('.gguf')) return !isGgufCompanionName(basename(path));
  return path.endsWith('.safetensors') || path.endsWith('.pdiparams');
}

/**
 * A checkpoint counts as installed only when its atomic-publish completion
 * marker is present, parses, lists BOTH a `config.json` and at least one weight
 * file, and every file it lists still exists on disk — never bare directory
 * existence (a half-written or aborted download with a partial `config.json` +
 * missing weights) nor a one-sided manifest (config-only or weights-only), both
 * of which `loadModel` would reject yet a laxer check would falsely report as
 * installed, hiding the retry.
 */
export function isModelInstalled(modelDir: string): boolean {
  const marker = readCompletion(modelDir);
  if (marker === undefined || marker.scope === 'partial') return false;
  const files = marker.files;
  // A loadable checkpoint needs a config AND a weight; a marker missing either
  // describes a hollow/one-sided publish that must NOT read as installed.
  const hasConfig = files.some((file) => file === 'config.json');
  const hasWeight = files.some((file) => typeof file === 'string' && isWeightFile(file));
  if (!hasConfig || !hasWeight) return false;
  return files.every((file) => typeof file === 'string' && isRegularFile(join(modelDir, file)));
}

/**
 * Whether `modelDir` holds a loadable checkpoint ON DISK — a `config.json` plus at
 * least one weight file, at the root or as a GGUF in a variant folder one level
 * down ({@link variantFolderGgufs}), or, with no `config.json`, a GGUF the engine
 * loads without asset files ({@link bareGgufFiles}) — regardless of the dashboard
 * completion marker. Unlike
 * {@link isModelInstalled} (downloader-OWNED, marker-gated) this also recognizes a
 * checkpoint installed by the agent wizard or an older `mlx download` CLI, which
 * did not write the marker (the current CLI writes the shared marker, making its
 * dirs downloader-owned). Callers use it to show such a model as PRESENT without
 * offering a dashboard install, which would refuse to overwrite the unowned
 * directory (`DownloadManager.refuseIfUnownedFinal`) and fail deterministically.
 *
 * FOLLOWS a symlink on the final component — `statSync`, not `lstatSync` — so a
 * symlinked model directory admits exactly like a real one. That mirrors the
 * loader and discovery (`isUsableDir`), which resolve a symlink to the
 * directory it points at: refusing to follow here would report `present=false`
 * for a checkpoint that loads and lists, and would flag it
 * `blockedByForeignDir` via the {@link isPathOccupied} fallback — a false
 * "foreign occupant" warning on a working install. A symlink whose target is
 * missing or is not a directory still reads as absent (a broken link admits
 * nothing); a symlinked models ROOT resolves the same way.
 */
export function isModelPresent(modelDir: string): boolean {
  try {
    if (!statSync(modelDir).isDirectory()) return false;
  } catch {
    return false;
  }
  if (!isRegularFile(join(modelDir, 'config.json'))) return bareGgufFiles(modelDir).length > 0;
  let entries: string[];
  try {
    entries = readdirSync(modelDir);
  } catch {
    return false;
  }
  // Presence is about a checkpoint the LOADER can open, so every payload below is
  // checked for its TYPE, never only its name. A directory named
  // `model.safetensors` is reachable without touching the disk by hand: a repo
  // holding `model.safetensors/x.safetensors` passes the download runner's path
  // filter, and publishing it runs `mkdir(dirname(dest), { recursive: true })`.
  // A name-only test then calls that wreckage present, which disables Install on
  // the card AND short-circuits a re-download to `done` — self-locking, with the
  // "needs cleanup" notice suppressed because `present` short-circuits it.
  if (isRegularFile(join(modelDir, 'model.safetensors'))) return true;
  if (isRegularFile(join(modelDir, 'weights.safetensors'))) return true;
  if (isRegularFile(join(modelDir, 'inference.pdiparams'))) return true;
  // A companion `.gguf` (mmproj/imatrix/dflash/draft) is NOT a loadable model:
  // counting one as present disables Install on the card for a directory the
  // loader never lists.
  if (rootGgufNames(modelDir, entries).length > 0) return true;
  if (variantFolderGgufs(modelDir, entries).length > 0) return true;
  // Sharded safetensors: every shard the index references must exist on disk —
  // otherwise an interrupted download that landed the index + only the first
  // shard would falsely read as present (matches the CLI's completeness check,
  // `isModelAlreadyDownloaded`). A lone shard without its index is not complete.
  return shardsComplete(modelDir);
}

/**
 * A primary safetensors name of the native loaders, the same set discovery reads (`PRIMARY_SAFETENSORS` in
 * `packages/lm/src/model-discovery.ts`): `model.safetensors`, `weights.safetensors`, or `model-…-of-….safetensors` /
 * `model.safetensors-…-of-….safetensors`.
 */
function isPrimarySafetensorsName(name: string): boolean {
  if (name === 'model.safetensors' || name === 'weights.safetensors') return true;
  const suffix = '.safetensors';
  if (!name.endsWith(suffix)) return false;
  let prefix: string;
  if (name.startsWith('model.safetensors-')) prefix = 'model.safetensors-';
  else if (name.startsWith('model-')) prefix = 'model-';
  else return false;
  const middle = name.slice(prefix.length, name.length - suffix.length);
  const of = middle.indexOf('-of-', 1);
  return of > 0 && of + '-of-'.length < middle.length;
}

/** Whether any of `names` in `dir` is a primary safetensors name on a regular file. */
function holdsPrimarySafetensors(dir: string, names: readonly string[]): boolean {
  return names.some((name) => isPrimarySafetensorsName(name) && isRegularFile(join(dir, name)));
}

/** The target GGUFs at the root of `modelDir` among its `entries`: regular files (links followed) ending in `.gguf` in any case, not companions. */
function rootGgufNames(modelDir: string, entries: readonly string[]): string[] {
  return entries.filter(
    (file) => file.toLowerCase().endsWith('.gguf') && !isGgufCompanionName(file) && isRegularFile(join(modelDir, file)),
  );
}

/**
 * The loadable GGUFs one level down in `modelDir`, in variant folders such as `UD-Q4_K_XL/`, which is how downloads
 * lay out a GGUF repository that keeps each build in its own folder. The walk is discovery's for a GGUF repository
 * folder (`appendVariantGguf` in `packages/lm/src/model-discovery.ts`: which folders, which companions, split sets);
 * its family name, header and sibling-asset checks are not applied. Only when the root holds no primary safetensors;
 * never a folder whose name starts with `.`; a folder, or a link to one, that holds primary safetensors is skipped;
 * nothing deeper is read. A target is a regular file (links followed) whose name ends in `.gguf` in any case and is
 * not a companion, and that the loader can open: a split set counts from its first part, and only with every part
 * beside it ({@link ggufSplitPartNames}).
 */
function variantFolderGgufs(modelDir: string, rootEntries: readonly string[]): (GgufFile & { folder: string })[] {
  const found: (GgufFile & { folder: string })[] = [];
  if (holdsPrimarySafetensors(modelDir, rootEntries)) return found;
  for (const folder of rootEntries) {
    if (folder.startsWith('.')) continue;
    const variantDir = join(modelDir, folder);
    let names: string[];
    try {
      if (!statSync(variantDir).isDirectory()) continue;
      names = readdirSync(variantDir);
    } catch {
      continue;
    }
    if (holdsPrimarySafetensors(variantDir, names)) continue;
    for (const name of names) {
      if (!name.toLowerCase().endsWith('.gguf') || isGgufCompanionName(name)) continue;
      const parts = ggufSplitPartNames(name, names);
      if (parts !== undefined && parts.every((part) => isRegularFile(join(variantDir, part))))
        found.push({ folder, name, parts });
    }
  }
  return found;
}

/**
 * The builds of the GGUF weights of `modelDir` (listed as `folderName`), each once, in file order; empty when its
 * weights are not GGUF: the root holds primary safetensors, or no GGUF counts by the rule of {@link isModelPresent}
 * (the root GGUFs, else the variant-folder ones). A root file is a build only by the variant walk's split rule
 * ({@link ggufSplitPartNames}: a first part with every part beside it). Each file's build is the one discovery names
 * it by (`ggufVariant`: the file name's quant tag, else its variant folder's).
 */
export function ggufBuilds(modelDir: string, folderName: string): string[] {
  return buildsOf(ggufWeightFiles(modelDir), folderName);
}

/**
 * A GGUF of a model folder: its name, the variant folder it is in (`undefined` at the root), and the files the loader
 * opens for it in that folder ({@link ggufSplitPartNames}).
 */
interface GgufFile {
  folder: string | undefined;
  name: string;
  parts: string[];
}

/**
 * The GGUFs {@link ggufBuilds} names builds from: none beside primary safetensors, else the root ones, else the ones in
 * variant folders.
 */
function ggufWeightFiles(modelDir: string): GgufFile[] {
  let entries: string[];
  try {
    entries = readdirSync(modelDir);
  } catch {
    return [];
  }
  if (holdsPrimarySafetensors(modelDir, entries)) return [];
  const root = rootGgufNames(modelDir, entries);
  if (root.length === 0) return variantFolderGgufs(modelDir, entries);
  return root.flatMap((name) => {
    const parts = ggufSplitPartNames(name, entries);
    return parts === undefined ? [] : [{ folder: undefined, name, parts }];
  });
}

/** The build of each of `files` (listed as `folderName`), each once, in file order. */
function buildsOf(files: readonly GgufFile[], folderName: string): string[] {
  const builds = new Set<string>();
  for (const { folder, name } of files) {
    const names = ggufNames(name, folderName, folder);
    builds.add(ggufVariant(names.stem, folderName, folder, names.id).variant);
  }
  return [...builds];
}

/**
 * The quant label of GGUF weights of these builds, or `undefined` for none. One build gives that build; several give
 * `GGUF`, because the folder is one row that serves each build under its own id. A `config.json` beside GGUF weights
 * is the base model's, so its quant block never applies.
 */
function ggufBuildLabel(builds: readonly string[]): string | undefined {
  if (builds.length === 0) return undefined;
  return builds.length === 1 ? builds[0]! : 'GGUF';
}

/** Header reads by path, kept while the file's `dev`, `ino`, size, `mtime` and `ctime` (a mode change) stay. */
const ggufArchitectures = new Map<string, { stamp: string; architecture: string | null }>();
const GGUF_ARCHITECTURE_CACHE_MAX = 256;

/** The `general.architecture` of the GGUF at `path` (a link followed), or `null` when its header cannot be read. */
function ggufArchitecture(path: string): string | null {
  let stamp: string;
  try {
    const stat = statSync(path, { bigint: true });
    stamp = `${stat.dev}:${stat.ino}:${stat.size}:${stat.mtimeNs}:${stat.ctimeNs}`;
  } catch {
    return null;
  }
  const known = ggufArchitectures.get(path);
  if (known?.stamp === stamp) return known.architecture;
  let architecture: string | null;
  try {
    architecture = readGgufArchitectureSync(path);
  } catch {
    architecture = null;
  }
  if (ggufArchitectures.size >= GGUF_ARCHITECTURE_CACHE_MAX) ggufArchitectures.clear();
  ggufArchitectures.set(path, { stamp, architecture });
  return architecture;
}

/**
 * The GGUFs of `modelDir` that the engine loads with no asset file beside them, each with its family: the folder has
 * no `config.json` (a dangling link counts as none, as for the engine), every part of the GGUF
 * ({@link ggufWeightFiles}) is a regular file (links followed), the real folder of the GGUF (where the engine looks
 * for its `config.json`) has no `config.json` either, and its header names a family whose GGUF discovery needs no
 * `config.json` or `tokenizer.json` and admits the file name (`bareGgufFamily`). A header that cannot be read admits
 * nothing.
 */
function bareGgufFiles(modelDir: string): (GgufFile & { family: string })[] {
  if (existsSync(join(modelDir, 'config.json'))) return [];
  const found: (GgufFile & { family: string })[] = [];
  for (const file of ggufWeightFiles(modelDir)) {
    const dir = file.folder === undefined ? modelDir : join(modelDir, file.folder);
    if (!file.parts.every((part) => isRegularFile(join(dir, part)))) continue;
    const path = join(dir, file.name);
    let realDir: string;
    try {
      realDir = dirname(realpathSync(path));
    } catch {
      continue;
    }
    if (existsSync(join(realDir, 'config.json'))) continue;
    const architecture = ggufArchitecture(path);
    const family = architecture === null ? undefined : bareGgufFamily(file.name, architecture);
    if (family !== undefined) found.push({ ...file, family });
  }
  return found;
}

/**
 * The model in `modelDir` (listed as `folderName`) as a catalog entry whose GGUF builds are of `family` reads it
 * (`undefined`: the entry names none), or `undefined` when there is none: {@link isModelPresent}, then with a
 * `config.json` the builds of {@link ggufBuilds}; without one (`configless`), only the builds of the GGUFs of
 * {@link bareGgufFiles} whose header states `family`, and no model when there are none.
 */
export function catalogFolderModel(
  modelDir: string,
  folderName: string,
  family: string | undefined,
): { configless: boolean; builds: string[] } | undefined {
  if (!isModelPresent(modelDir)) return undefined;
  if (existsSync(join(modelDir, 'config.json'))) return { configless: false, builds: ggufBuilds(modelDir, folderName) };
  const builds = buildsOf(
    bareGgufFiles(modelDir).filter((file) => file.family === family),
    folderName,
  );
  return builds.length > 0 ? { configless: true, builds } : undefined;
}

/** Entries {@link holdsModelFiles} reads before it gives up and answers "holds model files". */
export const HELD_SCAN_MAX_ENTRIES = 20_000;

/** File name endings of weights that discovery, the loaders or `mlx convert` read, compared in lower case. */
const MODEL_FILE_SUFFIXES = ['.safetensors', '.gguf', '.pdiparams', '.pdparams', '.npz', '.bin', '.pt', '.pth', '.pkl'];

export function isModelFileName(name: string): boolean {
  // Lower case: on a case-insensitive volume discovery and Delete open `Config.json` as `config.json`.
  const lower = name.toLowerCase();
  if (lower === 'config.json') return true;
  return MODEL_FILE_SUFFIXES.some((suffix) => lower.endsWith(suffix));
}

/**
 * Whether `dir` holds model files, so the catalog card offers Update for it: an entry named `config.json`, or named like
 * weights (see {@link MODEL_FILE_SUFFIXES}, companion GGUFs included), of any type, at any depth. Every layout that
 * discovery lists or the loader opens has one of these, so this cannot fall behind either of them.
 *
 * Follows a link at the root and links to folders inside it (discovery and the loader do too); a `dev:ino` set stops
 * a cycle. A folder that cannot be read, or more than `maxEntries` entries, counts as holding model files.
 */
export function holdsModelFiles(dir: string, maxEntries: number = HELD_SCAN_MAX_ENTRIES): boolean {
  try {
    if (!statSync(dir).isDirectory()) return false;
  } catch {
    return false;
  }
  let left = maxEntries;
  const visited = new Set<string>();
  const stack: string[] = [dir];
  while (stack.length > 0) {
    const current = stack.pop()!;
    let handle: Dir;
    try {
      const self = statSync(current);
      const key = `${self.dev}:${self.ino}`;
      if (visited.has(key)) continue;
      visited.add(key);
      handle = opendirSync(current);
    } catch {
      return true;
    }
    try {
      for (;;) {
        let entry: Dirent | null;
        try {
          entry = handle.readSync();
        } catch {
          return true;
        }
        if (entry === null) break;
        if (left <= 0) return true;
        left -= 1;
        if (isModelFileName(entry.name)) return true;
        const full = join(current, entry.name);
        if (isUsableDir(full, entry)) stack.push(full);
      }
    } finally {
      try {
        handle.closeSync();
      } catch {}
    }
  }
  return false;
}

/**
 * Every shard referenced by `model.safetensors.index.json`'s `weight_map` is a
 * regular file. Both the index and the shards it names go through the type gates:
 * a `weight_map` entry pointing at a directory describes nothing the loader can
 * open, and an index that is itself a FIFO would otherwise block this scan — and
 * with it the server — forever.
 */
function shardsComplete(modelDir: string): boolean {
  const raw = readRegularFile(join(modelDir, 'model.safetensors.index.json'));
  if (raw === undefined) return false;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    return false;
  }
  const weightMap =
    parsed !== null && typeof parsed === 'object' && 'weight_map' in parsed
      ? (parsed as { weight_map?: unknown }).weight_map
      : undefined;
  if (weightMap === null || typeof weightMap !== 'object') return false;
  const shards = new Set<string>();
  for (const value of Object.values(weightMap as Record<string, unknown>)) {
    if (typeof value === 'string') shards.add(value);
  }
  if (shards.size === 0) return false;
  for (const shard of shards) {
    if (!isRegularFile(join(modelDir, shard))) return false;
  }
  return true;
}

/**
 * Parse the atomic-publish completion marker in `finalDir` into its typed shape, or
 * `undefined` when the marker is absent or invalid. Unlike {@link isModelInstalled}
 * (a boolean "is a loadable checkpoint present") this surfaces the pinned snapshot's
 * IDENTITY — `repo` + `revision` — so a caller can tell WHICH revision a complete
 * install holds. The download runner uses it to avoid falsely reporting a complete
 * install of a DIFFERENT revision as already-done. It does not verify the listed
 * files still exist on disk (that is `isModelInstalled`'s job); callers pair the two.
 */
export function readCompletion(finalDir: string): DownloadCompletion | undefined {
  return parseCompletion(readMarkerFile(finalDir));
}

/** The parsed JSON of a completion marker as its typed shape, by the rules of {@link readCompletion}. */
export function parseCompletion(value: unknown): DownloadCompletion | undefined {
  const marker = asObject(value);
  if (
    marker === undefined ||
    typeof marker.repo !== 'string' ||
    typeof marker.revision !== 'string' ||
    typeof marker.completedAt !== 'string' ||
    !Array.isArray(marker.files) ||
    !marker.files.every((file) => typeof file === 'string') ||
    (marker.scope !== undefined && marker.scope !== 'full' && marker.scope !== 'partial') ||
    // Sidecar provenance is optional (pre-provenance markers lack it) but must
    // be a string when present — a numeric/object value is a malformed marker.
    (marker.assetsRepo !== undefined && typeof marker.assetsRepo !== 'string') ||
    (marker.assetsRevision !== undefined && typeof marker.assetsRevision !== 'string')
  ) {
    return undefined;
  }
  const adopted = readAdopted(marker.adopted);
  const located = readLocated(marker.located);
  return {
    repo: marker.repo,
    revision: marker.revision,
    files: marker.files,
    scope: marker.scope as DownloadCompletion['scope'],
    // The fields every consumer actually reads — a marker parsed here drops
    // anything not named, so new provenance MUST be threaded through or it
    // never reaches update discovery.
    ...(typeof marker.assetsRepo === 'string' ? { assetsRepo: marker.assetsRepo } : {}),
    ...(typeof marker.assetsRevision === 'string' ? { assetsRevision: marker.assetsRevision } : {}),
    completedAt: marker.completedAt,
    ...(adopted !== undefined ? { adopted } : {}),
    ...(located !== undefined ? { located } : {}),
  };
}

/** What a folder that Locate made uses, for the Models page and Local models. */
export interface LocatedState extends LocatedRecord {
  /** Every file the marker lists is there: the weights in `from` can be reached. */
  filesAvailable: boolean;
  /**
   * Deleting the folder leaves the model files on disk: each of its weight-like files is a link or has another name.
   * False when the folder holds the last name of one, so a delete frees its bytes.
   */
  filesStay: boolean;
}

/**
 * The {@link LocatedState} of `dir`, or `null` when it is not a real folder whose marker Locate wrote, or when none of
 * its weight-like files is a link or has another name any more.
 */
export function locatedState(dir: string): LocatedState | null {
  try {
    if (!lstatSync(dir).isDirectory()) return null;
  } catch {
    return null;
  }
  const completion = readCompletion(dir);
  if (completion?.located === undefined) return null;
  const { from, mode } = completion.located;
  const elsewhere = completion.files.filter(isLinkableFile).map((file) => {
    try {
      const stat = lstatSync(join(dir, file));
      return stat.isSymbolicLink() || stat.nlink >= 2;
    } catch {
      return false;
    }
  });
  // Every weight file is the folder's own (`mlx download` wrote new copies, or the located originals were removed):
  // the folder no longer uses the located files.
  if (!elsewhere.includes(true)) return null;
  return { from, mode, filesAvailable: isModelInstalled(dir), filesStay: !elsewhere.includes(false) };
}

/**
 * No-follow occupancy: is ANYTHING present at this path — including a dangling or
 * foreign symlink? `existsSync` FOLLOWS symlinks, so a `<modelsDir>/<slug>` that is
 * a dangling link (target on an unmounted volume, say) reads as ABSENT and slips
 * past the ownership preflight, triggering a full wasted download that only fails at
 * publish. `lstatSync` stats the LINK itself, so such a path reads as occupied and
 * is refused up front. {@link isDownloaderOwned} follows a LIVE link into a marked
 * directory (the target's marker is a valid ownership record — the publish swap
 * renames the LINK, never the target), so a live symlink into an unmarked dir or
 * a dangling one is what stays unowned and refused here.
 *
 * Shared with the catalog so the state the UI renders and the state the download
 * runner enforces are the SAME predicate: the runner refuses an occupied, unowned
 * final dir (`DownloadManager`'s ownership preflight), and the catalog marks exactly
 * that case blocked instead of offering an Install that would always error.
 */
export function isPathOccupied(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * Whether a directory was written by the dashboard downloader — i.e. it carries
 * a parseable completion marker of our shape. Distinct from {@link isModelInstalled}:
 * ownership does NOT require every listed file to still be present (a partial,
 * mid-upgrade owned dir is still ours). The downloader uses this to decide
 * whether it may overwrite a final dir: a dir WITHOUT our marker was placed by a
 * human or by an older `mlx download` CLI and must never be destroyed. The
 * current CLI writes the shared marker, so its installs count as
 * downloader-owned here.
 *
 * FOLLOWS a symlink on the final component — `statSync`, not `lstatSync`: a
 * symlinked model directory now lists, loads, and reports present like a real
 * one, so a live link into a marked install must read as owned too (its marker
 * is a valid record of THAT install; skip-detection reporting `done` through
 * the link is truthful). This stays safe at the destructive boundary because
 * every mutation of the final path acts on the LINK: `rename` moves it aside,
 * `rm` unlinks it — the foreign target is never written through. A symlink into
 * an UNMARKED dir is unowned exactly like that dir itself; a dangling link or a
 * non-directory resolves to false.
 *
 * The shape test is {@link readCompletion}, our FULL marker, not merely "carries a
 * `files` array". For a plain install this is the only gate between `POST /api/downloads`
 * and a `rename(finalDir → backup)` + `rm(backup, { recursive: true })`; only a
 * cleanup (`replaceBlocker`) passes an unowned entry, which it never removes with `rm`, and only one that still
 * matches what the user confirmed. A `{"files":[]}` file is not something we ever wrote — the marker has been the same
 * four fields since the first commit that emitted one — so accepting it authorized
 * the destructive swap on a directory we do not own. Going through the same reader
 * as every other ownership, identity and completeness check is also what stops those
 * from drifting apart.
 */
export function isDownloaderOwned(modelDir: string): boolean {
  try {
    if (!statSync(modelDir).isDirectory()) return false;
  } catch {
    return false;
  }
  return readCompletion(modelDir) !== undefined;
}

/**
 * Parse `<dir>/.mlx-download-complete.json` — the ONE reader every ownership,
 * identity and completeness check goes through, so none of them can drift.
 *
 * Opened `O_NOFOLLOW | O_NONBLOCK` and gated on `fstat().isFile()`, because
 * anything at that path which is not a regular file is not a marker we wrote,
 * and two of the alternatives are actively harmful:
 *
 * - A SYMLINK's target says nothing about THIS directory. Following one lets a
 *   link to any parseable marker vouch for a directory the downloader never
 *   created — and {@link isDownloaderOwned} is what permits the publish swap to
 *   move that directory aside and `rm -rf` the backup.
 * - A FIFO would BLOCK `readFileSync` until a writer appears. `/api/catalog` is
 *   synchronous and scans every child of the models dir, so one FIFO would wedge
 *   the whole single-threaded server, not merely this call. `O_NOFOLLOW` alone
 *   does not prevent that; the open must also be `O_NONBLOCK`, and the stat has
 *   to be on the DESCRIPTOR so nothing can be swapped in after the check.
 *
 * Nothing legitimate is lost: our own publish writes a regular file, and a
 * hard-linked copy (`cp -al`) is still a regular file.
 */
function readMarkerFile(dir: string): unknown {
  let fd: number | undefined;
  try {
    fd = openSync(
      join(dir, DOWNLOAD_COMPLETE_MARKER),
      constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK,
    );
    if (!fstatSync(fd).isFile()) return undefined;
    return JSON.parse(readFileSync(fd, 'utf-8')) as unknown;
  } catch {
    return undefined;
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

/**
 * Is `path` a REGULAR FILE — not a directory, FIFO, socket or device?
 *
 * FOLLOWS symlinks, like {@link isModelPresent} on the model ROOT and unlike
 * the no-follow gate {@link readMarkerFile} puts on the marker FILE, and the
 * difference is deliberate. The marker gate refuses a link because a link's
 * target is proof about the target, never about the directory holding the
 * link — while the model ROOT follows links because discovery admits a
 * symlinked directory wholesale. A weight FILE is the same case: every consumer
 * follows it — the loader opens it, and `walkDirStats` already `statSync`s it
 * precisely "so HF-cache-symlinked blobs count at their real byte size". A
 * no-follow test here would call a symlinked weight missing and hide a
 * checkpoint that loads perfectly, which is the regression this shape avoids.
 */
export function isRegularFile(path: string): boolean {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

/**
 * Read a regular file, or `undefined` for anything else — WITHOUT ever blocking.
 *
 * `readFileSync(path)` on a FIFO blocks until a writer appears, and both callers
 * sit on the synchronous request path, so one FIFO under the models dir wedges
 * the entire single-threaded server: not merely the request, but the event loop,
 * including the client's own abort timer and the process's signal handling (it
 * takes `SIGKILL` to reap). {@link readMarkerFile} already documents and solves
 * exactly this for the marker; these two readers are the rest of that fix.
 *
 * `O_NONBLOCK` makes the open return instead of waiting, and the `fstat` is on the
 * DESCRIPTOR so nothing can be swapped in after the check. `O_NOFOLLOW` is
 * deliberately NOT set, for the reason given on {@link isRegularFile}: a
 * HuggingFace snapshot stores `config.json` and the shard index as symlinks into
 * `blobs/`, and refusing those would stop reading checkpoints that load today.
 */
function readRegularFile(path: string): string | undefined {
  let fd: number | undefined;
  try {
    fd = openSync(path, constants.O_RDONLY | constants.O_NONBLOCK);
    if (!fstatSync(fd).isFile()) return undefined;
    return readFileSync(fd, 'utf-8');
  } catch {
    return undefined;
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

function asObject(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

/**
 * Whether `p` is a safe repo-relative path — one that, joined onto a base dir,
 * can only ever resolve INSIDE that base. Rejects anything a `join(base, p)`
 * could use to escape: an empty string; a backslash or NUL; a Windows drive
 * prefix (`C:`); an absolute (leading-slash) path; and any `/`-separated segment
 * that is empty (a leading/trailing/double slash), `.`, or `..`. This is the
 * shared component-rejection rule behind both {@link deleteLocalModel}'s stricter
 * single-child check and the download runner's gate on untrusted `listFiles`
 * paths before they become `join(stagingDir, path)` write targets.
 */
export function isSafeRelPath(p: string): boolean {
  if (p.length === 0) return false;
  if (p.includes('\\') || p.includes('\0')) return false;
  if (p.startsWith('/')) return false; // absolute (posix)
  if (/^[a-zA-Z]:/.test(p)) return false; // Windows drive prefix (e.g. C:)
  // Every path segment must be a real name — never empty (leading/trailing/double
  // slash) and never a `.`/`..` traversal component.
  for (const segment of p.split('/')) {
    if (segment === '' || segment === '.' || segment === '..') return false;
  }
  return true;
}

function hasClairicSidecars(modelDir: string): boolean {
  return (
    isRegularFile(join(modelDir, 'clair_head_config.json')) && isRegularFile(join(modelDir, 'clair_head.safetensors'))
  );
}

function hasClefSidecars(modelDir: string): boolean {
  return (
    isRegularFile(join(modelDir, 'joint_head_config.json')) && isRegularFile(join(modelDir, 'joint_head.safetensors'))
  );
}

/** Whether `dir` holds a target `.gguf` paired with Clairic head sidecars (valid without `config.json`). */
function isClairicGgufDir(dir: string): boolean {
  if (!hasClairicSidecars(dir)) return false;
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return false;
  }
  return entries.some(
    (name) => name.toLowerCase().endsWith('.gguf') && !isGgufCompanionName(name) && isRegularFile(join(dir, name)),
  );
}

/**
 * The family and builds of GGUF weights in `dir` that load without a `config.json`: a Clairic GGUF dir, or
 * {@link bareGgufFiles}. `undefined` when there are none.
 */
function configlessGguf(dir: string, folderName: string): { modelType: string; builds: string[] } | undefined {
  if (isClairicGgufDir(dir)) return { modelType: 'clairic', builds: ggufBuilds(dir, folderName) };
  const bare = bareGgufFiles(dir);
  return bare.length > 0 ? { modelType: bare[0]!.family, builds: buildsOf(bare, folderName) } : undefined;
}

/**
 * Family label from `config.json` via the registry's `matchFamily`, refined by
 * Qwen3.5 Clairic / CLEF head sidecars. A viewer must label, never throw, so
 * the shapes the loader fails closed on fall back to the raw `model_type`
 * (or `qwen3` when it is not a string).
 */
function detectModelTypeLabel(modelDir: string, config: Record<string, unknown>): string {
  try {
    const family = matchFamily(modelDir, config);
    if (family === 'qwen3_5') {
      if (hasClairicSidecars(modelDir)) return 'clairic';
      if (hasClefSidecars(modelDir)) return 'clef';
    }
    return family;
  } catch {
    return typeof config.model_type === 'string' ? config.model_type : 'qwen3';
  }
}

/**
 * Quantization label from `quantization` / `quantization_config`. Micro-scaling
 * and NVFP modes surface by name (`mxfp4`, `mxfp8`, `nvfp4`, `sym8`); a
 * modelopt checkpoint by its algorithm in lower case (`nvfp4`); affine
 * folds bits into the label (`affine-4bit`); a config with no quant block is
 * full-precision (`null`).
 */
function detectQuantLabel(config: Record<string, unknown>): string | null {
  const quant = asObject(config.quantization) ?? asObject(config.quantization_config);
  if (quant === undefined) return null;
  if (quant.quant_method === 'modelopt' && typeof quant.quant_algo === 'string' && quant.quant_algo !== '') {
    return quant.quant_algo.toLowerCase();
  }
  const mode =
    typeof quant.mode === 'string'
      ? quant.mode
      : typeof quant.quant_method === 'string'
        ? quant.quant_method
        : undefined;
  const bits = typeof quant.bits === 'number' && Number.isFinite(quant.bits) ? quant.bits : undefined;
  if (mode !== undefined && mode !== 'affine') return mode;
  if (bits !== undefined) return `affine-${bits}bit`;
  if (mode === 'affine') return 'affine';
  return null;
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

/** `value` when it is a positive safe integer: the `int(1e30)` "no limit" sentinel of tokenizer configs is not one. */
function positiveSafeInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

/**
 * `max_position_embeddings` at the root, then in `text_config`; without either, `model_max_length` in `text_config`
 * (Inkling-2 states its window only there), then at the root; else `null`.
 */
function detectContextWindow(config: Record<string, unknown>): number | null {
  const textConfig = asObject(config.text_config);
  return (
    positiveInteger(config.max_position_embeddings) ??
    positiveInteger(textConfig?.max_position_embeddings) ??
    positiveSafeInteger(textConfig?.model_max_length) ??
    positiveSafeInteger(config.model_max_length) ??
    null
  );
}

/**
 * Belt-and-braces cap on entries examined while sizing one checkpoint dir. A real
 * HF checkpoint holds a few dozen files; anything far larger is pathological (a
 * cycle the no-follow walk below already breaks, or a runaway tree), so the walk
 * aborts and returns a partial (lower-bound) size rather than wedging the
 * synchronous `/api/models` handler.
 */
const MAX_WALK_ENTRIES = 100_000;

export interface WalkMode {
  /** Stat the root and every entry with `lstat`: links count as files of 0 bytes and are never followed. */
  noFollow?: boolean;
  /** Keep the first N relative paths of the files counted (no-follow mode). */
  keepNames?: number;
}

/**
 * Iterative size + file count of a directory, bounded so it can never hang the
 * event loop. Traversal is an explicit worklist of directory PATHS, not recursion:
 * each directory's `Dir` handle is CLOSED before any child is opened, so at most
 * ONE handle is ever open — a deep tree can never hold one fd per level and hit
 * `EMFILE` on a child `opendirSync` (which a recursive descent, whose parent handle
 * stays open across the child call, does). The ROOT is followed once (`statSync`):
 * discovery now admits a symlinked model directory, so the walk must size the
 * checkpoint the loader opens rather than reporting 0 bytes for a linked install.
 * INSIDE the tree only REAL directories are descended into (`Dirent` carries lstat
 * semantics, so a symlink whose target is a directory reports
 * `isDirectory() === false` and is never pushed) — this is what breaks a
 * symlink cycle (`self -> .`, or two `-> .` links whose exponential fan-out used to
 * freeze the handler) and keeps an external symlinked tree's bytes out of the total.
 * A symlink is still `statSync`'d for its real byte size (preserving the HF-cache
 * blob-link intent) but never descended into. A `dev:ino` visited-set guards against
 * a real-directory hardlink/mount loop, and {@link MAX_WALK_ENTRIES} caps the total
 * work. A broken-symlink / unreadable FILE is a genuine per-file skip (silent).
 * `truncated` is set — the returned size is then a LOWER BOUND — when the budget was
 * hit OR when a resource failure (`opendirSync`/`readSync`) left a subtree unsized;
 * such a failure is never swallowed into an exact-looking undercount.
 */
export function walkDirStats(
  dir: string,
  maxEntries: number = MAX_WALK_ENTRIES,
  mode: WalkMode = {},
): { sizeBytes: number; fileCount: number; truncated: boolean; names: string[] } {
  const noFollow = mode.noFollow === true;
  const keepNames = mode.keepNames ?? 0;
  const names: string[] = [];
  let sizeBytes = 0;
  let fileCount = 0;
  let entriesSeen = 0;
  let truncated = false;
  // Hitting the entry budget must stop ALL further work (the budget bounds total
  // time so the synchronous `/api/models` handler can't wedge); a per-directory
  // resource failure only marks the result incomplete and skips that one dir.
  let budgetHit = false;
  const visited = new Set<string>();
  // Explicit worklist of directory PATHS. Bounded by the entry budget: every pushed
  // subdirectory was itself a counted entry, so the stack can never outgrow it.
  const stack: string[] = [dir];

  while (stack.length > 0 && !budgetHit) {
    const current = stack.pop()!;
    // Break any real-directory cycle (hardlink/bind loop) by identity. The no-follow
    // descent below already excludes symlinked dirs, so this is defense-in-depth for
    // the rare non-symlink loop. A vanished/unreadable dir here is a TOCTOU skip
    // (like the per-file skip), not a resource-exhaustion truncation.
    try {
      // The ROOT is followed (`statSync`) — discovery admits a symlinked model
      // dir, so its real tree must be sized; children stay no-follow (`lstatSync`)
      // so a symlink is never descended into and a cycle stays broken.
      const self = (current === dir && !noFollow ? statSync : lstatSync)(current);
      const key = `${self.dev}:${self.ino}`;
      if (visited.has(key)) continue;
      visited.add(key);
      // A queued path is opened later (deferred), so a real directory can be swapped
      // to a symlink between enqueue and pop. `opendirSync` FOLLOWS symlinks, so a
      // swapped-in link would be descended into and its external target sized. `lstat`
      // is no-follow — a symlink's `lstat` is never a directory — so reject any
      // non-directory here BEFORE `opendirSync`, closing the escape across the whole
      // enqueue→pop window. Only queued CHILDREN get the no-follow stat: the root
      // itself was followed once above (deliberately — discovery admits symlink
      // roots now), so it resolves to the real tree it names.
      //
      // ACCEPTED RESIDUAL: this no-follows only the FINAL component. Because the walk
      // traverses lexical paths, a concurrent adversary who swaps an ANCESTOR directory
      // (or the final component in the gap between this `lstat` and `opendirSync`) to a
      // symlink DURING the walk can still redirect sizing onto an external tree. This is
      // accepted, not defended: the output (`sizeBytes`/`fileCount`) is a best-effort
      // DISPLAY-ONLY estimate — it never drives a destructive or security decision
      // (`deleteLocalModel` deletes by the path-safe NAME, never by this size), so the
      // worst case under adversarial FS mutation of the user's OWN local store is a
      // wrong size NUMBER in the UI. A fully robust defense would need descriptor-
      // relative (`openat`/`O_NOFOLLOW`) traversal, which `node:fs`'s sync API does not
      // expose; the lexical walk is the deliberate trade-off.
      if (!self.isDirectory()) continue;
    } catch {
      continue;
    }

    // Enumerate INCREMENTALLY: `opendirSync` + `readSync` reads one `Dirent` at a
    // time, so the entry budget bounds BOTH memory and time even on a pathological
    // high-fan-out directory. `readdirSync({ withFileTypes: true })` would instead
    // first materialize the ENTIRE `Dirent[]` (millions of entries) and block the
    // synchronous `/api/models` handler before the per-entry budget could apply.
    let handle: Dir;
    try {
      handle = opendirSync(current);
    } catch {
      // Permission / fd-exhaustion (EMFILE): the dir's contents are unknown. Flag the
      // whole result incomplete (a lower bound) rather than silently dropping the
      // subtree, then keep sizing the rest of the worklist.
      truncated = true;
      continue;
    }
    try {
      for (;;) {
        let entry: Dirent | null;
        try {
          entry = handle.readSync();
        } catch {
          // A read fault mid-directory leaves the remainder of THIS dir unsized:
          // mark incomplete and move on rather than returning a silent undercount.
          truncated = true;
          break;
        }
        // Genuine end-of-directory: not a truncation.
        if (entry === null) break;
        // Budget check BEFORE processing this entry: a directory with more than
        // `maxEntries` entries stops here (with `truncated`) rather than draining
        // every remaining `Dirent`. Placed after the read so a directory holding
        // EXACTLY `maxEntries` entries reaches end-of-stream above and is NOT
        // falsely reported as truncated (matches the prior eager semantics).
        if (entriesSeen >= maxEntries) {
          truncated = true;
          budgetHit = true;
          break;
        }
        entriesSeen++;
        const full = join(current, entry.name);
        // Descend ONLY into a real directory; a symlink-to-directory is never entered.
        // PUSH the child path (processed after this handle is closed) — never recurse
        // while this handle is still open — so at most one `Dir` handle is ever live.
        if (entry.isDirectory()) {
          stack.push(full);
          continue;
        }
        if (noFollow) {
          // A foreign entry the app may remove: every non-directory is one file, and only a regular file adds
          // bytes. Nothing is followed or opened, so a link to a large external file or a FIFO costs nothing.
          fileCount += 1;
          if (names.length < keepNames) names.push(full.slice(dir.length + 1));
          if (entry.isFile()) {
            try {
              sizeBytes += lstatSync(full).size;
            } catch {
              // Gone since the read: counted, with no bytes.
            }
          }
          continue;
        }
        try {
          // `statSync` follows symlinks so HF-cache-symlinked blobs count at their
          // real byte size rather than the ~100-byte link. A symlink whose target is
          // a directory contributes neither bytes nor a descent (never followed).
          const stat = statSync(full);
          if (!stat.isDirectory()) {
            sizeBytes += stat.size;
            fileCount += 1;
          }
        } catch {
          // Broken symlink / unreadable file: a genuine per-file skip (not a
          // resource exhaustion), so it never flags the result incomplete.
        }
      }
    } finally {
      try {
        handle.closeSync();
      } catch {
        // Best-effort close; a failed close must never surface into `/api/models`.
      }
    }
  }

  return { sizeBytes, fileCount, truncated, names };
}

/**
 * Whether a discovery entry names a directory the loader could open: a REAL
 * directory, or a symlink whose target is a directory (`statSync` follows;
 * `Dirent.isDirectory()` is false for any symlink). Both are admitted — a
 * linked-in checkpoint must list like a copied one. A dangling or unreadable
 * link resolves to false (skipped silently, like sockets/fifos — a broken link
 * admits nothing).
 */
function isUsableDir(path: string, kind: Pick<Dirent, 'isDirectory' | 'isSymbolicLink'>): boolean {
  if (kind.isDirectory()) return true;
  if (!kind.isSymbolicLink()) return false;
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

/** A checkpoint's `config.json` as a JSON object, or why discovery skips the directory. */
function readCheckpointConfig(dir: string): { config: Record<string, unknown> } | { problem: string } {
  // Read through the type gate, not `readFileSync(path)`: discovery walks
  // every child of the models dir, so a single `config.json` FIFO would hang
  // this loop and the whole server with it.
  const raw = readRegularFile(join(dir, 'config.json'));
  if (raw === undefined) return { problem: 'no readable config.json' };
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    return { problem: 'no readable config.json' };
  }
  const config = asObject(parsed);
  return config === undefined ? { problem: 'config.json root is not a JSON object' } : { config };
}

/**
 * Discovery's rule for one entry of the models directory: a usable directory (real, or a live link to one) whose
 * name does not start with `.`, carrying a `config.json` that is a readable regular file with a JSON object root,
 * or a GGUF the engine loads without one ({@link bareGgufFiles}).
 */
function isListedEntry(path: string, kind: Pick<Dirent, 'isDirectory' | 'isSymbolicLink'>): boolean {
  return (
    !basename(path).startsWith('.') &&
    isUsableDir(path, kind) &&
    ('config' in readCheckpointConfig(path) || configlessGguf(path, basename(path)) !== undefined)
  );
}

/**
 * Scan `modelsDir` for checkpoint subdirectories. A subdirectory is a model
 * only when it carries a readable `config.json`, or holds a GGUF the engine loads
 * without one ({@link bareGgufFiles}); anything else is reported as
 * a warning and skipped (never guessed). A missing `modelsDir` yields empty
 * results with no warning. Draft-only checkpoints have a separate inventory
 * for storage accounting and deletion, never contributing to model counts.
 */
export function discoverLocalModels(
  modelsDir: string,
  options: { includeStats?: boolean } = {},
): {
  models: LocalModel[];
  companions: LocalCompanion[];
  warnings: string[];
} {
  const models: LocalModel[] = [];
  const companions: LocalCompanion[] = [];
  const warnings: string[] = [];

  let entries: Dirent[];
  try {
    entries = readdirSync(modelsDir, { withFileTypes: true });
  } catch {
    return { models, companions, warnings };
  }

  for (const entry of entries) {
    // A symlink whose target is a directory is admitted: the loader resolves it,
    // so it lists like a real checkpoint dir. A link to a file, a dangling link,
    // and every other non-directory is skipped before it can warn.
    const full = join(modelsDir, entry.name);
    if (!isUsableDir(full, entry)) continue;
    // Hidden dirs are runner scratch (`.staging/<slug>`), never checkpoints:
    // skip them silently so an in-flight download raises no spurious warning.
    if (entry.name.startsWith('.')) continue;
    const read = readCheckpointConfig(full);
    if ('problem' in read) {
      const label = configlessGguf(full, entry.name);
      if (label === undefined) {
        warnings.push(`${entry.name}: ${read.problem}; skipped`);
        continue;
      }
      const { sizeBytes, fileCount, truncated } =
        options.includeStats === false ? { sizeBytes: 0, fileCount: 0, truncated: false } : walkDirStats(full);
      if (truncated) {
        warnings.push(`${entry.name}: directory too large to size fully; reported size is a lower bound`);
      }
      models.push({
        name: entry.name,
        path: full,
        modelType: label.modelType,
        quant: ggufBuildLabel(label.builds) ?? null,
        contextWindow: null,
        sizeBytes,
        fileCount,
        servedIds: [],
        located: locatedState(full),
        link: entry.isSymbolicLink(),
      });
      continue;
    }
    const { config } = read;

    const { sizeBytes, fileCount, truncated } =
      options.includeStats === false ? { sizeBytes: 0, fileCount: 0, truncated: false } : walkDirStats(full);
    if (truncated) {
      warnings.push(`${entry.name}: directory too large to size fully; reported size is a lower bound`);
    }
    if (Array.isArray(config.architectures) && config.architectures.includes('DFlash2DraftModel')) {
      companions.push({ name: entry.name, path: full, sizeBytes, fileCount });
      continue;
    }
    models.push({
      name: entry.name,
      path: full,
      modelType: detectModelTypeLabel(full, config),
      quant: ggufBuildLabel(ggufBuilds(full, entry.name)) ?? detectQuantLabel(config),
      contextWindow: detectContextWindow(config),
      sizeBytes,
      fileCount,
      servedIds: [],
      located: locatedState(full),
      link: entry.isSymbolicLink(),
    });
  }

  models.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  companions.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return { models, companions, warnings };
}

/**
 * The path of `name` as a SINGLE direct child of `modelsDir`, the gate every removal under the models directory goes
 * through (a delete by route name, a cleanup of a catalog slug). Throws `Refusing to {verb} "{name}": …` when
 * `name` could reach anything else.
 */
export function modelsChildPath(modelsDir: string, name: string, verb: string): string {
  // A safe relative path (no `.`/`..`/backslash/NUL/drive prefix — shared with the download gate) that carries no
  // separator, so a decoded route param like `link%2Fvictim` or `../../etc` can never traverse out of the store.
  if (!isSafeRelPath(name) || name.includes('/')) {
    throw new Error(`Refusing to ${verb} "${name}": not a direct child of the models directory`);
  }
  // A legitimate checkpoint dir never starts with `.` — discovery skips dotdirs.
  // Rejecting them keeps the runner's reserved scratch (`.staging`, and any leaked
  // `<slug>.backup-*` / `<slug>@<sha>...` under it) out of reach.
  if (name.startsWith('.')) {
    throw new Error(`Refusing to ${verb} "${name}": reserved dashboard directory`);
  }
  const root = resolve(modelsDir);
  const target = join(root, name);
  if (dirname(target) !== root) {
    throw new Error(`Refusing to ${verb} "${name}": resolves outside the models directory`);
  }
  return target;
}

/** Entries a blocker summary walks before it reports `truncated`. */
export const BLOCKER_WALK_MAX_ENTRIES = 2_000;
/** Names are listed only for this few files. */
const BLOCKER_MAX_NAMES = 3;
const LINK_TARGET_MAX_CHARS = 200;

/**
 * What occupies a path the app did not create. The counts come from `lstat` and never follow a link or open a file;
 * `listed` reads the root `config.json` (or GGUF headers without one) and `holdsModel` lists the whole tree (both
 * through a link at the root too).
 */
export interface ForeignEntrySummary {
  /** `lstat` of the path, never followed. `file` = not a directory and not a link. */
  kind: 'directory' | 'symlink' | 'file';
  /** directory: files below it, recursive, bounded. file: 1. symlink: 0. */
  fileCount: number;
  /** directory: sum of the regular files. file: its size. symlink: 0. */
  sizeBytes: number;
  /** The walk stopped at its budget or could not read a part: both numbers are lower bounds. */
  truncated: boolean;
  /** Relative POSIX paths, sorted, only when `!truncated && fileCount <= 3`; else `[]`. */
  names: string[];
  /** Local models lists this entry (model or companion). */
  listed: boolean;
  /** A `config.json` of any type is at the root of the entry. */
  modelLike: boolean;
  /** {@link holdsModelFiles}: it holds a config or weights, so Update may check it in place. */
  holdsModel: boolean;
  /** symlink: {@link readLinkText}, else `null`. */
  linkTarget: string | null;
  /** symlink whose target does not exist. */
  dangling: boolean;
  /** symlink whose target exists and is not a folder. */
  linkToFile: boolean;
  /** A real directory the app has no write permission on: nothing can move it or add files to it. */
  locked: boolean;
}

/** The `readlink` text of `path`, cut to 200 characters plus `…`, or `null` when it is not a readable link. */
export function readLinkText(path: string): string | null {
  try {
    const text = readlinkSync(path);
    return text.length > LINK_TARGET_MAX_CHARS ? `${text.slice(0, LINK_TARGET_MAX_CHARS)}…` : text;
  } catch {
    return null;
  }
}

/** Whether Local models lists the entry at `path`, by discovery's own rule. */
export function isListedFolder(path: string): boolean {
  try {
    return isListedEntry(path, lstatSync(path));
  } catch {
    return false;
  }
}

/**
 * Measure what is at `path`, or `null` when nothing is. The walk never follows a link and never opens a file; to
 * decide `listed`, it reads the root `config.json`, or without one GGUF headers; `holdsModel` goes by file names.
 */
export function summarizeForeignEntry(
  path: string,
  maxEntries: number = BLOCKER_WALK_MAX_ENTRIES,
): ForeignEntrySummary | null {
  let stat;
  try {
    stat = lstatSync(path);
  } catch {
    return null;
  }
  if (stat.isSymbolicLink()) {
    const linkTarget = readLinkText(path);
    const dangling = !existsSync(path);
    let linkToFile = false;
    try {
      linkToFile = !statSync(path).isDirectory();
    } catch {}
    const holdsModel = holdsModelFiles(path);
    return {
      kind: 'symlink',
      fileCount: 0,
      sizeBytes: 0,
      truncated: false,
      names: [],
      listed: holdsModel && isListedEntry(path, stat),
      modelLike: !dangling && existsSync(join(path, 'config.json')),
      holdsModel,
      linkTarget,
      dangling,
      linkToFile,
      locked: false,
    };
  }
  if (!stat.isDirectory()) {
    return {
      kind: 'file',
      fileCount: 1,
      sizeBytes: stat.isFile() ? stat.size : 0,
      truncated: false,
      names: [],
      listed: false,
      modelLike: false,
      holdsModel: false,
      linkTarget: null,
      dangling: false,
      linkToFile: false,
      locked: false,
    };
  }
  let locked = false;
  try {
    accessSync(path, constants.W_OK);
  } catch {
    locked = true;
  }
  const walk = walkDirStats(path, maxEntries, { noFollow: true, keepNames: BLOCKER_MAX_NAMES + 1 });
  // `listed` lists folders without a budget, so the bounded walk of `holdsModel` (it counts every `config.json` and
  // `.gguf`) goes first.
  const holdsModel = holdsModelFiles(path);
  const names =
    !walk.truncated && walk.fileCount <= BLOCKER_MAX_NAMES
      ? [...walk.names].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
      : [];
  return {
    kind: 'directory',
    fileCount: walk.fileCount,
    sizeBytes: walk.sizeBytes,
    truncated: walk.truncated,
    names,
    listed: holdsModel && isListedEntry(path, stat),
    modelLike: isPathOccupied(join(path, 'config.json')),
    holdsModel,
    linkTarget: null,
    dangling: false,
    linkToFile: false,
    locked,
  };
}

/** The entry still matches what the user was shown. A truncated walk is compared by kind and link text only. */
export function sameForeignEntry(
  a: Pick<ForeignEntrySummary, 'kind' | 'fileCount' | 'sizeBytes' | 'truncated' | 'linkTarget'>,
  b: Pick<ForeignEntrySummary, 'kind' | 'fileCount' | 'sizeBytes' | 'truncated' | 'linkTarget'>,
): boolean {
  return (
    a.kind === b.kind &&
    a.truncated === b.truncated &&
    a.linkTarget === b.linkTarget &&
    (a.truncated || (a.fileCount === b.fileCount && a.sizeBytes === b.sizeBytes))
  );
}

/** A Remove that promised the model files stay on disk, when the folder now holds the last name of one. */
export class FilesNoLongerStayError extends Error {
  constructor(name: string) {
    super(`The model files of "${name}" are no longer anywhere else on disk. Delete now removes them.`);
    this.name = 'FilesNoLongerStayError';
  }
}

/**
 * Delete a checkpoint directory by its exact name. `name` must be a single
 * direct child of `modelsDir`: any path separator, `.`/`..`, or NUL is rejected,
 * so a decoded route param like `link%2Fvictim` (→ `link/victim`) or `../../etc`
 * can never traverse out of the store. The resolved child is `lstat`-ed without
 * following the final component. A symlinked child is admitted — discovery lists
 * it — but ONLY the link itself is removed (a non-recursive `rmSync` unlinks a
 * symlink rather than following it), so the foreign target outside the store is
 * never touched.
 *
 * `keepsFiles` is read once the name is found, so an unknown name stays "not found" before a bad request. When it
 * answers true (the Remove form told the user the model files stay on disk), the delete is refused with
 * {@link FilesNoLongerStayError} unless {@link locatedState} still says so.
 */
export function deleteLocalModel(modelsDir: string, name: string, keepsFiles: () => boolean = () => false): void {
  const target = modelsChildPath(modelsDir, name, 'delete');

  let stat;
  try {
    stat = lstatSync(target);
  } catch {
    throw new Error(`Model "${name}" not found`);
  }
  if (keepsFiles() && locatedState(target)?.filesStay !== true) throw new FilesNoLongerStayError(name);
  if (stat.isSymbolicLink()) {
    // A linked-in checkpoint is deletable like a real one — but only the LINK
    // is removed, never the foreign target it names. The checkpoint gate below
    // still applies through the link, so a link to a non-checkpoint is refused.
    if (!existsSync(join(target, 'config.json')) && configlessGguf(target, name) === undefined) {
      throw new Error(`Refusing to delete "${name}": not a model checkpoint directory`);
    }
    // Non-recursive `rmSync` unlinks a symlink — it never descends into the
    // target, so the external checkpoint survives intact.
    rmSync(target);
    return;
  }
  // Only delete a real checkpoint directory — a model or companion (a dir
  // carrying a `config.json`, or GGUF weights that load without one). The route takes an arbitrary name, so without
  // this a typo or hand-crafted request could `rmSync` an unrelated regular file
  // or non-model directory that merely happens to sit under `modelsDir`.
  //
  // `existsSync` and NOT the stricter `isRegularFile` the presence checks use:
  // this is the guard on the only route that can CLEAR a malformed checkpoint,
  // so a `config.json` that is a directory must still be deletable. Tightening
  // it here would make the wreckage those checks now report permanent.
  if (!stat.isDirectory() || (!existsSync(join(target, 'config.json')) && configlessGguf(target, name) === undefined)) {
    throw new Error(`Refusing to delete "${name}": not a model checkpoint directory`);
  }
  rmSync(target, { recursive: true, force: true });
}

/**
 * Default models directory, mirroring `resolveModelsDir` in
 * `@mlx-node/server/host/paths` (`packages/server/src/host/paths.ts`):
 * `MLX_MODELS_DIR` env → `modelsDir` field of `~/.mlx-node/config.json` →
 * `~/.mlx-node/models`. Unlike that helper this never creates the directory —
 * a viewer only reads.
 */
export function defaultModelsDir(): string {
  const envDir = process.env.MLX_MODELS_DIR;
  if (envDir !== undefined && envDir.length > 0) return resolve(envDir);

  const home = join(homedir(), '.mlx-node');
  const fromConfig = readModelsDirFromConfig(join(home, 'config.json'));
  if (fromConfig !== undefined) return resolve(fromConfig);

  return join(home, 'models');
}

/**
 * The plain sentence for a models directory that is a link whose target is missing (a folder on a disk that is not
 * connected); null for anything else, including a directory that does not exist yet. The same text as
 * `ModelsDirUnavailableError` in `@mlx-node/server/host/paths`, which the engine and the CLI throw.
 */
export function modelsDirUnavailable(modelsDir: string): string | null {
  try {
    if (!lstatSync(modelsDir).isSymbolicLink()) return null;
  } catch {
    return null;
  }
  try {
    statSync(modelsDir);
    return null;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') return null;
  }
  let target: string;
  try {
    target = resolve(dirname(modelsDir), readlinkSync(modelsDir));
  } catch {
    return null;
  }
  return `The models folder ${modelsDir} is a link to ${target}, which is not available. Connect that disk, or change the link, then try again.`;
}

function readModelsDirFromConfig(configPath: string): string | undefined {
  // Type-gated like every other config read in this module: a FIFO here would
  // block the server before it ever finished starting.
  const raw = readRegularFile(configPath);
  if (raw === undefined) return undefined;
  try {
    const parsed = JSON.parse(raw) as { modelsDir?: unknown };
    if (typeof parsed.modelsDir === 'string' && parsed.modelsDir.length > 0) return parsed.modelsDir;
  } catch {
    // Malformed config.json: fall through to the default.
  }
  return undefined;
}
