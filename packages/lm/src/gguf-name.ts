/** GGUF file-name parsers shared by discovery, the dashboard, and the CLI. Native-free. */

const GGUF_EXTENSION = '.gguf';

/** Characters that separate name tokens: `UD-Q4_K_XL`, `mmproj-F16`, `model.mtp`. */
function isNameSeparator(char: string | undefined): boolean {
  return char === '-' || char === '_' || char === '.';
}

function isDigit(char: string | undefined): boolean {
  return char !== undefined && char >= '0' && char <= '9';
}

/** The name without a case-insensitive `.gguf` extension, or undefined when it has none. */
function withoutGgufExtension(name: string): string | undefined {
  if (name.length <= GGUF_EXTENSION.length) return undefined;
  if (name.slice(-GGUF_EXTENSION.length).toLowerCase() !== GGUF_EXTENSION) return undefined;
  return name.slice(0, -GGUF_EXTENSION.length);
}

export interface GgufFileName {
  /** Path without the `.gguf` and `-NNNNN-of-NNNNN` suffixes. */
  readonly stem: string;
  /** 1-based shard number; 1 for a single file. */
  readonly ordinal: number;
  /** Shard count; 1 for a single file. */
  readonly shards: number;
}

const SPLIT_OF = '-of-';
const SPLIT_DIGITS = 5;

/** The fixed-width decimal at `text[start, start + SPLIT_DIGITS)`, or undefined. */
function splitNumber(text: string, start: number): number | undefined {
  if (start < 0) return undefined;
  let value = 0;
  for (let index = start; index < start + SPLIT_DIGITS; index++) {
    const char = text[index];
    if (!isDigit(char)) return undefined;
    value = value * 10 + Number(char);
  }
  return value;
}

/**
 * Parse `<stem>.gguf` or llama.cpp's split name `<stem>-NNNNN-of-NNNNN.gguf`
 * (the shape `resolve_gguf_shards` rebuilds; the stem may be empty, as there).
 * Undefined for a non-GGUF path.
 */
export function parseGgufFileName(path: string): GgufFileName | undefined {
  const base = withoutGgufExtension(path);
  if (base === undefined) return undefined;
  const single: GgufFileName = { stem: base, ordinal: 1, shards: 1 };
  // Read right to left: `-` ordinal `-of-` count.
  const countStart = base.length - SPLIT_DIGITS;
  const ofStart = countStart - SPLIT_OF.length;
  const ordinalStart = ofStart - SPLIT_DIGITS;
  const dash = ordinalStart - 1;
  if (dash < 0 || base[dash] !== '-' || base.slice(ofStart, countStart).toLowerCase() !== SPLIT_OF) return single;
  const ordinal = splitNumber(base, ordinalStart);
  const shards = splitNumber(base, countStart);
  if (ordinal === undefined || shards === undefined) return single;
  return { stem: base.slice(0, dash), ordinal, shards };
}

const MAX_SPLIT_SHARDS = 1024;

function splitDigits(value: number): string {
  return String(value).padStart(SPLIT_DIGITS, '0');
}

/**
 * The files the native loader opens for the GGUF file `name` (`resolve_gguf_shards`), found among the names in its
 * own folder `siblings` (the folder of the path as given, links not resolved): `name` alone when it is not split, or
 * for `PREFIX-00001-of-NNNNN.gguf` (`-of-` in small letters) exactly one `PREFIX-0000K-of-NNNNN` with a `.gguf` of
 * any case for every K from 2 to N, in part order. N comes from the name; the loader takes it from the header's
 * `split.count` and refuses a first part whose name says another count. Undefined when the loader cannot open it.
 */
export function ggufSplitPartNames(name: string, siblings: readonly string[]): string[] | undefined {
  const base = withoutGgufExtension(name);
  const parsed = parseGgufFileName(name);
  if (base === undefined || parsed === undefined) return undefined;
  if (parsed.stem.length === base.length) return [name];
  const { stem: prefix, ordinal, shards } = parsed;
  const suffix = `${SPLIT_OF}${splitDigits(shards)}`;
  if (ordinal !== 1 || shards < 1 || shards > MAX_SPLIT_SHARDS || !base.endsWith(suffix)) return undefined;
  const parts = [name];
  for (let part = 2; part <= shards; part++) {
    const want = `${prefix}-${splitDigits(part)}${suffix}`;
    const found = siblings.filter((sibling) => withoutGgufExtension(sibling) === want);
    if (found.length !== 1) return undefined;
    parts.push(found[0]!);
  }
  return parts;
}

/**
 * `mtp` is on the list because the catalog ships MTP weights BESIDE a target
 * (Qwen3.8's `MTP/mtp-*.gguf`, Gemma's `mtp-*.gguf`) and nothing in the runtime
 * pairs a standalone GGUF MTP file: counting one as weights certifies a
 * directory the loader cannot open, and discovery would list the sidecar as a model.
 */
const GGUF_COMPANION_TOKENS: ReadonlySet<string> = new Set(['imatrix', 'mmproj', 'dflash', 'draft', 'mtp']);

/**
 * True when a `.gguf` filename is a companion artifact (projector, calibration,
 * draft, MTP) rather than a loadable model payload: one of its `-`/`_`/`.`
 * separated tokens is a companion word, in any case. Discovery, the dashboard's
 * publish gate, and the CLI's weight classification all key off this one rule —
 * a file any of them certifies as weights is a file the others must accept.
 */
export function isGgufCompanionName(fileName: string): boolean {
  let start = 0;
  for (let index = 0; index <= fileName.length; index++) {
    if (index < fileName.length && !isNameSeparator(fileName[index])) continue;
    if (GGUF_COMPANION_TOKENS.has(fileName.slice(start, index).toLowerCase())) return true;
    start = index + 1;
  }
  return false;
}

/** `IQ*` mixes of the Unsloth Dynamic targets, upper case. */
const QWEN35_IQ_TAGS: readonly string[] = [
  'IQ1_S',
  'IQ1_M',
  'IQ2_XXS',
  'IQ2_XS',
  'IQ2_S',
  'IQ2_M',
  'IQ3_XXS',
  'IQ3_S',
  'IQ3_M',
  'IQ4_XS',
  'IQ4_NL',
];
/** `Q<digits><suffix>` targets, upper case: `Q4_K_XL`, `Q2_K_M`. */
const QWEN35_K_SUFFIXES: readonly string[] = ['_K_XL', '_K_M'];

/** True when a tag may start at `start`: the name start, or a separator before it. */
function tagStartsAt(base: string, start: number): boolean {
  return start === 0 || isNameSeparator(base[start - 1]);
}

/** Length of the `Q<digits><suffix>` tag ending `upper`, or 0. */
function kQuantTagLength(upper: string): number {
  for (const suffix of QWEN35_K_SUFFIXES) {
    if (!upper.endsWith(suffix)) continue;
    let start = upper.length - suffix.length;
    while (isDigit(upper[start - 1])) start--;
    if (start === upper.length - suffix.length || upper[start - 1] !== 'Q') continue;
    return upper.length - (start - 1);
  }
  return 0;
}

/**
 * The Qwen3.5/Qwen3.8 discovery filter admits the Unsloth Dynamic target names
 * users download: `Q*_K_XL` / `Q*_K_M` (`UD-Q2_K_XL` included) and the `UD-IQ*`
 * mixes in {@link QWEN35_IQ_TAGS}. Every ggml tensor type those mixes draw on
 * is importable (Q2_K..Q6_K, IQ4_NL, IQ4_XS, IQ3_S, the IQ1 / IQ2 / IQ3_XXS grid
 * formats, Q8_0, F32); only the per-tensor selection differs. Other variants
 * (`Q4_0`, `Q*_K_S`), split names, and companion artifacts such as
 * imatrix/mmproj/draft stay excluded. Gemma4 and Muse accept all supported
 * tensor formats, including Q4_0 QAT checkpoints.
 */
export function isQwen35XlGgufName(name: string): boolean {
  const base = withoutGgufExtension(name);
  if (base === undefined || isGgufCompanionName(name)) return false;
  const upper = base.toUpperCase();
  const kLength = kQuantTagLength(upper);
  if (kLength > 0 && tagStartsAt(upper, upper.length - kLength)) return true;
  return QWEN35_IQ_TAGS.some((tag) => upper.endsWith(tag) && tagStartsAt(upper, upper.length - tag.length));
}

const BONSAI_PQ2_GGUF_NAME = 'ternary-bonsai-2-27b-pq2_0.gguf';

/** The one Bonsai PQ2_0 checkpoint Qwen3.5 discovery admits, matched in any case. */
export function isBonsaiPq2GgufName(name: string): boolean {
  return name.toLowerCase() === BONSAI_PQ2_GGUF_NAME;
}
