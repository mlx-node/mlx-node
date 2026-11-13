/**
 * The base model and the build (`variant`) of a discovered checkpoint, from its names and its parsed `config.json`.
 * Native-free and pure. When the rule is not sure, a model is its own base: two models are never joined by a guess.
 */

import { parseGgufFileName } from './gguf-name.js';

export type ModelFormat = 'gguf' | 'mlx';

/** The base and the build of one model. */
export interface ModelVariantName {
  /** The model without its packaging: `Qwen3.8-Flash-Next`. Equal strings mean one model. */
  base: string;
  /** The build of that model: `UD-Q4_K_XL`, `MXFP4 Unsloth`, `BF16`. Never empty. */
  variant: string;
}

/** The trailing packaging words of a name. */
export interface NameTail {
  /** The tokens left of the tail, joined by `-`; empty when the whole name is tail tokens. */
  base: string;
  /** The quant tag, normalized: `UD-Q4_K_XL`, `NVFP4`, `4-bit`. */
  tag: string | undefined;
  /** Recipe names after the tag, in name order: `Unsloth`. */
  qualifiers: readonly string[];
  /** The tail holds the word `gguf`. */
  saysGguf: boolean;
}

const DROPPED_WORDS: ReadonlySet<string> = new Set(['GGUF', 'MLX', 'HF', 'MTP']);
const QUALIFIERS: ReadonlyMap<string, string> = new Map([['UNSLOTH', 'Unsloth']]);
const FLOAT_TAGS: ReadonlyMap<string, string> = new Map([
  ['BF16', 'BF16'],
  ['F16', 'F16'],
  ['FP16', 'F16'],
  ['F32', 'F32'],
  ['FP32', 'F32'],
  ['FP8', 'FP8'],
  ['FP8_E4M3', 'FP8'],
  ['MXFP4', 'MXFP4'],
  ['MXFP8', 'MXFP8'],
  ['NVFP4', 'NVFP4'],
  ['SYM8', 'SYM8'],
  ['MXFP4_MOE', 'MXFP4_MOE'],
]);
const DTYPE_TAGS: ReadonlySet<string> = new Set(['BF16', 'F16', 'F32']);
const TAG_FAMILIES: ReadonlyArray<readonly [string, readonly string[]]> = [
  ['Q', ['', '_0', '_1', '_K', '_K_S', '_K_M', '_K_L', '_K_XL', 'K']],
  ['IQ', ['_XXS', '_XS', '_S', '_M', '_NL']],
  ['PQ', ['_0']],
  ['TQ', ['_0']],
];

function digitsEnd(text: string, start: number): number {
  let index = start;
  while (index < text.length && text[index]! >= '0' && text[index]! <= '9') index++;
  return index;
}

/** The normalized quant tag a whole token spells, or undefined. */
export function quantTag(token: string): string | undefined {
  const upper = token.toUpperCase();
  const float = FLOAT_TAGS.get(upper);
  if (float !== undefined) return float;
  const bits = digitsEnd(upper, 0);
  if (bits > 0 && upper.slice(bits) === 'BIT') return `${upper.slice(0, bits)}-bit`;
  for (const [prefix, suffixes] of TAG_FAMILIES) {
    if (!upper.startsWith(prefix)) continue;
    const end = digitsEnd(upper, prefix.length);
    if (end > prefix.length && suffixes.includes(upper.slice(end))) return upper;
  }
  return undefined;
}

/** Read `name` from the right while its `-` tokens are packaging words. */
export function splitNameTail(name: string): NameTail {
  const tokens = name.split('-');
  let tag: string | undefined;
  const qualifiers: string[] = [];
  let saysGguf = false;
  let lastWasTag = false;
  while (tokens.length > 0) {
    const upper = tokens[tokens.length - 1]!.toUpperCase();
    const found = tag === undefined ? quantTag(upper) : undefined;
    const qualifier = QUALIFIERS.get(upper);
    let tookTag = false;
    if (DROPPED_WORDS.has(upper)) {
      if (upper === 'GGUF') saysGguf = true;
    } else if (found !== undefined) {
      tag = found;
      tookTag = true;
    } else if (upper === 'UD' && lastWasTag) {
      tag = `UD-${tag!}`;
    } else if (qualifier !== undefined && !qualifiers.includes(qualifier)) {
      qualifiers.unshift(qualifier);
    } else break;
    lastWasTag = tookTag;
    tokens.pop();
  }
  return { base: tokens.join('-'), tag, qualifiers, saysGguf };
}

function joinLabel(tag: string | undefined, qualifiers: readonly string[]): string {
  return [...(tag === undefined ? [] : [tag]), ...qualifiers].join(' ');
}

/** `prefix` is the start of `name` and ends at a `-` of it, without case. */
function startsAtToken(name: string, prefix: string): boolean {
  if (prefix.length === 0 || name.length <= prefix.length) return false;
  return name.slice(0, prefix.length).toLowerCase() === prefix.toLowerCase() && name[prefix.length] === '-';
}

/** `A<digits>B` or `A<digits>.<digits>B` in any case: the active parameters of a mixture of experts. */
function isActiveParameterWord(word: string): boolean {
  const upper = word.toUpperCase();
  if (upper.length < 3 || upper[0] !== 'A' || upper[upper.length - 1] !== 'B') return false;
  const body = upper.slice(1, -1);
  const whole = digitsEnd(body, 0);
  if (whole === 0) return false;
  if (whole === body.length) return true;
  return body[whole] === '.' && whole + 1 < body.length && digitsEnd(body, whole + 1) === body.length;
}

/** Every word `name` has after its start `prefix` (and the `-` there) is an active-parameter word. */
function addsActiveParametersOnly(name: string, prefix: string): boolean {
  return name
    .slice(prefix.length + 1)
    .split('-')
    .every(isActiveParameterWord);
}

/**
 * The model a file named only by its quant tag belongs to, from its folders: the variant folder's base when that
 * folder names a model (after the repo folder's base unless it starts with it, so two repos never share it), else the
 * repo folder's base. Empty when no folder names a model.
 */
function tagOnlyBase(repoFolder: string, variantFolder: string | undefined): string {
  const repo = splitNameTail(repoFolder).base;
  const own = variantFolder === undefined ? '' : splitNameTail(variantFolder).base;
  if (own === '') return repo;
  if (repo === '' || own.toLowerCase() === repo.toLowerCase() || startsAtToken(own, repo)) return own;
  return `${repo}-${own}`;
}

/**
 * The id of a split GGUF set whose split suffix is already cut (`stem`). A stem that is only a quant tag gets the
 * folder base that names its model in front (the repo folder's name when none does); a loose file in the models root
 * (`repoFolder` undefined) keeps its stem.
 */
export function ggufModelId(stem: string, repoFolder: string | undefined, variantFolder: string | undefined): string {
  if (repoFolder === undefined || splitNameTail(stem).base !== '') return stem;
  const named = tagOnlyBase(repoFolder, variantFolder);
  return `${named === '' ? repoFolder : named}-${stem}`;
}

/** The names discovery reads from a GGUF file name. */
export interface GgufNames {
  /** The id the file asks for: the file name without `.gguf`, or {@link ggufModelId} for a split set. */
  id: string;
  /** The name carries `-NNNNN-of-NNNNN`: it is the first part of a split set. */
  split: boolean;
  /** The name without `.gguf` and the split suffix (the whole name without `.gguf` when that leaves nothing). */
  stem: string;
}

/**
 * The names of the GGUF file `fileName`, in its repo folder (undefined for a loose file in the models root) and its
 * variant folder (undefined when the file sits in the repo folder itself).
 */
export function ggufNames(
  fileName: string,
  repoFolder: string | undefined,
  variantFolder: string | undefined,
): GgufNames {
  const whole = fileName.slice(0, -'.gguf'.length);
  const stem = parseGgufFileName(fileName)?.stem ?? whole;
  if (stem.length === whole.length) return { id: whole, split: false, stem: whole };
  if (stem === '') return { id: whole, split: true, stem: whole };
  return { id: ggufModelId(stem, repoFolder, variantFolder), split: true, stem };
}

/** The folder base `folder` is the model of a file whose base is `file`: the same name, or the start of it. */
function folderAgrees(folder: string, file: string): boolean {
  return folder.toLowerCase() === file.toLowerCase() || startsAtToken(file, folder);
}

/**
 * The base and build of a GGUF from its stem, its repo folder (undefined for a loose file in the models root), its
 * variant folder (undefined when the file sits in the repo folder itself) and its id. A file is its own base (`id`)
 * when no folder names its model, when its folder adds a word to its name that is not an active-parameter word, or
 * when its folders name a model and none of them is its name or the start of it.
 */
export function ggufVariant(
  stem: string,
  repoFolder: string | undefined,
  variantFolder: string | undefined,
  id: string,
): ModelVariantName {
  const own = splitNameTail(stem);
  const repo = repoFolder === undefined ? '' : splitNameTail(repoFolder).base;
  const sub = variantFolder === undefined ? '' : splitNameTail(variantFolder).base;
  const named = repoFolder === undefined ? '' : tagOnlyBase(repoFolder, variantFolder);
  const folders = [repo, sub].filter((folder) => folder !== '');
  let base: string;
  if (own.base === '') base = named !== '' ? named : id;
  else if (repo.length > own.base.length && startsAtToken(repo, own.base)) {
    base = addsActiveParametersOnly(repo, own.base) ? repo : id;
  } else if (folders.length > 0 && !folders.some((folder) => folderAgrees(folder, own.base))) base = id;
  else base = own.base;
  let variant = joinLabel(own.tag, own.qualifiers);
  if (variant === '' && variantFolder !== undefined) {
    const folder = splitNameTail(variantFolder);
    if (folder.base === '') variant = joinLabel(folder.tag, folder.qualifiers);
  }
  return { base, variant: variant === '' ? 'GGUF' : variant };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function modeWord(mode: string): string {
  const upper = mode.toUpperCase();
  return FLOAT_TAGS.get(upper) ?? upper;
}

interface QuantClass {
  bits: number;
  mode: string;
  top: boolean;
}

function classOf(block: Record<string, unknown>, top: boolean): QuantClass | undefined {
  if (typeof block.bits !== 'number' || !Number.isFinite(block.bits)) return undefined;
  const mode = typeof block.mode === 'string' && block.mode !== '' ? block.mode : 'affine';
  return { bits: block.bits, mode, top };
}

/**
 * The build label a `config.json` quantization block states. The top level covers every layer the per-layer entries do
 * not list. A class that is not affine wins, the one with the fewest bits (the top level's on a tie, then by name);
 * affine only gives one bit count or the range. Undefined when the config states no quantization.
 */
export function configQuantLabel(config: unknown): string | undefined {
  if (!isRecord(config)) return undefined;
  const own = config.quantization;
  const top = isRecord(own) ? classOf(own, true) : undefined;
  if (isRecord(own) && top !== undefined) {
    const classes = [top];
    for (const value of Object.values(own)) {
      if (!isRecord(value)) continue;
      const entry = classOf(value, false);
      if (entry !== undefined) classes.push(entry);
    }
    const exact = classes.filter((c) => c.mode !== 'affine');
    if (exact.length > 0) {
      exact.sort(
        (a, b) => a.bits - b.bits || Number(b.top) - Number(a.top) || (a.mode < b.mode ? -1 : a.mode > b.mode ? 1 : 0),
      );
      return modeWord(exact[0]!.mode);
    }
    let low = top.bits;
    let high = top.bits;
    for (const c of classes) {
      low = Math.min(low, c.bits);
      high = Math.max(high, c.bits);
    }
    return low === high ? `${low}-bit` : `${low}–${high}-bit`;
  }
  const foreign = config.quantization_config;
  if (!isRecord(foreign)) return undefined;
  if (foreign.quant_method === 'gemma' && typeof foreign.num_bits === 'number') return `${foreign.num_bits}-bit`;
  if (foreign.quant_method === 'compressed-tensors' && foreign.format === 'float-quantized') return 'FP8';
  return undefined;
}

/** `BF16` / `F16` / `F32` from the config's dtype, or undefined. */
export function configDtypeLabel(config: unknown): string | undefined {
  if (!isRecord(config)) return undefined;
  const text = isRecord(config.text_config) ? config.text_config : {};
  const dtype = config.torch_dtype ?? config.dtype ?? text.torch_dtype ?? text.dtype;
  if (dtype === 'bfloat16') return 'BF16';
  if (dtype === 'float16') return 'F16';
  if (dtype === 'float32') return 'F32';
  return undefined;
}

/** The base and build of an MLX model folder `folder` with its parsed `config.json` (undefined when unreadable). */
export function folderVariant(folder: string, config: unknown): ModelVariantName {
  const tail = splitNameTail(folder);
  const base = tail.base === '' || tail.saysGguf ? folder : tail.base;
  const fromConfig = configQuantLabel(config);
  const named =
    tail.tag !== undefined && !(DTYPE_TAGS.has(tail.tag) && fromConfig !== undefined) ? tail.tag : undefined;
  const tag = named ?? fromConfig ?? tail.tag ?? configDtypeLabel(config) ?? 'MLX';
  return { base, variant: joinLabel(tag, tail.qualifiers) };
}

/** A discovered model as {@link settleVariants} reads and writes it. */
export interface VariantMember {
  readonly name: string;
  readonly format: ModelFormat;
  readonly modelType: string;
  base: string;
  variant: string;
}

function groupsByBase(models: readonly VariantMember[]): Map<string, VariantMember[]> {
  const groups = new Map<string, VariantMember[]>();
  for (const model of models) {
    const key = model.base.toLowerCase();
    const members = groups.get(key);
    if (members === undefined) groups.set(key, [model]);
    else members.push(model);
  }
  return groups;
}

function mixesTypes(members: readonly VariantMember[]): boolean {
  return new Set(members.map((m) => m.modelType)).size > 1;
}

/**
 * The pass over a whole scan, grouped by `base` without case. A group with two model types is not a group: each member
 * becomes its own base, and the groups are formed again until none mixes types (a new base can meet another group).
 * Every member of a group takes the spelling that sorts first by code unit. Equal variants inside one base: the member
 * whose id is the base keeps the label when exactly one is; every other one gets ` (<id>)`.
 */
export function settleVariants(models: readonly VariantMember[]): void {
  let moved = true;
  while (moved) {
    moved = false;
    for (const members of groupsByBase(models).values()) {
      if (!mixesTypes(members)) continue;
      for (const m of members) {
        if (m.base !== m.name) moved = true;
        m.base = m.name;
        m.variant = m.format === 'gguf' ? 'GGUF' : 'MLX';
      }
    }
  }
  for (const [key, members] of groupsByBase(models)) {
    if (mixesTypes(members)) continue;
    let spelling = members[0]!.base;
    for (const m of members) if (m.base < spelling) spelling = m.base;
    const byLabel = new Map<string, VariantMember[]>();
    for (const m of members) {
      m.base = spelling;
      const same = byLabel.get(m.variant);
      if (same === undefined) byLabel.set(m.variant, [m]);
      else same.push(m);
    }
    for (const [label, tied] of byLabel) {
      if (tied.length < 2) continue;
      const plain = tied.filter((m) => m.name.toLowerCase() === key);
      for (const m of tied) if (plain.length !== 1 || m !== plain[0]) m.variant = `${label} (${m.name})`;
    }
  }
}
