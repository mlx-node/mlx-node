/**
 * J-lens SHIPPED-pack export (Task T4.0).
 *
 * Re-exports the already-final F32 master pack `lens-pack-v1.safetensors` as the
 * f16 (IEEE-754 half) SHIPPED pack `lens-pack-v1.f16.safetensors` (~46 MiB) plus
 * a D6 meta sidecar `lens-pack-v1.meta.json`. This is a RE-EXPORT, not a re-fit:
 * the 23 Jacobians (J.1..J.23) already exist and are frozen. Plan D6 says the
 * shipped artifact is f16 while the master checkpoint / F32 pack stay fp32; the
 * T3.1 fit shipped only the F32 pack, and this task closes that gap.
 *
 * INVARIANTS (do not violate):
 *   - The F32 pack (`lens-pack-v1.safetensors`) and the fp32 checkpoint
 *     (`v1.ckpt.safetensors`) are NEVER touched. This script only READS the F32
 *     pack. The f16 pack is a SEPARATE, additional artifact with a versioned name.
 *   - No native / model / GPU dependency: this file imports ONLY node builtins,
 *     so the encoder can be unit-tested (and this export run) without the serial
 *     GPU. The `verify-f16-parity.mts` harness (authored separately) is the piece
 *     that loads the model.
 *   - Deterministic output: NO wall-clock byte enters the pack or sidecar
 *     (`fit_date` is copied verbatim from the fit log), so two runs are
 *     byte-identical (proven by the self-verify determinism pass below).
 *
 * The `f32ToF16` / `f16ToF32` helpers are EXPORTED for the unit test; the actual
 * export only runs when this file is invoked as the entry script (see `isMain`).
 *
 * Run with: env PATH="/opt/homebrew/bin:$PATH" oxnode \
 *   packages/browser/scripts/jlens/export-shipped-pack.mts
 *   (NOT tsx/ts-node — repo convention.)
 */
import { createHash } from 'node:crypto';
import {
  existsSync,
  openSync,
  readFileSync,
  readSync,
  closeSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const CACHE_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';
const F32_PACK = join(CACHE_DIR, 'lens-pack-v1.safetensors'); // source (F32, read-only)
const F16_PACK = join(CACHE_DIR, 'lens-pack-v1.f16.safetensors'); // versioned shipped pack
const META_SIDECAR = join(CACHE_DIR, 'lens-pack-v1.meta.json'); // D6 sidecar
const FIT_LOG = join(CACHE_DIR, 'fit-v1-log.json'); // n_prompts + fit_date + provenance
const CORPUS_META = join(CACHE_DIR, 'corpus-128-v1-meta.json'); // corpus_sha

const HIDDEN = 1024;
const LO = 1; // exported boundaries J.1..J.23 (J.0 unfitted, J.24 = identity, never stored)
const HI = 23;
const F32_SPAN = HIDDEN * HIDDEN * 4; // 4,194,304 B per F32 [1024,1024]
const F16_SPAN = HIDDEN * HIDDEN * 2; // 2,097,152 B per F16 [1024,1024]

// ===========================================================================
// F32 <-> IEEE-754 half (F16), round-to-nearest-even. Bit-exact, no libraries.
// ===========================================================================
// A single shared scratch view to read the raw 32-bit pattern of a JS number.
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

/**
 * Encode a JS number (an f32 value) to its IEEE-754 binary16 (half) bit pattern
 * (a uint16). Handles: sign, exponent rebias (127 -> 15), round-to-nearest-even,
 * subnormals, overflow -> +/-Inf, and NaN preservation (a NaN never collapses to
 * Inf — the result keeps exponent all-ones AND a nonzero mantissa).
 */
export function f32ToF16(value: number): number {
  _f32[0] = value;
  const x = _u32[0];
  const sign = (x >>> 16) & 0x8000; // top half's sign bit, in the f16 position
  const exp = (x >>> 23) & 0xff; // 8-bit biased f32 exponent
  const mant = x & 0x7fffff; // 23-bit f32 mantissa

  // --- Inf / NaN (f32 exponent all ones) ---
  if (exp === 0xff) {
    if (mant !== 0) {
      // NaN: preserve NaN-ness. Take the top mantissa bits; force nonzero so it
      // stays a NaN (a zero mantissa with exp all-ones would be an Inf).
      let m16 = mant >>> 13;
      if (m16 === 0) m16 = 1;
      return sign | 0x7c00 | m16;
    }
    return sign | 0x7c00; // +/-Inf
  }

  // Rebias the exponent into the half's range.
  const e = exp - 127 + 15; // unbiased f32 exp + 15

  if (e >= 0x1f) {
    // Overflow (including anything that would round up to Inf) -> +/-Inf.
    return sign | 0x7c00;
  }

  if (e <= 0) {
    // Subnormal half, or an underflow to signed zero.
    if (e < -10) {
      // Magnitude < 2^-25 -> below half of the min subnormal -> signed zero.
      return sign;
    }
    // Restore the implicit leading 1 to form the full 24-bit significand, then
    // shift into the subnormal grid, rounding to nearest even.
    const full = mant | 0x800000; // 24-bit significand (1.mant)
    const shift = 14 - e; // in [14, 24] for e in [0, -10]
    let half = full >>> shift;
    const rem = full & ((1 << shift) - 1); // discarded low bits
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1) === 1)) half += 1;
    // `half` may carry from 0x3ff into the exp=1 slot (0x400) — that is the
    // correct promotion from the largest subnormal to the smallest normal.
    return sign | half;
  }

  // Normal half. Round the 23-bit mantissa down to 10 bits, nearest-even.
  const m16 = mant >>> 13; // top 10 bits
  const rem = mant & 0x1fff; // discarded low 13 bits
  const halfway = 0x1000; // 1 << 12
  let out = (e << 10) | m16;
  if (rem > halfway || (rem === halfway && (m16 & 1) === 1)) {
    // Round up. A mantissa carry (0x3ff -> 0x400) rolls into the exponent
    // correctly; if that pushes e to 0x1f the mantissa is 0 -> Inf, as intended.
    out += 1;
  }
  return sign | out;
}

/** Decode an IEEE-754 binary16 (half) bit pattern (uint16) back to an f32 value. */
export function f16ToF32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x03ff;
  if (exp === 0) {
    // Zero or subnormal: value = sign * mant * 2^-24.
    return sign * mant * 2 ** -24;
  }
  if (exp === 0x1f) {
    return mant === 0 ? sign * Infinity : NaN;
  }
  // Normal: value = sign * (1 + mant/1024) * 2^(exp-15).
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

// ===========================================================================
// safetensors helpers (read-only for the F32 source; hand-written writer for f16)
// ===========================================================================
// The header maps tensor names to TensorEntry, plus an optional `__metadata__`
// string-map — a heterogeneous object, so entries are cast to TensorEntry at use.
type StHeader = Record<string, unknown>;
type TensorEntry = { dtype: string; shape: number[]; data_offsets: [number, number] };

/** Parse the safetensors header of `path` (returns the raw JSON object + the
 *  absolute byte offset at which the tensor data buffer begins). */
function readHeader(path: string): { header: StHeader; dataStart: number } {
  const fd = openSync(path, 'r');
  try {
    const lenBuf = Buffer.alloc(8);
    if (readSync(fd, lenBuf, 0, 8, 0) < 8) throw new Error(`${path}: truncated (<8 bytes)`);
    const headerLen = Number(lenBuf.readBigUInt64LE(0));
    const hdrBuf = Buffer.alloc(headerLen);
    let got = 0;
    while (got < headerLen) {
      const n = readSync(fd, hdrBuf, got, headerLen - got, 8 + got);
      if (n <= 0) throw new Error(`${path}: header read short`);
      got += n;
    }
    return { header: JSON.parse(hdrBuf.toString('utf8')) as StHeader, dataStart: 8 + headerLen };
  } finally {
    closeSync(fd);
  }
}

/** Read J.LO..J.HI from the F32 source pack into fresh, aligned Float32Arrays.
 *  Asserts each is F32 [HIDDEN,HIDDEN] with a span of exactly F32_SPAN. */
function readF32Pack(path: string): { key: string; data: Float32Array }[] {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString('utf8')) as StHeader;
  const dataStart = 8 + headerLen;
  const out: { key: string; data: Float32Array }[] = [];
  for (let l = LO; l <= HI; l++) {
    const key = `J.${l}`;
    const t = header[key] as TensorEntry | undefined;
    if (!t) throw new Error(`F32 pack missing ${key}`);
    if (t.dtype !== 'F32') throw new Error(`${key} dtype ${t.dtype} != F32`);
    const sh = t.shape;
    if (sh.length !== 2 || sh[0] !== HIDDEN || sh[1] !== HIDDEN) {
      throw new Error(`${key} shape [${sh.join(',')}] != [${HIDDEN},${HIDDEN}]`);
    }
    const [lo, hi] = t.data_offsets;
    if (hi - lo !== F32_SPAN) throw new Error(`${key} span ${hi - lo} != ${F32_SPAN}`);
    // Copy to a fresh 0-offset ArrayBuffer so Float32Array alignment is guaranteed.
    const ab = buf.buffer.slice(buf.byteOffset + dataStart + lo, buf.byteOffset + dataStart + hi);
    out.push({ key, data: new Float32Array(ab) });
  }
  return out;
}

/** Downcast one F32 [HIDDEN*HIDDEN] array to a little-endian F16 byte Buffer. */
function encodeTensorF16(src: Float32Array): Buffer {
  const buf = Buffer.allocUnsafe(src.length * 2);
  for (let i = 0; i < src.length; i++) buf.writeUInt16LE(f32ToF16(src[i]), i * 2);
  return buf;
}

/** Read the exact D6 sidecar facts from the fit log + corpus meta. The log
 *  carries NO explicit `target` field, so `target` is derived from the frozen D1
 *  design decision (final residual, pre-final-norm) — but ONLY after asserting
 *  this log is the T3.1 production fit that D1 applies to. Everything else is a
 *  verbatim field read (real field names, not invented). */
function readMetaFacts(): {
  n_prompts: number;
  corpus_sha: string;
  target: string;
  layers: number[];
  fit_date: string;
} {
  const log = JSON.parse(readFileSync(FIT_LOG, 'utf8'));
  const corpusMeta = JSON.parse(readFileSync(CORPUS_META, 'utf8'));

  // n_prompts: the number of prompts actually fitted. Four fields carry it; they
  // must all agree or the log is inconsistent and we refuse to stamp a sidecar.
  const nCandidates = {
    'resume.final_n_done': log?.resume?.final_n_done,
    'settings.n_fit': log?.settings?.n_fit,
    'native_result.nDone': log?.native_result?.nDone,
    'pack.checkpoint_n_done': log?.pack?.checkpoint_n_done,
  };
  const nVals = Object.values(nCandidates).filter((v) => typeof v === 'number') as number[];
  if (nVals.length === 0)
    throw new Error(`fit log has no numeric n_prompts field (looked at ${Object.keys(nCandidates).join(', ')})`);
  const n_prompts = nVals[0];
  for (const [k, v] of Object.entries(nCandidates)) {
    if (typeof v === 'number' && v !== n_prompts) {
      throw new Error(`fit log n_prompts disagreement: ${k}=${v} != ${n_prompts} (log inconsistent)`);
    }
  }

  // corpus_sha: verbatim from the corpus-meta file (real field name = corpus_sha256).
  const corpus_sha = corpusMeta?.corpus_sha256;
  if (typeof corpus_sha !== 'string' || corpus_sha.length !== 64) {
    throw new Error(`corpus meta corpus_sha256 missing/not a 64-hex string: ${JSON.stringify(corpus_sha)}`);
  }

  // fit_date: verbatim from the log (real field name = generated_at).
  const fit_date = log?.generated_at;
  if (typeof fit_date !== 'string')
    throw new Error(`fit log generated_at missing/not a string: ${JSON.stringify(fit_date)}`);

  // target: the log carries no `target` field. Derive the D1-frozen target, gated
  // on this being the T3.1 production fit over the v1 corpus (so it is provenance-
  // checked, not hardcoded blindly). If a future log ever adds an explicit target
  // field, prefer it.
  const explicitTarget = log?.target ?? log?.settings?.target ?? log?.fit_target;
  let target: string;
  if (typeof explicitTarget === 'string') {
    target = explicitTarget;
  } else {
    if (log?.task !== 'T3.1') {
      throw new Error(
        `fit log has no target field and task=${JSON.stringify(log?.task)} != 'T3.1' — refusing to assume the D1 target`,
      );
    }
    // Cross-check the corpus the log was fit on matches the corpus-meta sha, so
    // the D1 target we stamp genuinely belongs to THIS corpus/fit.
    const logSha = log?.corpus_meta?.corpus_sha256;
    if (typeof logSha === 'string' && logSha !== corpus_sha) {
      throw new Error(`fit log corpus sha ${logSha} != corpus-meta sha ${corpus_sha} — provenance mismatch`);
    }
    target = 'final_residual_pre_final_norm'; // D1: fit target = FINAL residual, pre-final-norm
  }

  const layers = Array.from({ length: HI - LO + 1 }, (_, i) => LO + i); // [1..23]
  return { n_prompts, corpus_sha, target, layers, fit_date };
}

/** Serialize the f16 pack (header + 23 contiguous F16 bodies) to a Buffer. The
 *  header carries a `__metadata__` block (string map, per the safetensors spec)
 *  with the same D6 facts as the sidecar. Byte-deterministic: fixed key order,
 *  space-padded header to an 8-byte boundary, no wall-clock. */
function buildF16Pack(tensors: { key: string; data: Float32Array }[], facts: ReturnType<typeof readMetaFacts>): Buffer {
  const bodies: Buffer[] = [];
  const headerObj: Record<string, unknown> = {};
  // __metadata__ FIRST (safetensors requires all values to be strings).
  headerObj.__metadata__ = {
    n_prompts: String(facts.n_prompts),
    corpus_sha: facts.corpus_sha,
    target: facts.target,
    layers: `${facts.layers[0]}..${facts.layers[facts.layers.length - 1]}`,
    fit_date: facts.fit_date,
    dtype: 'F16',
    source_pack: 'lens-pack-v1.safetensors',
    task: 'T4.0',
  };
  let cursor = 0;
  for (const { key, data } of tensors) {
    const body = encodeTensorF16(data);
    if (body.length !== F16_SPAN) throw new Error(`${key} f16 body ${body.length} != ${F16_SPAN}`);
    headerObj[key] = { dtype: 'F16', shape: [HIDDEN, HIDDEN], data_offsets: [cursor, cursor + body.length] };
    bodies.push(body);
    cursor += body.length;
  }
  let headerJson = JSON.stringify(headerObj);
  // Pad the header with spaces so (8 + headerLen) is a multiple of 8 (official
  // safetensors convention; JSON.parse ignores the trailing spaces).
  const pad = (8 - ((8 + Buffer.byteLength(headerJson)) % 8)) % 8;
  headerJson += ' '.repeat(pad);
  const headerBuf = Buffer.from(headerJson, 'utf8');
  const lenBuf = Buffer.alloc(8);
  lenBuf.writeBigUInt64LE(BigInt(headerBuf.length));
  return Buffer.concat([lenBuf, headerBuf, ...bodies]);
}

/** Atomic write: `.tmp.<pid>` -> fsync-free rename (matches the repo convention). */
function writeAtomic(path: string, buf: Buffer): void {
  const tmp = `${path}.tmp.${process.pid}`;
  writeFileSync(tmp, buf);
  renameSync(tmp, path);
}

const MIB = 1024 * 1024;

// ===========================================================================
// Export + CPU self-verification (R5). Returns the produced pack bytes.
// ===========================================================================
function main(): void {
  console.log('==================== T4.0 f16 SHIPPED-PACK EXPORT ====================');
  if (!existsSync(F32_PACK)) throw new Error(`F32 source pack not found: ${F32_PACK}`);

  const facts = readMetaFacts();
  console.log(
    `meta facts: n_prompts=${facts.n_prompts} corpus_sha=${facts.corpus_sha.slice(0, 12)}… ` +
      `target=${facts.target} layers=[${facts.layers[0]}..${facts.layers[facts.layers.length - 1]}] fit_date=${facts.fit_date}`,
  );

  const tensors = readF32Pack(F32_PACK);
  console.log(
    `read F32 source: ${tensors.length} tensors J.${LO}..J.${HI}, ${(statSync(F32_PACK).size / MIB).toFixed(2)} MiB`,
  );

  // --- measure f16 round-trip error while encoding (R5) ---
  // Half precision gives <= 2^-11 RELATIVE error ONLY for values inside its
  // NORMAL range (|x| >= 2^-14). Values below the subnormal floor (|x| < 2^-24)
  // correctly UNDERFLOW to 0 (a dynamic-range property, not a precision failure)
  // whose relative error is 1.0 but whose ABSOLUTE error is < 6e-8. So the
  // precision assertion is scoped to the normal range; underflow/subnormal counts
  // are reported separately, and the absolute error is bounded across the board.
  const MIN_NORMAL = 2 ** -14; // smallest half normal
  const MIN_SUBNORMAL = 2 ** -24; // smallest half subnormal
  const HALF_ULP_REL = 2 ** -11; // round-to-nearest half rel-error bound (normals)
  let maxRelNormal = 0;
  let maxRelNormalAt = '';
  let maxAbsErr = 0;
  let maxAbsAt = '';
  let nUnderflow = 0; // 0 < |x| < 2^-24 -> flushed to 0
  let nSubnormal = 0; // 2^-24 <= |x| < 2^-14 -> subnormal half (degraded rel prec)
  let newNonFinite = 0;
  for (const { key, data } of tensors) {
    for (let i = 0; i < data.length; i++) {
      const orig = data[i];
      const rt = f16ToF32(f32ToF16(orig));
      if (Number.isFinite(orig) && !Number.isFinite(rt)) newNonFinite++;
      const abs = Math.abs(orig);
      const absErr = Math.abs(rt - orig);
      if (absErr > maxAbsErr) {
        maxAbsErr = absErr;
        maxAbsAt = `${key}[${i}] ${orig} -> ${rt}`;
      }
      if (abs >= MIN_NORMAL) {
        const rel = absErr / abs;
        if (rel > maxRelNormal) {
          maxRelNormal = rel;
          maxRelNormalAt = `${key}[${i}] ${orig} -> ${rt}`;
        }
      } else if (abs >= MIN_SUBNORMAL) {
        nSubnormal++;
      } else if (abs > 0) {
        nUnderflow++;
      }
    }
  }
  if (newNonFinite > 0)
    throw new Error(`${newNonFinite} finite F32 value(s) became NaN/Inf under f16 — refusing to ship`);
  // Normal-range round-trip must be at half precision (<= 2^-11 relative, tiny FP slack).
  if (maxRelNormal > HALF_ULP_REL * 1.0001) {
    throw new Error(
      `normal-range max f16 rel error ${maxRelNormal.toExponential(4)} > 2^-11 (${HALF_ULP_REL.toExponential(4)}) — not half precision`,
    );
  }

  const pack = buildF16Pack(tensors, facts);

  // --- determinism: rebuild from scratch, assert byte-identical (R5) ---
  const pack2 = buildF16Pack(readF32Pack(F32_PACK), readMetaFacts());
  if (!pack.equals(pack2)) throw new Error('non-deterministic export: two in-process builds differ');

  writeAtomic(F16_PACK, pack);

  // --- meta sidecar (D6 fields ONLY, natural JSON types) ---
  const sidecar = {
    n_prompts: facts.n_prompts,
    corpus_sha: facts.corpus_sha,
    target: facts.target,
    layers: facts.layers,
    fit_date: facts.fit_date,
  };
  writeAtomic(META_SIDECAR, Buffer.from(JSON.stringify(sidecar, null, 2) + '\n', 'utf8'));

  // --- verify the written file (R5): 23 keys, all F16 [HIDDEN,HIDDEN], no J.24,
  //     contiguous tiling, size within 46 MiB +/- 1 MiB ---
  const fileBytes = statSync(F16_PACK).size;
  const { header, dataStart } = readHeader(F16_PACK);
  const bodyBytes = fileBytes - dataStart;
  const keys = Object.keys(header)
    .filter((k) => k !== '__metadata__')
    .sort((a, b) => Number(a.slice(2)) - Number(b.slice(2)));
  const expected = Array.from({ length: HI - LO + 1 }, (_, i) => `J.${i + 1}`);
  if (keys.join(',') !== expected.join(',')) throw new Error(`written pack keys != J.1..J.23: [${keys.join(', ')}]`);
  if (header['J.24']) throw new Error('written pack contains J.24 (identity must never be stored)');
  const spans = keys.map((k) => (header[k] as TensorEntry).data_offsets).sort((a, b) => a[0] - b[0]);
  let cur = 0;
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    const t = header[k] as TensorEntry;
    if (t.dtype !== 'F16') throw new Error(`${k} dtype ${t.dtype} != F16`);
    if (t.shape.length !== 2 || t.shape[0] !== HIDDEN || t.shape[1] !== HIDDEN)
      throw new Error(`${k} shape != [${HIDDEN},${HIDDEN}]`);
    const [lo, hi] = spans[i];
    if (lo !== cur) throw new Error(`pack body not contiguous at ${cur}: next starts at ${lo}`);
    if (hi - lo !== F16_SPAN) throw new Error(`${k} span ${hi - lo} != ${F16_SPAN}`);
    cur = hi;
  }
  if (cur !== bodyBytes) throw new Error(`pack body ends at ${cur} but file body is ${bodyBytes}`);
  const sizeMiB = fileBytes / MIB;
  if (Math.abs(sizeMiB - 46) > 1) throw new Error(`pack size ${sizeMiB.toFixed(3)} MiB not within 46 +/- 1 MiB`);
  const metaBlock = header.__metadata__;
  if (metaBlock == null)
    throw new Error('written pack has no __metadata__ block (must not be null, unlike the F32 pack)');

  const sha = createHash('sha256').update(pack).digest('hex');
  console.log('');
  console.log(`[VERIFY] keys              : ${keys.length} = EXACTLY J.1..J.23  (J.24 absent: ${!header['J.24']})`);
  console.log(`[VERIFY] dtype/shape       : all F16 [${HIDDEN},${HIDDEN}], bodies tile contiguously`);
  console.log(
    `[VERIFY] size              : ${fileBytes} B = ${sizeMiB.toFixed(3)} MiB (expect ~46; ${(F16_SPAN / MIB).toFixed(2)} MiB x 23 + ${dataStart}B header)`,
  );
  console.log(`[VERIFY] __metadata__      : present (${JSON.stringify(metaBlock)})`);
  console.log(
    `[VERIFY] normal-range relErr: ${maxRelNormal.toExponential(4)} <= 2^-11 (${HALF_ULP_REL.toExponential(4)})  at ${maxRelNormalAt}`,
  );
  console.log(`[VERIFY] max abs error     : ${maxAbsErr.toExponential(4)}  at ${maxAbsAt}`);
  console.log(
    `[VERIFY] subnormal / underflow: ${nSubnormal} subnormal (|x|<2^-14), ${nUnderflow} flushed to 0 (|x|<2^-24); abs err of those < ${MIN_SUBNORMAL.toExponential(2)}`,
  );
  console.log(`[VERIFY] finite->NaN/Inf   : ${newNonFinite} (must be 0)`);
  console.log(`[VERIFY] determinism       : in-process rebuild byte-identical`);
  console.log(`[VERIFY] sha256(pack)      : ${sha}`);
  console.log('');
  console.log(`wrote ${F16_PACK}`);
  console.log(`wrote ${META_SIDECAR}`);
  console.log(`sidecar: ${JSON.stringify(sidecar)}`);
  console.log('\n==================== EXPORT OK ====================');
}

// Only run the export when invoked directly (so the unit test can import the pure
// helpers without triggering the export or any filesystem writes).
const isMain = process.argv[1] != null && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    main();
    process.exit(0);
  } catch (e) {
    console.error(`\nEXPORT FAILED: ${(e as Error).message ?? e}`);
    // Best-effort: never leave a stray tmp file behind.
    try {
      const tmp = `${F16_PACK}.tmp.${process.pid}`;
      if (existsSync(tmp)) unlinkSync(tmp);
    } catch {
      /* ignore */
    }
    process.exit(1);
  }
}
