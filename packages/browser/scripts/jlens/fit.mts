/**
 * J-lens PILOT fit driver (Task T1.3).
 *
 * Drives the T1.2 native offline-fit surface (`fitJacobianLens`,
 * `loadLensPackFromFile`, `lensReadout`) over the 128-token WikiText corpus built
 * by `prepare-corpus.mts`, producing the pilot Jacobian pack + the measured
 * seconds/prompt + per-prompt convergence diagnostics + an end-to-end
 * corpus->fit->checkpoint->resume->pack proof. oxnode scripting against the native
 * NAPI — NO native/Rust changes.
 *
 * ## Two roles (one file), dispatched by `__JLENS_FIT_WORKER`
 *
 *   ORCHESTRATOR (default): owns NO model. Reads the corpus, spawns short-lived
 *   WORKER subprocesses for each native operation, polls the fp32 master
 *   checkpoint on disk to extract PER-PROMPT convergence + timing (the native fit
 *   loop has no per-call prompt cap and only exports the pack at the end, so the
 *   running-mean intermediates are recovered by observing each atomic checkpoint
 *   write), folds the kill/resume proof into a single kill at n_done>=KILL_AT, and
 *   writes `pilot-fit-log.json`.
 *
 *   WORKER (`__JLENS_FIT_WORKER=1`): loads the model ONCE and executes ONE job
 *   (`sweep` | `fit` | `smoke`) read from a JSON job file (argv[2]), prints a
 *   result sentinel to stdout, and exits. A separate process per fit phase is what
 *   makes the kill/resume proof a genuine cross-process resume.
 *
 * ## Why poll the checkpoint instead of one-prompt-per-call
 *
 * The native checkpoint stores a corpus fingerprint (FNV hash + prompt_count +
 * maxSeqLen) that is revalidated on resume, so the SAME checkpoint can only ever
 * be resumed with the IDENTICAL fixed corpus — a growing/shrinking corpus is
 * rejected. And one `fitJacobianLens` call processes ALL prompts with
 * `idx >= next_idx` (no per-call cap). So per-prompt granularity is obtained by
 * running ONE cumulative fit over the fixed 10-corpus and reading the checkpoint's
 * `Jsum.*` after each prompt is committed (checkpoint is saved after every prompt).
 * The fit therefore runs each prompt EXACTLY ONCE (no double compute).
 *
 * Run:  env PATH="/opt/homebrew/bin:$PATH" oxnode \
 *         packages/browser/scripts/jlens/fit.mts
 * (arm64 Homebrew node ONLY; oxnode, never tsx. Requires prepare-corpus.mts first.)
 */
import { spawn, spawnSync } from 'node:child_process';
import {
  closeSync,
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  readSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const MODEL_PATH =
  '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const CACHE_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';
const CORPUS_PATH = join(CACHE_DIR, 'corpus-128.jsonl');
const CKPT_PATH = join(CACHE_DIR, 'pilot.ckpt.safetensors');
const PACK_PATH = join(CACHE_DIR, 'lens-pack-pilot.safetensors');
const LOG_PATH = join(CACHE_DIR, 'pilot-fit-log.json');
const SWEEP_DIR = join(CACHE_DIR, 'sweep');

// Pilot fit settings (design freeze).
const N_FIT = 10; // pilot = 10 prompts
const DIM_BATCH = 32; // KEEP 32 — fewer passes, same FLOPs; never exceed 32
const SKIP_FIRST = 16;
const MAX_SEQ_LEN = 128;
const USE_CHECKPOINTING = true; // bounds memory (~67GB pool otherwise)
const KILL_AT = 5; // kill Phase A once n_done reaches this -> cross-process resume proof

// Bv sweep (confirmatory, CHEAP — one short prompt).
const BVS = [8, 16, 32];
const SWEEP_TOKENS = 48; // truncate a corpus prompt so bv=8 is minutes, not ~20 min

// Model / pack geometry.
const HIDDEN = 1024;
const NUM_LAYERS = 24; // checkpoint stores Jsum.0..Jsum.23; pack exports J.1..J.23
const EXPORT_LO = 1; // pack boundaries 1..23 (J.0 embedding unfitted, J.24 identity)
const EXPORT_HI = 23;

// Smoke (caller contract: layers MUST be subset of 1..=24; the pack has no J.0).
const SMOKE_LAYER = 5;
const SMOKE_TOPK = 8;
// lensReadout caps the prompt at LENS_MAX_POSITIONS=48 positions (inspector.rs:
// LENS_MAX_POSITIONS — it bounds the transient [P, vocab] logits + layers×P host
// crossing). The corpus sequences are 128 tok, so the smoke MUST slice to <=48.
// 32 gives a real multi-token context while staying safely under the cap.
const SMOKE_PROMPT_TOKENS = 32;

const POLL_MS = 3000;
const SCRIPT_PATH = fileURLToPath(import.meta.url);

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------
function fail(msg: string): never {
  console.error(`\nFAIL: ${msg}`);
  process.exit(1);
}
function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
function loadCorpus(corpusPath: string = CORPUS_PATH): number[][] {
  if (!existsSync(corpusPath)) {
    fail(`corpus not found at ${corpusPath} — run prepare-corpus.mts first`);
  }
  const seqs = readFileSync(corpusPath, 'utf8')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l).ids as number[]);
  for (let i = 0; i < seqs.length; i++) {
    if (seqs[i].length !== MAX_SEQ_LEN) {
      fail(`corpus sequence ${i} length ${seqs[i].length} != ${MAX_SEQ_LEN}`);
    }
  }
  return seqs;
}

// ---------------------------------------------------------------------------
// safetensors reader (checkpoint introspection — orchestrator side, NO model)
// ---------------------------------------------------------------------------
type StHeader = Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;

/** Read only `meta.n_done` / `meta.next_idx` scalars via positioned fd reads
 *  (cheap — never materializes the ~96 MB `Jsum.*`). Returns null if unreadable. */
function readCheckpointScalars(path: string): { nDone: number; nextIdx: number } | null {
  let fd: number;
  try {
    fd = openSync(path, 'r');
  } catch {
    return null;
  }
  try {
    const lenBuf = Buffer.alloc(8);
    if (readSync(fd, lenBuf, 0, 8, 0) < 8) return null;
    const headerLen = Number(lenBuf.readBigUInt64LE(0));
    const hdrBuf = Buffer.alloc(headerLen);
    let got = 0;
    while (got < headerLen) {
      const n = readSync(fd, hdrBuf, got, headerLen - got, 8 + got);
      if (n <= 0) return null;
      got += n;
    }
    const header = JSON.parse(hdrBuf.toString('utf8')) as StHeader;
    const dataStart = 8 + headerLen;
    const readScalar = (key: string): number | null => {
      const t = header[key];
      if (!t) return null;
      const b = Buffer.alloc(4);
      if (readSync(fd, b, 0, 4, dataStart + t.data_offsets[0]) < 4) return null;
      return b.readFloatLE(0);
    };
    const nDone = readScalar('meta.n_done');
    const nextIdx = readScalar('meta.next_idx');
    if (nDone == null || nextIdx == null) return null;
    return { nDone: Math.round(nDone), nextIdx: Math.round(nextIdx) };
  } catch {
    return null;
  } finally {
    closeSync(fd);
  }
}

/** Read every `Jsum.{l}` (fp32 [1024,1024]) into a fresh, aligned Float32Array. */
function readJsumArrays(path: string): Float32Array[] {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString('utf8')) as StHeader;
  const dataStart = 8 + headerLen;
  const arrs: Float32Array[] = [];
  for (let l = 0; l < NUM_LAYERS; l++) {
    const t = header[`Jsum.${l}`];
    if (!t) throw new Error(`checkpoint missing Jsum.${l}`);
    if (t.dtype !== 'F32') throw new Error(`Jsum.${l} dtype ${t.dtype} != F32`);
    const s = dataStart + t.data_offsets[0];
    const e = dataStart + t.data_offsets[1];
    // Copy to a fresh 0-offset ArrayBuffer so Float32Array alignment is guaranteed.
    const ab = buf.buffer.slice(buf.byteOffset + s, buf.byteOffset + e);
    arrs.push(new Float32Array(ab));
  }
  return arrs;
}

/** Parse safetensors tensor-name keys (for the J.1..J.23 pack-key assertion). */
function safetensorsKeys(path: string): string[] {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString('utf8'));
  return Object.keys(header).filter((k) => k !== '__metadata__');
}

/** Read the full safetensors header map (name -> {dtype, shape, data_offsets}). */
function safetensorsHeader(path: string): StHeader {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  return JSON.parse(buf.subarray(8, 8 + headerLen).toString('utf8')) as StHeader;
}

// ---------------------------------------------------------------------------
// Pack export = the PROVEN native F32 path (T3.1). The native `export_pack`
// (crates/mlx-core .../qwen3_5/jlens_fit.rs) writes J.{l} = Jsum.{l}/n_done as
// F32 (~96 MB), the SAME serializer the pilot uses. We pass `packPath` to the
// fit worker so the native writes the pack on completion, and re-export from an
// already-complete checkpoint by invoking the native fit with 0 new prompts
// (export is gated on n_done>0, not prompts_this_run). f16 is DEFERRED to
// browser-shipping time (Phase 4): T3.2/T3.3 read this pack NATIVELY via
// lensReadout, so f16 adds no value here and a hand-written f16 pack would be an
// unnecessary correctness risk on the critical path of an ~11 h fit.
// ---------------------------------------------------------------------------
/** Assert a pack has EXACTLY keys J.1..J.23, all dtype F32, shape [HIDDEN,HIDDEN].
 *  Returns the sorted key list + file size in bytes. */
function assertPackF32(packPath: string): { keys: string[]; bytes: number } {
  if (!existsSync(packPath)) fail(`pack not found at ${packPath}`);
  const header = safetensorsHeader(packPath);
  const keys = Object.keys(header)
    .filter((k) => k !== '__metadata__')
    .sort((a, b) => Number(a.slice(2)) - Number(b.slice(2)));
  const expected = Array.from({ length: 23 }, (_, i) => `J.${i + 1}`);
  if (keys.join(',') !== expected.join(',')) fail(`pack keys != exactly J.1..J.23: got [${keys.join(', ')}]`);
  for (const k of keys) {
    if (header[k].dtype !== 'F32') fail(`pack ${k} dtype ${header[k].dtype} != F32`);
    const sh = header[k].shape;
    if (sh.length !== 2 || sh[0] !== HIDDEN || sh[1] !== HIDDEN) {
      fail(`pack ${k} shape [${sh.join(',')}] != [${HIDDEN},${HIDDEN}]`);
    }
  }
  return { keys, bytes: statSync(packPath).size };
}

// ---------------------------------------------------------------------------
// Convergence: running-mean pack stats from the live checkpoint
// ---------------------------------------------------------------------------
interface ConvState {
  prevJsum: Float32Array[] | null;
  prevN: number;
}
interface Snapshot {
  nDone: number;
  ckptMtimeMs: number;
  wallMs: number; // orchestrator observation time
  secPerPromptFit: number | null; // ckpt mtime delta from previous snapshot (pure fit time)
  maxJNormOverSqrtD: number;
  meanRelChange: number | null; // mean over layers 1..23 of ||pack_k-pack_{k-1}||_F/(||pack_k||_F+eps)
  maxRelChange: number | null;
}

/** On a newly committed prompt, read `Jsum.*`, form the running-mean pack
 *  (`J_l = Jsum_l / n`), and compute (a) the convergence proxy vs the previous
 *  pack and (b) `max_l ||J_l||_F / sqrt(hidden)`. Boundaries 1..23 (the exported
 *  set) only. Mutates `conv` (prevJsum/prevN) to this snapshot. */
function computeSnapshot(
  path: string,
  nDone: number,
  ckptMtimeMs: number,
  wallMs: number,
  prevWallMs: number | null,
  conv: ConvState,
): Snapshot {
  const cur = readJsumArrays(path);
  const sqrtD = Math.sqrt(HIDDEN);
  let maxNorm = 0;
  const rels: number[] = [];
  for (let l = EXPORT_LO; l <= EXPORT_HI; l++) {
    const c = cur[l];
    let curFrob2 = 0;
    let diffFrob2 = 0;
    if (conv.prevJsum) {
      const p = conv.prevJsum[l];
      const invN = 1 / nDone;
      const invP = 1 / conv.prevN;
      for (let i = 0; i < c.length; i++) {
        const pc = c[i] * invN;
        const d = pc - p[i] * invP;
        curFrob2 += pc * pc;
        diffFrob2 += d * d;
      }
    } else {
      const invN = 1 / nDone;
      for (let i = 0; i < c.length; i++) {
        const pc = c[i] * invN;
        curFrob2 += pc * pc;
      }
    }
    const curFrob = Math.sqrt(curFrob2);
    if (curFrob / sqrtD > maxNorm) maxNorm = curFrob / sqrtD;
    if (conv.prevJsum) rels.push(Math.sqrt(diffFrob2) / (curFrob + 1e-12));
  }
  conv.prevJsum = cur;
  conv.prevN = nDone;
  const meanRel = rels.length ? rels.reduce((a, b) => a + b, 0) / rels.length : null;
  const maxRel = rels.length ? Math.max(...rels) : null;
  return {
    nDone,
    ckptMtimeMs,
    wallMs,
    secPerPromptFit: null, // filled by caller from ckpt mtime deltas
    maxJNormOverSqrtD: maxNorm,
    meanRelChange: meanRel,
    maxRelChange: maxRel,
  };
}

// ---------------------------------------------------------------------------
// WORKER spawn + result capture
// ---------------------------------------------------------------------------
function writeJob(job: unknown): string {
  mkdirSync(CACHE_DIR, { recursive: true });
  const p = join(CACHE_DIR, `.job-${process.pid}-${Date.now()}.json`);
  writeFileSync(p, JSON.stringify(job));
  return p;
}

interface WorkerRun {
  code: number | null;
  signal: string | null;
  stdout: string;
  killed: boolean;
}

/** Spawn a WORKER for `job`. If `onPoll` is given, it is called every POLL_MS
 *  while the child runs (used to observe the checkpoint + decide when to kill).
 *  `onPoll` returns true to request a kill. Resolves when the child exits. */
function runWorker(
  job: unknown,
  onPoll?: () => boolean,
): Promise<WorkerRun> {
  const jobPath = writeJob(job);
  return new Promise<WorkerRun>((resolve) => {
    const child = spawn(process.execPath, [...process.execArgv, SCRIPT_PATH, jobPath], {
      env: { ...process.env, __JLENS_FIT_WORKER: '1' },
      stdio: ['ignore', 'pipe', 'inherit'],
    });
    let stdout = '';
    let killed = false;
    // 'exit' fires when the process ends, but the stdout pipe may not have
    // drained yet — reading `stdout` there can truncate/miss a valid
    // __FIT_RESULT__ / __SMOKE_RESULT__ / __SWEEP_RESULT__ sentinel and discard
    // an expensive fit. So capture the exit status here and only RESOLVE on
    // 'close' (fires after ALL stdio streams have flushed → stdout is complete).
    let exitCode: number | null = null;
    let exitSignal: string | null = null;
    let settled = false;
    child.stdout.on('data', (d: { toString(): string }) => {
      const s = d.toString();
      stdout += s;
      process.stdout.write(s); // mirror worker progress live
    });
    let timer: ReturnType<typeof setInterval> | null = null;
    if (onPoll) {
      timer = setInterval(() => {
        try {
          if (onPoll() && !killed) {
            killed = true;
            child.kill('SIGKILL');
          }
        } catch {
          /* poll errors are non-fatal */
        }
      }, POLL_MS);
    }
    // Single settle path: stop the poll timer + clean the job file + resolve
    // ONCE, whichever of 'close'/'error' arrives first.
    const finish = (code: number | null, signal: string | null): void => {
      if (settled) return;
      settled = true;
      if (timer) clearInterval(timer);
      try {
        rmSync(jobPath, { force: true });
      } catch {
        /* ignore */
      }
      resolve({ code, signal, stdout, killed });
    };
    child.on('exit', (code: number | null, signal: string | null) => {
      exitCode = code;
      exitSignal = signal;
    });
    child.on('close', () => finish(exitCode, exitSignal));
    // A spawn failure never emits 'exit'/'close' — resolve cleanly (with a null
    // status the callers already treat as failure) instead of hanging forever.
    child.on('error', () => finish(exitCode, exitSignal));
  });
}

function parseSentinel<T>(stdout: string, tag: string): T | null {
  const line = stdout.split('\n').find((l) => l.startsWith(tag));
  if (!line) return null;
  return JSON.parse(line.slice(tag.length).trim()) as T;
}

// ---------------------------------------------------------------------------
// A cumulative fit phase with checkpoint polling (convergence + optional kill)
// ---------------------------------------------------------------------------
interface FitJob {
  kind: 'fit';
  promptIds: number[][];
  dimBatch: number;
  skipFirst: number;
  maxSeqLen: number;
  useCheckpointing: boolean;
  resume: boolean;
  checkpointPath: string;
  packPath?: string;
}

/** Run a cumulative fit worker while polling the checkpoint. Records a snapshot
 *  for every newly committed prompt (nDone increment) into `snapshots`, sharing
 *  `conv` across phases so Phase B's first snapshot diffs against Phase A's last.
 *  If `killAt` is set, the worker is SIGKILLed once nDone >= killAt (crash/resume
 *  proof). Returns the worker run + the native JlensFitResult (null if killed). */
async function runFitPhase(
  job: FitJob,
  snapshots: Snapshot[],
  conv: ConvState,
  killAt: number | null,
): Promise<{ run: WorkerRun; result: any | null }> {
  let lastNDone = conv.prevN; // continue numbering + convergence chaining across phases
  // Reset per phase: there is no clean per-prompt fit time across the process gap
  // (kill + mismatch worker + Phase B model reload), so the first snapshot of each
  // phase reports secPerPromptFit=null; the native's Phase-B secPerPrompt is the
  // authoritative per-prompt price.
  let lastMtimeMs: number | null = null;
  let prevWallMs: number | null = null;

  const onPoll = (): boolean => {
    let mtimeMs: number;
    try {
      mtimeMs = statSync(job.checkpointPath).mtimeMs;
    } catch {
      return false; // checkpoint not written yet
    }
    if (lastMtimeMs != null && mtimeMs === lastMtimeMs) return killAt != null && lastNDone >= killAt;
    const scal = readCheckpointScalars(job.checkpointPath);
    if (!scal) return false;
    if (scal.nDone > lastNDone) {
      // One (or more) new prompt(s) committed since last observation.
      if (scal.nDone > lastNDone + 1) {
        // Two+ commits landed between polls (or in the final catch-up). The
        // checkpoint only stores the LATEST Jsum, so the intermediate prompts'
        // snapshots are unrecoverable — this row's secPerPromptFit/meanRelChange
        // is a MULTI-PROMPT delta, not per-prompt. Warn loudly (don't hard-fail
        // mid-fit: the fit itself is fine, only the diagnostic series is coarse).
        console.log(
          `    [poll] WARNING: n_done jumped ${lastNDone} -> ${scal.nDone}; ` +
            `snapshots for n_done ${lastNDone + 1}..${scal.nDone - 1} are UNRECOVERABLE ` +
            `(checkpoint keeps only the latest Jsum) — the n_done=${scal.nDone} row is a multi-prompt delta`,
        );
      }
      const wallMs = Date.now();
      const snap = computeSnapshot(job.checkpointPath, scal.nDone, mtimeMs, wallMs, prevWallMs, conv);
      // Pure per-prompt fit time = checkpoint mtime delta from the previous save.
      snap.secPerPromptFit =
        lastMtimeMs != null ? (mtimeMs - lastMtimeMs) / 1000 : null;
      snapshots.push(snap);
      console.log(
        `    [poll] n_done=${snap.nDone} ` +
          `dt_fit=${snap.secPerPromptFit != null ? snap.secPerPromptFit.toFixed(1) + 's' : 'n/a'} ` +
          `maxJ/sqrtD=${snap.maxJNormOverSqrtD.toFixed(4)} ` +
          `meanRelChg=${snap.meanRelChange != null ? snap.meanRelChange.toExponential(3) : 'n/a'} ` +
          `maxRelChg=${snap.maxRelChange != null ? snap.maxRelChange.toExponential(3) : 'n/a'}`,
      );
      lastNDone = scal.nDone;
      lastMtimeMs = mtimeMs;
      prevWallMs = wallMs;
    }
    return killAt != null && lastNDone >= killAt;
  };

  const run = await runWorker(job, onPoll);
  // Final catch-up: if the worker finished the last prompt(s) between polls.
  onPoll();
  const result = run.killed ? null : parseSentinel<any>(run.stdout, '__FIT_RESULT__');
  return { run, result };
}

// ---------------------------------------------------------------------------
// ORCHESTRATOR
// ---------------------------------------------------------------------------
async function orchestrator(): Promise<void> {
  const t0 = Date.now();
  const corpus = loadCorpus();
  if (corpus.length < N_FIT + 1) {
    fail(`corpus has ${corpus.length} sequences, need >= ${N_FIT + 1}`);
  }
  const fitCorpus = corpus.slice(0, N_FIT);
  mkdirSync(SWEEP_DIR, { recursive: true });

  // Clean any stale pilot checkpoint/pack so the run starts fresh + deterministic.
  for (const p of [CKPT_PATH, PACK_PATH]) rmSync(p, { force: true });

  console.log(`\n==================== T1.3 PILOT FIT ====================`);
  console.log(`corpus: ${corpus.length} seqs (128 tok); fitting first ${N_FIT}; dimBatch=${DIM_BATCH}\n`);

  // ---------- Phase 0: Bv sweep (confirmatory, cheap) ----------
  console.log(`[SWEEP] dimBatch ∈ {${BVS.join(', ')}} on ONE ${SWEEP_TOKENS}-tok prompt (throwaway ckpts)`);
  rmSync(SWEEP_DIR, { recursive: true, force: true });
  mkdirSync(SWEEP_DIR, { recursive: true });
  const sweepPrompt = fitCorpus[0].slice(0, SWEEP_TOKENS);
  const sweepRun = await runWorker({
    kind: 'sweep',
    promptIds: [sweepPrompt],
    bvs: BVS,
    dir: SWEEP_DIR,
    skipFirst: SKIP_FIRST,
    maxSeqLen: SWEEP_TOKENS,
    useCheckpointing: USE_CHECKPOINTING,
  });
  const sweep = parseSentinel<Array<{ bv: number; secPerPrompt: number; peakMemoryMb: number; nValidLast: number }>>(
    sweepRun.stdout,
    '__SWEEP_RESULT__',
  );
  if (!sweep || sweepRun.code !== 0) fail(`Bv sweep worker failed (code ${sweepRun.code})`);
  console.log(`[SWEEP] done: ${sweep.map((s) => `bv=${s.bv}:${s.secPerPrompt.toFixed(1)}s`).join('  ')}`);
  const fastest = sweep.reduce((a, b) => (b.secPerPrompt < a.secPerPrompt ? b : a));
  console.log(`[SWEEP] fastest per-prompt: bv=${fastest.bv} (@${SWEEP_TOKENS} tok)\n`);

  // ---------- Phase A: cumulative fit, killed at n_done>=KILL_AT ----------
  const snapshots: Snapshot[] = [];
  const conv: ConvState = { prevJsum: null, prevN: 0 };
  console.log(`[PHASE A] fit ${N_FIT}-corpus (resume=false); will SIGKILL at n_done>=${KILL_AT}`);
  const phaseA = await runFitPhase(
    {
      kind: 'fit',
      promptIds: fitCorpus,
      dimBatch: DIM_BATCH,
      skipFirst: SKIP_FIRST,
      maxSeqLen: MAX_SEQ_LEN,
      useCheckpointing: USE_CHECKPOINTING,
      resume: false,
      checkpointPath: CKPT_PATH,
      packPath: PACK_PATH,
    },
    snapshots,
    conv,
    KILL_AT,
  );
  const afterKill = readCheckpointScalars(CKPT_PATH);
  if (!afterKill) fail(`no checkpoint after Phase A kill`);
  console.log(
    `[PHASE A] worker exited (killed=${phaseA.run.killed}, signal=${phaseA.run.signal}); ` +
      `checkpoint n_done=${afterKill.nDone} next_idx=${afterKill.nextIdx}`,
  );
  const killedNDone = afterKill.nDone;
  // Resume proof is only genuine if Phase A REALLY crashed mid-fit with prompts
  // left for Phase B to finish. Without these, a raced SIGKILL (worker finishes
  // all 10 before the kill lands, or the poll misses the window) would let
  // Phase B do 0 prompts and the `promptsThisRun === N_FIT - killedNDone` check
  // degenerate to `0 !== 0` = false → a VACUOUS pass that proved nothing.
  if (!phaseA.run.killed) {
    fail(`Phase A worker was not killed (killed=false) — no crash to resume from; resume proof is vacuous`);
  }
  if (phaseA.run.signal !== 'SIGKILL') {
    fail(`Phase A exit signal=${phaseA.run.signal} != SIGKILL — the SIGKILL never landed; resume proof is vacuous`);
  }
  if (killedNDone < KILL_AT) {
    fail(`Phase A committed ${killedNDone} < KILL_AT=${KILL_AT} — kill raced ahead of the crash point; resume proof is vacuous`);
  }
  if (killedNDone >= N_FIT) {
    fail(
      `Phase A committed ${killedNDone} >= N_FIT=${N_FIT} — the worker finished ALL prompts before the kill landed, ` +
        `so Phase B has nothing to resume; resume proof is vacuous`,
    );
  }

  // ---------- Corpus-mismatch resume MUST be rejected (T1.2 guard) ----------
  console.log(`\n[MISMATCH] resume same checkpoint with ONE flipped id -> expect clean rejection`);
  const flipped = fitCorpus.map((ids) => ids.slice());
  flipped[0][64] = flipped[0][64] === 12345 ? 12346 : 12345; // flip one id -> different corpus hash
  const mismatchRun = await runWorker({
    kind: 'fit',
    promptIds: flipped,
    dimBatch: DIM_BATCH,
    skipFirst: SKIP_FIRST,
    maxSeqLen: MAX_SEQ_LEN,
    useCheckpointing: USE_CHECKPOINTING,
    resume: true,
    checkpointPath: CKPT_PATH,
    packPath: PACK_PATH,
  });
  const mismatchErr = parseSentinel<{ message: string }>(mismatchRun.stdout, '__WORKER_ERROR__');
  const mismatchRejected =
    mismatchRun.code !== 0 &&
    !!mismatchErr &&
    /different prompt corpus|resume=false|different corpus|maxSeqLen/i.test(mismatchErr.message);
  console.log(
    `[MISMATCH] worker code=${mismatchRun.code} rejected=${mismatchRejected} ` +
      `msg="${mismatchErr?.message?.slice(0, 120) ?? '(none)'}"`,
  );
  const afterMismatch = readCheckpointScalars(CKPT_PATH);
  if (!mismatchRejected) fail(`corpus-mismatch resume was NOT rejected — the T1.2 guard failed`);
  if (!afterMismatch || afterMismatch.nDone !== killedNDone) {
    fail(`mismatch resume mutated the checkpoint (n_done ${afterMismatch?.nDone} != ${killedNDone})`);
  }
  console.log(`[MISMATCH] rejected + checkpoint untouched (n_done still ${killedNDone})  -> PASS`);

  // ---------- Phase B: fresh process resumes SAME corpus -> continues to 10 ----------
  console.log(`\n[PHASE B] fresh worker resume=true (same corpus) -> skip ${killedNDone}, continue to ${N_FIT}`);
  const phaseB = await runFitPhase(
    {
      kind: 'fit',
      promptIds: fitCorpus,
      dimBatch: DIM_BATCH,
      skipFirst: SKIP_FIRST,
      maxSeqLen: MAX_SEQ_LEN,
      useCheckpointing: USE_CHECKPOINTING,
      resume: true,
      checkpointPath: CKPT_PATH,
      packPath: PACK_PATH,
    },
    snapshots,
    conv,
    null,
  );
  if (!phaseB.result) fail(`Phase B worker did not return a fit result (code ${phaseB.run.code})`);
  const resB = phaseB.result;
  console.log(
    `[PHASE B] nDone=${resB.nDone} promptsThisRun=${resB.promptsThisRun} skipped=${resB.promptsSkipped} ` +
      `secPerPrompt=${resB.secPerPrompt.toFixed(2)} maxJ/sqrtD=${resB.maxJNormOverSqrtD.toFixed(4)} ` +
      `peakMb=${resB.peakMemoryMb.toFixed(0)}`,
  );

  // Resume assertions.
  if (resB.nDone !== N_FIT) fail(`final n_done=${resB.nDone} != ${N_FIT}`);
  // Phase B must have actually resumed work — a fresh process that did 0 prompts
  // proves nothing about cross-process resume (guards the vacuous `0 === 0` case
  // together with the Phase-A `killedNDone < N_FIT` assertion above).
  if (resB.promptsThisRun <= 0) {
    fail(`Phase B promptsThisRun=${resB.promptsThisRun} <= 0 — the fresh process resumed NO work; resume proof is vacuous`);
  }
  if (resB.promptsThisRun !== N_FIT - killedNDone) {
    fail(`Phase B promptsThisRun=${resB.promptsThisRun} != ${N_FIT - killedNDone} (resume skip broken)`);
  }
  console.log(
    `[RESUME] Phase A committed ${killedNDone}, Phase B (fresh process) continued ${resB.promptsThisRun} ` +
      `-> ${resB.nDone}/${N_FIT}  -> PASS`,
  );

  // ---------- Pack key assertion (exactly J.1..J.23) ----------
  if (!existsSync(PACK_PATH)) fail(`pilot pack not exported at ${PACK_PATH}`);
  const keys = safetensorsKeys(PACK_PATH).sort((a, b) => Number(a.slice(2)) - Number(b.slice(2)));
  const expected = Array.from({ length: 23 }, (_, i) => `J.${i + 1}`);
  if (keys.join(',') !== expected.join(',')) {
    fail(`pack keys != exactly J.1..J.23: got [${keys.join(', ')}]`);
  }
  console.log(`[PACK] ${keys.length} keys = EXACTLY J.1..J.23  -> PASS`);

  // ---------- Smoke: load pack + J-lens vs logit lens ----------
  // Smoke is a GATE, but it runs AFTER the ~68-min fit + pack export, so a smoke
  // failure must NOT discard the fit data: any problem is collected into
  // smokeOk/smokeFailReason, the log is written unconditionally below, and the
  // process exits non-zero only at the very end (after the log is safely on disk).
  console.log(`\n[SMOKE] loadLensPackFromFile + lensReadout(useJacobian, layers=[${SMOKE_LAYER}])`);
  // A held-out (UNfit) corpus sequence, sliced to <=LENS_MAX_POSITIONS (48).
  const smokePrompt = corpus[N_FIT].slice(0, SMOKE_PROMPT_TOKENS);
  const smokeRun = await runWorker({
    kind: 'smoke',
    packPath: PACK_PATH,
    promptIds: smokePrompt,
    layer: SMOKE_LAYER,
    topK: SMOKE_TOPK,
  });
  const smoke = parseSentinel<{
    loaded: number;
    jacobianApplied: boolean;
    // # of logit-lens top-K ids that the J-lens demotes OUT of its own top-K
    // (J-lens full-vocab rank > K). >= 1 proves a real per-layer J moved the
    // readout — robust to a single rank flip (unlike the old top-K-order proxy).
    jDemotedLogitTop: number;
    logitTopRanksUnderJ: number[];
    jIds: number[];
    jTexts: string[];
    lIds: number[];
    lTexts: string[];
    anyNaN: boolean;
  }>(smokeRun.stdout, '__SMOKE_RESULT__');
  const smokeErr = parseSentinel<{ message: string }>(smokeRun.stdout, '__WORKER_ERROR__');
  let smokeOk = true;
  let smokeFailReason: string | null = null;
  const smokeFail = (r: string): void => {
    if (smokeOk) {
      smokeOk = false;
      smokeFailReason = r;
    }
  };
  if (!smoke || smokeRun.code !== 0) {
    smokeFail(`smoke worker failed (code ${smokeRun.code}): ${smokeErr?.message ?? 'no __SMOKE_RESULT__'}`);
  } else {
    if (smoke.loaded !== 23) smokeFail(`pack loaded ${smoke.loaded} Jacobians, expected 23`);
    if (!smoke.jacobianApplied) smokeFail(`jacobianApplied=false after loading a pack`);
    if (smoke.anyNaN) smokeFail(`J-lens readout produced NaN logits`);
    if (!(smoke.jDemotedLogitTop >= 1)) {
      smokeFail(
        `J-lens demoted 0 of the logit-lens top-${SMOKE_TOPK} ids out of its top-${SMOKE_TOPK} at ℓ${SMOKE_LAYER} ` +
          `(jDemotedLogitTop=${smoke.jDemotedLogitTop}) — J≈I, not doing anything`,
      );
    }
  }
  if (smokeOk && smoke) {
    console.log(
      `[SMOKE] loaded=${smoke.loaded} jacobianApplied=${smoke.jacobianApplied} ` +
        `jDemotedLogitTop=${smoke.jDemotedLogitTop}/${SMOKE_TOPK} (J!=logit)  -> PASS`,
    );
    console.log(`  J-lens ℓ${SMOKE_LAYER} top: [${smoke.jIds.slice(0, 5).join(', ')}] ("${smoke.jTexts.slice(0, 4).join('","')}")`);
    console.log(`  logit  ℓ${SMOKE_LAYER} top: [${smoke.lIds.slice(0, 5).join(', ')}] ("${smoke.lTexts.slice(0, 4).join('","')}")`);
    console.log(`  logit-top-${SMOKE_TOPK} ranks under J: [${smoke.logitTopRanksUnderJ.join(', ')}] (rank>${SMOKE_TOPK} = demoted)`);
  } else {
    console.error(`[SMOKE] FAILED: ${smokeFailReason}  (the fit data below is still valid and will be written)`);
  }

  // ---------- Assemble per-prompt table + log ----------
  const perPrompt = snapshots.map((s) => ({
    n_done: s.nDone,
    secPerPromptFit: s.secPerPromptFit,
    maxJNormOverSqrtD: s.maxJNormOverSqrtD,
    meanRelChange: s.meanRelChange,
    maxRelChange: s.maxRelChange,
  }));
  // Authoritative per-prompt price = native's Phase B measurement (clean, load-excluded).
  const secPerPromptAuthoritative = resB.secPerPrompt;
  const totalWallSec = (Date.now() - t0) / 1000;

  // Convergence trend check: mean_rel_change should trend DOWN (~1/n).
  const rels = perPrompt.filter((p) => p.meanRelChange != null).map((p) => p.meanRelChange as number);
  const convergesDown = rels.length >= 2 && rels[rels.length - 1] < rels[0];

  // The per-prompt convergence series is LOAD-BEARING (secPerPromptFit +
  // meanRelChange are read per-prompt). It is only trustworthy if it is exactly
  // the contiguous run 1..N_FIT (one snapshot per committed prompt). A poll that
  // missed a commit collapses two prompts into one row (see the [poll] jump
  // WARNING) — surface that here as a non-fatal flag + a loud stderr warning so
  // the corruption is visible, not silent. Exit stays governed by the smoke gate.
  let convergenceContiguous = perPrompt.length === N_FIT;
  for (let i = 0; i < perPrompt.length; i++) {
    if (perPrompt[i].n_done !== i + 1) {
      convergenceContiguous = false;
      break;
    }
  }
  if (!convergenceContiguous) {
    console.error(
      `[CONVERGENCE] WARNING: per-prompt n_done series is NOT contiguous 1..${N_FIT} ` +
        `(len=${perPrompt.length}, got [${perPrompt.map((p) => p.n_done).join(',')}]); ` +
        `rows spanning a poll gap are multi-prompt deltas — the convergence series is partially corrupted ` +
        `(non-fatal; convergence.contiguous=false is recorded in the log).`,
    );
  }

  const log = {
    task: 'T1.3',
    generated_at: new Date().toISOString(),
    model: MODEL_PATH,
    corpus_path: CORPUS_PATH,
    corpus_meta: JSON.parse(readFileSync(join(CACHE_DIR, 'corpus-meta.json'), 'utf8')),
    settings: {
      n_fit: N_FIT,
      dim_batch: DIM_BATCH,
      skip_first: SKIP_FIRST,
      max_seq_len: MAX_SEQ_LEN,
      use_checkpointing: USE_CHECKPOINTING,
      kill_at: KILL_AT,
    },
    bv_sweep: { tokens: SWEEP_TOKENS, results: sweep, fastest_bv: fastest.bv },
    per_prompt: perPrompt,
    convergence: {
      mean_rel_change_series: rels,
      trends_down: convergesDown,
      contiguous: convergenceContiguous,
      note:
        'mean_rel_change = mean over J.1..J.23 of ||pack_n - pack_{n-1}||_F / (||pack_n||_F + 1e-12); ' +
        'running-mean pack -> ~1/n decay expected.',
    },
    resume_proof: {
      phase_a_committed: killedNDone,
      phase_a_killed: phaseA.run.killed,
      phase_a_signal: phaseA.run.signal,
      phase_b_prompts_this_run: resB.promptsThisRun,
      final_n_done: resB.nDone,
      mismatch_rejected: mismatchRejected,
      mismatch_error: mismatchErr?.message ?? null,
    },
    timing: {
      sec_per_prompt_authoritative: secPerPromptAuthoritative,
      total_wall_sec: totalWallSec,
      projected_100_prompt_hours: (secPerPromptAuthoritative * 100) / 3600,
      projected_1000_prompt_hours: (secPerPromptAuthoritative * 1000) / 3600,
    },
    native_result_phase_b: resB,
    pack_path: PACK_PATH,
    checkpoint_path: CKPT_PATH,
    smoke_ok: smokeOk,
    smoke_fail_reason: smokeFailReason,
    smoke: smoke ?? { error: smokeErr?.message ?? 'no __SMOKE_RESULT__' },
  };
  writeFileSync(LOG_PATH, JSON.stringify(log, null, 2) + '\n');

  // ---------- Final summary ----------
  console.log(`\n==================== PILOT SUMMARY ====================`);
  console.log(`  per-prompt table (n_done | secFit | maxJ/sqrtD | meanRelChg | maxRelChg):`);
  for (const p of perPrompt) {
    console.log(
      `    ${String(p.n_done).padStart(2)} | ` +
        `${p.secPerPromptFit != null ? p.secPerPromptFit.toFixed(1).padStart(6) + 's' : '   n/a'} | ` +
        `${p.maxJNormOverSqrtD.toFixed(4)} | ` +
        `${p.meanRelChange != null ? p.meanRelChange.toExponential(3) : '     n/a  '} | ` +
        `${p.maxRelChange != null ? p.maxRelChange.toExponential(3) : '     n/a  '}`,
    );
  }
  console.log(`  Bv sweep (@${SWEEP_TOKENS}tok): ${sweep.map((s) => `bv${s.bv}=${s.secPerPrompt.toFixed(1)}s`).join(' ')}  fastest=bv${fastest.bv}`);
  console.log(`  secPerPrompt (authoritative, Phase B): ${secPerPromptAuthoritative.toFixed(2)}s`);
  console.log(`  projected: 100 prompts ~ ${((secPerPromptAuthoritative * 100) / 60).toFixed(1)} min; 1000 ~ ${((secPerPromptAuthoritative * 1000) / 3600).toFixed(1)} h`);
  console.log(`  convergence trends_down=${convergesDown} (series ${rels.map((r) => r.toExponential(2)).join(' -> ')})`);
  console.log(`  resume: A=${killedNDone} + B=${resB.promptsThisRun} -> ${resB.nDone}/${N_FIT}; mismatch rejected=${mismatchRejected}`);
  console.log(`  pack: ${PACK_PATH}`);
  console.log(`  log:  ${LOG_PATH}`);
  console.log(`  total wall: ${(totalWallSec / 60).toFixed(1)} min`);
  if (smokeOk) {
    console.log(`\n==================== ALL GATES PASS ====================`);
    process.exit(0);
  }
  console.error(`\n============ SMOKE GATE FAILED (fit data preserved at ${LOG_PATH}) ============`);
  console.error(`  reason: ${smokeFailReason}`);
  process.exit(1);
}

// ===========================================================================
// PRODUCTION ORCHESTRATOR (T3.1) — selected by JLENS_MODE=production.
//
// A single STRAIGHT fit 0 -> N_FIT (default 100) over the v1 corpus with:
//   - NO artificial self-kill (the pilot's KILL_AT was only a resume proof);
//   - per-prompt checkpointing (the native saves the checkpoint after EVERY
//     prompt when useCheckpointing=true, so a crash at n_done=k resumes from k);
//   - a Bv sweep re-run at the REAL window length (SWEEP_TOKENS=128) to pick the
//     fastest dimBatch that fits memory;
//   - an F32 pack export (J.1..J.23) via the proven native export_pack;
//   - an "already-complete" fast path (n_done>=N_FIT -> skip fit, re-export pack);
//   - the pilot's convergence logging + final lensReadout smoke gate.
//
// The pilot orchestrator above is UNCHANGED; when JLENS_MODE is unset the pilot
// path runs byte-identically. All v1 paths default under .cache/jlens and are
// individually overridable, so the smoke uses fully separate files.
// ===========================================================================
interface ProdConfig {
  corpusPath: string;
  ckptPath: string;
  packPath: string;
  logPath: string;
  sweepDir: string;
  nFit: number;
  sweepTokens: number;
  bvs: number[];
  dimBatchPinned: number | null;
  testKillAt: number | null;
}
interface SweepRow {
  bv: number;
  secPerPrompt: number;
  peakMemoryMb: number;
  nValidLast: number;
}
interface SmokeResult {
  loaded: number;
  jacobianApplied: boolean;
  jDemotedLogitTop: number;
  logitTopRanksUnderJ: number[];
  jIds: number[];
  jTexts: string[];
  lIds: number[];
  lTexts: string[];
  anyNaN: boolean;
}

function prodEnvInt(name: string, dflt: number): number {
  const v = process.env[name];
  if (v == null || v === '') return dflt;
  const n = Number(v);
  if (!Number.isInteger(n) || n <= 0) fail(`env ${name}=${JSON.stringify(v)} is not a positive integer`);
  return n;
}

function prodConfig(): ProdConfig {
  const bvs = (process.env.JLENS_BVS ?? '8,16,32')
    .split(',')
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isInteger(n) && n > 0);
  if (bvs.length === 0) fail(`JLENS_BVS parsed to no valid ints`);
  return {
    corpusPath: process.env.JLENS_CORPUS_PATH ?? join(CACHE_DIR, 'corpus-128-v1.jsonl'),
    ckptPath: process.env.JLENS_CKPT_PATH ?? join(CACHE_DIR, 'v1.ckpt.safetensors'),
    packPath: process.env.JLENS_PACK_PATH ?? join(CACHE_DIR, 'lens-pack-v1.safetensors'),
    logPath: process.env.JLENS_LOG_PATH ?? join(CACHE_DIR, 'fit-v1-log.json'),
    sweepDir: process.env.JLENS_SWEEP_DIR ?? join(CACHE_DIR, 'sweep-v1'),
    nFit: prodEnvInt('JLENS_N_FIT', 100),
    sweepTokens: prodEnvInt('JLENS_SWEEP_TOKENS', 128),
    bvs,
    dimBatchPinned: process.env.JLENS_DIM_BATCH ? Number(process.env.JLENS_DIM_BATCH) : null,
    testKillAt: process.env.JLENS_TEST_KILL_AT ? Number(process.env.JLENS_TEST_KILL_AT) : null,
  };
}

/** Best-effort corpus-meta read (informational log field only; never fatal). */
function readCorpusMeta(corpusPath: string): unknown | null {
  const candidates = [
    process.env.JLENS_META_PATH,
    corpusPath.replace(/\.jsonl$/, '-meta.json'),
    join(CACHE_DIR, 'corpus-meta.json'),
  ].filter(Boolean) as string[];
  for (const c of candidates) {
    try {
      if (existsSync(c)) return JSON.parse(readFileSync(c, 'utf8'));
    } catch {
      /* ignore */
    }
  }
  return null;
}

/** Load pack + lensReadout(useJacobian) smoke — the production gate. */
async function runReadoutSmoke(
  cfg: ProdConfig,
  corpus: number[][],
  smokeIdx: number,
): Promise<{ ok: boolean; reason: string | null; smoke: SmokeResult | { error: string } }> {
  console.log(
    `\n[SMOKE] loadLensPackFromFile + lensReadout(useJacobian, layer=${SMOKE_LAYER}) on corpus seq #${smokeIdx}`,
  );
  const smokePrompt = corpus[smokeIdx].slice(0, SMOKE_PROMPT_TOKENS);
  const smokeRun = await runWorker({
    kind: 'smoke',
    packPath: cfg.packPath,
    promptIds: smokePrompt,
    layer: SMOKE_LAYER,
    topK: SMOKE_TOPK,
  });
  const smoke = parseSentinel<SmokeResult>(smokeRun.stdout, '__SMOKE_RESULT__');
  const smokeErr = parseSentinel<{ message: string }>(smokeRun.stdout, '__WORKER_ERROR__');
  if (!smoke || smokeRun.code !== 0) {
    return {
      ok: false,
      reason: `smoke worker failed (code ${smokeRun.code}): ${smokeErr?.message ?? 'no __SMOKE_RESULT__'}`,
      smoke: smoke ?? { error: smokeErr?.message ?? 'no __SMOKE_RESULT__' },
    };
  }
  if (smoke.loaded !== 23) return { ok: false, reason: `pack loaded ${smoke.loaded} Jacobians, expected 23`, smoke };
  if (!smoke.jacobianApplied) return { ok: false, reason: `jacobianApplied=false after loading a pack`, smoke };
  if (smoke.anyNaN) return { ok: false, reason: `J-lens readout produced NaN logits`, smoke };
  if (!(smoke.jDemotedLogitTop >= 1)) {
    return {
      ok: false,
      reason: `J-lens demoted 0 of the logit-top-${SMOKE_TOPK} ids at ℓ${SMOKE_LAYER} (J≈I, not doing anything)`,
      smoke,
    };
  }
  console.log(
    `[SMOKE] loaded=${smoke.loaded} jacobianApplied=${smoke.jacobianApplied} ` +
      `jDemotedLogitTop=${smoke.jDemotedLogitTop}/${SMOKE_TOPK} (J!=logit)  -> PASS`,
  );
  console.log(`  J-lens ℓ${SMOKE_LAYER} top: [${smoke.jIds.slice(0, 5).join(', ')}] ("${smoke.jTexts.slice(0, 4).join('","')}")`);
  console.log(`  logit  ℓ${SMOKE_LAYER} top: [${smoke.lIds.slice(0, 5).join(', ')}] ("${smoke.lTexts.slice(0, 4).join('","')}")`);
  return { ok: true, reason: null, smoke };
}

async function productionOrchestrator(): Promise<void> {
  const t0 = Date.now();
  const cfg = prodConfig();
  mkdirSync(CACHE_DIR, { recursive: true });

  const corpus = loadCorpus(cfg.corpusPath);
  if (corpus.length < cfg.nFit) {
    fail(`corpus has ${corpus.length} sequences at ${cfg.corpusPath}, need >= N_FIT=${cfg.nFit}`);
  }
  const fitCorpus = corpus.slice(0, cfg.nFit);
  // Held-out seq for the readout smoke; when corpus == N_FIT, use the last fitted.
  const smokeIdx = corpus.length > cfg.nFit ? cfg.nFit : cfg.nFit - 1;

  console.log(`\n==================== T3.1 PRODUCTION FIT ====================`);
  console.log(`corpus: ${corpus.length} seqs (128 tok) @ ${cfg.corpusPath}; fitting first ${cfg.nFit}`);
  console.log(`  ckpt=${cfg.ckptPath}`);
  console.log(`  pack=${cfg.packPath}  (F32)`);
  console.log(`  log =${cfg.logPath}`);

  const existing = readCheckpointScalars(cfg.ckptPath);
  if (existing) {
    console.log(`[RESUME] existing checkpoint n_done=${existing.nDone} next_idx=${existing.nextIdx}`);
  } else {
    console.log(`[FRESH] no checkpoint at ${cfg.ckptPath} — straight fit 0 -> ${cfg.nFit}`);
  }

  const snapshots: Snapshot[] = [];
  let sweep: SweepRow[] | null = null;
  let dimBatch: number | null = null;
  let res: any | null = null;
  let mode: 'reexport-complete' | 'fresh-fit' | 'resume-fit';

  if (existing && existing.nDone >= cfg.nFit) {
    // -------- Already complete: skip fit + sweep, re-export the F32 pack via the
    // native fit with 0 NEW prompts. `export_pack` is gated on n_done>0 (not
    // prompts_this_run), so a fully-complete resumed checkpoint still re-writes the
    // pack. Requires the IDENTICAL corpus (the checkpoint fingerprint is
    // revalidated on resume) — we pass the same fitCorpus. --------
    mode = 'reexport-complete';
    console.log(`[COMPLETE] n_done=${existing.nDone} >= N_FIT=${cfg.nFit}; skipping fit, re-exporting F32 pack (0 new prompts)`);
    const reexportRun = await runWorker({
      kind: 'fit',
      promptIds: fitCorpus,
      dimBatch: cfg.dimBatchPinned ?? 8, // irrelevant with 0 new prompts
      skipFirst: SKIP_FIRST,
      maxSeqLen: MAX_SEQ_LEN,
      useCheckpointing: USE_CHECKPOINTING,
      resume: true,
      checkpointPath: cfg.ckptPath,
      packPath: cfg.packPath,
    });
    res = parseSentinel<any>(reexportRun.stdout, '__FIT_RESULT__');
    if (!res || reexportRun.code !== 0) {
      const err = parseSentinel<{ message: string }>(reexportRun.stdout, '__WORKER_ERROR__');
      fail(`re-export worker failed (code ${reexportRun.code}): ${err?.message ?? 'no __FIT_RESULT__'}`);
    }
    if (res.promptsThisRun !== 0) {
      console.log(`[COMPLETE] note: re-export processed ${res.promptsThisRun} prompt(s) (checkpoint was not fully complete)`);
    }
  } else {
    mode = existing ? 'resume-fit' : 'fresh-fit';

    // -------- Bv sweep at the REAL window length (unless pinned) --------
    if (cfg.dimBatchPinned != null) {
      dimBatch = cfg.dimBatchPinned;
      if (dimBatch > 32) fail(`pinned dimBatch=${dimBatch} exceeds the 32 ceiling (never exceed 32)`);
      console.log(`[SWEEP] skipped — dimBatch pinned to ${dimBatch} via JLENS_DIM_BATCH`);
    } else {
      console.log(`[SWEEP] dimBatch ∈ {${cfg.bvs.join(', ')}} on ONE ${cfg.sweepTokens}-tok prompt (throwaway ckpts)`);
      rmSync(cfg.sweepDir, { recursive: true, force: true });
      mkdirSync(cfg.sweepDir, { recursive: true });
      const sweepPrompt = fitCorpus[0].slice(0, cfg.sweepTokens);
      const sweepRun = await runWorker({
        kind: 'sweep',
        promptIds: [sweepPrompt],
        bvs: cfg.bvs,
        dir: cfg.sweepDir,
        skipFirst: SKIP_FIRST,
        maxSeqLen: cfg.sweepTokens,
        useCheckpointing: USE_CHECKPOINTING,
      });
      sweep = parseSentinel<SweepRow[]>(sweepRun.stdout, '__SWEEP_RESULT__');
      if (!sweep || sweepRun.code !== 0) fail(`Bv sweep worker failed (code ${sweepRun.code})`);
      const fastest = sweep.reduce((a, b) => (b.secPerPrompt < a.secPerPrompt ? b : a));
      dimBatch = fastest.bv;
      console.log(
        `[SWEEP] @${cfg.sweepTokens}tok: ` +
          `${sweep.map((s) => `bv${s.bv}=${s.secPerPrompt.toFixed(1)}s/${s.peakMemoryMb.toFixed(0)}MB`).join('  ')}`,
      );
      console.log(`[SWEEP] winner: bv=${dimBatch} (fastest per-prompt @${cfg.sweepTokens} tok)`);
    }
    if (dimBatch > 32) fail(`dimBatch=${dimBatch} exceeds the 32 ceiling (never exceed 32)`);

    // -------- Straight fit, per-prompt ckpt, NO artificial kill --------
    const conv: ConvState = { prevJsum: null, prevN: existing?.nDone ?? 0 };
    const resume = existing != null;
    if (resume && existsSync(cfg.ckptPath)) {
      // Seed convergence chain so the first post-resume prompt has a real delta.
      try {
        conv.prevJsum = readJsumArrays(cfg.ckptPath);
        conv.prevN = existing!.nDone;
      } catch {
        /* non-fatal — first post-resume row just reports meanRelChange=null */
      }
    }
    const killAt = cfg.testKillAt; // null in real production; SMOKE-ONLY crash injection
    if (killAt != null) {
      console.log(`[TEST-KILL] SMOKE-ONLY crash injection: SIGKILL once n_done>=${killAt}`);
    }
    console.log(
      `[FIT] resume=${resume} dimBatch=${dimBatch} skipFirst=${SKIP_FIRST} ` +
        `maxSeqLen=${MAX_SEQ_LEN} useCheckpointing=${USE_CHECKPOINTING} (per-prompt checkpoint)`,
    );
    const fitPhase = await runFitPhase(
      {
        kind: 'fit',
        promptIds: fitCorpus,
        dimBatch,
        skipFirst: SKIP_FIRST,
        maxSeqLen: MAX_SEQ_LEN,
        useCheckpointing: USE_CHECKPOINTING,
        resume,
        checkpointPath: cfg.ckptPath,
        packPath: cfg.packPath, // native writes the F32 pack on completion (proven pilot path)
      },
      snapshots,
      conv,
      killAt,
    );

    if (fitPhase.run.killed) {
      // SMOKE-ONLY crash injection landed. Do NOT export — leave the partial
      // checkpoint so a re-run resumes from it. Exit non-zero (simulated crash).
      const afterKill = readCheckpointScalars(cfg.ckptPath);
      console.log(
        `[TEST-KILL] SIGKILL landed; checkpoint n_done=${afterKill?.nDone} next_idx=${afterKill?.nextIdx} — re-run to resume`,
      );
      process.exit(137);
    }
    if (!fitPhase.result) {
      fail(`fit worker returned no result (code ${fitPhase.run.code}; signal ${fitPhase.run.signal})`);
    }
    res = fitPhase.result;
    console.log(
      `[FIT] nDone=${res.nDone} promptsThisRun=${res.promptsThisRun} skipped=${res.promptsSkipped} ` +
        `secPerPrompt=${res.secPerPrompt.toFixed(1)}s maxJ/sqrtD=${res.maxJNormOverSqrtD.toFixed(4)} peakMb=${res.peakMemoryMb.toFixed(0)}`,
    );
    if (res.nDone !== cfg.nFit) fail(`final n_done=${res.nDone} != N_FIT=${cfg.nFit}`);
  }

  // ==================== FINALIZE (common to all modes) ====================
  // The native fit already wrote the F32 pack to packPath (on completion, or on
  // the 0-prompt re-export). Assert it is EXACTLY J.1..J.23 / F32 [HIDDEN,HIDDEN].
  const { keys: packKeys, bytes: packBytes } = assertPackF32(cfg.packPath);
  console.log(
    `[PACK] ${packKeys.length} keys = EXACTLY J.1..J.23, all F32 [${HIDDEN},${HIDDEN}], ` +
      `${(packBytes / 1e6).toFixed(1)} MB (checkpoint n_done=${res ? res.nDone : existing?.nDone})  -> PASS`,
  );

  // Readout smoke gate.
  const smokeRes = await runReadoutSmoke(cfg, corpus, smokeIdx);

  // Convergence series (mean_rel_change should trend down ~1/n).
  const perPrompt = snapshots.map((s) => ({
    n_done: s.nDone,
    secPerPromptFit: s.secPerPromptFit,
    maxJNormOverSqrtD: s.maxJNormOverSqrtD,
    meanRelChange: s.meanRelChange,
    maxRelChange: s.maxRelChange,
  }));
  const rels = perPrompt.filter((p) => p.meanRelChange != null).map((p) => p.meanRelChange as number);
  const convergesDown = rels.length >= 2 && rels[rels.length - 1] < rels[0];
  const finalNDone = res ? res.nDone : existing!.nDone;
  // Re-export runs 0 prompts, so its native secPerPrompt is meaningless — null it.
  const secPerPrompt = res && res.promptsThisRun > 0 ? res.secPerPrompt : null;
  const totalWallSec = (Date.now() - t0) / 1000;

  const log = {
    task: 'T3.1',
    mode,
    generated_at: new Date().toISOString(),
    model: MODEL_PATH,
    corpus_path: cfg.corpusPath,
    corpus_meta: readCorpusMeta(cfg.corpusPath),
    settings: {
      n_fit: cfg.nFit,
      dim_batch: dimBatch,
      dim_batch_pinned: cfg.dimBatchPinned,
      skip_first: SKIP_FIRST,
      max_seq_len: MAX_SEQ_LEN,
      use_checkpointing: USE_CHECKPOINTING,
      sweep_tokens: cfg.sweepTokens,
      kill_at: null,
      test_kill_at: cfg.testKillAt,
    },
    bv_sweep: sweep ? { tokens: cfg.sweepTokens, results: sweep, fastest_bv: dimBatch } : null,
    per_prompt: perPrompt,
    convergence: {
      mean_rel_change_series: rels,
      trends_down: convergesDown,
      note:
        'mean_rel_change = mean over J.1..J.23 of ||pack_n - pack_{n-1}||_F / (||pack_n||_F + 1e-12); ' +
        'running-mean pack -> ~1/n decay expected.',
    },
    resume: {
      resumed: mode !== 'fresh-fit',
      resumed_from_n_done: existing?.nDone ?? 0,
      prompts_this_run: res ? res.promptsThisRun : 0,
      final_n_done: finalNDone,
    },
    timing: {
      sec_per_prompt: secPerPrompt,
      total_wall_sec: totalWallSec,
      projected_100_prompt_hours: secPerPrompt != null ? (secPerPrompt * 100) / 3600 : null,
    },
    native_result: res,
    pack: { path: cfg.packPath, dtype: 'F32', keys: packKeys.length, bytes: packBytes, checkpoint_n_done: finalNDone },
    checkpoint_path: cfg.ckptPath,
    smoke_ok: smokeRes.ok,
    smoke_fail_reason: smokeRes.reason,
    smoke: smokeRes.smoke,
  };
  writeFileSync(cfg.logPath, JSON.stringify(log, null, 2) + '\n');

  // ---------- Summary ----------
  console.log(`\n==================== PRODUCTION SUMMARY (${mode}) ====================`);
  if (perPrompt.length) {
    console.log(`  per-prompt (n_done | secFit | maxJ/sqrtD | meanRelChg):`);
    for (const p of perPrompt) {
      console.log(
        `    ${String(p.n_done).padStart(3)} | ` +
          `${p.secPerPromptFit != null ? p.secPerPromptFit.toFixed(1).padStart(6) + 's' : '   n/a'} | ` +
          `${p.maxJNormOverSqrtD.toFixed(4)} | ` +
          `${p.meanRelChange != null ? p.meanRelChange.toExponential(3) : '   n/a'}`,
      );
    }
  }
  if (sweep) {
    console.log(
      `  Bv sweep (@${cfg.sweepTokens}tok): ${sweep.map((s) => `bv${s.bv}=${s.secPerPrompt.toFixed(1)}s`).join(' ')}  winner=bv${dimBatch}`,
    );
  }
  if (secPerPrompt != null) {
    console.log(`  secPerPrompt @128 (dimBatch=${dimBatch}): ${secPerPrompt.toFixed(1)}s`);
    console.log(`  projected 100 prompts: ${((secPerPrompt * 100) / 3600).toFixed(2)} h`);
  }
  console.log(`  convergence trends_down=${convergesDown} (series ${rels.map((r) => r.toExponential(2)).join(' -> ')})`);
  console.log(`  resume: from n_done=${existing?.nDone ?? 0}, +${res ? res.promptsThisRun : 0} this run -> ${finalNDone}/${cfg.nFit}`);
  console.log(`  pack: ${cfg.packPath} (F32, ${(packBytes / 1e6).toFixed(1)} MB)`);
  console.log(`  log:  ${cfg.logPath}`);
  console.log(`  total wall: ${(totalWallSec / 60).toFixed(1)} min`);
  if (smokeRes.ok) {
    console.log(`\n==================== ALL GATES PASS ====================`);
    process.exit(0);
  }
  console.error(`\n============ SMOKE GATE FAILED (fit data preserved at ${cfg.logPath}) ============`);
  console.error(`  reason: ${smokeRes.reason}`);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// WORKER (loads the model; runs ONE job)
// ---------------------------------------------------------------------------
async function worker(): Promise<void> {
  const jobPath = process.argv[2];
  if (!jobPath) {
    console.log('__WORKER_ERROR__ ' + JSON.stringify({ message: 'no job file argv' }));
    process.exit(1);
  }
  const job = JSON.parse(readFileSync(jobPath, 'utf8'));
  const { Qwen35Model } = await import('@mlx-node/lm');
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;

  try {
    if (job.kind === 'sweep') {
      const results: Array<{ bv: number; secPerPrompt: number; peakMemoryMb: number; nValidLast: number }> = [];
      for (const bv of job.bvs as number[]) {
        const ckpt = join(job.dir as string, `sweep-bv${bv}.ckpt.safetensors`);
        rmSync(ckpt, { force: true });
        const r = await model.fitJacobianLens({
          promptIds: job.promptIds,
          dimBatch: bv,
          skipFirst: job.skipFirst,
          maxSeqLen: job.maxSeqLen,
          useCheckpointing: job.useCheckpointing,
          checkpointPath: ckpt,
          resume: false,
        });
        console.log(`    [sweep] bv=${bv} secPerPrompt=${r.secPerPrompt.toFixed(2)} nValid=${r.nValidLast} peakMb=${r.peakMemoryMb.toFixed(0)}`);
        results.push({ bv, secPerPrompt: r.secPerPrompt, peakMemoryMb: r.peakMemoryMb, nValidLast: r.nValidLast });
        rmSync(ckpt, { force: true });
      }
      console.log('__SWEEP_RESULT__ ' + JSON.stringify(results));
      process.exit(0);
    }

    if (job.kind === 'fit') {
      const r = await model.fitJacobianLens({
        promptIds: job.promptIds,
        dimBatch: job.dimBatch,
        skipFirst: job.skipFirst,
        maxSeqLen: job.maxSeqLen,
        useCheckpointing: job.useCheckpointing,
        resume: job.resume,
        checkpointPath: job.checkpointPath,
        packPath: job.packPath,
      });
      console.log('__FIT_RESULT__ ' + JSON.stringify(r));
      process.exit(0);
    }

    if (job.kind === 'smoke') {
      const loaded = await model.loadLensPackFromFile(job.packPath);
      const P = (job.promptIds as number[]).length;
      // 1) Logit lens FIRST (no Jacobian) — its top-K ids are the reference set.
      const lro = await model.lensReadout(job.promptIds, { layers: [job.layer], topK: job.topK, useJacobian: false });
      const lCell = lro.cells.find((c: { position: number }) => c.position === P - 1);
      if (!lCell) {
        console.log('__WORKER_ERROR__ ' + JSON.stringify({ message: `no last-position logit cell at layer ${job.layer}` }));
        process.exit(1);
      }
      const lIds: number[] = lCell.topKIds;
      // 2) J lens, PINNING the logit-lens top-K ids so we get their J-lens
      //    full-vocab ranks (pinnedIds is capped at LENS_MAX_PINNED=8; topK<=8).
      const pinnedIds = lIds.slice(0, 8);
      const jro = await model.lensReadout(job.promptIds, {
        layers: [job.layer],
        topK: job.topK,
        pinnedIds,
        useJacobian: true,
      });
      const jCellIdx = jro.cells.findIndex((c: { position: number }) => c.position === P - 1);
      const jCell = jCellIdx >= 0 ? jro.cells[jCellIdx] : undefined;
      if (!jCell) {
        console.log('__WORKER_ERROR__ ' + JSON.stringify({ message: `no last-position J cell at layer ${job.layer}` }));
        process.exit(1);
      }
      const jLogits = Array.from(jCell.topKLogits as Float32Array);
      const anyNaN = jLogits.some((x) => !Number.isFinite(x));
      const jIds: number[] = jCell.topKIds;
      // `pinned[k].ranks` is aligned to `cells` (layer-major, position-minor);
      // for our single requested layer the readout cell is at index jCellIdx.
      // Count how many logit-lens top-K ids J demotes OUT of its own top-K
      // (1-based full-vocab rank > K). A real per-layer J moves at least one.
      const K = pinnedIds.length;
      const logitTopRanksUnderJ: number[] = (jro.pinned as Array<{ ranks: Int32Array }>).map(
        (p) => p.ranks[jCellIdx],
      );
      const jDemotedLogitTop = logitTopRanksUnderJ.filter((r) => r > K).length;
      console.log(
        '__SMOKE_RESULT__ ' +
          JSON.stringify({
            loaded,
            jacobianApplied: jro.jacobianApplied,
            jDemotedLogitTop,
            logitTopRanksUnderJ,
            anyNaN,
            jIds,
            jTexts: jCell.topKTexts,
            lIds,
            lTexts: lCell.topKTexts,
          }),
      );
      process.exit(0);
    }

    console.log('__WORKER_ERROR__ ' + JSON.stringify({ message: `unknown job kind ${job.kind}` }));
    process.exit(1);
  } catch (e: any) {
    console.log('__WORKER_ERROR__ ' + JSON.stringify({ message: String(e?.message ?? e) }));
    process.exit(1);
  }
}

// ---------------------------------------------------------------------------
if (process.env.__JLENS_FIT_WORKER) {
  // WORKER is mode-agnostic (all paths come from the job file) — dispatch first.
  worker().catch((e) => {
    console.log('__WORKER_ERROR__ ' + JSON.stringify({ message: String(e?.message ?? e) }));
    process.exit(1);
  });
} else if ((process.env.JLENS_MODE ?? 'pilot') === 'production') {
  productionOrchestrator().catch((e) => {
    console.error(e);
    process.exit(1);
  });
} else {
  orchestrator().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
