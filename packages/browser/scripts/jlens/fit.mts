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
function loadCorpus(): number[][] {
  if (!existsSync(CORPUS_PATH)) {
    fail(`corpus not found at ${CORPUS_PATH} — run prepare-corpus.mts first`);
  }
  const seqs = readFileSync(CORPUS_PATH, 'utf8')
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
    child.on('exit', (code: number | null, signal: string | null) => {
      if (timer) clearInterval(timer);
      try {
        rmSync(jobPath, { force: true });
      } catch {
        /* ignore */
      }
      resolve({ code, signal, stdout, killed });
    });
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
  if (afterKill.nDone < 1) fail(`Phase A committed no prompts (n_done=${afterKill.nDone}) — fit may be broken`);
  const killedNDone = afterKill.nDone;

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
    sameOrder: boolean;
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
    if (smoke.sameOrder) smokeFail(`J-lens == logit lens at ℓ${SMOKE_LAYER} (J≈I, not doing anything)`);
  }
  if (smokeOk && smoke) {
    console.log(`[SMOKE] loaded=${smoke.loaded} jacobianApplied=${smoke.jacobianApplied} J!=logit  -> PASS`);
    console.log(`  J-lens ℓ${SMOKE_LAYER} top: [${smoke.jIds.slice(0, 5).join(', ')}] ("${smoke.jTexts.slice(0, 4).join('","')}")`);
    console.log(`  logit  ℓ${SMOKE_LAYER} top: [${smoke.lIds.slice(0, 5).join(', ')}] ("${smoke.lTexts.slice(0, 4).join('","')}")`);
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
      const jro = await model.lensReadout(job.promptIds, { layers: [job.layer], topK: job.topK, useJacobian: true });
      const lro = await model.lensReadout(job.promptIds, { layers: [job.layer], topK: job.topK, useJacobian: false });
      const jCell = jro.cells.find((c: { position: number }) => c.position === P - 1);
      const lCell = lro.cells.find((c: { position: number }) => c.position === P - 1);
      if (!jCell || !lCell) {
        console.log('__WORKER_ERROR__ ' + JSON.stringify({ message: `no last-position cell at layer ${job.layer}` }));
        process.exit(1);
      }
      const jLogits = Array.from(jCell.topKLogits as Float32Array);
      const anyNaN = jLogits.some((x) => !Number.isFinite(x));
      const jIds: number[] = jCell.topKIds;
      const lIds: number[] = lCell.topKIds;
      const sameOrder = jIds.length === lIds.length && jIds.every((v, i) => v === lIds[i]);
      console.log(
        '__SMOKE_RESULT__ ' +
          JSON.stringify({
            loaded,
            jacobianApplied: jro.jacobianApplied,
            sameOrder,
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
  worker().catch((e) => {
    console.log('__WORKER_ERROR__ ' + JSON.stringify({ message: String(e?.message ?? e) }));
    process.exit(1);
  });
} else {
  orchestrator().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
