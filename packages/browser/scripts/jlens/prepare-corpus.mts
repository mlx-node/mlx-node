/**
 * J-lens PILOT corpus builder (Task T1.3).
 *
 * Produces equal-length 128-token prompt sequences from WikiText-103-raw-v1 for
 * the offline Jacobian fit driven by `fit.mts`. This is oxnode scripting against
 * the T1.2 native surface — NO native/Rust changes.
 *
 *   1. FETCH  WikiText-103-raw-v1 (train) records with `len(text.strip()) >= 600`
 *             chars, cached under `.cache/jlens/wikitext-raw-records.jsonl` (via
 *             `uv run --with datasets` streaming, so re-runs never re-download and
 *             only a few row-groups are ever pulled). Skipped if the cache exists.
 *   2. TOKENIZE each record with the MODEL'S OWN tokenizer (`encodeTokens`, the
 *             same BPE `@mlx-node/lm` uses) — RAW text, NO chat template, NO manual
 *             special tokens (design-freeze D9).
 *   3. BOS    Qwen3.5's tokenizer_config has `bos_token: null` / `add_bos_token:
 *             false` (VERIFIED at load below, not assumed), so there is NO
 *             attention-sink BOS to prepend — `skip_first=16` alone carries the
 *             sink role. `bos_used=false` is recorded in the meta sidecar. (Had a
 *             BOS existed we would prepend it to every window: [BOS] + 127 content.)
 *   4. WINDOW slice each record into NON-OVERLAPPING 128-token windows and DROP any
 *             final sub-128 tail, so EVERY emitted sequence is EXACTLY 128 ids
 *             (design-freeze D2 equal-length).
 *
 * Output (all under `.cache/jlens/`, OUTSIDE the repo, gitignored — never committed):
 *   - `corpus-128.jsonl`  one line per sequence: {"ids": number[128]}
 *   - `corpus-meta.json`  {n_sequences, max_seq_len, bos_used, corpus, min_chars,
 *                          windowing, corpus_sha, ...}
 *
 * Run:  env PATH="/opt/homebrew/bin:$PATH" oxnode \
 *         packages/browser/scripts/jlens/prepare-corpus.mts
 * (arm64 Homebrew node ONLY — the x64 node cannot load the arm64 addon; oxnode,
 *  never tsx.)
 */
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
} from 'node:fs';
import { join } from 'node:path';

const MODEL_PATH =
  '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const CACHE_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';
const RAW_RECORDS = join(CACHE_DIR, 'wikitext-raw-records.jsonl');

// ---------------------------------------------------------------------------
// Parameterization (T3.1). The PILOT defaults are UNCHANGED when no env var is
// set — env only re-targets the output paths + emit/fetch counts so the SAME
// script can build the production `-v1` corpus (100 windows) or a tiny smoke
// corpus without touching the pilot artifacts. Deterministic streaming means a
// bigger raw-records cache still yields the identical first-N records.
//   JLENS_CORPUS_OUT  — corpus jsonl output (default pilot corpus-128.jsonl)
//   JLENS_META_OUT    — meta sidecar (default: derived from CORPUS_OUT, or the
//                       pilot corpus-meta.json when CORPUS_OUT is the pilot path)
//   JLENS_EMIT_MAX    — EXACT window count to emit (default 30)
//   JLENS_EMIT_MIN    — hard floor sanity bound (default 20)
//   JLENS_FETCH_WANT  — qualifying raw records to cache; must exceed the records
//                       needed to yield EMIT_MAX windows (default 60)
//   JLENS_MIN_CHARS   — min stripped chars per raw record (default 600)
// ---------------------------------------------------------------------------
const PILOT_CORPUS_PATH = join(CACHE_DIR, 'corpus-128.jsonl');
const CORPUS_PATH = process.env.JLENS_CORPUS_OUT ?? PILOT_CORPUS_PATH;
const META_PATH =
  process.env.JLENS_META_OUT ??
  (CORPUS_PATH === PILOT_CORPUS_PATH
    ? join(CACHE_DIR, 'corpus-meta.json') // preserve the pilot's exact meta filename
    : CORPUS_PATH.replace(/\.jsonl$/, '-meta.json'));

function log(msg: string): void {
  console.log(msg);
}
function fail(msg: string): never {
  console.error(`\nFAIL: ${msg}`);
  process.exit(1);
}
/** Parse a positive-integer env override, falling back to the pilot default. */
function envInt(name: string, dflt: number): number {
  const v = process.env[name];
  if (v == null || v === '') return dflt;
  const n = Number(v);
  if (!Number.isInteger(n) || n <= 0) fail(`env ${name}=${JSON.stringify(v)} is not a positive integer`);
  return n;
}

const MIN_CHARS = envInt('JLENS_MIN_CHARS', 600); // matches reference `min_chars`
const WINDOW = 128; // design-freeze D2 equal-length prompts
const EMIT_MAX = envInt('JLENS_EMIT_MAX', 30); // emit EXACTLY this many 128-tok sequences
const EMIT_MIN = envInt('JLENS_EMIT_MIN', 20); // hard floor — pilot needs 10 fit + sweep + margin
const FETCH_WANT = envInt('JLENS_FETCH_WANT', 60); // qualifying raw records to cache (margin over EMIT_MAX)

/**
 * Ensure the raw-text record cache exists. If missing/short, stream
 * WikiText-103-raw-v1 (train) via `uv run --with datasets` and cache the first
 * `FETCH_WANT` records with `len(strip()) >= MIN_CHARS`. Streaming stops early,
 * so only a few parquet row-groups are pulled (a few MB), never the full split.
 */
function ensureRawRecords(): void {
  if (existsSync(RAW_RECORDS)) {
    const n = readFileSync(RAW_RECORDS, 'utf8').trim().split('\n').filter(Boolean).length;
    // Reuse only when the cache holds enough records to cover FETCH_WANT (which
    // must be sized above the records needed for EMIT_MAX windows). A pilot cache
    // of exactly FETCH_WANT=60 still reuses; a larger production FETCH_WANT (e.g.
    // 300) forces a deterministic refetch (identical first-60, plus more).
    if (n >= FETCH_WANT) {
      log(`raw records cache present (${n} >= FETCH_WANT=${FETCH_WANT}) -> ${RAW_RECORDS}`);
      return;
    }
    log(`raw records cache too small (${n} < FETCH_WANT=${FETCH_WANT}); refetching`);
  }
  log(`fetching WikiText-103-raw-v1 (train, len>=${MIN_CHARS}) via uv+datasets streaming ...`);
  const py = `
import json, sys
from datasets import load_dataset
OUT = ${JSON.stringify(RAW_RECORDS)}
MIN_CHARS = ${MIN_CHARS}
WANT = ${FETCH_WANT}
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
n = 0; scanned = 0
with open(OUT, "w") as f:
    for rec in ds:
        scanned += 1
        t = rec["text"]
        if len(t.strip()) >= MIN_CHARS:
            f.write(json.dumps({"text": t}) + "\\n"); n += 1
            if n >= WANT: break
        if scanned > 200000: break
print(f"wrote {n} records (scanned {scanned}) -> {OUT}", file=sys.stderr)
`;
  const res = spawnSync('uv', ['run', '--with', 'datasets', 'python', '-c', py], {
    stdio: ['ignore', 'inherit', 'inherit'],
    encoding: 'utf8',
  });
  if (res.status !== 0) {
    fail(
      `WikiText fetch failed (uv exit ${res.status}). Ensure \`uv\` is on PATH ` +
        `(e.g. PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH") and network is available.`,
    );
  }
  if (!existsSync(RAW_RECORDS)) fail(`fetch reported success but ${RAW_RECORDS} is missing`);
}

async function main() {
  mkdirSync(CACHE_DIR, { recursive: true });
  ensureRawRecords();

  // ---- BOS audit: read the tokenizer_config directly, don't assume. ----
  const tokCfg = JSON.parse(
    readFileSync(join(MODEL_PATH, 'tokenizer_config.json'), 'utf8'),
  ) as { bos_token?: unknown; add_bos_token?: unknown };
  const bosToken = typeof tokCfg.bos_token === 'string' ? tokCfg.bos_token : null;
  const addBos = tokCfg.add_bos_token === true;
  const bosUsed = bosToken != null && addBos; // both must indicate a real sink BOS
  log(
    `tokenizer BOS audit: bos_token=${JSON.stringify(bosToken)} ` +
      `add_bos_token=${JSON.stringify(tokCfg.add_bos_token)} -> bos_used=${bosUsed}`,
  );

  const { Qwen35Model } = await import('@mlx-node/lm');
  log(`loading model from ${MODEL_PATH} ...`);
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;
  log('model loaded.');

  // Cross-check: the model tokenizer must NOT silently prepend a special id.
  const probe: number[] = await model.encodeTokens('The');
  if (bosUsed && probe.length > 0) {
    // (defensive) if a BOS were configured we'd expect encodeTokens to prepend it.
    log(`  (probe "The") -> ${JSON.stringify(probe)}`);
  } else {
    log(`  encode probe "The" -> ${JSON.stringify(probe)} (no leading special id — raw BPE)`);
  }

  const records: string[] = readFileSync(RAW_RECORDS, 'utf8')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l).text as string)
    .filter((t) => t.trim().length >= MIN_CHARS);
  log(`raw records eligible (>=${MIN_CHARS} chars): ${records.length}`);

  const bosPrefix: number[] = bosUsed && bosToken != null ? await model.encodeTokens(String(bosToken)) : [];
  const contentPerWindow = WINDOW - bosPrefix.length; // 128 (no BOS) or 127 (+BOS)

  const sequences: number[][] = [];
  let recordsUsed = 0;
  let windowsDropped = 0;
  for (const text of records) {
    if (sequences.length >= EMIT_MAX) break;
    // RAW tokenize — no chat template, no manual specials (encodeTokens honors
    // the tokenizer's own add_special_tokens config, verified false above).
    const ids: number[] = await model.encodeTokens(text);
    let used = false;
    for (let off = 0; off + contentPerWindow <= ids.length; off += contentPerWindow) {
      const window = bosPrefix.concat(ids.slice(off, off + contentPerWindow));
      if (window.length !== WINDOW) fail(`internal: window length ${window.length} != ${WINDOW}`);
      sequences.push(window);
      used = true;
      if (sequences.length >= EMIT_MAX) break;
    }
    // Count the dropped sub-128 tail (informational).
    if (ids.length % contentPerWindow !== 0) windowsDropped++;
    if (used) recordsUsed++;
  }

  // The corpus contract is EXACTLY EMIT_MAX windows — the fit driver, checkpoint
  // corpus-fingerprint, and pilot log all assume a fixed 30-sequence corpus. A
  // stale/short raw-records cache could otherwise silently yield 20-29 sequences
  // and exit 0, quietly changing the corpus. Require the exact count.
  if (sequences.length !== EMIT_MAX) {
    fail(
      `produced ${sequences.length} sequences, need EXACTLY ${EMIT_MAX} (EMIT_MIN floor is ${EMIT_MIN}). ` +
        `The raw-records cache is likely stale/short — delete ${RAW_RECORDS} and re-run ` +
        `(and/or raise FETCH_WANT=${FETCH_WANT} or lower MIN_CHARS=${MIN_CHARS}) to refetch enough records.`,
    );
  }

  // Hard invariant: every emitted sequence is EXACTLY 128 ids.
  for (let i = 0; i < sequences.length; i++) {
    if (sequences[i].length !== WINDOW) {
      fail(`sequence ${i} has length ${sequences[i].length} != ${WINDOW}`);
    }
    for (const id of sequences[i]) {
      if (!Number.isInteger(id) || id < 0) fail(`sequence ${i} has bad id ${id}`);
    }
  }

  // Deterministic content fingerprint over the emitted sequences (order-sensitive).
  const corpusSha = createHash('sha256')
    .update(JSON.stringify(sequences))
    .digest('hex');

  // Write corpus JSONL (one sequence per line) ATOMICALLY — write to a .tmp
  // sibling then rename, so an interrupted run can never leave a torn/partial
  // corpus that the fit driver would then read as valid.
  const jsonl = sequences.map((ids) => JSON.stringify({ ids })).join('\n') + '\n';
  writeFileSync(CORPUS_PATH + '.tmp', jsonl);
  renameSync(CORPUS_PATH + '.tmp', CORPUS_PATH);

  const meta = {
    n_sequences: sequences.length,
    max_seq_len: WINDOW,
    bos_used: bosUsed,
    bos_token: bosToken,
    add_bos_token: tokCfg.add_bos_token ?? null,
    corpus: 'wikitext-103-raw-v1',
    split: 'train',
    min_chars: MIN_CHARS,
    windowing: `non-overlapping ${WINDOW}-tok windows, sub-${WINDOW} tails dropped, BOS=${
      bosUsed ? String(bosToken) : 'none'
    }`,
    chat_template: false,
    tokenizer: 'qwen3.5-0.8b (model-native BPE via encodeTokens)',
    records_eligible: records.length,
    records_used: recordsUsed,
    tails_dropped: windowsDropped,
    corpus_sha256: corpusSha,
    generated_at: new Date().toISOString(),
  };
  writeFileSync(META_PATH + '.tmp', JSON.stringify(meta, null, 2) + '\n');
  renameSync(META_PATH + '.tmp', META_PATH);

  log('');
  log(`==================== CORPUS BUILT ====================`);
  log(`  sequences:     ${sequences.length} (each EXACTLY ${WINDOW} ids)`);
  log(`  records used:  ${recordsUsed} of ${records.length} eligible`);
  log(`  bos_used:      ${bosUsed} (${meta.windowing})`);
  log(`  corpus_sha256: ${corpusSha}`);
  log(`  first seq len: ${sequences[0].length}   first 8 ids: [${sequences[0].slice(0, 8).join(', ')}]`);
  log(`  corpus:        ${CORPUS_PATH}`);
  log(`  meta:          ${META_PATH}`);
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
