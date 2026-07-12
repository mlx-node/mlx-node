/**
 * J-lens byte-escape MIGRATION (one-shot data fix).
 *
 * The committed baked frames (demo/learn/widgets/jlens/baked/<slug>.json) and
 * their byte-identical /jspace starters (demo/jspace/starters/<slug>.json) were
 * produced by a bake that decoded each lens token STANDALONE, so byte-level-BPE
 * fragment tokens collapsed to the Unicode replacement char `�` (see
 * src/lens-byte-escape.ts for the full root cause). This script rewrites those
 * `�` in-place to faithful hex escapes like `‹E5›`, reconstructing the raw bytes
 * from each token id via the shipped tokenizer's byte-level vocab.
 *
 * NO WASM rebuild, NO inference, NO re-bake: this is a pure DATA transform over
 * the already-committed JSON. It mirrors the live worker fix (mlx-worker.ts) and
 * the forward-looking bake patch (bake.mts) so all three stay in lockstep.
 *
 * Genuine `�` tokens (bytes `EF BF BD`, a real U+FFFD in the vocab piece) are
 * VALID UTF-8 and are intentionally preserved — only lossy fragments escape.
 *
 * Formatting: files are re-emitted with `JSON.stringify(x, null, 2) + '\n'`,
 * exactly matching bake.mts:256, so the diff touches only the escaped strings and
 * the two dirs stay byte-identical.
 *
 * Run (NO model / GPU needed):
 *   oxnode packages/browser/scripts/jlens/fix-byte-escapes.mts
 * Override the tokenizer with JLENS_TOKENIZER (a path to a tokenizer.json).
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { SerializedRun } from '../../demo/jlens-core/revive.ts';
import { buildIdToBytes, escapeByteFragments } from '../../src/lens-byte-escape.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const BROWSER_ROOT = join(HERE, '../..');

const STARTER_DIR = join(BROWSER_ROOT, 'demo/jspace/starters');
const BAKED_DIR = join(BROWSER_ROOT, 'demo/learn/widgets/jlens/baked');
const SLUGS = ['arithmetic-parens', 'french-season', 'spanish-opposite'];

// Local tokenizer.json (dev symlink), with a fallback to a downloaded checkpoint.
const TOKENIZER_CANDIDATES = [
  process.env.JLENS_TOKENIZER,
  join(BROWSER_ROOT, 'demo/public/model/tokenizer.json'),
  join(BROWSER_ROOT, 'test-output/final/tokenizer.json'),
].filter((p): p is string => Boolean(p));

function findTokenizer(): string {
  for (const p of TOKENIZER_CANDIDATES) {
    if (existsSync(p)) return p;
  }
  throw new Error(
    `no tokenizer.json found; looked at:\n  ${TOKENIZER_CANDIDATES.join('\n  ')}\n` +
      `set JLENS_TOKENIZER=/path/to/tokenizer.json`,
  );
}

type Envelope = { logit: SerializedRun; jacobian: SerializedRun; [k: string]: unknown };

/** Rewrite every `�`-bearing string in a run: cells[].topKTexts (aligned with
 *  topKIds), tokens[].text (aligned with tokens[].id), and pinned[].tokenText
 *  (aligned with pinned[].tokenId). Covering pinned keeps the migration in
 *  lockstep with the live worker + bake, so a future re-bake stays byte-identical
 *  to the committed data even if a concept ever pins a fragment token. Returns
 *  #strings changed. */
function fixRun(run: SerializedRun, idToBytes: (id: number) => Uint8Array | null): number {
  let changed = 0;
  for (const cell of run.cells) {
    for (let i = 0; i < cell.topKTexts.length; i++) {
      const original = cell.topKTexts[i]!;
      const escaped = escapeByteFragments(original, cell.topKIds[i]!, idToBytes);
      if (escaped !== original) {
        cell.topKTexts[i] = escaped;
        changed += 1;
      }
    }
  }
  for (const token of run.tokens) {
    const escaped = escapeByteFragments(token.text, token.id, idToBytes);
    if (escaped !== token.text) {
      token.text = escaped;
      changed += 1;
    }
  }
  for (const pin of run.pinned) {
    const escaped = escapeByteFragments(pin.tokenText, pin.tokenId, idToBytes);
    if (escaped !== pin.tokenText) {
      pin.tokenText = escaped;
      changed += 1;
    }
  }
  return changed;
}

function fixFile(path: string, idToBytes: (id: number) => Uint8Array | null): number {
  const envelope = JSON.parse(readFileSync(path, 'utf8')) as Envelope;
  const changed = fixRun(envelope.logit, idToBytes) + fixRun(envelope.jacobian, idToBytes);
  if (changed > 0) {
    // Match bake.mts:256 formatting exactly so the diff is minimal + starters==baked.
    writeFileSync(path, JSON.stringify(envelope, null, 2) + '\n');
  }
  return changed;
}

function main(): void {
  const tokenizerPath = findTokenizer();
  console.log(`tokenizer: ${tokenizerPath}`);
  const idToBytes = buildIdToBytes(readFileSync(tokenizerPath, 'utf8'));

  let totalChanged = 0;
  for (const slug of SLUGS) {
    const starterPath = join(STARTER_DIR, `${slug}.json`);
    const bakedPath = join(BAKED_DIR, `${slug}.json`);
    const starterChanged = fixFile(starterPath, idToBytes);
    const bakedChanged = fixFile(bakedPath, idToBytes);
    totalChanged += starterChanged + bakedChanged;

    // Post-condition: the two committed copies MUST stay byte-identical.
    const identical = readFileSync(starterPath, 'utf8') === readFileSync(bakedPath, 'utf8');
    console.log(
      `[${slug.padEnd(18)}] starter:${starterChanged} baked:${bakedChanged} ` +
        `identical:${identical ? 'YES' : 'NO !!!'}`,
    );
    if (!identical) {
      throw new Error(`${slug}: starter and baked diverged after migration — aborting`);
    }
  }
  console.log(`\nMIGRATION: OK — ${totalChanged} string(s) escaped across ${SLUGS.length * 2} files`);
}

main();
