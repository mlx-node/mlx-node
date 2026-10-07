/// <reference types="node" />

import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

// Teacher-forced DFlash2 acceptance (splash-qwen38.md §6). Run from the
// repository root. `record` writes one reference per case; `force` replays
// those references through the real verify path and summarizes
// `<dir>/cycles-<label>.jsonl`. One fresh benchmark.ts process per case.
const usage = `Usage:
  oxnode docs/research/splash-qwen38/tf-acceptance.ts record <addon.node> <ref-dir> [short,6k,32k] [tokens=1024]
  oxnode docs/research/splash-qwen38/tf-acceptance.ts force <addon.node> <ref-dir> <label> [short,6k,32k] [tokens=1024]
  oxnode docs/research/splash-qwen38/tf-acceptance.ts summary <ref-dir> <label> [tokens=1024]`;
const [command, ...rest] = process.argv.slice(2);
const benchmark = resolve('docs/research/splash-qwen38/benchmark.ts');

function run(binding: string, dir: string, label: string, names: string, tokens: string, env: NodeJS.ProcessEnv) {
  mkdirSync(dir, { recursive: true });
  for (const name of names.split(',')) {
    const out = join(dir, `bench-${label}-${name}.json`);
    const child = spawnSync('oxnode', [benchmark, binding, out, 'dflash', name, '1', tokens], {
      stdio: 'inherit',
      env: { ...process.env, ...env },
    });
    if (child.status !== 0) throw new Error(`benchmark.ts failed for ${name} (exit ${child.status})`);
  }
}

type Row = [pos: number, len: number, aRef: number, aLive: number, flip: number, wallUs?: number];
interface Turn {
  key: string;
  maxNew: number;
  generated: number;
  cycles: Row[];
}

function summarize(rows: Row[]) {
  const drafted = rows.filter((r) => r[1] > 0);
  const committed = rows.reduce((sum, r) => sum + r[3] + 1, 0);
  const wallUs = rows.reduce((sum, r) => sum + (r[5] ?? 0), 0);
  const depth = Math.max(0, ...drafted.map((r) => r[1]));
  const byPosition = Array.from({ length: depth }, (_, i) => {
    const attempted = drafted.filter((r) => r[1] > i).length;
    return attempted ? drafted.filter((r) => r[3] > i).length / attempted : 0;
  });
  return {
    cycles: rows.length,
    draftCycles: drafted.length,
    committed,
    meanCommitted: rows.length ? committed / rows.length : 0,
    meanARef: drafted.length ? drafted.reduce((sum, r) => sum + r[2], 0) / drafted.length : 0,
    flipRate: rows.length ? rows.filter((r) => r[4] >= 0).length / rows.length : 0,
    byPosition,
    ...(wallUs > 0 ? { msPerCommitted: wallUs / 1000 / committed } : {}),
  };
}

function summary(dir: string, label: string, tokens: string) {
  const turns: Turn[] = readFileSync(join(dir, `cycles-${label}.jsonl`), 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  // benchmark.ts warms up with a 16-token turn on the same prompt; keep only
  // full-length turns.
  const kept = turns.filter((t) => t.maxNew === Number(tokens));
  console.log(
    JSON.stringify(
      {
        label,
        turns: kept.map((t) => ({ key: t.key, generated: t.generated, ...summarize(t.cycles) })),
        total: summarize(kept.flatMap((t) => t.cycles)),
      },
      null,
      2,
    ),
  );
}

if (command === 'record') {
  const [binding, dir, names = 'short,6k,32k', tokens = '1024'] = rest;
  if (!binding || !dir || rest.length > 4) throw new Error(usage);
  run(binding, resolve(dir), 'record', names, tokens, { MLX_DFLASH2_TF_RECORD: resolve(dir) });
} else if (command === 'force') {
  const [binding, dir, label, names = 'short,6k,32k', tokens = '1024'] = rest;
  if (!binding || !dir || !label || rest.length > 5) throw new Error(usage);
  run(binding, resolve(dir), label, names, tokens, { MLX_DFLASH2_TF_DIR: resolve(dir), MLX_DFLASH2_TF_LABEL: label });
  summary(resolve(dir), label, tokens);
} else if (command === 'summary') {
  const [dir, label, tokens = '1024'] = rest;
  if (!dir || !label || rest.length > 3) throw new Error(usage);
  summary(resolve(dir), label, tokens);
} else {
  throw new Error(usage);
}
