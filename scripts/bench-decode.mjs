#!/usr/bin/env node
// bench-decode.mjs — headless decode benchmark for the WebGPU browser backend.
//
// Drives the existing demo with Playwright, fires N chats, captures the
// `[profile]` output that the demo already emits with `?profile=1`, and reports
// median / p5 / p95 tok/s plus mean per-opcode RPC counts.
//
// Prerequisites (run these in separate terminals once):
//   vp install
//   vp build                                        # builds WASM blob + demo
//   vp preview --port 5173                          # serves dist/
//   node scripts/bench-decode.mjs                   # or with options below
//
// Options (all env-overridable):
//   URL=http://localhost:5173           # demo URL; bench appends ?profile=1
//   RUNS=8                              # number of decode samples
//   PROMPT="Write a haiku about atoms." # prompt fired for each run
//   WARMUP=1                            # burn this many runs before collecting
//   HEADED=0                            # 1 to show the browser window
//   TIMEOUT_MS=180000                   # per-run timeout
//
// Output: pretty table + a JSON blob the calling commit can paste into the
// commit message. Stats preserved: tok/s median+p5+p95, gpuRPCs/tok median,
// pool hit rate, bind-group cache hit rate, spin hit rate, per-opcode means.

import { chromium } from 'playwright';

const URL_BASE = process.env.URL ?? 'http://localhost:5173';
const RUNS = Number(process.env.RUNS ?? 8);
const WARMUP = Number(process.env.WARMUP ?? 1);
const HEADED = process.env.HEADED === '1';
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS ?? 180_000);
const PROMPT = process.env.PROMPT ?? 'Write a short haiku about atoms.';

// Regexes matched against the demo's log panel output.
// Source: packages/browser/demo/app.ts 'profile' case.
const RE_DECODE_LINE = /^\[profile\] Decode (\d+) tok \| dispatches=(\d+) .* gpuRPCs=(\d+) \(([\d.]+)\/tok\)/;
const RE_POOL = /^\[profile\] pool: hits=(\d+) .* misses=(\d+) .* hitRate=([\d.]+)%/;
const RE_GPU_POOL = /^\[profile\] gpu-pool: hits=(\d+) .* misses=(\d+) .* hitRate=([\d.]+)%/;
const RE_BG_CACHE = /^\[profile\] bg-cache: hits=(\d+) .* misses=(\d+) .* hitRate=([\d.]+)%/;
const RE_SPIN = /^\[profile\] spin: hits=(\d+) misses=(\d+) hitRate=([\d.]+)% budget=(\d+)/;
const RE_OPCODES = /^\[profile\] opcodes: (.+)$/;
// Status bar reports tok/s via setStatus('... | Decode XX.X tok/s', 'ready')
// — keep a second source in case the profile line is truncated.
const RE_TOKPS_STATUS = /Decode ([\d.]+) tok\/s/;

function percentile(sortedNums, p) {
  if (sortedNums.length === 0) return 0;
  const idx = Math.min(sortedNums.length - 1, Math.max(0, Math.floor(p * sortedNums.length)));
  return sortedNums[idx];
}

function parseOpcodes(str) {
  // "matmul(123, 5.0/t), softmax(45, 2.0/t), ..."
  const out = {};
  for (const part of str.split(/, /)) {
    const m = /^(\w+)\((\d+),/.exec(part);
    if (m) out[m[1]] = Number(m[2]);
  }
  return out;
}

async function runOnce(page, runIdx, isWarmup) {
  const tag = isWarmup ? `warmup ${runIdx}` : `run ${runIdx}`;

  // Clear log panel by reloading. Reloading also resets bridgeStub state
  // and the gpu-worker handle table — keeps runs independent.
  await page.goto(`${URL_BASE}/?profile=1`, { waitUntil: 'domcontentloaded' });

  // Wait for model to load and Ready status.
  await page.waitForFunction(() => document.getElementById('status')?.textContent?.includes('Ready'), {
    timeout: TIMEOUT_MS,
  });

  // Capture all log lines appended after we hit send.
  const logLines = [];
  const statusLines = [];
  let tokPerSec = NaN;

  await page.exposeFunction('__benchOnLog', (line) => {
    logLines.push(line);
  });
  await page.exposeFunction('__benchOnStatus', (text) => {
    statusLines.push(text);
    const m = RE_TOKPS_STATUS.exec(text);
    if (m) tokPerSec = Number(m[1]);
  });

  // Install observers for the log panel and status bar.
  await page.evaluate(() => {
    const logEl = document.getElementById('log');
    const statusEl = document.getElementById('status');
    const logObs = new MutationObserver(() => {
      const children = Array.from(logEl?.children ?? []);
      const last = children[children.length - 1];
      if (last?.textContent) window.__benchOnLog(last.textContent);
    });
    if (logEl) logObs.observe(logEl, { childList: true });
    const statusObs = new MutationObserver(() => {
      window.__benchOnStatus(statusEl?.textContent ?? '');
    });
    if (statusEl) statusObs.observe(statusEl, { childList: true, characterData: true, subtree: true });
  });

  // Fire the prompt.
  await page.fill('#prompt', PROMPT);
  await page.click('#send');

  // Wait for the profile line to arrive (emitted after decode completes).
  const deadline = Date.now() + TIMEOUT_MS;
  let profileLine = null;
  while (Date.now() < deadline) {
    profileLine = logLines.find((l) => RE_DECODE_LINE.test(l));
    if (profileLine) break;
    await page.waitForTimeout(250);
  }
  if (!profileLine) throw new Error(`${tag}: no [profile] line within ${TIMEOUT_MS}ms`);

  // Parse.
  const m1 = RE_DECODE_LINE.exec(profileLine);
  const numTokens = Number(m1[1]);
  const dispatches = Number(m1[2]);
  const gpuRpcs = Number(m1[3]);
  const gpuRpcsPerTok = Number(m1[4]);

  const pool = findAndMatch(logLines, RE_POOL, (m) => ({
    hits: Number(m[1]),
    misses: Number(m[2]),
    rate: Number(m[3]),
  })) ?? { hits: 0, misses: 0, rate: 0 };
  const gpuPool = findAndMatch(logLines, RE_GPU_POOL, (m) => ({
    hits: Number(m[1]),
    misses: Number(m[2]),
    rate: Number(m[3]),
  })) ?? { hits: 0, misses: 0, rate: 0 };
  const bg = findAndMatch(logLines, RE_BG_CACHE, (m) => ({
    hits: Number(m[1]),
    misses: Number(m[2]),
    rate: Number(m[3]),
  })) ?? { hits: 0, misses: 0, rate: 0 };
  const spin = findAndMatch(logLines, RE_SPIN, (m) => ({
    hits: Number(m[1]),
    misses: Number(m[2]),
    rate: Number(m[3]),
    budget: Number(m[4]),
  })) ?? { hits: 0, misses: 0, rate: 0, budget: 0 };
  const opcodes = findAndMatch(logLines, RE_OPCODES, (m) => parseOpcodes(m[1])) ?? {};

  return {
    tag,
    tokPerSec,
    numTokens,
    dispatches,
    gpuRpcs,
    gpuRpcsPerTok,
    pool,
    gpuPool,
    bg,
    spin,
    opcodes,
  };
}

function findAndMatch(lines, re, fn) {
  for (const l of lines) {
    const m = re.exec(l);
    if (m) return fn(m);
  }
  return null;
}

function summarize(label, nums) {
  const sorted = [...nums].sort((a, b) => a - b);
  return {
    label,
    median: percentile(sorted, 0.5).toFixed(2),
    p5: percentile(sorted, 0.05).toFixed(2),
    p95: percentile(sorted, 0.95).toFixed(2),
    min: sorted[0]?.toFixed(2) ?? 'n/a',
    max: sorted[sorted.length - 1]?.toFixed(2) ?? 'n/a',
    n: nums.length,
  };
}

async function main() {
  console.log(`bench-decode: URL=${URL_BASE} runs=${RUNS} warmup=${WARMUP} headed=${HEADED}`);
  const browser = await chromium.launch({
    headless: !HEADED,
    // WebGPU requires specific flags; these are Chromium-only.
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,SharedArrayBuffer',
      '--disable-dawn-features=disallow_unsafe_apis',
    ],
  });
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.error('pageerror:', e.message));

  const results = [];
  for (let i = 0; i < WARMUP; i++) {
    await runOnce(page, i + 1, true);
  }
  for (let i = 0; i < RUNS; i++) {
    const r = await runOnce(page, i + 1, false);
    console.log(
      `  ${r.tag}: ${r.tokPerSec.toFixed(2)} tok/s | ${r.numTokens} tok | ` +
        `gpuRPCs/tok=${r.gpuRpcsPerTok.toFixed(1)} | pool=${r.pool.rate}% | spin=${r.spin.rate}% (budget=${r.spin.budget})`,
    );
    results.push(r);
  }

  const tokps = summarize(
    'tok/s',
    results.map((r) => r.tokPerSec),
  );
  const rpcs = summarize(
    'gpuRPCs/tok',
    results.map((r) => r.gpuRpcsPerTok),
  );
  const poolRate = summarize(
    'pool-hit%',
    results.map((r) => r.pool.rate),
  );
  const spinRate = summarize(
    'spin-hit%',
    results.map((r) => r.spin.rate),
  );
  const spinBudget = summarize(
    'spin-budget',
    results.map((r) => r.spin.budget),
  );

  // Mean per-opcode RPC count across runs.
  const opcodeSums = {};
  for (const r of results) {
    for (const [name, count] of Object.entries(r.opcodes)) {
      opcodeSums[name] = (opcodeSums[name] ?? 0) + count;
    }
  }
  const opcodeMeans = Object.fromEntries(
    Object.entries(opcodeSums).map(([n, s]) => [n, Number((s / results.length).toFixed(1))]),
  );

  console.log('\n=== Summary ===');
  for (const s of [tokps, rpcs, poolRate, spinRate, spinBudget]) {
    console.log(
      `  ${s.label.padEnd(14)} median=${s.median.padStart(8)}  p5=${s.p5.padStart(8)}  p95=${s.p95.padStart(8)}  min=${s.min.padStart(8)}  max=${s.max.padStart(8)}`,
    );
  }
  console.log('\nper-opcode mean (sorted):');
  const opcodesSorted = Object.entries(opcodeMeans).sort((a, b) => b[1] - a[1]);
  for (const [name, mean] of opcodesSorted.slice(0, 25)) {
    console.log(`  ${name.padEnd(40)} ${mean}`);
  }

  // JSON blob for commit-message paste.
  const json = {
    url: URL_BASE,
    runs: RUNS,
    warmup: WARMUP,
    prompt: PROMPT,
    tokps,
    rpcs,
    poolRate,
    spinRate,
    spinBudget,
    opcodeMeans,
  };
  console.log('\n=== JSON (paste into commit) ===');
  console.log(JSON.stringify(json, null, 2));

  await browser.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
