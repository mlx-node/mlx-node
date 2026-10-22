/** Real inference cancellation, distinct from cancellation during tokenization. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import { XlmRobertaRerankerModel, getMemorySnapshot } from '../../../packages/core/index.cjs';

const [modelPath, output] = process.argv.slice(2);
if (!output) throw Error('Usage: node cancellation.mjs MODEL REPORT');
const memory = () =>
  Object.fromEntries(Object.entries(getMemorySnapshot()).map(([key, value]) => [key, Number(value)]));
const hash = async (path) => {
  const digest = createHash('sha256');
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest('hex');
};
const addonPath = process.env.NAPI_RS_NATIVE_LIBRARY_PATH ?? resolve('packages/core/mlx-core.darwin-arm64.node');
const before = memory();
const model = await XlmRobertaRerankerModel.load(modelPath);
const resident = memory();
const modelInfo = JSON.parse(model.infoJson);
const short = JSON.stringify({ query: 'query', documents: ['cat', 'cat cat'] });
// The pinned tokenizer makes four 2,048-token pairs. All token/position/type/mask
// inputs together occupy 128 KiB; 1 MiB beyond resident weights requires an
// forward activation allocation, not just tokenizer host allocations. Allocation
// precedes command submission; this does not claim to interrupt an active kernel.
const request = JSON.stringify({ query: 'query', documents: Array(4).fill('cat '.repeat(2042)) });
const rows = [];
async function recover() {
  const task = model.startRerank(short);
  try {
    await task.result();
  } finally {
    task.cancel();
  }
}
function pending(input) {
  const task = model.startRerank(input);
  let settled = false;
  const result = task.result().then(
    () => {
      settled = true;
      return false;
    },
    (error) => {
      settled = true;
      assert.match(error.message, /rerank_cancelled/);
      return true;
    },
  );
  void result.catch(() => {}); // Every owner awaits this result during cleanup.
  return { task, result, settled: () => settled };
}
async function observedInference(job, baseline) {
  const started = performance.now();
  while (!job.settled()) {
    const snapshot = memory();
    if (snapshot.activeBytes > baseline.activeBytes + 1024 * 1024)
      return { observedAfterMs: performance.now() - started, activeBytes: snapshot.activeBytes };
    assert.ok(performance.now() - started < 10000, 'No inference allocation observed');
    await new Promise((done) => setTimeout(done, 5));
  }
  throw Error('Request completed before inference cancellation could be observed');
}
try {
  await recover();
  for (let i = 0; i < 8; i++) {
    const baseline = memory();
    const job = pending(request);
    try {
      const observation = await observedInference(job, baseline);
      const at = performance.now();
      job.task.cancel();
      assert.equal(await job.result, true);
      const cancellationMs = performance.now() - at;
      await recover();
      assert.equal(memory().activeBytes, baseline.activeBytes);
      rows.push({ baseline, ...observation, cancellationMs, afterRecovery: memory() });
    } finally {
      job.task.cancel();
      await job.result;
    }
  }
  const baseline = memory();
  const active = pending(request);
  const queued = pending(short);
  try {
    await observedInference(active, baseline);
    await Promise.all([model.dispose(), model.dispose()]);
    assert.equal(await active.result, true);
    assert.equal(await queued.result, true);
  } finally {
    active.task.cancel();
    queued.task.cancel();
    await Promise.all([active.result, queued.result]);
  }
} finally {
  await model.dispose();
}
const after = memory();
assert.equal(after.activeBytes, before.activeBytes);
const checksums = {};
for (const name of (await readdir(modelPath)).sort()) {
  if (name.endsWith('.json') || name.endsWith('.safetensors')) checksums[name] = await hash(join(modelPath, name));
}
await writeFile(
  output,
  JSON.stringify(
    {
      modelPath: resolve(modelPath),
      node: process.version,
      scriptSha256: await hash(new URL(import.meta.url)),
      modelInfo,
      checksums,
      config: JSON.parse(await readFile(join(modelPath, 'config.json'), 'utf8')),
      addonSha256: await hash(addonPath),
      metalSha256: await hash(join(dirname(addonPath), 'mlx.metallib')),
      before,
      resident,
      rows,
      after,
      forwardAllocationCancellationObserved: true,
      queuedAndActiveShutdownJoined: true,
    },
    null,
    2,
  ) + '\n',
);
