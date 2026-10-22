/** Requires the built addon; failures are exercised before any model inference. */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

for (const script of ['compare.mjs', 'benchmark.mjs'])
  void test(`${script} invalidates an old report when a fresh run fails`, async () => {
    const root = await mkdtemp(join(tmpdir(), 'rerank-report-'));
    try {
      const output = join(root, 'report.json');
      await writeFile(output, JSON.stringify({ passed: true, stale: true }));
      const args = [join(root, 'missing-model')];
      if (script === 'compare.mjs') args.push(join(root, 'missing-reference.json'));
      args.push(output);
      const result = spawnSync(process.execPath, [fileURLToPath(new URL(script, import.meta.url)), ...args], {
        encoding: 'utf8',
      });
      assert.notEqual(result.status, 0);
      await assert.rejects(readFile(output), { code: 'ENOENT' });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

void test('comparison refuses to overwrite or remove its reference', async () => {
  const root = await mkdtemp(join(tmpdir(), 'rerank-reference-'));
  try {
    const reference = join(root, 'reference.json');
    const original = '[]\n';
    await writeFile(reference, original);
    const result = spawnSync(
      process.execPath,
      [fileURLToPath(new URL('./compare.mjs', import.meta.url)), root, reference, reference],
      { encoding: 'utf8' },
    );
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Report must not overwrite the reference/);
    assert.equal(await readFile(reference, 'utf8'), original);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
