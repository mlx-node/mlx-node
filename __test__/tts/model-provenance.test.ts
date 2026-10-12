import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, expect, it } from 'vite-plus/test';

import { readModelProvenance } from '../../scripts/tts/model-provenance.js';

const directories: string[] = [];
function checkpoint() {
  const directory = mkdtempSync(join(tmpdir(), 'tts-provenance-'));
  directories.push(directory);
  return directory;
}
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

it('reads explicit source provenance before the ordinary download marker', async () => {
  const directory = checkpoint();
  const source = { repo: 'Qwen/source', revision: 'a'.repeat(40) };
  writeFileSync(join(directory, 'reference-revision.json'), JSON.stringify(source));
  writeFileSync(
    join(directory, '.mlx-download-complete.json'),
    JSON.stringify({ repo: 'community/converted', revision: 'b'.repeat(40) }),
  );
  expect(await readModelProvenance(directory)).toEqual(source);
});

it('reads standard download provenance without copying marker bookkeeping', async () => {
  const directory = checkpoint();
  const source = { repo: 'Qwen/download', revision: 'a'.repeat(40) };
  writeFileSync(
    join(directory, '.mlx-download-complete.json'),
    JSON.stringify({ ...source, scope: 'full', files: ['config.json'], completedAt: new Date().toISOString() }),
  );
  expect(await readModelProvenance(directory)).toEqual(source);
});

it('reports unknown provenance for an ordinary checkpoint without metadata', async () => {
  expect(await readModelProvenance(checkpoint())).toBeNull();
});

it.each(['reference-revision.json', '.mlx-download-complete.json'])('rejects damaged JSON in %s', async (name) => {
  const directory = checkpoint();
  writeFileSync(join(directory, '.mlx-download-complete.json'), JSON.stringify({ repo: 'Qwen/model', revision: 'a' }));
  writeFileSync(join(directory, name), '{broken');
  await expect(readModelProvenance(directory)).rejects.toBeInstanceOf(SyntaxError);
});

it('rejects malformed provenance instead of pretending it is missing', async () => {
  const directory = checkpoint();
  writeFileSync(join(directory, 'reference-revision.json'), JSON.stringify({ repo: 'Qwen/model', revision: null }));
  await expect(readModelProvenance(directory)).rejects.toThrow('Invalid model provenance');
});

it('does not hide filesystem errors other than absent files', async () => {
  const directory = checkpoint();
  mkdirSync(join(directory, 'reference-revision.json'));
  await expect(readModelProvenance(directory)).rejects.toMatchObject({ code: 'EISDIR' });
});
