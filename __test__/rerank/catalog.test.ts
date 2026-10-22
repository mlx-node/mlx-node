import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

import { expect, it } from 'vite-plus/test';
it('imports catalog and types when native loading is forbidden', async () => {
  const catalog = new URL('../../packages/rerank/src/catalog.ts', import.meta.url).href;
  const types = new URL('../../packages/rerank/src/types.ts', import.meta.url).href;
  const { stdout } = await promisify(execFile)(process.execPath, [
    '--import',
    '@oxc-node/core/register',
    '--input-type=module',
    '-e',
    `process.dlopen = () => { throw new Error('native addon loaded'); }; const catalog = await import(${JSON.stringify(catalog)}); await import(${JSON.stringify(types)}); console.log(catalog.isRerankerConfig({architectures:['JinaForRanking']}));`,
  ]);
  expect(stdout.trim()).toBe('true');
});
