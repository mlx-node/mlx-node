/** Real-weight SDK/HTTP/CLI score identity and BGE-only serving. */
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';

import { loadReranker } from '@mlx-node/rerank';
import { rerankWireResponse } from '@mlx-node/rerank/protocol';

import { createInferenceHost } from '../../../packages/server/src/host/index.js';

const [path, output] = process.argv.slice(2);
if (!output) throw Error('Usage: oxnode surfaces.ts MODEL REPORT');
const modelPath = resolve(path);
const root = await mkdtemp(join(tmpdir(), 'mlx-bge-surfaces-'));
await symlink(modelPath, join(root, 'bge'));
const body = {
  model: 'bge',
  query: '中国的首都是哪里？',
  documents: ['北京是中国的首都。', 'Paris is the capital of France.', '', '<pad>', '北京是中国的首都。'],
  return_documents: true,
};
const model = await loadReranker(modelPath);
let expected;
try {
  expected = rerankWireResponse(
    'bge',
    await model.rerank({ query: body.query, documents: body.documents, returnDocuments: true }),
  );
} finally {
  await model.dispose();
}
const host = await createInferenceHost({
  modelsDir: root,
  port: 0,
  disableStore: true,
  sweepOrphanTempRoots: false,
  logDir: join(root, 'logs'),
});
try {
  assert.equal(host.models.length, 0);
  const response = await fetch(host.url + '/v1/rerank', { method: 'POST', body: JSON.stringify(body) });
  assert.equal(response.status, 200);
  const http = await response.json();
  assert.deepEqual(http, expected);
  const wrong = await fetch(host.url + '/v1/responses', {
    method: 'POST',
    body: JSON.stringify({ model: 'bge', input: 'hello' }),
  });
  assert.equal(wrong.status, 400);
  assert.equal((await wrong.json()).error.code, 'model_capability_mismatch');
} finally {
  await host.close();
}
try {
  const input = join(root, 'request.json');
  const { model: _, ...request } = body;
  await writeFile(input, JSON.stringify(request));
  const { stdout } = await promisify(execFile)(
    process.execPath,
    ['packages/cli/dist/cli.js', 'rerank', '--model', modelPath, '--input', input],
    { maxBuffer: 4 * 1024 * 1024 },
  );
  const cli = JSON.parse(stdout);
  assert.deepEqual({ ...cli, model: 'bge' }, expected);
  assert.ok(expected.results.every((r) => r.relevance_score >= 0 && r.relevance_score <= 1));
  assert.ok(!/logit/i.test(JSON.stringify(expected)));
  await writeFile(
    output,
    JSON.stringify(
      {
        modelPath,
        sdkHttpCliIdentical: true,
        bgeOnlyServing: true,
        chatRejected: true,
        normalizedOnly: true,
        scores: expected.results.map(({ index, relevance_score }) => ({ index, relevance_score })),
        usage: expected.usage,
        meta: expected.meta,
      },
      null,
      2,
    ) + '\n',
  );
} finally {
  await rm(root, { recursive: true, force: true });
}
