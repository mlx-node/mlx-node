import { readFile, stat } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { parseRerankBody, rerankWireResponse } from '@mlx-node/rerank/protocol';
import { resolveModelsDir } from '@mlx-node/server/host/paths';
export async function run(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      model: { type: 'string', short: 'm' },
      input: { type: 'string', short: 'i' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: true,
  });
  if (values.help) {
    console.log(
      'Usage: mlx rerank --model <path|local-name> --input <JSON-file|->\nJSON fields: query, documents, top_n?, return_documents?, truncate?',
    );
    return;
  }
  if (!values.model || !values.input) throw new Error('--model and --input are required');
  let modelPath = resolve(values.model);
  try {
    await stat(modelPath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
    modelPath = join(resolveModelsDir(), values.model);
  }
  let text: string;
  if (values.input === '-') {
    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
    text = Buffer.concat(chunks).toString('utf8');
  } else text = await readFile(values.input, 'utf8');
  const body: unknown = JSON.parse(text);
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error('Expected JSON object');
  const { request } = parseRerankBody({ ...body, model: values.model });
  const { loadReranker } = await import('@mlx-node/rerank');
  const model = await loadReranker(modelPath);
  const controller = new AbortController();
  const abort = () => controller.abort(new Error('Interrupted'));
  process.once('SIGINT', abort);
  process.once('SIGTERM', abort);
  try {
    const result = await model.rerank({ ...request, signal: controller.signal });
    console.log(JSON.stringify(rerankWireResponse(values.model, result)));
  } finally {
    process.off('SIGINT', abort);
    process.off('SIGTERM', abort);
    await model.dispose();
  }
}
