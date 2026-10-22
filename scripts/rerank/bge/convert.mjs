import { join } from 'node:path';

import { convertModel } from '../../../packages/core/index.cjs';
const [source, root] = process.argv.slice(2);
if (!source || !root) throw Error('Usage: node scripts/rerank/bge/convert.mjs SOURCE OUTPUT_ROOT');
for (const [name, dtype, mode] of [
  ['fp16', 'float16'],
  ['bf16', 'bfloat16'],
  ['affine', 'float16', 'affine'],
  ['mxfp4', 'float16', 'mxfp4'],
]) {
  const start = performance.now();
  const result = await convertModel({
    inputDir: source,
    outputDir: join(root, name),
    dtype,
    ...(mode ? { quantize: true, quantBits: 4, quantGroupSize: mode === 'affine' ? 64 : 32, quantMode: mode } : {}),
  });
  console.log(
    JSON.stringify({
      name,
      milliseconds: performance.now() - start,
      numTensors: result.numTensors,
      numParameters: result.numParameters,
      outputPath: result.outputPath,
    }),
  );
}
