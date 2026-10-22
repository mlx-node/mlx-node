/** Convert the two pinned unquantized source checkpoints into the validation matrix. */
import { resolve } from 'node:path';

import { convertModel } from '../../packages/core/index.cjs';
const root = resolve(process.argv[2] ?? '.cache/rerank-validation');
for (const profile of ['v3', 'v35'])
  for (const mode of ['affine', 'mxfp4']) {
    const outputDir = resolve(root, `${profile}-${mode}`);
    const result = await convertModel({
      inputDir: resolve(root, profile),
      outputDir,
      dtype: 'bfloat16',
      quantize: true,
      quantBits: 4,
      quantGroupSize: mode === 'affine' ? 64 : 32,
      quantMode: mode,
    });
    console.log(
      JSON.stringify({ profile, mode, outputDir, tensors: result.numTensors, parameters: result.numParameters }),
    );
  }
