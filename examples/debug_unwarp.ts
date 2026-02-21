import { readFileSync, writeFileSync } from 'node:fs';
import { DocUnwarpModel } from '@mlx-node/core';

const imagePath = process.argv[2] || './examples/ocr.png';
const unwarpModelPath = '.cache/models/UVDoc-mlx';

const model = DocUnwarpModel.load(unwarpModelPath);
const imageBuffer = readFileSync(imagePath);
console.log(`Input image: ${imageBuffer.length} bytes`);

const result = model.unwarp(imageBuffer);
console.log(`Output image: ${result.image.length} bytes`);

writeFileSync('/tmp/unwarped_debug.png', result.image);
console.log('Saved to /tmp/unwarped_debug.png');
