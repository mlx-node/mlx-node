import { writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
const [binding, output, revision] = process.argv.slice(2);
if (!binding || !output || !revision) throw new Error('binding output revision required');
const core: typeof import('../../../../packages/core/index.cjs') = createRequire(import.meta.url)(binding);
const runs = [];
for (const size of [32, 32768, 262144]) {
  const array = core.MxArray.fromFloat32(new Float32Array(size).fill(0.25), new BigInt64Array([BigInt(size)]));
  array.eval();
  for (let round = -1; round < 7; round++) {
    const start = performance.now();
    for (let i = 0; i < 200; i++) {
      const values = array.toFloat32();
      if (values.length !== size || values[size - 1] !== 0.25) throw new Error('export mismatch');
    }
    if (round >= 0) runs.push({ size, round, us: ((performance.now() - start) * 1000) / 200 });
  }
}
await writeFile(output, JSON.stringify({ revision, runs }, null, 2));
console.log(JSON.stringify({ revision, runs }));
