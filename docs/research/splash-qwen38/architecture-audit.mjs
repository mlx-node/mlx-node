// Offline inventory only: this script never loads model tensors or runs the GPU.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const [artifactArg, splashArg] = process.argv.slice(2);
assert(artifactArg && splashArg, 'usage: node architecture-audit.mjs ARTIFACT_DIR SPLASH_DIR');
const artifacts = resolve(artifactArg);
const splash = resolve(splashArg);
const sources = {};
function read(path) {
  const bytes = readFileSync(path);
  sources[path] = createHash('sha256').update(bytes).digest('hex');
  return bytes.toString('utf8');
}
const header = JSON.parse(read(resolve(artifacts, 'native-weight-header.json')));
const nativeConfig = JSON.parse(read(resolve(artifacts, 'native-model-config.json')));
const config = nativeConfig.text_config;
const manifest = JSON.parse(read(resolve(splash, 'install/models/incoai/Qwen3.8-27B-Splash/manifest.json')));
const bytesFor = (predicate) =>
  Object.entries(header)
    .filter(([key]) => key !== '__metadata__' && predicate(key))
    .reduce((sum, [, value]) => {
      const [start, end] = value.data_offsets;
      assert(Number.isSafeInteger(start) && Number.isSafeInteger(end) && end >= start);
      return sum + end - start;
    }, 0);
const splashBytesFor = (predicate) =>
  manifest.artifacts.filter((entry) => predicate(entry.path)).reduce((sum, entry) => sum + entry.size, 0);
const targetBytes = bytesFor((key) => key.startsWith('model.layers.') || key.startsWith('lm_head.'));
const splashTargetBytes = splashBytesFor((path) => path.startsWith('target/') && path !== 'target/embedding.bin');
const layers = config.layer_types.filter((kind) => kind === 'full_attention').length;
const gqa = config.num_attention_heads / config.num_key_value_heads;
assert.equal(config.torch_dtype, 'bfloat16');
assert.equal(layers, 16);
assert.equal(gqa, 6);
const rows = 8;
const tail = Math.min(Math.floor(32 / gqa), rows - 1);
const quant = nativeConfig.quantization;
let sameFormatGateUpPairs = 0;
for (let layer = 0; layer < config.num_hidden_layers; layer++) {
  const prefix = `language_model.model.layers.${layer}.mlp.`;
  const gate = quant[`${prefix}gate_proj`] ?? quant;
  const up = quant[`${prefix}up_proj`] ?? quant;
  if (['mode', 'bits', 'group_size'].every((key) => gate[key] === up[key])) sameFormatGateUpPairs++;
}
assert.equal(sameFormatGateUpPairs, 39);
const bytesPerPrefixToken = layers * config.num_key_value_heads * config.head_dim * 2 * 2;
const prefixCopies = [87, 6219, 32768].map((tokens) => ({
  tokens,
  allLayerPrefixBytes: tokens * bytesPerPrefixToken,
  // The full block and truncated head each concatenate K and V. Each copy
  // reads and writes the prefix once; subsequent attention reads are excluded.
  doubleConcatReadWriteBytes: tokens * bytesPerPrefixToken * 4,
  idealMillisecondsAt614GBps: (tokens * bytesPerPrefixToken * 4 * 1000) / 614e9,
}));
const tracePath = resolve(artifacts, '../splash-qwen38-phase2/phase-profile.log');
const lines = read(tracePath).split('\n');
const segments = [];
let start = null;
for (const [index, line] of lines.entries()) {
  if (line.startsWith('[dflash2-phase] draft+selector:')) start = index;
  if (!line.startsWith('[dflash2-phase] verify:') || start === null) continue;
  const primitives = {};
  const submissions = {};
  let zeroPrimitiveFinalizations = 0;
  for (const item of lines.slice(start + 1, index)) {
    if (item.startsWith('[metal-op] ')) {
      const name = item.slice('[metal-op] '.length);
      primitives[name] = (primitives[name] ?? 0) + 1;
    }
    const event = /^\[metal-eval\] (\w+): (\d+) ops/.exec(item);
    if (event) {
      submissions[event[1]] = (submissions[event[1]] ?? 0) + 1;
      if (event[1] === 'finalize' && Number(event[2]) === 0) zeroPrimitiveFinalizations++;
    }
  }
  segments.push({ startLine: start + 1, endLine: index + 1, primitives, submissions, zeroPrimitiveFinalizations });
  start = null;
}
assert.equal(segments.length, 33, 'expected the retained trace, including warmup and request tail');
const representative = segments[Math.floor(segments.length / 2)];
assert.equal(representative.primitives.QuantizedMatmul, 383);
assert.equal(representative.submissions.commit, 80);
const summary = {
  sources,
  inventory: {
    targetLayerAndHeadBytes: targetBytes,
    splashTargetLayerAndHeadArtifactBytes: splashTargetBytes,
    ratio: targetBytes / splashTargetBytes,
    splashDraftArtifactBytes: splashBytesFor((path) => path.startsWith('draft/')),
    caveat:
      'Stored tensors/artifacts, not measured per-cycle DRAM traffic. Splash includes alignment; embeddings, vision, and native MTP are excluded from the target comparison.',
  },
  attention: {
    fullAttentionLayers: layers,
    rows,
    gqa,
    headRows: rows - tail,
    tailRows: tail,
    prefixCopies,
    caveat:
      'Logical BF16 tensor copy volume for the compiled split verifier. Peak-bandwidth arithmetic is an ideal lower time estimate, not measured latency or a promised saving.',
  },
  fusion: {
    sameFormatGateUpPairs,
    mixedFormatGateUpPairs: config.num_hidden_layers - sameFormatGateUpPairs,
    sameFormatGateUpIntermediateReadWriteBytes: sameFormatGateUpPairs * rows * config.intermediate_size * 2 * 2 * 2,
    allGateUpIntermediateReadWriteBytes: config.num_hidden_layers * rows * config.intermediate_size * 2 * 2 * 2,
    downResidualIntermediateReadWriteBytes: config.num_hidden_layers * rows * config.hidden_size * 2 * 2,
    caveat:
      'Logical avoided intermediate traffic only. Quantization metadata compatibility does not replace runtime merge eligibility checks or a measured fused-kernel gain.',
  },
  trace: {
    segmentCount: segments.length,
    representativePrimitiveEvents: Object.values(representative.primitives).reduce((sum, count) => sum + count, 0),
    representative,
    caveat:
      'Instrumented historical trace, with extra evaluation boundaries. Primitive events are not dispatch counts; submissions are not host waits. gpu_total is asynchronous, so this script does not estimate GPU idle time.',
  },
};
writeFileSync(resolve(artifacts, 'architecture-audit.json'), `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
