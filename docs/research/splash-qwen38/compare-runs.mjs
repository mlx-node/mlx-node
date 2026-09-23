// Compare matched uninstrumented benchmark artifacts, including output parity.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const [baselinePath, candidatePath, outputPath] = process.argv.slice(2);
assert(baselinePath && candidatePath && outputPath, 'usage: node compare-runs.mjs BASELINE CANDIDATE OUTPUT');
const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
const candidate = JSON.parse(readFileSync(candidatePath, 'utf8'));
assert.equal(candidate.target, baseline.target);
assert.equal(candidate.draft, baseline.draft);
assert.deepEqual(candidate.config, baseline.config);
assert.equal(candidate.records.length, baseline.records.length);
for (const artifact of [baseline, candidate]) {
  for (const key of ['MLX_METAL_COMMAND_TRACE', 'MLX_METAL_OP_TRACE', 'MLX_DFLASH2_PHASE_TIME']) {
    assert(!artifact.environment[key], `instrumented result: ${key}`);
  }
}
const comparisons = baseline.records.map((before) => {
  const matches = candidate.records.filter((row) => row.name === before.name && row.run === before.run);
  assert.equal(matches.length, 1, `missing or duplicate ${before.name}/${before.run}`);
  const after = matches[0];
  for (const key of [
    'inputHash',
    'outputHash',
    'mode',
    'numTokens',
    'promptTokens',
    'reasoningTokens',
    'finishReason',
    'cachedTokens',
  ]) {
    assert.equal(after[key], before[key], `${before.name}/${before.run}: ${key}`);
  }
  for (const key of ['mtpCycles', 'mtpMeanAcceptedTokensTotal', 'mtpMeanDepth']) {
    assert.equal(after.performance[key], before.performance[key], `${before.name}/${before.run}: ${key}`);
  }
  assert.deepEqual(after.performance.mtpAcceptanceByPosition, before.performance.mtpAcceptanceByPosition);
  const beforeRate = before.performance.decodeTokensPerSecond;
  const afterRate = after.performance.decodeTokensPerSecond;
  assert(beforeRate > 0 && afterRate > 0, 'missing decode rate');
  return {
    name: before.name,
    run: before.run,
    outputHash: before.outputHash,
    baselineDecodeTokensPerSecond: beforeRate,
    candidateDecodeTokensPerSecond: afterRate,
    decodeChangePercent: 100 * (afterRate / beforeRate - 1),
    wallChangePercent: 100 * (after.wallMs / before.wallMs - 1),
    baselineTtftMs: before.performance.ttftMs,
    candidateTtftMs: after.performance.ttftMs,
  };
});
const result = { baselinePath, candidatePath, comparisons };
writeFileSync(outputPath, `${JSON.stringify(result, null, 2)}\n`);
console.log(JSON.stringify(result, null, 2));
