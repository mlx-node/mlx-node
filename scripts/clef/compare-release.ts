/// <reference types="node" />

/** Compare verify-release.ts results with the upstream Python reference output. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import type { ClefResult } from '@mlx-node/lm';

interface NativeRun {
  checkpoint: string;
  loadMs: number;
  memory: { active: number; peak: number; cache: number; wiredLimit: number };
  repeat: string;
  midInferenceCancellation: string;
  http: string;
  media: string;
  sdk: string;
  cases: { id: string; milliseconds: number; result: ClefResult; expectedPassed: boolean }[];
}
interface ReferenceRun {
  device: string;
  dtype: string;
  torch: string;
  referenceSha256: string;
  cases: {
    id: string;
    input_tokens: number;
    questions: { id: string; options: string[]; probabilities: number[] }[];
  }[];
}
const native: NativeRun = JSON.parse(await readFile(process.argv[2], 'utf8'));
const reference: ReferenceRun = JSON.parse(await readFile(process.argv[3], 'utf8'));
const marker = JSON.parse(await readFile(join(native.checkpoint, '.mlx-download-complete.json'), 'utf8'));
assert.equal(native.cases.length, reference.cases.length);
let maxProbabilityError = 0;
let comparedProbabilities = 0;
const cases = native.cases.map((run) => {
  const expected = reference.cases.find((test) => test.id === run.id)!;
  assert.ok(expected, `Missing reference case ${run.id}`);
  assert.equal(run.result.usage.input_tokens, expected.input_tokens);
  assert.equal(Object.keys(run.result.answers).length, expected.questions.length);
  let maxError = 0;
  for (const question of expected.questions) {
    const answer = run.result.answers[question.id];
    assert.ok(answer, `Missing question ${question.id}`);
    const actual =
      answer.type === 'noul'
        ? question.options.map((option) => (option === 'true' ? answer.noul : 1 - answer.noul))
        : question.options.map((option) => answer.probabilities[option]);
    assert.ok(actual.every(Number.isFinite));
    assert.equal(
      actual.indexOf(Math.max(...actual)),
      question.probabilities.indexOf(Math.max(...question.probabilities)),
      `Modal decision changed: ${run.id}/${question.id}`,
    );
    for (const [index, probability] of actual.entries()) {
      maxError = Math.max(maxError, Math.abs(probability - question.probabilities[index]));
      comparedProbabilities++;
    }
  }
  maxProbabilityError = Math.max(maxProbabilityError, maxError);
  return {
    id: run.id,
    inputTokens: expected.input_tokens,
    milliseconds: run.milliseconds,
    expectedPassed: run.expectedPassed,
    maxProbabilityError: maxError,
    answers: run.result.answers,
  };
});
const report = {
  repo: marker.repo,
  revision: marker.revision,
  referenceSourceSha256: reference.referenceSha256,
  reference: { torch: reference.torch, device: reference.device, dtype: reference.dtype },
  loadMs: native.loadMs,
  memory: native.memory,
  comparedProbabilities,
  maxProbabilityError,
  repeat: native.repeat,
  midInferenceCancellation: native.midInferenceCancellation,
  http: native.http,
  sdk: native.sdk,
  media: native.media,
  cases,
};
await writeFile(process.argv[4], JSON.stringify(report, null, 2) + '\n');
assert.ok(cases.every((test) => test.expectedPassed));
assert.ok(maxProbabilityError < 0.01, `BF16 probability drift ${maxProbabilityError} exceeds 0.01`);
console.log(
  JSON.stringify({
    repo: marker.repo,
    revision: marker.revision,
    comparedProbabilities,
    maxProbabilityError,
    cases: cases.length,
  }),
);
