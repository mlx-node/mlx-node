/**
 * Smoke test for the `mlx redact` CLI command.
 *
 * Imports the command's `run()` directly and invokes it with a synthetic
 * argv — sidesteps any dependency on `packages/cli/dist/` being built and
 * avoids capturing process-level stdin/stdout. We always pass `--input`
 * and `--output` so all I/O goes through files.
 *
 * Gated on `PRIVACY_FILTER_MODEL_DIR` so CI without weights stays green
 * (same convention as `packages/privacy/__test__/load.test.ts`).
 */
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { run as runRedact } from '../src/commands/redact.js';

const MODEL_DIR = process.env.PRIVACY_FILTER_MODEL_DIR;
const modelAvailable = !!MODEL_DIR && existsSync(MODEL_DIR);

describe.skipIf(!modelAvailable)('mlx redact CLI', () => {
  it('redacts entities from --input to --output and emits a JSON sidecar when --json is set', async () => {
    const tmp = mkdtempSync(join(tmpdir(), 'mlx-redact-cmd-'));
    const inputPath = join(tmp, 'input.txt');
    const outputPath = join(tmp, 'redacted.txt');
    const entitiesPath = `${outputPath}.entities.json`;
    writeFileSync(inputPath, "Hi, I'm Harry Potter, email: harry@hogwarts.edu", 'utf-8');

    await runRedact(['--model', MODEL_DIR!, '--input', inputPath, '--output', outputPath, '--json']);

    const redacted = readFileSync(outputPath, 'utf-8');
    expect(redacted).toContain('[private_person]');
    expect(redacted).toContain('[private_email]');
    expect(redacted).not.toContain('Harry Potter');
    expect(redacted).not.toContain('harry@hogwarts.edu');

    const entities = JSON.parse(readFileSync(entitiesPath, 'utf-8')) as Array<{ label: string }>;
    expect(entities.length).toBeGreaterThan(0);
    const labels = entities.map((e) => e.label).sort();
    expect(labels).toContain('private_person');
    expect(labels).toContain('private_email');
  });
});
