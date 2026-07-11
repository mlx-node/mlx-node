/**
 * Gated end-to-end smoke for `mlx agent`: spawns the REAL built CLI as a
 * SUBPROCESS and asserts it produces a model answer, fully offline.
 *
 * Why a subprocess and not in-process `runAgent`: pi's `main()` calls
 * `process.exit()` on the print path, so driving it in-process would tear
 * down the test worker. The subprocess is exactly what a user runs —
 * `node packages/cli/dist/cli.js agent …` — with stdin ignored so the pi
 * print-mode wrapper never blocks reading it.
 *
 * Availability convention (mirrors `packages/agent/__test__/provider-live.test.ts`):
 * the smallest local qwen3.5 checkpoint, overridable via
 * `MLX_AGENT_TEST_MODEL`. Skips cleanly when no candidate model exists
 * (CI has none) or when the CLI has not been built. GPU work never runs
 * concurrently — the suite already pins `maxWorkers`/`maxConcurrency` to 1.
 */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

const CANDIDATES = [
  process.env.MLX_AGENT_TEST_MODEL,
  join(homedir(), '.mlx-node', 'models', 'qwen3.5-0.8b-mlx-bf16'),
  join(homedir(), '.cache', 'models', 'qwen3.5-0.8b-mlx-bf16'),
].filter((p): p is string => typeof p === 'string' && p.length > 0);

const MODEL_PATH = CANDIDATES.find((p) => existsSync(join(p, 'config.json')));

// The test file lives at <repoRoot>/__test__/cli/agent-e2e.test.ts.
const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
// The built CLI a user's `mlx` bin points at (`tsc -b` output). Absent
// until `yarn build:ts` runs — gate on it so the suite skips clean rather
// than shelling out to a package manager to build on the fly.
const CLI_ENTRY = join(REPO_ROOT, 'packages', 'cli', 'dist', 'cli.js');

// Model load dominates; keep the kill well under the `it` timeout so a
// hang surfaces as a diagnostic rather than an opaque Vitest timeout.
const RUN_TIMEOUT = 240_000;
const EXPECTED = 'hello from mlx agent';

interface CliResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

function runCli(args: string[], env: NodeJS.ProcessEnv): Promise<CliResult> {
  return new Promise((resolve, reject) => {
    // stdin 'ignore' → the pi print-mode wrapper sees EOF immediately and
    // never blocks reading stdin.
    const child = spawn(process.execPath, [CLI_ENTRY, ...args], {
      cwd: REPO_ROOT,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(
        new Error(`mlx agent timed out after ${RUN_TIMEOUT}ms\n--- stdout ---\n${stdout}\n--- stderr ---\n${stderr}`),
      );
    }, RUN_TIMEOUT);

    child.stdout!.setEncoding('utf8');
    child.stderr!.setEncoding('utf8');
    child.stdout!.on('data', (chunk: string) => {
      stdout += chunk;
    });
    child.stderr!.on('data', (chunk: string) => {
      stderr += chunk;
    });
    child.on('error', (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on('close', (code) => {
      clearTimeout(timer);
      resolve({ code, stdout, stderr });
    });
  });
}

describe.skipIf(!MODEL_PATH || !existsSync(CLI_ENTRY))('mlx agent CLI e2e smoke', () => {
  let agentDir: string;

  beforeAll(async () => {
    agentDir = await mkdtemp(join(tmpdir(), 'mlx-agent-e2e-'));
  });

  afterAll(async () => {
    if (agentDir) {
      await rm(agentDir, { recursive: true, force: true });
    }
  });

  it(
    'runs a real print-mode turn to exit 0 with the model answer on stdout',
    async () => {
      const modelsDir = dirname(MODEL_PATH!);
      const modelId = basename(MODEL_PATH!);
      const result = await runCli(
        [
          'agent',
          '--models-dir',
          modelsDir,
          '--model',
          `mlx/${modelId}`,
          '-p',
          `Reply with exactly: ${EXPECTED}`,
          '--no-session',
        ],
        {
          ...process.env,
          MLX_AGENT_AUTO_APPROVE: '1',
          PI_CODING_AGENT_DIR: agentDir,
        },
      );

      expect(result.code, `stderr:\n${result.stderr}\nstdout:\n${result.stdout}`).toBe(0);
      // A 0.8B model is stochastic: it echoes the phrase but may vary case,
      // trailing punctuation, or line-wrapping. This is a smoke test (does the
      // real CLI run a turn offline and answer coherently?), so normalize case
      // and whitespace rather than demand a byte-exact match.
      const normalized = result.stdout.toLowerCase().replace(/\s+/g, ' ');
      expect(normalized, `stdout:\n${result.stdout}`).toContain(EXPECTED);
    },
    RUN_TIMEOUT + 30_000,
  );
});
