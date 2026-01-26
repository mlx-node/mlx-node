/**
 * Execution Helper for GitHub Tool Training
 *
 * Executes model-generated code in a subprocess with:
 * - Mock Octokit that asserts params match expected patterns
 * - Mock git context (owner, repo, branch)
 * - Timeout protection with SIGKILL fallback
 * - Assertion result capturing
 *
 * ISOLATION LIMITATIONS:
 * This is NOT a security sandbox. The subprocess:
 * - Inherits the parent's environment variables
 * - Has full filesystem read/write access
 * - HTTP_PROXY vars discourage but don't prevent network access
 *
 * Only suitable for local experimentation with trusted or reviewed code.
 * For untrusted code, use proper isolation (containers, VMs, etc.).
 */

import { spawn } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync, mkdirSync, cpSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

import { getScenarioSnapshot, serializeSnapshot, type ScenarioSnapshot } from './scenario-snapshots.js';

const require = createRequire(import.meta.url);

const oxnode = dirname(require.resolve('@oxc-node/core'));
const oxnodeNative = dirname(
  require.resolve(`@oxc-node/core-${process.platform}-${process.arch}`, {
    paths: [oxnode],
  }),
);
const pirates = dirname(
  require.resolve('pirates', {
    paths: [oxnode],
  }),
);

// Resolve paths
const __dirname = fileURLToPath(new URL('.', import.meta.url));
const MOCK_OCTOKIT_SRC = join(__dirname, 'mock-octokit');
const MOCK_SIMPLE_GIT_SRC = join(__dirname, 'mock-simple-git');

export interface ResultValidation {
  valid: boolean;
  error?: string;
  missing?: string[];
  undefinedKeys?: string[];
  isErrorCase?: boolean;
  skipped?: boolean;
}

export interface AssertionResult {
  matched: boolean;
  callIndex: number;
  method: string;
  expected: {
    method: string;
    params: Record<string, unknown>;
  } | null;
  actual: {
    method: string;
    params: Record<string, unknown>;
  };
  mismatches?: Array<{
    key: string;
    expected: unknown;
    actual: unknown;
  }>;
  error?: string;
}

export interface ExecutionResult {
  /** Whether execution completed without errors */
  success: boolean;
  /** Result from the default export function */
  result?: unknown;
  /** Error message if execution failed */
  error?: string;
  /** Type of error */
  errorType?: 'syntax' | 'runtime' | 'timeout' | 'wrong_api' | 'no_snapshot';
  /** Execution time in milliseconds */
  executionTimeMs: number;
  /** Assertion results from mock octokit */
  assertions?: AssertionResult[];
  /** Number of actual API calls made by the model */
  callCount?: number;
  /** Number of expected API calls (including optional) */
  expectedCallCount?: number;
  /** Number of required (non-optional) expected calls */
  requiredCallCount?: number;
  /** Number of required calls that were successfully matched */
  matchedRequiredCalls?: number;
  /** Whether all assertions matched (no param mismatches) */
  allAssertionsMatched?: boolean;
  /** Whether all required calls were made */
  allRequiredCallsMade?: boolean;
  /** Result validation against expected return keys */
  resultValidation?: ResultValidation;
}

export interface ExecutionOptions {
  /** Timeout in milliseconds (default: 5000) */
  timeoutMs?: number;
  /** Mock git context */
  context?: {
    owner: string;
    repo: string;
    branch: string;
  };
}

const DEFAULT_CONTEXT = {
  owner: 'test-owner',
  repo: 'test-repo',
  branch: 'test-feature',
};

/**
 * Execute model-generated code with mock octokit.
 *
 * @param code - The generated JavaScript code to execute
 * @param scenarioId - The scenario ID (e.g., 'get-inline-comments' or 'get-inline-comments-0')
 * @param options - Execution options
 */
export async function executeGitHubCode(
  code: string,
  scenarioId: string,
  options: ExecutionOptions = {},
): Promise<ExecutionResult> {
  const startTime = Date.now();
  const timeoutMs = options.timeoutMs ?? 5000;
  const context = options.context ?? DEFAULT_CONTEXT;

  // Get snapshot for this scenario
  const snapshot = getScenarioSnapshot(scenarioId);
  if (!snapshot) {
    return {
      success: false,
      error: `No snapshot found for scenario: ${scenarioId}`,
      errorType: 'no_snapshot',
      executionTimeMs: Date.now() - startTime,
    };
  }

  // Create temp directory
  const tempDir = mkdtempSync(join(tmpdir(), 'github-exec-'));

  try {
    // Set up directory structure:
    // tempDir/
    //   node_modules/
    //     octokit/         <- Copy mock-octokit here
    //     @napi-rs/        <- Symlink from project
    //   _generated.mjs    <- Model's code (modified)
    //   _wrapper.mjs      <- Wrapper that runs the code

    const nodeModulesDir = join(tempDir, 'node_modules');
    mkdirSync(nodeModulesDir, { recursive: true });

    // Copy mock octokit
    const mockOctokitDest = join(nodeModulesDir, 'octokit');
    cpSync(MOCK_OCTOKIT_SRC, mockOctokitDest, { recursive: true });

    // Copy mock @napi-rs/simple-git package
    const napiDir = join(nodeModulesDir, '@napi-rs');
    mkdirSync(napiDir, { recursive: true });
    cpSync(MOCK_SIMPLE_GIT_SRC, join(napiDir, 'simple-git'), { recursive: true });

    const oxnodeDir = join(nodeModulesDir, '@oxc-node');
    mkdirSync(join(oxnodeDir, 'core'), { recursive: true });
    mkdirSync(join(oxnodeDir, `core-${process.platform}-${process.arch}`), { recursive: true });
    cpSync(oxnode, join(oxnodeDir, 'core'), { recursive: true });
    cpSync(oxnodeNative, join(oxnodeDir, `core-${process.platform}-${process.arch}`), { recursive: true });

    mkdirSync(join(nodeModulesDir, 'pirates'), { recursive: true });
    cpSync(pirates, join(nodeModulesDir, 'pirates'), { recursive: true });

    // Modify code to inject mock git context
    // Replace the git setup section with mock values
    const modifiedCode = injectMockGitContext(code, context);

    // Write generated code
    writeFileSync(join(tempDir, '_generated.mjs'), modifiedCode);

    // Write utils.mjs that provides mock context (resolves `import { owner, repo, ... } from './utils'`)
    const utilsCode = `
export const owner = '${context.owner}';
export const repo = '${context.repo}';
export const currentBranch = '${context.branch}';
export { Octokit } from 'octokit';
import { Octokit } from 'octokit';
export const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });
`;
    writeFileSync(join(tempDir, 'utils.mjs'), utilsCode);

    // Write wrapper that executes the code and captures result
    const wrapper = `
import fn from './_generated.mjs';

try {
  const result = await fn();
  // The mock octokit outputs results via beforeExit handler
  // Store result for it to include
  if (typeof global !== 'undefined') {
    global.__EXECUTION_RESULT__ = result;
  }
} catch (error) {
  console.error('Execution error:', error.message);
  process.exit(1);
}
`;
    writeFileSync(join(tempDir, '_wrapper.mjs'), wrapper);

    // Execute with timeout
    return await executeWithTimeout(tempDir, snapshot, context, timeoutMs, startTime);
  } finally {
    // Cleanup
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Inject mock git context into the code.
 * Replaces boilerplate git setup with mock values.
 *
 * The boilerplate prepends code like:
 *   const [owner, repo] = parseGitHub(remoteUrl);
 *   const branch = _repo.head().shorthand();
 *
 * We replace these with direct mock assignments.
 */
function injectMockGitContext(code: string, context: { owner: string; repo: string; branch: string }): string {
  let modifiedCode = code;

  // Replace the destructuring assignment for owner/repo (lowercase)
  // Pattern: const [owner, repo] = parseGitHub(remoteUrl);
  modifiedCode = modifiedCode.replace(
    /const \[owner, repo\] = parseGitHub\(remoteUrl\);/g,
    `const owner = '${context.owner}';\nconst repo = '${context.repo}';`,
  );

  // Replace branch extraction (new pattern)
  // Pattern: const branch = _repo.head().shorthand();
  modifiedCode = modifiedCode.replace(
    /const branch = _repo\.head\(\)\.shorthand\(\);/g,
    `const branch = '${context.branch}';`,
  );

  // Also handle any other calls to _repo.head().shorthand()
  modifiedCode = modifiedCode.replace(/_repo\.head\(\)\.shorthand\(\)/g, `'${context.branch}'`);

  return modifiedCode;
}

/**
 * Execute code in subprocess with timeout
 */
async function executeWithTimeout(
  tempDir: string,
  snapshot: ScenarioSnapshot,
  context: { owner: string; repo: string; branch: string },
  timeoutMs: number,
  startTime: number,
): Promise<ExecutionResult> {
  return new Promise((resolve) => {
    const child = spawn('node', ['--import', '@oxc-node/core/register', '_wrapper.mjs'], {
      cwd: tempDir,
      env: {
        ...process.env,
        SCENARIO_SNAPSHOT: serializeSnapshot(snapshot),
        MOCK_CONTEXT: JSON.stringify(context),
        // Block network access
        HTTP_PROXY: 'http://127.0.0.1:0',
        HTTPS_PROXY: 'http://127.0.0.1:0',
        NO_PROXY: '',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    // Timeout handler with SIGKILL fallback
    let killTimeoutId: ReturnType<typeof setTimeout> | null = null;
    const timeoutId = setTimeout(() => {
      timedOut = true;
      child.kill('SIGTERM');
      // If SIGTERM doesn't work, force kill after 1 second
      killTimeoutId = setTimeout(() => {
        child.kill('SIGKILL');
      }, 1000);
    }, timeoutMs);

    child.on('close', (code) => {
      clearTimeout(timeoutId);
      if (killTimeoutId) clearTimeout(killTimeoutId);
      const executionTimeMs = Date.now() - startTime;

      if (timedOut) {
        resolve({
          success: false,
          error: `Execution timeout (${timeoutMs}ms)`,
          errorType: 'timeout',
          executionTimeMs,
        });
        return;
      }

      // Parse mock output
      const mockOutput = parseMockOutput(stdout);

      if (code !== 0) {
        // Determine error type
        let errorType: ExecutionResult['errorType'] = 'runtime';
        if (stderr.includes('SyntaxError')) {
          errorType = 'syntax';
        } else if (mockOutput && !mockOutput.success) {
          errorType = 'wrong_api';
        }

        resolve({
          success: false,
          error: stderr.slice(0, 500) || `Exit code: ${code}`,
          errorType,
          executionTimeMs,
          assertions: mockOutput?.assertions,
          callCount: mockOutput?.callCount,
          expectedCallCount: mockOutput?.expectedCallCount,
          requiredCallCount: mockOutput?.requiredCallCount,
          matchedRequiredCalls: mockOutput?.matchedRequiredCalls,
          allAssertionsMatched: mockOutput?.allAssertionsMatched,
          allRequiredCallsMade: mockOutput?.allRequiredCallsMade,
          resultValidation: mockOutput?.resultValidation,
        });
        return;
      }

      // Success
      resolve({
        success: mockOutput?.success ?? true,
        result: mockOutput?.result,
        executionTimeMs,
        assertions: mockOutput?.assertions,
        callCount: mockOutput?.callCount,
        expectedCallCount: mockOutput?.expectedCallCount,
        requiredCallCount: mockOutput?.requiredCallCount,
        matchedRequiredCalls: mockOutput?.matchedRequiredCalls,
        allAssertionsMatched: mockOutput?.allAssertionsMatched,
        allRequiredCallsMade: mockOutput?.allRequiredCallsMade,
        resultValidation: mockOutput?.resultValidation,
      });
    });

    child.on('error', (err) => {
      clearTimeout(timeoutId);
      resolve({
        success: false,
        error: err.message,
        errorType: 'runtime',
        executionTimeMs: Date.now() - startTime,
      });
    });
  });
}

/**
 * Parse mock output from stdout
 */
function parseMockOutput(stdout: string): {
  success: boolean;
  result: unknown;
  assertions: AssertionResult[];
  callCount: number;
  expectedCallCount: number;
  requiredCallCount: number;
  matchedRequiredCalls: number;
  allAssertionsMatched: boolean;
  allRequiredCallsMade: boolean;
  resultValidation: ResultValidation;
} | null {
  // Look for our special output marker
  const marker = '__MOCK_OUTPUT__';
  const markerIndex = stdout.indexOf(marker);

  if (markerIndex === -1) {
    return null;
  }

  try {
    const jsonStr = stdout.slice(markerIndex + marker.length).trim();
    // Find the end of the JSON (first newline or end of string)
    const endIndex = jsonStr.indexOf('\n');
    const json = endIndex === -1 ? jsonStr : jsonStr.slice(0, endIndex);
    return JSON.parse(json);
  } catch {
    return null;
  }
}

/**
 * Batch execute multiple code snippets.
 * Currently sequential, could be parallelized with worker threads.
 */
export async function batchExecuteGitHubCode(
  items: Array<{ code: string; scenarioId: string }>,
  options: ExecutionOptions = {},
): Promise<ExecutionResult[]> {
  const results: ExecutionResult[] = [];

  for (const item of items) {
    results.push(await executeGitHubCode(item.code, item.scenarioId, options));
  }

  return results;
}
