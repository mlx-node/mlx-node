import type { RewardOutput } from '@mlx-node/trl';

import { validateGeneratedCode } from './code-validator';
import { executeGitHubCode } from './execute-github-code';

/**
 * GitHub tool-use reward function using gate-based scoring.
 *
 * Gate-based scoring system (hard threshold at 7.0):
 * - Score < 7.0 = BROKEN (can't run or wrong result)
 * - Score >= 7.0 = CORRECT (runs and returns right data)
 * - Score 7-10 = Style bonuses only
 *
 * Gates:
 * - GATE 1: FORMAT (0-2) - Tool call structure
 * - GATE 2: SYNTAX (2-4) - Code validity
 * - GATE 3: API (4-6) - Correct API usage
 * - GATE 4: EXECUTION (5.5-6.5) - Runtime validation
 * - ALL GATES PASSED: Base score = 9.0
 * - STYLE BONUSES (9-10): error handling, typed errors
 *
 * Gate values in details.gate_failed:
 * - 0 = PASSED (no gate failed)
 * - 1 = FORMAT
 * - 2 = SYNTAX
 * - 3 = API
 * - 4 = EXECUTION
 *
 * @param outputs - Array of reward outputs to score
 * @param contexts - Array of contexts (one per output) containing scenario metadata for validation
 */
export async function githubToolReward(outputs: RewardOutput[], contexts: string[]): Promise<Float32Array> {
  const rewards = new Float32Array(outputs.length);
  const DEBUG = process.env.DEBUG_REWARDS === '1';

  for (let i = 0; i < outputs.length; i++) {
    const output = outputs[i];
    const context = contexts[i] ?? '';
    const { completion } = output;

    // GATE 1: FORMAT (0-2)
    if (completion.toolCalls.length === 0) {
      rewards[i] = completion.rawText.includes('<tool_call>') ? 1.0 : 0.0;
      continue;
    }

    const call = completion.toolCalls[0];
    if (call.status !== 'ok' || call.name !== 'run_js') {
      rewards[i] = 2.0;
      if (DEBUG) console.log(`[${i}] GATE 1 FAIL: Invalid tool call (status=${call.status}, name=${call.name})`);
      continue;
    }

    // GATE 2: SYNTAX (2-4)
    // Use pre-parsed arguments from toolCalls instead of regex extraction
    const code =
      typeof call.arguments === 'object' && call.arguments !== null
        ? ((call.arguments as Record<string, unknown>).code as string | undefined)
        : undefined;

    if (!code || typeof code !== 'string') {
      rewards[i] = 2.5;
      if (DEBUG) console.log(`[${i}] GATE 2 FAIL: No code in arguments`);
      continue;
    }

    const validation = validateGeneratedCode(code);
    if (!validation.syntaxValid) {
      rewards[i] = 3.0;
      if (DEBUG) console.log(`[${i}] GATE 2 FAIL: Syntax errors`);
      continue;
    }
    if (!validation.hasDefaultExport || !validation.hasAsyncFunction) {
      rewards[i] = 3.5;
      if (DEBUG) console.log(`[${i}] GATE 2 FAIL: Missing export/async`);
      continue;
    }
    if (!validation.importsValid) {
      rewards[i] = 4.0;
      if (DEBUG) console.log(`[${i}] GATE 2 FAIL: Invalid imports`);
      continue;
    }
    if (!validation.hasUtilsImport) {
      rewards[i] = 4.0;
      if (DEBUG) console.log(`[${i}] GATE 2 FAIL: Missing './utils' import`);
      continue;
    }

    try {
      // Extract scenario ID from context JSON (e.g., "get-pr-0" -> "get-pr")
      let scenarioId = 'unknown';
      try {
        const ctx = JSON.parse(context);
        // Remove numeric suffix (e.g., "get-pr-0" -> "get-pr")
        scenarioId = ctx.id?.replace(/-\d+$/, '') ?? 'unknown';
      } catch {
        scenarioId = context; // Fallback to raw context if not JSON
      }
      const execResult = await executeGitHubCode(code, scenarioId, { timeoutMs: 5000 });

      if (!execResult.success) {
        rewards[i] = 5.5;
        if (DEBUG) console.log(`[${i}] GATE 4 FAIL: Execution failed (${execResult.errorType})`);
        continue;
      }
      if (!execResult.allAssertionsMatched) {
        rewards[i] = 6.0;
        if (DEBUG) console.log(`[${i}] GATE 4 FAIL: Assertion mismatches`);
        continue;
      }
      if (!execResult.resultValidation?.valid) {
        rewards[i] = 6.5;
        if (DEBUG) console.log(`[${i}] GATE 4 FAIL: Result validation failed:`, execResult.resultValidation);
        continue;
      }
    } catch (err) {
      rewards[i] = 5.5;
      if (DEBUG) console.error(`[${i}] GATE 4 FAIL: Execution error:`, err);
      continue;
    }

    const score = 10.0;

    rewards[i] = Math.min(10.0, Math.round(score * 10) / 10);

    if (DEBUG) console.log(`[${i}] ALL GATES PASSED: score=${rewards[i]}`);
  }

  return rewards;
}
