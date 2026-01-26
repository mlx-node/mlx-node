import { parseArgs } from 'node:util';
import { createWriteStream, existsSync, readFileSync } from 'node:fs';

import { AsyncEntry } from '@napi-rs/keyring';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { input } from '@inquirer/prompts';
import { generateText, tool, stepCountIs, hasToolCall, type StepResult } from 'ai';
import { z } from 'zod';

import { queryMethod, initLspService, type LspService } from '../grpo/lsp';
import { validateGeneratedCode } from '../grpo/code-validator';
import { executeGitHubCode } from '../grpo/execute-github-code';
import { SYSTEM_PROMPT, GET_PR_PROMPTS } from './prompts';

// ============================================================================
// CLI Arguments
// ============================================================================

const { values } = parseArgs({
  options: {
    model: {
      type: 'string',
      short: 'm',
      default: 'xiaomi/mimo-v2-flash:free',
    },
    target: {
      type: 'string',
      short: 't',
      default: '1000',
    },
    output: {
      type: 'string',
      short: 'o',
      default: 'examples/sft/github-get-pr.jsonl',
    },
    resume: {
      type: 'boolean',
      default: false,
    },
    debug: {
      type: 'boolean',
      short: 'd',
      default: false,
    },
    interactive: {
      type: 'boolean',
      short: 'i',
      default: false,
    },
    ['delete-openrouter-key']: {
      type: 'boolean',
    },
  },
});

// ============================================================================
// OpenRouter Setup
// ============================================================================

const keyring = new AsyncEntry('mlx-node', 'openrouter');

let openrouterApiKey = (await keyring.getPassword()) ?? process.env.OPENROUTER_API_KEY;

if (values['delete-openrouter-key']) {
  await keyring.deletePassword();
  console.log('OpenRouter API Key deleted');
  process.exit(0);
}

if (!openrouterApiKey) {
  if (values.interactive) {
    openrouterApiKey = await input({
      message: 'Input your OpenRouter API Key',
    });
    if (openrouterApiKey) {
      await keyring.setPassword(openrouterApiKey);
    }
  } else {
    throw new Error('OPENROUTER_API_KEY is not set. Use --interactive to set it.');
  }
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  tool_calls?: Array<{ id: string; name: string; arguments: string }>;
  tool_call_id?: string;
}

interface SFTRecord {
  messages: ChatMessage[];
  metadata: {
    scenarioId: string;
    category: string;
    model: string;
    hasThinking: boolean;
    thinkingContent: string;
    lspQuery: string;
  };
}

// ============================================================================
// Statistics Tracking
// ============================================================================

interface Stats {
  total: number;
  accepted: number;
  rejected: {
    wrongLspCount: number;
    wrongRunJsCount: number;
    codeValidationFailed: number;
    executionFailed: number;
  };
}

// ============================================================================
// Validation
// ============================================================================

interface ValidationResult {
  valid: boolean;
  reason?: string;
  lspCallCount: number;
  runJsCallCount: number;
}

function validateToolPattern(steps: StepResult<any>[]): ValidationResult {
  let lspCalls = 0;
  let runJsCalls = 0;

  for (const step of steps) {
    for (const tc of step.toolCalls || []) {
      if (tc.toolName === 'lsp') lspCalls++;
      if (tc.toolName === 'run_js') runJsCalls++;
    }
  }

  const valid = lspCalls >= 1 && runJsCalls === 1;

  let reason: string | undefined;
  if (!valid) {
    const issues: string[] = [];
    if (lspCalls < 1) issues.push(`lsp=${lspCalls}(need 1)`);
    if (runJsCalls !== 1) issues.push(`run_js=${runJsCalls}(need 1)`);
    reason = issues.join(', ');
  }

  return { valid, reason, lspCallCount: lspCalls, runJsCallCount: runJsCalls };
}

// ============================================================================
// Count Existing Records (for --resume)
// ============================================================================

function countExistingRecords(outputPath: string): number {
  if (!existsSync(outputPath)) {
    return 0;
  }

  const content = readFileSync(outputPath, 'utf-8');
  const lines = content
    .trim()
    .split('\n')
    .filter((line) => line.trim());
  return lines.length;
}

// ============================================================================
// Extract Thinking Content
// ============================================================================

function extractThinkingContent(steps: StepResult<any>[]): string {
  const thinkingParts: string[] = [];
  for (const step of steps) {
    if (step.reasoningText) {
      thinkingParts.push(step.reasoningText.trim());
    }
  }
  return thinkingParts.join('\n');
}

// ============================================================================
// Extract LSP Query
// ============================================================================

function extractLspQuery(steps: StepResult<any>[]): string {
  for (const step of steps) {
    for (const tc of step.toolCalls || []) {
      if (tc.toolName === 'lsp') {
        return JSON.stringify(tc.input || {});
      }
    }
  }
  return '';
}

// ============================================================================
// Create SFT Record (Multi-turn format)
// ============================================================================

async function createSFTRecord(
  prompt: string,
  requireThinking: boolean,
  lspService: LspService,
  model: string,
  stats: Stats,
  debug: boolean = false,
): Promise<SFTRecord | null> {
  let codeValid = false;
  let executionValid = false;

  const provider = createOpenRouter({
    apiKey: openrouterApiKey,

    extraBody: {
      reasoning: requireThinking
        ? {
            enabled: true,
          }
        : {
            enabled: false,
          },
    },
  });

  const result = await generateText({
    model: provider(model),
    system: SYSTEM_PROMPT,
    prompt,
    tools: {
      lsp: tool({
        description:
          'Query API documentation. Use this first to check parameters before writing code. Returns parameter names, types, and descriptions.',
        inputSchema: z.object({
          method: z.string().describe('Method path on octokit instance, e.g., "octokit.rest.pulls.list"'),
          kind: z.enum(['parameters', 'return']).describe('What to query: parameters or return type'),
        }),
        execute: ({ method, kind }) => {
          return queryMethod(lspService, { method, kind });
        },
      }),
      run_js: tool({
        description:
          'Execute JavaScript code. Use this after checking API docs with lsp. Code must import from "./utils" and export default async function.',
        inputSchema: z.object({
          code: z.string().describe('JavaScript code to execute'),
        }),
        execute: async ({ code }) => {
          if (debug) console.log(`    run_js: code length ${code.length}`);
          const validator = validateGeneratedCode(code);
          if (validator.valid) {
            codeValid = true;
            const executeResult = await executeGitHubCode(code, 'get-pr');
            if (debug) {
              console.log(
                `    Execution: success=${executeResult.success}, assertions=${executeResult.allAssertionsMatched}`,
              );
              if (!executeResult.success) {
                console.log(`    Error: ${executeResult.error} (${executeResult.errorType})`);
              }
            }
            if (executeResult.success && executeResult.allAssertionsMatched) {
              executionValid = true;
              return { success: true, result: executeResult.result };
            }
            return {
              success: false,
              error: executeResult.error || 'Assertion mismatch',
              errorType: executeResult.errorType,
            };
          }
          if (debug) console.log(`    Validation failed: ${JSON.stringify(validator)}`);
          return { success: false, error: 'Code validation failed', issues: validator };
        },
      }),
    },
    stopWhen: [stepCountIs(5), hasToolCall('run_js')],
  });

  if (debug) {
    console.log(
      `  Steps: ${result.steps.length}, codeValid=${codeValid}, executionValid=${executionValid}, finish=${result.finishReason}`,
    );
    for (let i = 0; i < result.steps.length; i++) {
      const step = result.steps[i];
      const tools = step.toolCalls?.map((tc) => tc.toolName).join(', ') || 'none';
      console.log(`    Step ${i + 1}: tools=[${tools}]`);
    }
  }

  // Validate tool pattern
  const validation = validateToolPattern(result.steps);

  if (!validation.valid) {
    // Update rejection stats
    if (validation.lspCallCount < 1) {
      stats.rejected.wrongLspCount++;
    }
    if (validation.runJsCallCount !== 1) {
      stats.rejected.wrongRunJsCount++;
    }
    if (debug) {
      console.log(`  Rejected: ${validation.reason}`);
    }
    return null;
  }

  // Check code validation
  if (!codeValid) {
    stats.rejected.codeValidationFailed++;
    if (debug) {
      console.log(`  Rejected: code validation failed`);
    }
    return null;
  }

  // Check execution
  if (!executionValid) {
    stats.rejected.executionFailed++;
    if (debug) {
      console.log(`  Rejected: execution failed`);
    }
    return null;
  }

  // Build multi-turn messages array
  const messages: ChatMessage[] = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: prompt },
  ];

  let toolCallIdCounter = 0;

  for (const step of result.steps) {
    // Build content with <think> tags if reasoning exists
    let content = '';
    if (step.reasoningText) {
      content = `<think>${step.reasoningText}</think>`;
    }
    if (step.text) {
      content += step.text;
    }

    // Build assistant message with content and tool_calls
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content,
    };

    if (step.toolCalls && step.toolCalls.length > 0) {
      assistantMessage.tool_calls = step.toolCalls.map((tc) => {
        toolCallIdCounter++;
        return {
          id: `call_${toolCallIdCounter}`,
          name: tc.toolName,
          arguments: JSON.stringify(tc.input || {}),
        };
      });
    }

    messages.push(assistantMessage);

    // Add tool response messages
    if (step.toolResults && step.toolResults.length > 0) {
      for (let i = 0; i < step.toolResults.length; i++) {
        const tr = step.toolResults[i];
        const toolCallId =
          assistantMessage.tool_calls?.[i]?.id || `call_${toolCallIdCounter - step.toolResults.length + i + 1}`;
        messages.push({
          role: 'tool',
          content: JSON.stringify(tr.output, null, 2),
          tool_call_id: toolCallId,
        });
      }
    }
  }

  // Add final text if different from last step
  if (result.text && result.steps.length > 0 && result.steps[result.steps.length - 1].text !== result.text) {
    messages.push({
      role: 'assistant',
      content: result.text,
    });
  }

  const thinkingContent = extractThinkingContent(result.steps);
  const lspQuery = extractLspQuery(result.steps);

  const sftRecord: SFTRecord = {
    messages,
    metadata: {
      scenarioId: 'get-pr',
      category: 'github-api',
      model,
      hasThinking: requireThinking,
      thinkingContent,
      lspQuery,
    },
  };

  return sftRecord;
}

// ============================================================================
// Main
// ============================================================================

console.log('Initializing LSP service...');
const lspService = await initLspService();

const target = parseInt(values.target!);
let completed = values.resume ? countExistingRecords(values.output!) : 0;
const outputStream = createWriteStream(values.output!, { flags: values.resume ? 'a' : 'w' });

// Statistics tracking
const stats: Stats = {
  total: 0,
  accepted: 0,
  rejected: {
    wrongLspCount: 0,
    wrongRunJsCount: 0,
    codeValidationFailed: 0,
    executionFailed: 0,
  },
};

// 80/20 thinking split tracking
const THINKING_RATIO = 0.8;
const splitStats = { withThinking: 0, withoutThinking: 0 };

console.log(`\nConfiguration:`);
console.log(`  Model: ${values.model}`);
console.log(`  Target: ${target}`);
console.log(`  Output: ${values.output}`);
console.log(`  Resume: ${values.resume} (starting at ${completed})`);
console.log(`  Prompts: ${GET_PR_PROMPTS.length} variations`);
console.log(`  Thinking ratio: ${THINKING_RATIO * 100}%\n`);

let failures = 0;
const startTime = Date.now();

while (completed < target) {
  const promptIndex = completed % GET_PR_PROMPTS.length;
  const prompt = GET_PR_PROMPTS[promptIndex];

  // Determine if this sample should require thinking (80/20 split)
  // Use randomization to avoid chicken-and-egg problem where no thinking records ever pass
  const requireThinking = Math.random() < THINKING_RATIO;

  stats.total++;

  try {
    const record = await createSFTRecord(prompt, requireThinking, lspService, values.model!, stats, values.debug);

    if (record) {
      outputStream.write(JSON.stringify(record) + '\n');
      completed++;
      stats.accepted++;
      failures = 0; // Reset failure counter on success

      // Update split stats
      if (record.metadata.hasThinking) {
        splitStats.withThinking++;
      } else {
        splitStats.withoutThinking++;
      }

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const thinkingIndicator = record.metadata.hasThinking ? '[T]' : '[.]';
      console.log(
        `[${completed}/${target}] ${thinkingIndicator} Generated (prompt #${promptIndex + 1}) [${elapsed}s elapsed]`,
      );
    } else {
      failures++;
      console.log(`[${completed}/${target}] x Failed validation (prompt #${promptIndex + 1}, failures: ${failures})`);

      // If too many consecutive failures, warn but continue
      if (failures >= 10) {
        console.log(`  Warning: ${failures} consecutive failures. Consider checking the model or prompts.`);
      }
    }

    // Log stats every 50 attempts
    if (stats.total % 50 === 0) {
      const acceptRate = ((stats.accepted / stats.total) * 100).toFixed(1);
      const thinkingRate =
        (splitStats.withThinking / (splitStats.withThinking + splitStats.withoutThinking + 0.001)) * 100;
      console.log(`\n--- Stats after ${stats.total} attempts ---`);
      console.log(`  Accepted: ${stats.accepted} (${acceptRate}%)`);
      console.log(
        `  Rejected: wrongLsp=${stats.rejected.wrongLspCount}, wrongRunJs=${stats.rejected.wrongRunJsCount}, codeValidation=${stats.rejected.codeValidationFailed}, execution=${stats.rejected.executionFailed}`,
      );
      console.log(
        `  Thinking split: ${splitStats.withThinking}/${splitStats.withoutThinking} (${thinkingRate.toFixed(1)}% with thinking)`,
      );
      console.log('');
    }
  } catch (error) {
    failures++;
    console.error(`[${completed}/${target}] x Error:`, error instanceof Error ? error.message : error);

    // If too many consecutive failures, warn but continue
    if (failures >= 10) {
      console.log(`  Warning: ${failures} consecutive failures. Consider checking the API key or model.`);
    }
  }
}

outputStream.end();

const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
const acceptRate = ((stats.accepted / stats.total) * 100).toFixed(1);
const thinkingRate = (splitStats.withThinking / (splitStats.withThinking + splitStats.withoutThinking + 0.001)) * 100;

console.log(`\n=== Final Summary ===`);
console.log(`Generated ${completed} records to ${values.output}`);
console.log(`Total attempts: ${stats.total}`);
console.log(`Accept rate: ${acceptRate}%`);
console.log(
  `Thinking split: ${splitStats.withThinking}/${splitStats.withoutThinking} (${thinkingRate.toFixed(1)}% with thinking)`,
);
console.log(`\nRejection breakdown:`);
console.log(`  Wrong LSP count: ${stats.rejected.wrongLspCount}`);
console.log(`  Wrong run_js count: ${stats.rejected.wrongRunJsCount}`);
console.log(`  Code validation failed: ${stats.rejected.codeValidationFailed}`);
console.log(`  Execution failed: ${stats.rejected.executionFailed}`);
console.log(`\nTotal time: ${totalTime}s`);
console.log(`Avg time per record: ${(parseFloat(totalTime) / completed).toFixed(2)}s`);
