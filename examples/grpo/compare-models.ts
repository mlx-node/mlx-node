/**
 * Model Comparison: Fine-tuned vs Base Model
 *
 * Compares the GRPO-trained model against the base model on ast-grep tool use tasks.
 * Uses the SAME patterns from training to measure in-distribution improvement.
 *
 * Usage:
 *   yarn oxnode examples/grpo/compare-models.ts [options]
 *
 * Options:
 *   --base-model <path>       Base model path (default: .cache/models/qwen3-0.6b-mlx-bf16)
 *   --finetuned-model <path>  Fine-tuned model path (default: outputs/grpo-tool-use/final)
 *   --samples <n>             Samples per test case (default: 1)
 *   --verbose                 Show all samples, not just first
 */

import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { parse, Lang } from '@ast-grep/napi';
import { ModelLoader } from '@mlx-node/lm';
import type { Qwen3Model } from '@mlx-node/lm';
import type { RewardOutput } from '@mlx-node/trl';
import { ALL_PATTERNS, type AstGrepPattern } from './ast-grep-dataset';

// ============================================================================
// Configuration
// ============================================================================

const DEFAULT_BASE_MODEL = resolve(process.cwd(), '.cache', 'models', 'qwen3-0.6b-mlx-bf16');
const DEFAULT_FINETUNED_MODEL = resolve(process.cwd(), 'outputs', 'grpo-tool-use', 'final');

// Same system prompt as training for fair comparison
const SYSTEM_PROMPT = `You are a code search assistant that ALWAYS uses the ast_grep tool.

IMPORTANT: You MUST call the ast_grep tool for EVERY request. Do NOT provide text answers - ONLY use tool calls.

## ast-grep Metavariable Syntax
- $NAME - Matches exactly ONE AST node (identifier, expression, statement)
- $$$NAME - Matches ZERO OR MORE nodes (for argument lists, statement blocks)
- Names must be UPPERCASE with underscores: $VAR, $FUNC, $$$ARGS

## Pattern Examples
| Description | Pattern |
|-------------|---------|
| Function declaration | function $NAME($$$ARGS) { $$$BODY } |
| Arrow function | $PARAM => $BODY |
| Const declaration | const $NAME = $VALUE |
| Let declaration | let $NAME = $VALUE |
| Var declaration | var $NAME = $VALUE |
| React useState | const [$STATE, $SETTER] = useState($INIT) |
| Import named | import { $$$NAMES } from $SOURCE |
| Class with extends | class $NAME extends $PARENT { $$$BODY } |
| Try-catch | try { $$$TRY } catch ($ERR) { $$$CATCH } |
| Method call | $OBJ.$METHOD($$$ARGS) |
| Await expression | await $EXPR |

## Response Format
1. Brief thinking in <think> tags (optional, keep short)
2. Call ast_grep with correct JSON format

## Example Response
<think>Looking for arrow functions with single expression body.</think>
<tool_call>
{"name": "ast_grep", "arguments": {"pattern": "$PARAM => $BODY"}}
</tool_call>

## Another Example
<think>Finding const declarations.</think>
<tool_call>
{"name": "ast_grep", "arguments": {"pattern": "const $NAME = $VALUE"}}
</tool_call>

CRITICAL: The tool_call MUST contain valid JSON with "name" and "arguments" keys.`;

// ============================================================================
// Reward Function (same as training)
// ============================================================================

function getLang(language: string): Lang {
  const normalized = language.toLowerCase();
  if (normalized === 'typescript' || normalized === 'ts') return Lang.TypeScript;
  if (normalized === 'tsx') return Lang.Tsx;
  if (normalized === 'jsx') return Lang.JavaScript;
  return Lang.JavaScript;
}

interface EvalResult {
  pattern: AstGrepPattern;
  completion: string;
  reward: number;
  toolCallPresent: boolean;
  correctToolName: boolean;
  validJson: boolean;
  patternProvided: boolean;
  patternExecutes: boolean;
  exactMatch: boolean;
  thinking: boolean;
}

function evaluateCompletion(completion: RewardOutput, pattern: AstGrepPattern): EvalResult {
  const { rawText, toolCalls, thinking } = completion.completion;

  let score = 0;
  let toolCallPresent = false;
  let correctToolName = false;
  let validJson = false;
  let patternProvided = false;
  let patternExecutes = false;
  let exactMatch = false;
  let hasThinking = !!thinking;

  if (hasThinking) score += 0.5;

  // 1. Tool call present
  if (toolCalls.length === 0) {
    if (rawText.includes('ast_grep') || rawText.includes('ast-grep')) {
      score += 0.2;
    }
    return {
      pattern,
      completion: rawText,
      reward: score,
      toolCallPresent,
      correctToolName,
      validJson,
      patternProvided,
      patternExecutes,
      exactMatch,
      thinking: hasThinking,
    };
  }
  toolCallPresent = true;
  score += 1;

  const call = toolCalls[0];

  // 2. Correct tool name
  if (call.name !== 'ast_grep') {
    score += 0.2;
    return {
      pattern,
      completion: rawText,
      reward: score,
      toolCallPresent,
      correctToolName,
      validJson,
      patternProvided,
      patternExecutes,
      exactMatch,
      thinking: hasThinking,
    };
  }
  correctToolName = true;
  score += 1;

  // 3. Valid JSON
  if (call.status !== 'ok') {
    return {
      pattern,
      completion: rawText,
      reward: score,
      toolCallPresent,
      correctToolName,
      validJson,
      patternProvided,
      patternExecutes,
      exactMatch,
      thinking: hasThinking,
    };
  }
  validJson = true;
  score += 1;

  // 4. Pattern provided
  const args = call.arguments as Record<string, unknown>;
  const extractedPattern = args?.pattern;
  if (!extractedPattern || typeof extractedPattern !== 'string') {
    return {
      pattern,
      completion: rawText,
      reward: score,
      toolCallPresent,
      correctToolName,
      validJson,
      patternProvided,
      patternExecutes,
      exactMatch,
      thinking: hasThinking,
    };
  }
  patternProvided = true;
  score += 1;

  // 5. Pattern executes + 6. Match count
  try {
    const lang = getLang(pattern.language);
    const root = parse(lang, pattern.codeContext).root();
    const matches = root.findAll(extractedPattern);
    patternExecutes = true;
    score += 1;

    const actual = matches.length;
    const expected = pattern.expectedMatches;

    if (actual === expected) {
      exactMatch = true;
      score += 3;
    } else if (expected > 0 && actual > 0 && Math.abs(actual - expected) <= 1) {
      score += 1.5;
    } else if (actual > 0) {
      score += 0.5;
    }
  } catch {
    // Pattern doesn't execute
  }

  // 7. Metavariable quality
  if (/\$\$\$[A-Z_]+/.test(extractedPattern)) {
    score += 0.3;
  }
  if (/\$[A-Z_]+/.test(extractedPattern) && !/\$\$/.test(extractedPattern.replace(/\$\$\$/g, ''))) {
    score += 0.3;
  }
  if (/\$[a-z]/.test(extractedPattern)) {
    score -= 0.3;
  }
  const metavarCount = (extractedPattern.match(/\$+[A-Z_]+/g) || []).length;
  if (metavarCount >= 2) score += 0.2;
  if (metavarCount >= 3) score += 0.2;

  return {
    pattern,
    completion: rawText,
    reward: Math.round(Math.max(0, Math.min(score, 10)) * 10) / 10,
    toolCallPresent,
    correctToolName,
    validJson,
    patternProvided,
    patternExecutes,
    exactMatch,
    thinking: hasThinking,
  };
}

// ============================================================================
// Metrics Computation
// ============================================================================

interface Metrics {
  meanReward: number;
  stdReward: number;
  toolCallRate: number;
  correctToolNameRate: number;
  validJsonRate: number;
  patternProvidedRate: number;
  patternExecuteRate: number;
  exactMatchRate: number;
  thinkingRate: number;
}

function computeMetrics(results: EvalResult[]): Metrics {
  const n = results.length;
  if (n === 0) {
    return {
      meanReward: 0,
      stdReward: 0,
      toolCallRate: 0,
      correctToolNameRate: 0,
      validJsonRate: 0,
      patternProvidedRate: 0,
      patternExecuteRate: 0,
      exactMatchRate: 0,
      thinkingRate: 0,
    };
  }

  const rewards = results.map((r) => r.reward);
  const meanReward = rewards.reduce((a, b) => a + b, 0) / n;
  const variance = rewards.reduce((sum, r) => sum + (r - meanReward) ** 2, 0) / n;
  const stdReward = Math.sqrt(variance);

  return {
    meanReward,
    stdReward,
    toolCallRate: results.filter((r) => r.toolCallPresent).length / n,
    correctToolNameRate: results.filter((r) => r.correctToolName).length / n,
    validJsonRate: results.filter((r) => r.validJson).length / n,
    patternProvidedRate: results.filter((r) => r.patternProvided).length / n,
    patternExecuteRate: results.filter((r) => r.patternExecutes).length / n,
    exactMatchRate: results.filter((r) => r.exactMatch).length / n,
    thinkingRate: results.filter((r) => r.thinking).length / n,
  };
}

// ============================================================================
// Evaluation
// ============================================================================

async function evaluateModel(
  model: Qwen3Model,
  patterns: AstGrepPattern[],
  samplesPerPattern: number,
): Promise<EvalResult[]> {
  const results: EvalResult[] = [];

  for (const pattern of patterns) {
    const userContent = `Find all ${pattern.description} in this code:

\`\`\`${pattern.language}
${pattern.codeContext}
\`\`\``;

    const messages = [
      { role: 'system' as const, content: SYSTEM_PROMPT },
      { role: 'user' as const, content: userContent },
    ];

    for (let s = 0; s < samplesPerPattern; s++) {
      const result = await model.chat(messages, {
        maxNewTokens: 512,
        temperature: 0.7,
        topP: 0.95,
      });

      const evalResult = evaluateCompletion(
        {
          prompt: messages[1].content,
          completion: {
            text: result.text,
            rawText: result.rawText,
            toolCalls: result.toolCalls,
            thinking: result.thinking ?? undefined,
            numTokens: result.numTokens,
            finishReason: result.finishReason,
          },
          expectedAnswer: JSON.stringify({
            correctPattern: pattern.pattern,
            expectedMatches: pattern.expectedMatches,
            language: pattern.language,
            codeContext: pattern.codeContext,
          }),
        },
        pattern,
      );

      results.push(evalResult);
    }
  }

  return results;
}

// ============================================================================
// Display
// ============================================================================

function formatPercent(value: number): string {
  return (value * 100).toFixed(1) + '%';
}

function formatImprovement(base: number, ft: number): string {
  if (base === 0 && ft === 0) return '  0.0pp';
  if (base === 0) return '+' + formatPercent(ft).padStart(6);
  const diff = (ft - base) * 100;
  const sign = diff >= 0 ? '+' : '';
  return sign + diff.toFixed(1) + 'pp';
}

function printComparison(
  baseResults: EvalResult[],
  ftResults: EvalResult[],
  patterns: AstGrepPattern[],
  verbose: boolean,
) {
  const baseMetrics = computeMetrics(baseResults);
  const ftMetrics = computeMetrics(ftResults);

  console.log('\n');
  console.log('═══════════════════════════════════════════════════════════════════════');
  console.log('                    Model Comparison: ast-grep Tool Use                 ');
  console.log('═══════════════════════════════════════════════════════════════════════');
  console.log();

  // Aggregate metrics table
  console.log('───────────────────────────────────────────────────────────────────────');
  console.log('                          AGGREGATE METRICS                             ');
  console.log('───────────────────────────────────────────────────────────────────────');
  console.log('Metric                      Base          Fine-tuned    Improvement');
  console.log('───────────────────────────────────────────────────────────────────────');

  const rows = [
    [
      'Mean Reward (0-10)',
      `${baseMetrics.meanReward.toFixed(1)} ± ${baseMetrics.stdReward.toFixed(1)}`,
      `${ftMetrics.meanReward.toFixed(1)} ± ${ftMetrics.stdReward.toFixed(1)}`,
      `${ftMetrics.meanReward > baseMetrics.meanReward ? '+' : ''}${(ftMetrics.meanReward - baseMetrics.meanReward).toFixed(1)}`,
    ],
    [
      'Tool Call Rate',
      formatPercent(baseMetrics.toolCallRate),
      formatPercent(ftMetrics.toolCallRate),
      formatImprovement(baseMetrics.toolCallRate, ftMetrics.toolCallRate),
    ],
    [
      'Correct Tool Name',
      formatPercent(baseMetrics.correctToolNameRate),
      formatPercent(ftMetrics.correctToolNameRate),
      formatImprovement(baseMetrics.correctToolNameRate, ftMetrics.correctToolNameRate),
    ],
    [
      'Valid JSON Rate',
      formatPercent(baseMetrics.validJsonRate),
      formatPercent(ftMetrics.validJsonRate),
      formatImprovement(baseMetrics.validJsonRate, ftMetrics.validJsonRate),
    ],
    [
      'Pattern Execute Rate',
      formatPercent(baseMetrics.patternExecuteRate),
      formatPercent(ftMetrics.patternExecuteRate),
      formatImprovement(baseMetrics.patternExecuteRate, ftMetrics.patternExecuteRate),
    ],
    [
      'Exact Match Rate',
      formatPercent(baseMetrics.exactMatchRate),
      formatPercent(ftMetrics.exactMatchRate),
      formatImprovement(baseMetrics.exactMatchRate, ftMetrics.exactMatchRate),
    ],
    [
      'Thinking Present',
      formatPercent(baseMetrics.thinkingRate),
      formatPercent(ftMetrics.thinkingRate),
      formatImprovement(baseMetrics.thinkingRate, ftMetrics.thinkingRate),
    ],
  ];

  for (const [name, base, ft, imp] of rows) {
    console.log(`${name.padEnd(24)} ${base.padEnd(14)} ${ft.padEnd(14)} ${imp}`);
  }
  console.log('───────────────────────────────────────────────────────────────────────');
  console.log();

  // Sample comparisons
  console.log('───────────────────────────────────────────────────────────────────────');
  console.log('                         SAMPLE COMPARISONS                             ');
  console.log('───────────────────────────────────────────────────────────────────────');

  const samplesPerPattern = baseResults.length / patterns.length;

  for (let i = 0; i < patterns.length; i++) {
    const pattern = patterns[i];
    const baseIdx = i * samplesPerPattern;
    const baseResult = baseResults[baseIdx];
    const ftResult = ftResults[baseIdx];

    console.log();
    console.log(`[${pattern.id}] Find ${pattern.description}`);
    console.log(`Expected pattern: ${pattern.pattern}`);
    console.log(`Expected matches: ${pattern.expectedMatches}`);
    console.log();

    console.log(`BASE (reward: ${baseResult.reward.toFixed(1)}):`);
    console.log(baseResult.completion.slice(0, 300) + (baseResult.completion.length > 300 ? '...' : ''));
    console.log();

    console.log(`FINE-TUNED (reward: ${ftResult.reward.toFixed(1)}):`);
    console.log(ftResult.completion.slice(0, 300) + (ftResult.completion.length > 300 ? '...' : ''));
    console.log();
    console.log('─'.repeat(71));

    if (!verbose && i >= 4) {
      console.log(`\n... (${patterns.length - 5} more test cases, use --verbose to see all)`);
      break;
    }
  }

  console.log();
  console.log('═══════════════════════════════════════════════════════════════════════');
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const { values } = parseArgs({
    options: {
      'base-model': { type: 'string', default: DEFAULT_BASE_MODEL },
      'finetuned-model': { type: 'string', default: DEFAULT_FINETUNED_MODEL },
      samples: { type: 'string', default: '1' },
      'num-patterns': { type: 'string', default: '20' },
      verbose: { type: 'boolean', default: false },
    },
  });

  const basePath = resolve(values['base-model']!);
  const ftPath = resolve(values['finetuned-model']!);
  const samplesPerPattern = parseInt(values['samples']!, 10);
  const numPatterns = parseInt(values['num-patterns']!, 10);
  const verbose = values['verbose'] || false;

  // Use training patterns for in-distribution evaluation (shuffle and take subset)
  const shuffled = [...ALL_PATTERNS].sort(() => Math.random() - 0.5);
  const patterns = shuffled.slice(0, Math.min(numPatterns, ALL_PATTERNS.length));

  console.log('═══════════════════════════════════════════════════════════════════════');
  console.log('                    Model Comparison: ast-grep Tool Use                 ');
  console.log('              (Using SAME patterns as training for fair eval)           ');
  console.log('═══════════════════════════════════════════════════════════════════════');
  console.log();
  console.log(`Base Model:       ${basePath}`);
  console.log(`Fine-tuned Model: ${ftPath}`);
  console.log(`Test Patterns:    ${patterns.length} (from training set)`);
  console.log(`Samples/Pattern:  ${samplesPerPattern}`);
  console.log(`Total Samples:    ${patterns.length * samplesPerPattern} per model`);
  console.log();

  // Check paths
  if (!existsSync(basePath)) {
    console.error(`Error: Base model not found at ${basePath}`);
    process.exit(1);
  }
  if (!existsSync(ftPath)) {
    console.error(`Error: Fine-tuned model not found at ${ftPath}`);
    process.exit(1);
  }

  // Load models
  console.log('Loading base model...');
  const baseModel = await ModelLoader.loadPretrained(basePath);
  console.log('Loading fine-tuned model...');
  const ftModel = await ModelLoader.loadPretrained(ftPath);
  console.log();

  // Evaluate
  console.log(`Evaluating base model on ${patterns.length} patterns...`);
  const baseResults = await evaluateModel(baseModel, patterns, samplesPerPattern);
  console.log(`  Mean reward: ${computeMetrics(baseResults).meanReward.toFixed(2)}`);

  console.log();
  console.log(`Evaluating fine-tuned model on ${patterns.length} patterns...`);
  const ftResults = await evaluateModel(ftModel, patterns, samplesPerPattern);
  console.log(`  Mean reward: ${computeMetrics(ftResults).meanReward.toFixed(2)}`);

  // Display comparison
  printComparison(baseResults, ftResults, patterns, verbose);
}

main().catch((err) => {
  console.error('Error:', err);
  process.exit(1);
});
