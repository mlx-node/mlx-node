/**
 * GRPO Training Demo: Teaching Tool Use (ast-grep)
 *
 * This demo shows how to train a language model to use tools effectively.
 * We'll teach Qwen3-0.6B to generate correct ast-grep commands for code search tasks.
 *
 * Key improvements over v1:
 * - Uses Qwen3's native tool calling format (<tool_call>JSON</tool_call>)
 * - Leverages pre-trained tool calling capability
 * - Simplified reward functions using parseToolCallsFromText
 * - Expanded dataset with 50+ unique patterns
 * - Curriculum learning (basic → intermediate → advanced)
 *
 * Usage:
 *   yarn oxnode examples/grpo/train-tool-use.ts [options]
 *
 * Options:
 *   --model-path <path>     Path to model (default: .cache/models/qwen3-0.6b-mlx-f32)
 *   --num-examples <n>      Number of examples (default: 100)
 *   --output-dir <path>     Output directory (default: outputs/grpo-tool-use)
 *   --dry-run               Just show dataset, don't train
 *   --resume, -r            Resume training from latest checkpoint
 */

import { parseArgs } from 'node:util';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { parse, Lang } from '@ast-grep/napi';
import {
  GRPOTrainer,
  createTrainingLogger,
  type GRPOConfig,
  type DatasetExample,
  type RewardOutput,
} from '@mlx-node/trl';
import { generateCurriculumDataset, type AstGrepPattern } from './ast-grep-dataset';

const DEFAULT_MODEL_PATH = resolve(process.cwd(), 'outputs/sft-ast-grep/final-bf16');
const DEFAULT_NUM_EXAMPLES = 100;
const DEFAULT_OUTPUT_DIR = resolve(process.cwd(), 'outputs', 'grpo-tool-use');

// ============================================================================
// System Prompt
// ============================================================================

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
// Dataset Generation
// ============================================================================

/**
 * Convert ast-grep patterns to GRPO training examples.
 * Metadata is stored in the answer field as JSON for the reward function.
 */
function patternsToDatasetExamples(patterns: AstGrepPattern[]): DatasetExample[] {
  return patterns.map((pattern) => {
    // Format the user message without metadata leakage
    const userContent = `Find all ${pattern.description.toLowerCase()} in this code:

\`\`\`${pattern.language}
${pattern.codeContext}
\`\`\``;

    // Store metadata as JSON in the answer field for reward function
    const metadata = JSON.stringify({
      correctPattern: pattern.pattern,
      expectedMatches: pattern.expectedMatches,
      language: pattern.language,
      codeContext: pattern.codeContext,
      category: pattern.category,
    });

    return {
      prompt: [
        { role: 'system' as const, content: SYSTEM_PROMPT },
        { role: 'user' as const, content: userContent },
      ],
      answer: metadata, // Reward function will parse this
      metadata: {
        id: pattern.id,
        description: pattern.description,
        category: pattern.category,
      },
    };
  });
}

// ============================================================================
// Reward Function (Simplified with Tool Calling)
// ============================================================================

/**
 * Get the language enum for ast-grep
 */
function getLang(language: string): Lang {
  const normalized = language.toLowerCase();
  if (normalized === 'typescript' || normalized === 'ts') return Lang.TypeScript;
  if (normalized === 'tsx') return Lang.Tsx;
  if (normalized === 'jsx') return Lang.JavaScript;
  return Lang.JavaScript;
}

/**
 * Tool-use reward function using structured RewardOutput.
 *
 * The new API provides pre-parsed tool calls and thinking content via RewardOutput:
 * - output.completion.toolCalls - pre-parsed tool calls (no need to call parseToolCallsFromText)
 * - output.completion.thinking - extracted thinking content (no regex needed)
 * - output.completion.rawText - raw output for fallback matching
 * - output.expectedAnswer - metadata from dataset
 *
 * Reward components (0-10 points):
 * 1. Tool call present (0-1)
 * 2. Correct tool name (0-1)
 * 3. Valid JSON (0-1)
 * 4. Pattern provided (0-1)
 * 5. Pattern executes (0-1)
 * 6. Correct match count (0-3)
 * 7. Metavariable quality (0-1)
 * 8. Thinking present (0-1)
 */
async function toolUseReward(outputs: RewardOutput[]): Promise<Float32Array> {
  const rewards = new Float32Array(outputs.length);

  const DEBUG = process.env.DEBUG_REWARDS === '1';

  for (let i = 0; i < outputs.length; i++) {
    const output = outputs[i];
    const { completion, expectedAnswer } = output;

    // Parse metadata from expectedAnswer (stored as JSON in dataset)
    let metadata: {
      correctPattern?: string;
      expectedMatches?: number;
      language?: string;
      codeContext?: string;
    } = {};

    try {
      if (expectedAnswer) {
        metadata = JSON.parse(expectedAnswer);
      }
    } catch {
      // If parsing fails, use empty metadata
    }

    let score = 0;

    // Tool calls are pre-parsed in completion.toolCalls
    const toolCalls = completion.toolCalls;

    // Check for thinking (pre-extracted in completion.thinking)
    if (completion.thinking) {
      score += 0.5;
    }

    // 1. Tool call present (0-1)
    if (toolCalls.length === 0) {
      // Give small credit if completion mentions ast-grep at all
      if (completion.rawText.includes('ast_grep') || completion.rawText.includes('ast-grep')) {
        score += 0.2;
      }
      rewards[i] = score;
      if (DEBUG) {
        console.log(`\n======== REWARD DEBUG ${i} =========`);
        console.log('Completion length:', completion.rawText.length, 'tokens:', completion.numTokens);
        console.log('Completion:', completion.rawText);
        console.log('Finish reason:', completion.finishReason);
        console.log('No tool calls found');
        console.log('Score:', score);
        console.log('=======================================\n');
      }
      continue;
    }
    score += 1;

    const call = toolCalls[0];

    // 2. Correct tool name (0-1)
    if (call.name !== 'ast_grep') {
      score += 0.2; // Partial credit for calling some tool
      rewards[i] = score;
      continue;
    }
    score += 1;

    // 3. Valid JSON (0-1)
    if (call.status !== 'ok') {
      rewards[i] = score;
      continue;
    }
    score += 1;

    // 4. Pattern provided (0-1)
    const args = call.arguments as Record<string, unknown>;
    const pattern = args?.pattern;
    if (!pattern || typeof pattern !== 'string') {
      rewards[i] = score;
      continue;
    }
    score += 1;

    // 5. Pattern executes (0-1) + 6. Correct match count (0-3)
    try {
      const lang = getLang(metadata.language || 'javascript');
      const codeContext = metadata.codeContext || '';

      if (codeContext) {
        const root = parse(lang, codeContext).root();
        const matches = root.findAll(pattern);
        score += 1; // Pattern executes successfully

        // 6. Correct match count
        const expected = metadata.expectedMatches || 0;
        const actual = matches.length;

        if (actual === expected) {
          score += 3; // Exact match
        } else if (expected > 0 && actual > 0 && Math.abs(actual - expected) <= 1) {
          score += 1.5; // Close
        } else if (actual > 0) {
          score += 0.5; // Found something
        }
      } else {
        // No code context to validate against - just check if pattern is reasonable
        score += 0.5;
      }
    } catch {
      // Pattern doesn't execute - no additional points
    }

    // 7. Metavariable quality (0-1)
    if (/\$\$\$[A-Z_]+/.test(pattern)) {
      score += 0.3; // Correct variadic ($$$)
    }
    if (/\$[A-Z_]+/.test(pattern) && !/\$\$/.test(pattern.replace(/\$\$\$/g, ''))) {
      score += 0.3; // Correct single ($) without $$
    }
    if (/\$[a-z]/.test(pattern)) {
      score -= 0.3; // Lowercase penalty
    }
    // Bonus for having multiple metavariables (more expressive pattern)
    const metavarCount = (pattern.match(/\$+[A-Z_]+/g) || []).length;
    if (metavarCount >= 2) {
      score += 0.2;
    }
    if (metavarCount >= 3) {
      score += 0.2;
    }

    // Round to 1 decimal place to avoid float32 precision artifacts (5.800000190734863 → 5.8)
    rewards[i] = Math.round(Math.max(0, Math.min(score, 10)) * 10) / 10;

    // Debug logging
    if (DEBUG) {
      console.log(`\n======== REWARD DEBUG ${i} ==========`);
      console.log('Completion:', completion.rawText.slice(0, 500));
      console.log('Tool call:', call.name, call.arguments);
      console.log('Pattern:', pattern);
      console.log('Expected matches:', metadata.expectedMatches);
      console.log('Score:', rewards[i]);
      console.log('=======================================\n');
    }
  }

  return rewards;
}

// ============================================================================
// Main Training Function
// ============================================================================

async function main() {
  const { values } = parseArgs({
    options: {
      'model-path': {
        type: 'string',
        short: 'm',
        default: DEFAULT_MODEL_PATH,
      },
      'num-examples': {
        type: 'string',
        short: 'n',
        default: String(DEFAULT_NUM_EXAMPLES),
      },
      'output-dir': {
        type: 'string',
        short: 'o',
        default: DEFAULT_OUTPUT_DIR,
      },
      'dry-run': {
        type: 'boolean',
        default: false,
      },
      resume: {
        type: 'boolean',
        short: 'r',
        default: false,
      },
    },
  });

  const modelPath = resolve(values['model-path']!);
  const numExamples = Number.parseInt(values['num-examples']!, 10);
  const outputDir = resolve(values['output-dir']!);
  const dryRun = values['dry-run'] || false;
  const resume = values['resume'] || false;

  // Create logger (auto-detects TUI mode from environment)
  const logger = createTrainingLogger({ outputDir });

  // Decorative banner (suppressed in TUI mode)
  logger.banner(
    '╔════════════════════════════════════════════════════════╗',
    '║   GRPO Training: Teaching Tool Use (ast-grep)          ║',
    '║   Using native tool calling format                     ║',
    '╚════════════════════════════════════════════════════════╝',
    '',
  );

  // Generate dataset using curriculum learning
  logger.info('📚 Generating ast-grep training dataset with curriculum learning...');
  const patterns = generateCurriculumDataset(numExamples);
  const examples = patternsToDatasetExamples(patterns);

  // Count categories
  const categoryCounts = patterns.reduce(
    (acc, p) => {
      acc[p.category] = (acc[p.category] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>,
  );

  logger.info(`✓ Generated ${examples.length} training examples`);
  logger.info(`  - Basic: ${categoryCounts.basic || 0}`);
  logger.info(`  - Intermediate: ${categoryCounts.intermediate || 0}`);
  logger.info(`  - Advanced: ${categoryCounts.advanced || 0}`);

  // Show sample (suppressed in TUI mode)
  logger.banner(
    '',
    'Sample training example:',
    '─────────────────────────────────────────────────────',
    `Task: ${patterns[0].description}`,
    `Pattern: ${patterns[0].pattern}`,
    `Category: ${patterns[0].category}`,
    `Expected matches: ${patterns[0].expectedMatches}`,
    '─────────────────────────────────────────────────────',
    '',
  );

  if (dryRun) {
    console.log('Dry run mode - showing more examples:\n');
    for (let i = 0; i < Math.min(10, patterns.length); i++) {
      console.log(`Example ${i + 1} [${patterns[i].category}]:`);
      console.log('  Task:', patterns[i].description);
      console.log('  Pattern:', patterns[i].pattern);
      console.log('  Language:', patterns[i].language);
      console.log();
    }
    return;
  }

  // Check if model exists
  if (!existsSync(modelPath)) {
    logger.error(`❌ Model not found at: ${modelPath}`);
    logger.error('Please download the model first:');
    logger.error('   yarn download:qwen3');
    process.exitCode = 1;
    return;
  }

  logger.banner(
    'Configuration:',
    '─────────────────────────────────────────────────────',
    `Model: ${modelPath}`,
    `Training examples: ${numExamples}`,
    `Output directory: ${outputDir}`,
    `Group size: 4 generations per prompt`,
    `Learning rate: 1e-6`,
    `Epochs: 10`,
    'Format: Native tool calling (<tool_call>JSON</tool_call>)',
    '─────────────────────────────────────────────────────',
    '',
  );

  // Configure GRPO trainer with improved settings
  const config: GRPOConfig = {
    // Model
    modelConfig: 'qwen3-0.6b',
    modelPath,

    // Training hyperparameters
    learningRate: 1e-6,
    numEpochs: 5,
    batchSize: 1,
    gradientAccumulationSteps: 8,

    // GRPO parameters
    groupSize: 4,
    clipEpsilon: 0.2,
    klCoef: 0.0, // KL penalty requires reference model (not supported in autograd yet)
    advantageNormalization: true,
    maxCompletionLengthForTraining: 1024,

    // Generation parameters
    maxNewTokens: 4096,
    temperature: 0.8,
    topP: 0.95,
    topK: 50,
    repetitionPenalty: 1.1, // Prevent repetitive outputs

    // Reward configuration
    rewardType: 'function',
    rewardFunction: toolUseReward,

    // Loss configuration
    lossType: 'grpo',

    // Optimization
    weightDecay: 0.01,
    gradientClipNorm: 1.0,

    // Logging and checkpointing
    logInterval: 5,
    saveInterval: 25,
    evalInterval: 50, // Regular validation
    outputDir,
    logConsole: process.env.MLX_TUI_MODE !== '1',
    logJsonl: process.env.MLX_TUI_MODE !== '1',
    runName: 'ast-grep-tool-use',
    device: 'metal',

    // Resume from checkpoint if --resume flag is set
    resumeFromCheckpoint: resume ? 'latest' : undefined,
  };

  logger.banner(
    '🚀 Starting GRPO training with tool calling format...',
    '',
    'Reward components (0-10 points):',
    '  • Tool call present (0-1 pt)',
    '  • Correct tool name (0-1 pt)',
    '  • Valid JSON (0-1 pt)',
    '  • Pattern provided (0-1 pt)',
    '  • Pattern executes (0-1 pt)',
    '  • Correct match count (0-3 pts)',
    '  • Metavariable quality (0-1 pt)',
    '  • Thinking present (0-0.5 pt)',
    '',
    '═══════════════════════════════════════════════════════',
    '',
  );

  // Create trainer and start training
  const trainer = await GRPOTrainer.create(config);

  try {
    await trainer.train(examples);

    logger.banner(
      '',
      '═══════════════════════════════════════════════════════',
      '✅ Training complete!',
      '',
      `Results saved to: ${outputDir}`,
      '',
      'Next steps:',
      '  1. Test the model: node examples/grpo/test-generation.ts',
      `  2. Check logs: cat ${resolve(outputDir, 'ast-grep-tool-use-v2.jsonl')}`,
      `  3. Inspect checkpoint: ${resolve(outputDir, 'final')}`,
      '═══════════════════════════════════════════════════════',
      '',
    );
  } catch (error) {
    logger.error(`❌ Training failed: ${error}`);
    if (error instanceof Error) {
      logger.error(`Error details: ${error.message}`);
      logger.error(`Stack trace: ${error.stack}`);
    }
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error('[train-tool-use] Fatal error:', error);
  process.exitCode = 1;
});
