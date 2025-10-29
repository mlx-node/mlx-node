/**
 * GRPO Training Demo: Teaching Tool Use (ast-grep)
 *
 * This demo shows how to train a language model to use tools effectively.
 * We'll teach Qwen3-0.6B to generate correct ast-grep commands for code search tasks.
 *
 * Key concepts:
 * - Task-specific dataset generation
 * - Execution-based reward functions
 * - Safe tool execution in sandbox
 * - Multi-component rewards (syntax, execution, correctness)
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
import { GRPOTrainer, createTrainingLogger, type GRPOConfig, type DatasetExample } from '@mlx-node/trl';

const DEFAULT_MODEL_PATH = resolve(process.cwd(), '.cache', 'models', 'qwen3-0.6b-mlx-bf16');
const DEFAULT_NUM_EXAMPLES = 100;
const DEFAULT_OUTPUT_DIR = resolve(process.cwd(), 'outputs', 'grpo-tool-use');

// ============================================================================
// Dataset: ast-grep Code Search Tasks
// ============================================================================

interface AstGrepTask {
  description: string; // Natural language description
  codeContext: string; // Sample code to search
  correctPattern: string; // Correct ast-grep pattern
  language: string; // Programming language
  expectedMatches: number; // Expected number of matches
}

/**
 * Generate synthetic dataset for ast-grep training
 *
 * In production, you'd:
 * 1. Collect real code search queries from developers
 * 2. Have experts annotate correct ast-grep patterns
 * 3. Validate patterns execute correctly
 */
function generateAstGrepDataset(numExamples: number): AstGrepTask[] {
  const tasks: AstGrepTask[] = [
    // Basic patterns
    {
      description: 'Find all function definitions',
      codeContext: 'function foo() {}\\nconst bar = () => {};\\nfunction baz(x) { return x; }',
      correctPattern: 'function $NAME($$$ARGS) { $$$BODY }',
      language: 'javascript',
      expectedMatches: 2,
    },
    {
      description: 'Find arrow functions',
      codeContext: 'const foo = () => {};\\nconst bar = (x) => x * 2;\\nfunction baz() {}',
      correctPattern: 'const $NAME = ($$$ARGS) => $BODY',
      language: 'javascript',
      expectedMatches: 2,
    },
    {
      description: 'Find async functions',
      codeContext: 'async function foo() {}\\nfunction bar() {}\\nasync function baz() {}',
      correctPattern: 'async function $NAME($$$ARGS) { $$$BODY }',
      language: 'javascript',
      expectedMatches: 2,
    },

    // Import patterns
    {
      description: 'Find all import statements',
      codeContext: "import { foo } from 'bar';\\nimport baz from 'qux';\\nconst x = 1;",
      correctPattern: 'import $$$SPEC from $SOURCE',
      language: 'javascript',
      expectedMatches: 2,
    },
    {
      description: 'Find named imports',
      codeContext: "import { foo, bar } from 'baz';\\nimport qux from 'quux';",
      correctPattern: 'import { $$$NAMES } from $SOURCE',
      language: 'javascript',
      expectedMatches: 1,
    },

    // Class patterns
    {
      description: 'Find class declarations',
      codeContext: 'class Foo {}\\nconst bar = 1;\\nclass Baz extends Qux {}',
      correctPattern: 'class $NAME { $$$BODY }',
      language: 'javascript',
      expectedMatches: 1,
    },
    {
      description: 'Find classes with extends',
      codeContext: 'class Foo {}\\nclass Bar extends Baz {}\\nclass Qux extends Quux {}',
      correctPattern: 'class $NAME extends $PARENT { $$$BODY }',
      language: 'javascript',
      expectedMatches: 2,
    },

    // Variable patterns
    {
      description: 'Find const declarations',
      codeContext: 'const foo = 1;\\nlet bar = 2;\\nconst baz = 3;\\nvar qux = 4;',
      correctPattern: 'const $NAME = $VALUE',
      language: 'javascript',
      expectedMatches: 2,
    },
    {
      description: 'Find let declarations',
      codeContext: 'const foo = 1;\\nlet bar = 2;\\nlet baz = 3;',
      correctPattern: 'let $NAME = $VALUE',
      language: 'javascript',
      expectedMatches: 2,
    },

    // TypeScript patterns
    {
      description: 'Find TypeScript interfaces',
      codeContext: 'interface Foo {}\\ntype Bar = {};\\ninterface Baz { x: number; }',
      correctPattern: 'interface $NAME { $$$BODY }',
      language: 'typescript',
      expectedMatches: 2,
    },
    {
      description: 'Find type aliases',
      codeContext: 'type Foo = string;\\ninterface Bar {}\\ntype Baz = number;',
      correctPattern: 'type $NAME = $TYPE',
      language: 'typescript',
      expectedMatches: 2,
    },

    // React patterns
    {
      description: 'Find React functional components',
      codeContext: 'function MyComponent() { return <div />; }\\nconst x = 1;\\nfunction Other() { return <span />; }',
      correctPattern: 'function $NAME($$$ARGS) { return <$$$JSX />; }',
      language: 'javascript',
      expectedMatches: 2,
    },
    {
      description: 'Find useState calls',
      codeContext: 'const [count, setCount] = useState(0);\\nconst x = 1;\\nconst [name, setName] = useState("");',
      correctPattern: 'const [$VAR, $SETTER] = useState($INIT)',
      language: 'javascript',
      expectedMatches: 2,
    },

    // Error handling patterns
    {
      description: 'Find try-catch blocks',
      codeContext: 'try { foo(); } catch (e) { bar(); }\\nconst x = 1;\\ntry { baz(); } catch (err) { qux(); }',
      correctPattern: 'try { $$$TRY } catch ($ERR) { $$$CATCH }',
      language: 'javascript',
      expectedMatches: 2,
    },

    // Conditional patterns
    {
      description: 'Find all if statements',
      codeContext: 'if (x) { foo(); }\\nif (y) { bar(); } else { baz(); }\\nif (z) { qux(); }',
      correctPattern: 'if ($COND) { $$$BODY }',
      language: 'javascript',
      expectedMatches: 3,
    },
  ];

  // Duplicate tasks to reach desired number
  const result: AstGrepTask[] = [];
  while (result.length < numExamples) {
    result.push(...tasks);
  }

  return result.slice(0, numExamples);
}

/**
 * Convert ast-grep tasks to GRPO training examples
 */
function tasksToDatasetExamples(tasks: AstGrepTask[]): DatasetExample[] {
  return tasks.map((task) => {
    const systemContent = `You are an ast-grep expert. Generate ast-grep patterns for code search tasks.

Output format: <answer>ast-grep -p 'PATTERN'</answer>

Metavariables: $NAME (single), $$ARGS (list), $$$BODY (any)
Keep thinking brief and focused.`;

    const userContent = `Task: ${task.description}

Code context:
\`\`\`${task.language}
${task.codeContext.replace(/\\n/g, '\n')}
\`\`\`

Expected matches: ${task.expectedMatches}

Generate the ast-grep pattern. Be concise.

[METADATA: correctPattern="${task.correctPattern}", expectedMatches=${task.expectedMatches}, language="${task.language}"]`;

    return {
      prompt: [
        { role: 'system' as const, content: systemContent },
        { role: 'user' as const, content: userContent },
      ],
      answer: `ast-grep -p '${task.correctPattern}'`,
      metadata: {
        task: task.description,
        language: task.language,
        correctPattern: task.correctPattern,
        expectedMatches: task.expectedMatches,
        codeContext: task.codeContext,
      },
    };
  });
}

// ============================================================================
// Reward Functions for Tool Use
// ============================================================================

/**
 * Extract ast-grep command from model output
 */
function extractAstGrepCommand(completion: string): string | null {
  // Try to extract from <answer> tags
  const answerMatch = completion.match(/<answer>\s*(.*?)\s*<\/answer>/s);
  if (answerMatch) {
    return answerMatch[1].trim();
  }

  // Try to extract ast-grep command directly
  const cmdMatch = completion.match(/ast-grep\s+.*?(?:\n|$)/);
  if (cmdMatch) {
    return cmdMatch[0].trim();
  }

  return null;
}

/**
 * Extract pattern from ast-grep command
 * Helper to eliminate duplication across reward functions
 */
function extractPattern(command: string | null): string | null {
  if (!command) return null;
  const match = command.match(/ast-grep\s+(?:-p|--pattern)\s+['"](.+?)['"]/);
  return match ? match[1] : null;
}

/**
 * Reward 1: Syntax Correctness (0-1 points)
 * Does the command have valid ast-grep syntax?
 */
function syntaxReward(command: string | null): number {
  if (!command) return 0;

  // Must start with ast-grep
  if (!command.startsWith('ast-grep')) return 0;

  // Must have -p flag and pattern
  if (!command.includes('-p')) return 0;
  if (!command.includes("'") && !command.includes('"')) return 0;

  return 1.0;
}

/**
 * Reward 2: Pattern Validation (0-2 points)
 * Does the pattern use valid ast-grep syntax?
 */
function patternReward(command: string | null): number {
  const pattern = extractPattern(command);
  if (!pattern) return 0;

  let score = 0.0; // Valid pattern

  // Bonus for code structure (not just metavariables)
  const withoutMetavars = pattern.replace(/\$+\w+/g, '');
  if (/[a-zA-Z(){}[\]]/.test(withoutMetavars)) score += 0.5;

  // Round to avoid floating point precision issues
  return Math.round(Math.min(score, 2.0) * 100) / 100;
}

/**
 * Reward 3: Execution Success (0-1 points)
 * Does the command execute without errors?
 */
function executionReward(command: string | null, codeContext: string, language: string): number {
  const pattern = extractPattern(command);
  if (!pattern) return 0;

  try {
    const lang = language === 'typescript' ? Lang.TypeScript : Lang.JavaScript;
    const root = parse(lang, codeContext).root();
    root.findAll(pattern); // Execute to verify it works
    return 1.0;
  } catch {
    return 0; // Execution failed
  }
}

/**
 * Reward 4: Correctness (0-3 points)
 * Does the command produce the expected results?
 */
function correctnessReward(
  command: string | null,
  correctPattern: string,
  codeContext: string,
  language: string,
  expectedMatches: number,
): number {
  const pattern = extractPattern(command);
  if (!pattern) return 0;

  // Need correct pattern to check correctness
  if (!correctPattern || correctPattern.trim().length === 0) return 0;

  try {
    // Both patterns are safe, proceed with ast-grep
    const lang = language === 'typescript' ? Lang.TypeScript : Lang.JavaScript;
    const root = parse(lang, codeContext).root();

    // Count matches from generated pattern
    const generatedMatches = root.findAll(pattern);
    const generatedCount = generatedMatches.length;

    // Count matches from correct pattern
    const correctMatches = root.findAll(correctPattern);
    const correctCount = correctMatches.length;

    let score = 0;

    // Exact pattern match (perfect!)
    if (pattern === correctPattern) {
      score += 3.0;
    }
    // Found exactly the same matches as correct pattern
    else if (generatedCount === correctCount && correctCount === expectedMatches) {
      score += 2.5;
    }
    // Same number of matches as expected
    else if (generatedCount === expectedMatches) {
      score += 2.0;
    }
    // Close to expected matches
    else if (Math.abs(generatedCount - expectedMatches) <= 1) {
      score += 1.0;
    }
    // At least found something
    else if (generatedCount > 0) {
      score += 0.5;
    }

    return score;
  } catch {
    return 0;
  }
}

/**
 * Reward 5: Format Compliance (0-1 point)
 * Does the output follow the required XML format?
 * Note: Qwen3 automatically generates <think> tags
 */
function formatReward(completion: string): number {
  let score = 0;

  // Check for thinking (model generates <think> automatically)
  const hasThink = /<think>[\s\S]*?<\/think>/.test(completion);

  // Reward for having closed thinking tags
  if (hasThink) {
    score += 0.2;
  }

  // Has <answer> tags (REQUIRED) - this is the main goal
  const answerMatch = completion.match(/<answer>([\s\S]*?<\/answer>)/);
  if (answerMatch) {
    score += 0.5; // Main points for having complete answer tag

    // Answer contains ast-grep command
    const answerContent = answerMatch[1];
    if (answerContent.includes('ast-grep')) {
      score += 0.2;

      // Has complete command structure with -p flag and pattern in quotes
      if (/ast-grep\s+(?:-p|--pattern)\s+['"][^'"]+['"]/.test(answerContent)) {
        score += 0.1; // Bonus for complete command
      }
    }
  }

  // Bootstrap rewards: Give partial credit for incomplete but promising outputs
  if (!answerMatch) {
    // If has opening <answer> but no closing (ran out of tokens)
    if (/<answer>/.test(completion) && !/<\/answer>/.test(completion)) {
      score += 0.2; // Partial credit for starting answer

      // Still check if ast-grep is present after <answer>
      const afterAnswer = completion.split('<answer>')[1] || '';
      if (afterAnswer.includes('ast-grep')) {
        score += 0.1;
      }
    }
    // If no answer tags at all, minimal credit
    else if (/ast-grep/.test(completion)) {
      score += 0.1;
    }

    // Reward having pattern structure anywhere
    if (/\$\w+/.test(completion)) {
      score += 0.05;
    }
  }

  // Round to avoid floating point precision issues (0.9999999999999999 -> 1.0)
  return Math.round(Math.min(score, 1.0) * 100) / 100;
}

/**
 * Combined reward function for tool-use training
 */
async function toolUseReward(
  prompts: string[],
  completions: string[],
  answers: (string | null)[],
): Promise<Float32Array> {
  const numCompletions = completions.length;
  const rewards = new Float32Array(numCompletions);

  // Debug flag - set to true to see what model generates
  const DEBUG = process.env.DEBUG_REWARDS === '1';

  for (let i = 0; i < numCompletions; i++) {
    const completion = completions[i];
    const prompt = prompts[i];

    // Extract metadata from prompt
    let codeContext = '';
    let language = 'javascript';
    let expectedMatches = 1;
    let correctPattern = '';

    // Parse code context and language from prompt
    const contextMatch = prompt.match(/Code context:\s*```(\w+)\s*([\s\S]*?)```/);
    if (contextMatch) {
      language = contextMatch[1];
      codeContext = contextMatch[2];
    }

    // Extract metadata from embedded METADATA tag in prompt
    const metadataMatch = prompt.match(
      /\[METADATA: correctPattern="([^"]+)", expectedMatches=(\d+), language="([^"]+)"\]/,
    );
    if (metadataMatch) {
      correctPattern = metadataMatch[1];
      expectedMatches = Number.parseInt(metadataMatch[2], 10);
      language = metadataMatch[3];
    } else {
      // Fallback: try to extract from answers (if provided)
      if (answers[i]) {
        const answerPatternMatch = answers[i]!.match(/ast-grep\s+(?:-p|--pattern)\s+['"](.+?)['"]/);
        if (answerPatternMatch) {
          correctPattern = answerPatternMatch[1];
        }
      }
    }

    // Extract command from completion
    const command = extractAstGrepCommand(completion);

    // Calculate component rewards
    const r1 = syntaxReward(command); // 0-1
    const r2 = patternReward(command); // 0-2
    const r3 = executionReward(command, codeContext, language); // 0-1
    const r4 = correctnessReward(command, correctPattern, codeContext, language, expectedMatches); // 0-3
    const r5 = formatReward(completion); // 0-1

    // Debug logging
    if (DEBUG && i === 0) {
      console.log('\n=== REWARD DEBUG (first completion) ===');
      console.log('Completion:', completion);
      console.log('Extracted command:', command || '(null)');
      console.log('Metadata extracted:');
      console.log('  - correctPattern:', correctPattern || '(empty)');
      console.log('  - expectedMatches:', expectedMatches);
      console.log('  - language:', language);
      console.log('Rewards: syntax=' + r1, 'pattern=' + r2, 'exec=' + r3, 'correct=' + r4, 'format=' + r5);
      console.log('Total reward:', r1 + r2 + r3 + r4 + r5);
      console.log('=======================================\n');
    }

    // Total: 0-8 points
    rewards[i] = r1 + r2 + r3 + r4 + r5;
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
    '╚════════════════════════════════════════════════════════╝',
    '',
  );

  // Generate dataset
  logger.info('📚 Generating ast-grep training dataset...');
  const tasks = generateAstGrepDataset(numExamples);
  const examples = tasksToDatasetExamples(tasks);

  logger.info(`✓ Generated ${examples.length} training examples`);

  // Show sample (suppressed in TUI mode)
  logger.banner(
    '',
    'Sample training example:',
    '─────────────────────────────────────────────────────',
    `Task: ${tasks[0].description}`,
    `Pattern: ${tasks[0].correctPattern}`,
    `Expected matches: ${tasks[0].expectedMatches}`,
    '─────────────────────────────────────────────────────',
    '',
  );

  if (dryRun) {
    console.log('Dry run mode - showing more examples:\n');
    for (let i = 0; i < Math.min(5, tasks.length); i++) {
      console.log(`Example ${i + 1}:`);
      console.log('  Task:', tasks[i].description);
      console.log('  Pattern:', tasks[i].correctPattern);
      console.log('  Language:', tasks[i].language);
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
    `Learning rate: 5e-7`,
    '─────────────────────────────────────────────────────',
    '',
  );

  // Configure GRPO trainer
  const config: GRPOConfig = {
    // Model
    modelConfig: 'qwen3-0.6b',
    modelPath, // Load pretrained weights from disk

    // Training hyperparameters
    learningRate: 5e-7, // Lower LR for tool-use fine-tuning
    numEpochs: 2, // Multiple passes for better learning
    batchSize: 1,
    gradientAccumulationSteps: 4,

    // GRPO parameters
    groupSize: 4, // 4 generations per prompt for diversity
    clipEpsilon: 0.2,
    klCoef: 0.0,
    advantageNormalization: true,

    // Generation parameters
    maxNewTokens: 2048, // Sufficient for thinking + answer
    temperature: 0.7, // Moderate temperature for focused exploration
    topP: 0.95,
    topK: 50,

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
    evalInterval: 25,
    outputDir,
    // Logger auto-detects TUI mode from MLX_TUI_MODE env var
    // In TUI mode, the TUI handles all display; in CLI mode, use console/jsonl
    logConsole: process.env.MLX_TUI_MODE !== '1',
    logJsonl: process.env.MLX_TUI_MODE !== '1',
    runName: 'ast-grep-tool-use',
    device: 'metal',

    // Resume from checkpoint if --resume flag is set
    resumeFromCheckpoint: resume ? 'latest' : undefined,
  };

  logger.banner(
    '🚀 Starting GRPO training...',
    '',
    'Reward components:',
    '  • Syntax correctness (0-1 pt)',
    '  • Pattern validation (0-2 pts)',
    '  • Execution success (0-1 pt)',
    '  • Result correctness (0-3 pts)',
    '  • Format compliance (0-1 pt)',
    '  Total: 0-8 points',
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
      `  2. Check logs: cat ${resolve(outputDir, 'ast-grep-tool-use.jsonl')}`,
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
