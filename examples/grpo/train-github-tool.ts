/**
 * GRPO Training Demo: Teaching GitHub Tool Use (Octokit) with Multi-Turn LSP
 *
 * This demo shows how to train a language model to interact with GitHub via Octokit.
 * The model learns to:
 * 1. Query API documentation using the `lsp` tool when unsure
 * 2. Generate ESM JavaScript code using the `run_js` tool
 *
 * Multi-Turn Training Flow:
 * - Generate first response
 * - If model calls `lsp`: execute lsp tool, append response, generate again
 * - If model calls `run_js`: evaluate with reward function
 * - Max 2 turns (lsp → run_js)
 *
 * Usage:
 *   yarn oxnode examples/grpo/train-github-tool.ts [options]
 *
 * Options:
 *   --target <id>, -t       Focus training on specific scenarios (can be used multiple times)
 *                           Valid IDs: get-inline-comments, get-comments, get-file-diff,
 *                           reply-inline-comment, resolve-inline-comment, batch-resolve-comments,
 *                           change-pr-title, check-workflow-status, get-ci-outputs, trigger-workflow
 *   --model-path <path>     Path to model (default: .cache/models/qwen3-1.7b-mlx-bf16)
 *   --num-examples <n>      Number of examples (default: 100)
 *   --output-dir <path>     Output directory (default: outputs/grpo-github-tool)
 *   --run-name <name>       Name for this training run (auto-generated from target if not provided)
 *   --dry-run               Just show dataset, don't train
 *   --resume, -r            Resume training from latest checkpoint
 *
 * Examples:
 *   yarn oxnode examples/grpo/train-github-tool.ts                    # Interactive multi-select
 *   yarn oxnode examples/grpo/train-github-tool.ts -t get-comments    # Train on get-comments only
 *   yarn oxnode examples/grpo/train-github-tool.ts -t get-comments -t get-file-diff  # Train on multiple
 *   yarn oxnode examples/grpo/train-github-tool.ts -t get-file-diff --dry-run  # Preview dataset
 */

import { parseArgs } from 'node:util';
import { resolve, join } from 'node:path';
import { existsSync } from 'node:fs';
import { checkbox, Separator } from '@inquirer/prompts';
import {
  GRPOTrainer,
  createTrainingLogger,
  type GRPOTrainerConfig,
  type DatasetExample,
  type ChatMessage,
  type ToolDefinition,
} from '@mlx-node/trl';
import { buildRewardOutputs } from '@mlx-node/core';
import { createToolDefinition } from '@mlx-node/lm';
import {
  generateCurriculumDataset,
  generateMultiScenarioDataset,
  getAllScenarioInfo,
  countByCategory,
  type GitHubScenario,
} from './github-dataset';
import { githubToolReward } from './github-tool-reward';
import { initLspService, executeToolCall, type LspService } from './lsp';

import { SYSTEM_PROMPT } from '../sft/prompts';

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format a tool response for inclusion in a message.
 * Creates a properly formatted tool response string wrapped in `<tool_response>` tags.
 */
function formatToolResponse(content: unknown): string {
  const contentStr = typeof content === 'string' ? content : JSON.stringify(content);
  return `<tool_response>
${contentStr}
</tool_response>`;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_MODEL_PATH = resolve(process.cwd(), '.cache/models/qwen3-0.8b-mlx-bf16');
const SFT_MODEL_PATH = resolve(process.cwd(), 'outputs/sft-octokit/final');
const DEFAULT_NUM_EXAMPLES = 100;

// Global LSP service instance (initialized once at startup)
let lspService: LspService | null = null;

// ============================================================================
// Tool Definitions - These are passed to the model during generation
// ============================================================================

/**
 * LSP tool - Query Octokit REST API documentation
 */
const lspTool: ToolDefinition = createToolDefinition(
  'lsp',
  'Query Octokit REST API documentation. Use this BEFORE writing code to understand the API parameters and return types.',
  {
    method: {
      type: 'string',
      description: 'API method to query (e.g., "octokit.rest.pulls.list", "octokit.rest.issues.createComment")',
    },
    kind: {
      type: 'string',
      enum: ['parameters', 'return'],
      description: 'What to query: "parameters" for input parameters, "return" for response type',
    },
  },
  ['method', 'kind'],
);

/**
 * run_js tool - Execute JavaScript code using Octokit
 */
const runJsTool: ToolDefinition = createToolDefinition(
  'run_js',
  'Execute JavaScript code that uses the Octokit API. The code should be a complete ESM module with an async main() function.',
  {
    code: {
      type: 'string',
      description: 'JavaScript code to execute. Must export an async main(octokit) function.',
    },
  },
  ['code'],
);

/**
 * All tools available for training
 */
const TRAINING_TOOLS: ToolDefinition[] = [lspTool, runJsTool];

/**
 * Execute an LSP tool call and return the formatted response
 */
function executeLspTool(args: { method?: string; kind?: string }): string {
  if (!lspService) {
    return 'Error: LSP service not initialized';
  }
  if (!args.method || !args.kind) {
    return 'Error: Missing required arguments "method" and "kind"';
  }
  if (args.kind !== 'parameters' && args.kind !== 'return') {
    return `Error: Invalid kind "${args.kind}". Must be "parameters" or "return"`;
  }
  return executeToolCall(lspService, { method: args.method, kind: args.kind });
}

/**
 * Parse tool calls from raw completion text.
 * Returns the first tool call found, or null if none.
 */
function parseToolCallFromText(text: string): { name: string; arguments: Record<string, unknown> } | null {
  const toolCallMatch = text.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/);
  if (!toolCallMatch) return null;

  try {
    const json = JSON.parse(toolCallMatch[1]);
    if (json.name && typeof json.arguments === 'object') {
      return { name: json.name, arguments: json.arguments };
    }
  } catch {
    // Invalid JSON, return null
  }
  return null;
}

/**
 * Interactive CLI prompt to select training scenarios (multi-select)
 */
async function selectScenarios(): Promise<string[]> {
  const scenarios = getAllScenarioInfo();

  const readScenarios = scenarios.filter((s) => s.category === 'read');
  const writeScenarios = scenarios.filter((s) => s.category === 'write');
  const workflowScenarios = scenarios.filter((s) => s.category === 'workflow');

  const selected = await checkbox({
    message: 'Select training scenarios (Space to toggle, Enter to confirm):',
    choices: [
      new Separator('── Read Operations ──'),
      ...readScenarios.map((s) => ({
        value: s.id,
        name: `${s.id} - ${s.description}`,
      })),
      new Separator('── Write Operations ──'),
      ...writeScenarios.map((s) => ({
        value: s.id,
        name: `${s.id} - ${s.description}`,
      })),
      new Separator('── Workflow Operations ──'),
      ...workflowScenarios.map((s) => ({
        value: s.id,
        name: `${s.id} - ${s.description}`,
      })),
    ],
  });

  return selected;
}

/**
 * Build choices for TUI prompt (simpler format than @inquirer/prompts).
 */
function buildTuiChoices(): Array<{ value: string; label: string; description?: string }> {
  const scenarios = getAllScenarioInfo();
  const choices: Array<{ value: string; label: string; description?: string }> = [];

  // "All scenarios" option with empty string value
  choices.push({
    value: '',
    label: 'All Scenarios',
    description: 'Train on all 10 APIs (harder)',
  });

  // Read operations
  for (const s of scenarios.filter((s) => s.category === 'read')) {
    choices.push({
      value: s.id,
      label: `[Read] ${s.id}`,
      description: `${s.apiMethod} (${s.numPromptVariations} variations)`,
    });
  }

  // Write operations
  for (const s of scenarios.filter((s) => s.category === 'write')) {
    choices.push({
      value: s.id,
      label: `[Write] ${s.id}`,
      description: `${s.apiMethod} (${s.numPromptVariations} variations)`,
    });
  }

  // Workflow operations
  for (const s of scenarios.filter((s) => s.category === 'workflow')) {
    choices.push({
      value: s.id,
      label: `[Workflow] ${s.id}`,
      description: `${s.apiMethod} (${s.numPromptVariations} variations)`,
    });
  }

  return choices;
}

// ============================================================================
// Dataset Generation
// ============================================================================

/**
 * Convert GitHub scenarios to GRPO training examples.
 * When usePerScenarioPrompt is true (default), each example gets a tailored
 * system prompt with the exact code pattern for that scenario.
 *
 * @param scenarios - Array of GitHub scenarios to convert
 * @param systemPrompt - Optional system prompt to use for all examples (when usePerScenarioPrompt is false)
 * @param usePerScenarioPrompt - Whether to use per-scenario prompts (default: true)
 */
function scenariosToDatasetExamples(scenarios: GitHubScenario[], systemPrompt: string): DatasetExample[] {
  return scenarios.map((scenario) => {
    // Create user message
    const userContent = scenario.prompt;

    return {
      prompt: [
        { role: 'system' as const, content: systemPrompt },
        { role: 'user' as const, content: userContent },
      ],
      // Store all scenario metadata for the reward function
      metadata: {
        id: scenario.id,
        description: scenario.description,
        category: scenario.category,
        expectedApiCall: scenario.expectedApiCall,
        expectedApiMethod: scenario.expectedApiMethod,
        expectedReturnShape: scenario.expectedReturnShape,
      },
    };
  });
}

// ============================================================================
// Main Training Function
// ============================================================================

async function main() {
  const { values } = parseArgs({
    options: {
      target: {
        type: 'string',
        short: 't',
        multiple: true, // Allow multiple -t flags
      },
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
      },
      'run-name': {
        type: 'string',
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

  // Prefer SFT checkpoint if it exists (unless explicit model path is provided)
  const explicitModelPath = values['model-path'] !== DEFAULT_MODEL_PATH;
  let modelPath: string;
  if (explicitModelPath) {
    modelPath = resolve(values['model-path']!);
  } else if (existsSync(SFT_MODEL_PATH)) {
    modelPath = SFT_MODEL_PATH;
    console.log(`📦 Found SFT checkpoint at ${SFT_MODEL_PATH}, using it as base model`);
  } else {
    modelPath = resolve(values['model-path']!);
  }

  const numExamples = Number.parseInt(values['num-examples']!, 10);
  const dryRun = values['dry-run'] || false;
  const resume = values['resume'] || false;

  // Check for TUI mode and create preliminary logger for prompt
  const isTuiMode = process.env.MLX_TUI_MODE === '1';
  const prelimLogger = createTrainingLogger({});

  // Get targets from CLI (could be array or single value)
  const cliTargets = values['target'] ? (Array.isArray(values['target']) ? values['target'] : [values['target']]) : [];

  let targetScenarios: string[] = [];

  if (cliTargets.length > 0) {
    targetScenarios = cliTargets;
  } else if (isTuiMode) {
    // TUI mode: use multi-select prompt
    const selected = await prelimLogger.promptMulti(
      'training-targets',
      'Select training scenarios (Space to toggle, Enter to confirm):',
      buildTuiChoices(),
    );
    targetScenarios = selected?.filter((v) => v !== '') ?? [];
  } else {
    // CLI mode: use @inquirer/prompts checkbox
    targetScenarios = await selectScenarios();
  }

  // Output directory and run name are unified - checkpoints are shared across targets
  const outputDir = values['output-dir']
    ? resolve(values['output-dir'])
    : resolve(process.cwd(), 'outputs', 'grpo-github-tool');
  const runName = values['run-name'] || 'github-tool-use';

  // Create logger (auto-detects TUI mode from environment)
  const logger = createTrainingLogger({ outputDir });

  // Decorative banner
  const bannerTitle =
    targetScenarios.length > 0
      ? `   GRPO Training: ${targetScenarios.join(', ')}   `
      : '   GRPO Training: Teaching GitHub Tool Use (Octokit)  ';
  logger.banner(
    '',
    '======================================================',
    bannerTitle,
    '   Model learns to generate ESM JavaScript code       ',
    '======================================================',
    '',
  );

  // Generate dataset based on targets
  const scenarios =
    targetScenarios.length > 0
      ? generateMultiScenarioDataset(targetScenarios, numExamples)
      : generateCurriculumDataset(numExamples);

  // Use per-scenario prompts (each example gets system prompt with exact code hint)
  const examples = scenariosToDatasetExamples(scenarios, SYSTEM_PROMPT);

  // Count categories
  const categoryCounts = countByCategory(scenarios);

  logger.info(`Generated ${examples.length} training examples`);
  if (targetScenarios.length > 0) {
    logger.info(`  - Targets: ${targetScenarios.join(', ')} (focused training)`);
    logger.info(`  - Prompt variations: ${scenarios.length}`);
  } else {
    logger.info(`  - Read operations: ${categoryCounts.read}`);
    logger.info(`  - Write operations: ${categoryCounts.write}`);
    logger.info(`  - Workflow operations: ${categoryCounts.workflow}`);
  }

  // Show sample
  logger.banner(
    '',
    'Sample training example:',
    '-----------------------------------------------------',
    `Scenario: ${scenarios[0].description}`,
    `Prompt: ${scenarios[0].prompt}`,
    `Category: ${scenarios[0].category}`,
    `Expected API: ${scenarios[0].expectedApiCall}`,
    '-----------------------------------------------------',
    '',
  );

  if (dryRun) {
    console.log('Dry run mode - showing more examples:\n');
    for (let i = 0; i < Math.min(10, scenarios.length); i++) {
      console.log(`Example ${i + 1} [${scenarios[i].category}]:`);
      console.log('  Prompt:', scenarios[i].prompt);
      console.log('  API:', scenarios[i].expectedApiCall);
      console.log();
    }
    return;
  }

  // Initialize LSP service for multi-turn generation
  logger.info('Initializing LSP service for API documentation...');
  lspService = await initLspService();
  logger.info(`LSP service ready: ${lspService.namespaces.size} namespaces, ${lspService.methods.size} methods`);

  // Check if model exists
  if (!existsSync(modelPath)) {
    logger.error(`Model not found at: ${modelPath}`);
    logger.error('Please download the model first:');
    logger.error('   yarn download:qwen3');
    process.exitCode = 1;
    return;
  }

  logger.banner(
    'Configuration:',
    '-----------------------------------------------------',
    `Model: ${modelPath}`,
    `Training examples: ${numExamples}`,
    `Output directory: ${outputDir}`,
    `Group size: 6 generations per prompt`,
    `Learning rate: 1e-6`,
    `Epochs: 10`,
    'Format: Tool calling (<tool_call>JSON</tool_call>)',
    '-----------------------------------------------------',
    '',
  );

  // Configure GRPO trainer
  // Targets: avg≥8.5, high(≥8)≥95%, perfect(≥9)≥90%, fail(≤2)≤1%, truncation≤0.1%, variance≤0.5
  const config: GRPOTrainerConfig<string> = {
    // Model
    modelName: 'qwen3',
    modelPath,

    // Training hyperparameters
    learningRate: 1e-6,
    numEpochs: 10,
    // GRPO parameters
    groupSize: 5,
    gradientAccumulationSteps: 1,
    batchSize: 2,
    lmHeadChunkSize: 2,
    forwardChunkSize: 1,
    clipEpsilon: 0.2,
    klCoef: 0.0,
    advantageNormalization: true,

    maxCompletionLength: 1536,
    temperature: 0.7,
    topP: 0.9,
    topK: 50,
    repetitionPenalty: 1.1,

    // Tool calling - pass tool definitions so model can generate <tool_call> tags
    tools: TRAINING_TOOLS,
    enableThinking: true, // Allow model to think before tool calls

    // Reward configuration
    // Note: We use a custom training loop below that calls githubToolReward directly
    // with per-output contexts, so we don't set rewardFunction in the config.
    // The reward function is called manually in the training loop (line ~636).

    // Loss configuration
    lossType: 'grpo',

    // Optimization
    weightDecay: 0.01,
    gradientClipNorm: 1.0,

    // Logging and checkpointing
    logInterval: 5,
    saveInterval: 8,
    evalInterval: 50,
    outputDir,
    logConsole: process.env.MLX_TUI_MODE !== '1',
    logJsonl: process.env.MLX_TUI_MODE !== '1',
    runName,
    device: 'metal',

    // Resume from checkpoint
    resumeFromCheckpoint: resume ? 'latest' : undefined,

    // Output recording (SQLite database for debugging)
    outputStore: {
      enabled: true,
      localPath: join(outputDir, 'outputs.db'),
    },
  };

  logger.banner(
    'Starting GRPO training with GitHub tool format...',
    '',
    'Reward components (exactly 10 points):',
    '  1. Thinking quality    (0-1): reasoning about approach',
    '  2. Tool format         (0-1): <tool_call> with run_js',
    '  3. Code structure      (0-3): syntax, imports, async+try/catch',
    '  4. API correctness     (0-4): method + params + execution',
    '  5. Execution bonus     (0-1): all assertions matched',
    '',
    '=====================================================',
    '',
  );

  // Create trainer
  const trainer = await GRPOTrainer.create(config);
  const engine = trainer.getNativeEngine();
  const groupSize = config.groupSize ?? 4;

  try {
    // Multi-turn training loop
    // Instead of using trainer.train() which doesn't support multi-turn,
    // we use the low-level API with custom generation logic.

    const numEpochs = config.numEpochs ?? 10;
    const batchSize = config.batchSize ?? 1;
    const saveInterval = config.saveInterval ?? 25;

    // Track LSP usage statistics
    let totalLspCalls = 0;
    let totalCompletions = 0;

    // Calculate steps per epoch for TUI display
    const stepsPerEpoch = Math.ceil(examples.length / batchSize);

    // Get starting position from trainer (handles checkpoint resumption)
    const startStep = trainer.getStep();
    let startEpoch = trainer.getEpoch();
    const startBatchIdx = startStep > 0 ? startStep % stepsPerEpoch : 0;

    // Validate epoch matches step (handles corrupted training_state.json)
    // If training_state.json was corrupted, epoch may be wrong even if step was derived from checkpoint name
    if (startStep > 0) {
      const calculatedEpoch = Math.floor(startStep / stepsPerEpoch);
      if (startEpoch !== calculatedEpoch) {
        logger.warn(
          `Epoch mismatch: training_state.json says epoch ${startEpoch}, but step ${startStep} suggests epoch ${calculatedEpoch}. Using calculated.`,
        );
        startEpoch = calculatedEpoch;
      }
    }

    // Send TUI init message (required for TUI to show Running state and config)
    const modelName = modelPath.split('/').pop() ?? 'Unknown';
    logger.init(
      modelName,
      {
        trainingType: 'grpo',
        numEpochs,
        batchSize,
        groupSize,
        learningRate: config.learningRate ?? 1e-6,
      },
      examples.length,
    );

    // Initialize output store for database recording (sends databasePath to TUI)
    await trainer.ensureOutputStoreInitialized();

    if (startStep > 0) {
      logger.info(
        `Resuming from step ${startStep} (epoch ${startEpoch + 1}, batch ${startBatchIdx + 1}/${stepsPerEpoch})`,
      );
    }

    for (let epoch = startEpoch; epoch < numEpochs; epoch++) {
      trainer.startEpoch();
      const epochStart = Date.now();

      // Send TUI epoch_start message
      logger.epochStart(epoch, numEpochs, stepsPerEpoch);

      // Calculate starting batch for this epoch (only applies to resumed epoch)
      const epochStartBatch = epoch === startEpoch && startStep > 0 ? startBatchIdx * batchSize : 0;

      // Validate resume position doesn't exceed dataset size
      if (epochStartBatch >= examples.length) {
        logger.warn(
          `Resume batch ${epochStartBatch} exceeds dataset size ${examples.length}. Starting from beginning of next epoch.`,
        );
        continue;
      }

      for (let i = epochStartBatch; i < examples.length; i += batchSize) {
        const batch = examples.slice(i, Math.min(i + batchSize, examples.length));
        const prompts = batch.map((ex) => ex.prompt);
        // Context contains scenario metadata stored as JSON in the metadata field
        const contexts = batch.map((ex) =>
          ex.metadata
            ? JSON.stringify({
                id: ex.metadata.id,
                category: ex.metadata.category,
                expectedApiCall: ex.metadata.expectedApiCall,
                expectedApiMethod: ex.metadata.expectedApiMethod,
                expectedReturnShape: ex.metadata.expectedReturnShape,
              })
            : '',
        );

        // =====================================================================
        // MULTI-TURN TRAINING IMPLEMENTATION
        // =====================================================================
        // The goal is to train the model on BOTH:
        // 1. Turn 1: The LSP call (or direct run_js)
        // 2. Turn 2: The run_js code after receiving LSP response
        //
        // We achieve this by:
        // 1. Generate turn 1 completions → capture tokens/logprobs
        // 2. For completions with LSP calls → generate turn 2 → capture tokens/logprobs
        // 3. CONCATENATE turn 1 + turn 2 tokens/logprobs for LSP completions
        // 4. Build combined GenerateBatchResult with proper lengths
        // 5. Train using the combined result so model learns both turns
        // =====================================================================

        // Step 1: Generate first-turn completions using low-level API for full data
        const turn1Result = await engine.generateBatchForTraining(prompts);
        const firstTurnCompletions = turn1Result.completionTexts;

        // Step 2: Process each completion - check for LSP calls and generate turn 2 if needed
        // We need to build a combined GenerateBatchResult with concatenated tokens/logprobs
        const finalCompletionTexts: string[] = [];
        const combinedTokens: number[] = [];
        const combinedLogprobs: number[] = [];
        const combinedLengths: number[] = [];
        const combinedFinishReasons: string[] = [];
        let stepLspCalls = 0;

        // Track offset into turn1 flattened arrays
        let turn1Offset = 0;

        for (let j = 0; j < firstTurnCompletions.length; j++) {
          const turn1Completion = firstTurnCompletions[j];
          const turn1Length = turn1Result.completionLengths[j];
          const turn1Tokens = turn1Result.completionTokens.slice(turn1Offset, turn1Offset + turn1Length);
          const turn1Logprobs = turn1Result.completionLogprobs.slice(turn1Offset, turn1Offset + turn1Length);
          turn1Offset += turn1Length;

          const toolCall = parseToolCallFromText(turn1Completion);

          if (toolCall?.name === 'lsp') {
            // Execute LSP tool and generate second turn
            stepLspCalls++;
            const lspResult = executeLspTool(toolCall.arguments as { method?: string; kind?: string });
            const toolResponse = formatToolResponse(lspResult);

            // Build extended conversation for second turn
            const promptIdx = Math.floor(j / groupSize);
            const originalPrompt = prompts[promptIdx];

            // Create second-turn prompt with tool response
            const secondTurnPrompt: ChatMessage[] = [
              ...originalPrompt,
              { role: 'assistant' as const, content: turn1Completion },
              { role: 'tool' as const, content: toolResponse },
            ];

            // Generate second turn using generateBatchForTraining to get tokens/logprobs
            // We pass a single-element array since we're generating one completion at a time
            const turn2Result = await engine.generateBatchForTraining([secondTurnPrompt]);

            // Get turn 2 data (only one completion)
            const turn2Completion = turn2Result.completionTexts[0] ?? '';
            const turn2Length = turn2Result.completionLengths[0] ?? 0;
            const turn2Tokens = turn2Result.completionTokens.slice(0, turn2Length);
            const turn2Logprobs = turn2Result.completionLogprobs.slice(0, turn2Length);
            const turn2FinishReason = turn2Result.finishReasons[0] ?? 'eos';

            // CONCATENATE turn 1 + turn 2 tokens and logprobs
            // This is the key fix: we train on ALL tokens from both turns
            combinedTokens.push(...turn1Tokens, ...turn2Tokens);
            combinedLogprobs.push(...turn1Logprobs, ...turn2Logprobs);
            combinedLengths.push(turn1Length + turn2Length);

            // Final text is turn 2 (what we score on)
            // Finish reason comes from turn 2
            finalCompletionTexts.push(turn2Completion);
            combinedFinishReasons.push(turn2FinishReason);
          } else {
            // No LSP call - use turn 1 as-is
            combinedTokens.push(...turn1Tokens);
            combinedLogprobs.push(...turn1Logprobs);
            combinedLengths.push(turn1Length);
            finalCompletionTexts.push(turn1Completion);
            combinedFinishReasons.push(turn1Result.finishReasons[j] ?? 'eos');
          }
        }

        totalLspCalls += stepLspCalls;
        totalCompletions += firstTurnCompletions.length;

        // Step 3: Build the combined GenerateBatchResult for training
        // This includes tokens/logprobs from BOTH turns for LSP completions
        const combinedGenResult = {
          completionTexts: finalCompletionTexts,
          completionTokens: combinedTokens,
          completionLogprobs: combinedLogprobs,
          completionLengths: combinedLengths,
          finishReasons: combinedFinishReasons,
        };

        // Step 4: Score final completions (based on turn 2 for LSP, turn 1 for non-LSP)
        const promptTexts = prompts.map((msgs) => msgs.map((m) => `${m.role}: ${m.content}`).join('\n'));
        const tokenCounts = combinedLengths;
        const finishReasons = combinedFinishReasons;

        const rewardOutputs = buildRewardOutputs(
          promptTexts,
          finalCompletionTexts,
          tokenCounts,
          finishReasons,
          groupSize,
        );

        // Expand contexts to match rewardOutputs: each prompt's context repeated groupSize times
        // e.g., [ctx1, ctx2] with groupSize=4 → [ctx1, ctx1, ctx1, ctx1, ctx2, ctx2, ctx2, ctx2]
        const expandedContexts = contexts.flatMap((ctx) => Array(groupSize).fill(ctx));

        const rewards = await githubToolReward(rewardOutputs, expandedContexts);

        // Step 5: Train with the COMBINED tokens/logprobs from both turns
        // The reward from turn 2 quality is propagated to train BOTH turns
        // This teaches the model to: (1) make good LSP calls AND (2) use the response well
        const stepMetrics = await engine.trainStepWithGenerations(prompts, Array.from(rewards), combinedGenResult);

        // Increment trainer's step counter (for checkpoint naming and resumption)
        trainer.incrementStep();
        const currentStep = trainer.getStep();

        // Record step to database for metrics restoration on resume
        await trainer.recordStepToDatabase(
          currentStep,
          stepMetrics,
          finalCompletionTexts,
          Array.from(rewards),
          promptTexts,
        );

        // Calculate step number within epoch
        const batchIdx = Math.floor(i / batchSize);

        // Send TUI step message for real-time metrics display
        logger.step(
          {
            step: stepMetrics.step,
            loss: stepMetrics.loss,
            totalTokens: stepMetrics.totalTokens,
            meanReward: stepMetrics.meanReward,
            stdReward: stepMetrics.stdReward,
            meanAdvantage: stepMetrics.meanAdvantage,
            generationTimeMs: stepMetrics.generationTimeMs,
            trainingTimeMs: stepMetrics.trainingTimeMs,
            peakMemoryMb: stepMetrics.peakMemoryMb,
            activeMemoryMb: stepMetrics.activeMemoryMb,
          },
          batchIdx,
          stepsPerEpoch,
        );

        // Send TUI generation samples for the Samples tab
        for (let j = 0; j < finalCompletionTexts.length; j++) {
          const promptIdx = Math.floor(j / groupSize);
          const promptText = promptTexts[promptIdx] ?? '';
          logger.generation({
            index: j,
            prompt: promptText, // Truncate for display
            completion: finalCompletionTexts[j],
            reward: rewards[j],
            tokens: tokenCounts[j] ?? 0,
          });
        }

        // Save checkpoint periodically (using trainer's step counter for consistency)
        if (currentStep > 0 && currentStep % saveInterval === 0) {
          await trainer.saveCheckpoint();
        }
      }

      const epochTime = (Date.now() - epochStart) / 1000;
      const epochMetrics = trainer.endEpoch(epochTime);

      // Send TUI epoch_end message
      logger.epochEnd(epoch, numEpochs, epochTime);

      logger.info(
        `Epoch ${epoch + 1} complete: avg_loss=${epochMetrics.avgLoss.toFixed(4)}, ` +
          `avg_reward=${epochMetrics.avgReward.toFixed(2)}, time=${epochTime.toFixed(1)}s`,
      );
    }

    // Save final checkpoint
    await trainer.saveCheckpoint('final');

    // Send TUI complete message with final step count
    const finalStep = numEpochs * stepsPerEpoch;
    logger.complete(finalStep);

    logger.banner(
      '',
      '=====================================================',
      'Training complete!',
      '',
      `Results saved to: ${outputDir}`,
      `LSP tool usage: ${totalLspCalls}/${totalCompletions} completions (${((totalLspCalls / totalCompletions) * 100).toFixed(1)}%)`,
      '',
      'Next steps:',
      '  1. Test the model: node examples/grpo/test-generation.ts',
      `  2. Check logs: cat ${resolve(outputDir, 'github-tool-use.jsonl')}`,
      `  3. Inspect checkpoint: ${resolve(outputDir, 'final')}`,
      '=====================================================',
      '',
    );
  } catch (error) {
    logger.error(`Training failed: ${error as Error}`);
    if (error instanceof Error) {
      logger.error(`Error details: ${error.message}`);
      logger.error(`Stack trace: ${error.stack}`);
    }
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error('[train-github-tool] Fatal error:', error);
  process.exitCode = 1;
});
