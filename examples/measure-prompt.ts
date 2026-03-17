import { resolve } from 'node:path';
import { execSync } from 'node:child_process';
import { GRPOTrainer, type GRPOTrainerConfig } from '@mlx-node/trl';
import { createToolDefinition } from '@mlx-node/lm';
import { SYSTEM_PROMPT } from './sft/prompts';

function showMemory(label: string) {
  const pid = process.pid;
  const psOut = execSync(`ps -o rss=,vsz= -p ${pid}`).toString().trim();
  const [rssKB] = psOut.split(/\s+/).map(Number);
  console.log(`[${label}] RSS: ${(rssKB / 1024 / 1024).toFixed(2)} GB`);
}

const lspTool = createToolDefinition(
  'lsp',
  'Query Octokit REST API documentation.',
  {
    method: { type: 'string', description: 'API method to query' },
    kind: { type: 'string', enum: ['parameters', 'return'], description: 'What to query' },
  },
  ['method', 'kind'],
);

const runJsTool = createToolDefinition(
  'run_js',
  'Execute JavaScript code using Octokit.',
  {
    code: { type: 'string', description: 'JavaScript code to execute.' },
  },
  ['code'],
);

async function main() {
  showMemory('start');

  const modelPath = resolve(process.cwd(), '.cache/models/qwen3.5-0.8b-mlx-bf16');

  // Match the actual training config from train-github-tool.ts
  const config: GRPOTrainerConfig<string> = {
    modelName: 'qwen3.5',
    modelPath,
    learningRate: 1e-6,
    numEpochs: 1,
    groupSize: 5, // actual config
    batchSize: 2, // actual config
    gradientAccumulationSteps: 1,
    lmHeadChunkSize: 2,
    forwardChunkSize: 1,
    clipEpsilon: 0.2,
    klCoef: 0.0,
    maxCompletionLength: 1536, // actual config
    temperature: 0.7,
    topP: 0.9,
    topK: 50,
    repetitionPenalty: 1.1,
    tools: [lspTool, runJsTool],
    enableThinking: true,
    lossType: 'grpo',
    weightDecay: 0.01,
    gradientClipNorm: 1.0,
    logInterval: 1,
    saveInterval: 999,
    outputDir: resolve(process.cwd(), 'outputs/mem-test'),
    logConsole: false,
    logJsonl: false,
    runName: 'mem-test',
    device: 'metal',
  };

  console.log('Creating trainer...');
  const trainer = await GRPOTrainer.create(config);
  const engine = trainer.getNativeEngine();
  showMemory('after model load');

  // 2 prompts (batchSize=2)
  const prompts = [
    [
      { role: 'system' as const, content: SYSTEM_PROMPT },
      { role: 'user' as const, content: 'Show me the PR comments' },
    ],
    [
      { role: 'system' as const, content: SYSTEM_PROMPT },
      { role: 'user' as const, content: 'List all discussion comments on the current PR' },
    ],
  ];

  // Phase 1: Generation (2 prompts × 5 group = 10 completions)
  console.log('\n=== Phase 1: Generation (2 prompts × 5 group = 10 completions, max 1536 tokens) ===');
  showMemory('before generation');
  const t0 = Date.now();
  const genResult = await engine.generateBatchForTraining(prompts);
  const t1 = Date.now();
  showMemory('after generation');
  console.log(`Generation: ${genResult.completionTexts.length} completions in ${((t1 - t0) / 1000).toFixed(1)}s`);
  console.log(`Completion lengths: ${genResult.completionLengths.join(', ')}`);
  const maxLen = Math.max(...genResult.completionLengths);
  const avgLen = genResult.completionLengths.reduce((a, b) => a + b, 0) / genResult.completionLengths.length;
  console.log(`Max completion: ${maxLen}, Avg: ${avgLen.toFixed(0)}`);

  // Phase 2: Training (autograd)
  console.log('\n=== Phase 2: Autograd (10 completions, forwardChunkSize=1, lmHeadChunkSize=2) ===');
  const rewards = Array.from({ length: genResult.completionTexts.length }, () => 5.0);
  showMemory('before trainStep');
  const t2 = Date.now();
  try {
    const metrics = await engine.trainStepWithGenerations(prompts, rewards, genResult);
    const t3 = Date.now();
    showMemory('after trainStep');
    console.log(`Training: loss=${metrics.loss.toFixed(4)}, time=${((t3 - t2) / 1000).toFixed(1)}s`);
    console.log(`Peak metal: ${metrics.peakMemoryMb.toFixed(0)}MB, active: ${metrics.activeMemoryMb.toFixed(0)}MB`);
    console.log(`Total tokens: ${metrics.totalTokens}`);
  } catch (e) {
    showMemory('after trainStep ERROR');
    console.error(`Training failed: ${String(e)}`);
  }

  showMemory('final');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
