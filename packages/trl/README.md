# @mlx-node/trl

Training library for language models on Apple Silicon. Supports GRPO (Group Relative Policy Optimization) and SFT (Supervised Fine-Tuning) with Metal GPU acceleration, built-in reward functions, dataset handling, and checkpoint management.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Node.js 18+

## Installation

```bash
npm install @mlx-node/trl
```

## Quick Start

### GRPO Training

```typescript
import { GRPOTrainer } from '@mlx-node/trl';

const trainer = await GRPOTrainer.create({
  modelPath: './models/Qwen3-0.6B',
  outputDir: './output/grpo-run',
  learningRate: 1e-6,
  groupSize: 4,
  maxCompletionLength: 256,
  temperature: 0.8,
  rewardFunction: async (outputs) => {
    return outputs.map((o) => (o.text.includes('correct') ? 1.0 : 0.0));
  },
});

const dataset = await loadDataset('train');
await trainer.train(dataset);
```

### SFT Training

```typescript
import { SFTTrainer } from '@mlx-node/trl';

const trainer = await SFTTrainer.create({
  modelName: './models/Qwen3-0.6B',
  outputDir: './output/sft-run',
  learningRate: 2e-5,
  batchSize: 4,
  numEpochs: 3,
  completionOnly: true,
});

await trainer.train('./data/training.jsonl');
```

## GRPO Training

GRPO generates multiple completions per prompt, scores them with reward functions, and trains the model to prefer higher-reward outputs.

### Loss Variants

| Loss Type | Description                                 |
| --------- | ------------------------------------------- |
| `grpo`    | Standard Group Relative Policy Optimization |
| `dapo`    | Dynamic sampling with adaptive clipping     |
| `dr_grpo` | Dr.GRPO with improved gradient estimation   |
| `bnpo`    | Batch-normalized policy optimization        |

### Configuration

```typescript
import { GRPOTrainer, GRPOTrainerConfig } from '@mlx-node/trl';

const config: GRPOTrainerConfig = {
  // Model
  modelPath: './models/Qwen3-0.6B',
  outputDir: './output',

  // Training
  learningRate: 1e-6,
  batchSize: 1,
  numEpochs: 1,
  gradientAccumulationSteps: 1,
  gradientClipNorm: 1.0,
  weightDecay: 0.01,

  // GRPO
  groupSize: 4, // completions per prompt
  clipEpsilon: 0.2, // PPO clipping
  klCoef: 0.0, // KL divergence coefficient
  lossType: 'grpo', // grpo | dapo | dr_grpo | bnpo

  // Generation
  maxCompletionLength: 256,
  temperature: 0.8,
  topP: 0.95,
  repetitionPenalty: 1.1,

  // Tool calling
  tools: [toolDef],
  enableThinking: true,

  // Rewards
  rewardFunction: myRewardFn,

  // Memory optimization
  gradientCheckpointing: true,
  lmHeadChunkSize: 2,
  vocabChunkSize: 65536,

  // Checkpointing
  saveInterval: 100,
  maxCheckpoints: 3,
  resumeFromCheckpoint: './output/checkpoint-500',

  // Optimizer
  optimizerType: 'adamw', // adamw | sgd
};
```

### TOML Configuration

Load training config from a TOML file:

```typescript
import { loadTomlConfig, applyOverrides } from '@mlx-node/trl';

const config = loadTomlConfig('./train.toml');
applyOverrides(config, ['learningRate=2e-6', 'batchSize=2']);
```

### Built-in Rewards

Register native Rust reward functions for high-performance scoring:

```typescript
trainer.registerBuiltinReward({
  type: 'ToolUse',
  weight: 1.0,
  allowedTools: ['get_weather', 'search'],
});

trainer.registerBuiltinReward({
  type: 'XmlFormat',
  weight: 0.5,
  requiredTags: ['reasoning', 'answer'],
});

trainer.registerBuiltinReward({
  type: 'Length',
  weight: 0.3,
  min: 50,
  max: 500,
});

trainer.registerBuiltinReward({
  type: 'JsonSchema',
  weight: 1.0,
});
```

### Custom Reward Functions

```typescript
import { RewardFunction, RewardOutput } from '@mlx-node/trl';

const reward: RewardFunction = async (outputs: RewardOutput[]) => {
  return outputs.map((output) => {
    let score = 0;
    if (output.toolCalls?.length) score += 0.5;
    if (output.text.length > 100) score += 0.3;
    return score;
  });
};

trainer.setRewardFunction(reward);
```

### Custom Training Loop

For advanced use cases, use the low-level API:

```typescript
const trainer = await GRPOTrainer.create(config);

for (const batch of dataset) {
  const generations = await trainer.generateBatch(batch.prompts);
  const rewards = await trainer.scoreGenerations(batch.prompts, generations.completions, context);
  const metrics = trainer.trainStep(batch.prompts, context);
  trainer.incrementStep();

  if (metrics.step % 100 === 0) {
    await trainer.saveCheckpoint();
  }
}
```

### Output Store (SQLite)

Record all training generations and metrics to SQLite for analysis:

```typescript
const trainer = await GRPOTrainer.create({
  ...config,
  outputStore: {
    enabled: true,
    database: './output/training.db',
  },
});
```

## SFT Training

Supervised fine-tuning with autograd, gradient accumulation, and completion-only masking.

### Dataset Formats

Two formats are auto-detected from JSONL files:

**Prompt-Completion:**

```json
{ "prompt": [{ "role": "user", "content": "Hello" }], "completion": { "role": "assistant", "content": "Hi!" } }
```

**Conversation:**

```json
{
  "messages": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi!" }
  ]
}
```

### SFT Configuration

```typescript
import { SFTTrainer, SFTTrainerConfig } from '@mlx-node/trl';

const config: SFTTrainerConfig = {
  modelName: './models/Qwen3-0.6B',
  outputDir: './output/sft',
  learningRate: 2e-5,
  batchSize: 4,
  gradientAccumulationSteps: 8,
  numEpochs: 3,
  maxSeqLength: 2048,
  completionOnly: true, // only compute loss on assistant tokens
  labelSmoothing: 0.1,
  maxGradNorm: 1.0,
  weightDecay: 0.01,
  loggingSteps: 10,
  saveSteps: 100,
  maxCheckpoints: 3,
  gradientCheckpointing: true,
};
```

### Programmatic Dataset

```typescript
import { SFTDataset, createSFTDataset } from '@mlx-node/trl';

const dataset = createSFTDataset(examples, tokenizer, {
  maxSeqLength: 2048,
  completionOnly: true,
});

const trainer = await SFTTrainer.create(config);
await trainer.train(dataset);
```

## Datasets

### GSM8K Loader

Built-in loader for the GSM8K math dataset:

```typescript
import { loadLocalGsm8kDataset, LocalGsm8kDatasetLoader } from '@mlx-node/trl';

// Direct load
const examples = await loadLocalGsm8kDataset('train', { limit: 1000 });

// Via DatasetLoader interface
const loader = new LocalGsm8kDatasetLoader('./data/gsm8k');
const trainData = await loader.load('train');
```

### Custom Datasets

Implement the `DatasetLoader` interface:

```typescript
import { DatasetLoader, DatasetExample } from '@mlx-node/trl';

class MyDataset implements DatasetLoader {
  async load(split: 'train' | 'test', limit?: number): Promise<DatasetExample[]> {
    return examples.map((e) => ({
      prompt: [
        { role: 'system', content: 'You are helpful.' },
        { role: 'user', content: e.question },
      ],
      metadata: { answer: e.answer },
    }));
  }
}
```

## Utilities

### XML Chain-of-Thought Parser

Parse `<reasoning>...</reasoning><answer>...</answer>` format:

```typescript
import { parseXmlCot, extractXmlAnswer } from '@mlx-node/trl';

const result = parseXmlCot(modelOutput);
// { reasoning: "...", answer: "42", isStrictMatch: true, isSoftMatch: true, errors: [] }

const answer = extractXmlAnswer(modelOutput);
// "42"
```

### Model Conversion

Re-exported from `@mlx-node/core`:

```typescript
import { convertModel, convertParquetToJsonl } from '@mlx-node/trl';
```

## Features

- **Checkpoint resume** — automatic state restoration including optimizer, step count, and dataset position
- **Emergency save** — catches NaN gradients and SIGTERM/SIGINT for safe recovery
- **TUI mode** — interactive terminal UI with pause/resume/stop (via `mlx-tui` binary)
- **JSONL logging** — structured training logs for external monitoring
- **Multi-model** — supports Qwen3, Qwen3.5 Dense, and Qwen3.5 MoE architectures
- **Reward timeout** — configurable timeout for async reward functions (default 60s)
- **Path security** — traversal prevention for dataset file loading

## API Reference

### Trainers

| Class         | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `GRPOTrainer` | GRPO training with generation, rewards, and policy optimization |
| `SFTTrainer`  | Supervised fine-tuning with completion-only masking             |

### Datasets

| Export                    | Description                                      |
| ------------------------- | ------------------------------------------------ |
| `loadLocalGsm8kDataset()` | Load GSM8K JSONL dataset                         |
| `LocalGsm8kDatasetLoader` | `DatasetLoader` implementation for GSM8K         |
| `SFTDataset`              | Tokenized SFT dataset with padding and shuffling |
| `loadSFTDataset()`        | Load SFT dataset from JSONL file                 |
| `createSFTDataset()`      | Create SFT dataset from in-memory examples       |

### Configuration

| Export                  | Description                       |
| ----------------------- | --------------------------------- |
| `GRPOTrainerConfig`     | Full GRPO configuration interface |
| `SFTTrainerConfig`      | Full SFT configuration interface  |
| `loadTomlConfig()`      | Load GRPO config from TOML file   |
| `loadSFTTomlConfig()`   | Load SFT config from TOML file    |
| `getDefaultConfig()`    | Default GRPO config               |
| `getDefaultSFTConfig()` | Default SFT config                |

### Types

| Type                  | Description                                        |
| --------------------- | -------------------------------------------------- |
| `DatasetExample`      | Training example with prompt messages and metadata |
| `RewardFunction<T>`   | Custom reward function signature                   |
| `RewardOutput`        | Structured completion data for reward scoring      |
| `XmlParseResult`      | Result of XML chain-of-thought parsing             |
| `TrainStepMetrics`    | Per-step training metrics                          |
| `BuiltinRewardConfig` | Configuration for native reward functions          |

## License

[MIT](https://github.com/mlx-node/mlx-node/blob/main/LICENSE)
