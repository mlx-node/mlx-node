/**
 * TOML Configuration Loader for GRPO Training
 *
 * Loads and validates GRPO training configuration from TOML files.
 */

import { readFileSync } from 'node:fs';
import { parse as parseToml } from '@std/toml';
import type { GRPOConfig, RewardFunction, RewardOutput } from '@mlx-node/trl';
import { ALL_REWARD_FUNCTIONS } from '@mlx-node/trl';

/**
 * TOML configuration structure
 */
interface TomlConfig {
  model?: {
    name?: string;
    path?: string;
  };
  dataset?: {
    split?: string;
    max_train_samples?: number | null;
    include_one_shot?: boolean;
  };
  training?: {
    learning_rate?: number;
    num_epochs?: number;
    batch_size?: number;
    gradient_accumulation_steps?: number;
    weight_decay?: number;
    gradient_clip_norm?: number;
    gradient_clip_value?: number;
  };
  grpo?: {
    group_size?: number;
    clip_epsilon?: number;
    kl_coef?: number;
    advantage_normalization?: boolean;
    loss_type?: 'grpo' | 'dapo' | 'dr_grpo' | 'bnpo';
  };
  generation?: {
    max_new_tokens?: number;
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
  };
  reward?: {
    type?: 'function' | 'model';
    use_correctness?: boolean;
    use_integer?: boolean;
    use_strict_format?: boolean;
    use_soft_format?: boolean;
    use_xml_count?: boolean;
    model_path?: string;
  };
  logging?: {
    console?: boolean;
    jsonl?: boolean;
    log_interval?: number;
    save_interval?: number;
    eval_interval?: number;
    output_dir?: string;
    run_name?: string;
  };
  advanced?: {
    device?: string;
    seed?: number;
  };
}

/**
 * Dataset configuration
 */
export interface DatasetConfig {
  split: string;
  maxTrainSamples?: number;
  includeOneShot: boolean;
}

/**
 * Combined reward function based on configuration
 * Uses the unified reward function signature: (outputs: RewardOutput[]) => number[] | Float32Array
 */
function createRewardFunction(config: TomlConfig): RewardFunction {
  const rewardConfig = config.reward ?? {};

  // Determine which reward functions to use
  const enabledFunctions: RewardFunction[] = [];

  if (rewardConfig.use_correctness !== false) {
    enabledFunctions.push(ALL_REWARD_FUNCTIONS[0]); // correctnessReward
  }
  if (rewardConfig.use_integer !== false) {
    enabledFunctions.push(ALL_REWARD_FUNCTIONS[1]); // integerReward
  }
  if (rewardConfig.use_strict_format !== false) {
    enabledFunctions.push(ALL_REWARD_FUNCTIONS[2]); // strictFormatReward
  }
  if (rewardConfig.use_soft_format !== false) {
    enabledFunctions.push(ALL_REWARD_FUNCTIONS[3]); // softFormatReward
  }
  if (rewardConfig.use_xml_count !== false) {
    enabledFunctions.push(ALL_REWARD_FUNCTIONS[4]); // xmlCountReward
  }

  // If no functions enabled, use all
  const funcs = enabledFunctions.length > 0 ? enabledFunctions : ALL_REWARD_FUNCTIONS;

  // Return combined reward function with unified signature
  return async (outputs: RewardOutput[]): Promise<Float32Array> => {
    // Apply all reward functions with the structured outputs
    const allScores = await Promise.all(funcs.map((fn) => fn(outputs)));
    const numOutputs = outputs.length;
    const combinedScores = new Float32Array(numOutputs);

    for (const scores of allScores) {
      const scoresArray = scores instanceof Float32Array ? Array.from(scores) : scores;
      for (let i = 0; i < numOutputs; i++) {
        combinedScores[i] += scoresArray[i];
      }
    }

    return combinedScores;
  };
}

/**
 * Load GRPO configuration from TOML file
 */
export function loadConfigFromToml(configPath: string): {
  grpoConfig: Partial<GRPOConfig>;
  datasetConfig: DatasetConfig;
} {
  // Read and parse TOML file
  const configContent = readFileSync(configPath, 'utf-8');
  const toml = parseToml(configContent) as TomlConfig;

  // Extract model configuration
  const modelConfig = toml.model?.path ?? toml.model?.name ?? 'qwen3-0.6b';

  // Extract dataset configuration
  const datasetConfig: DatasetConfig = {
    split: toml.dataset?.split ?? 'train',
    maxTrainSamples: toml.dataset?.max_train_samples ?? undefined,
    includeOneShot: toml.dataset?.include_one_shot ?? true,
  };

  // Build GRPO configuration
  const grpoConfig: Partial<GRPOConfig> = {
    // Model
    modelConfig,

    // Training hyperparameters
    learningRate: toml.training?.learning_rate ?? 1e-6,
    numEpochs: toml.training?.num_epochs ?? 1,
    batchSize: toml.training?.batch_size ?? 1,
    gradientAccumulationSteps: toml.training?.gradient_accumulation_steps ?? 4,
    weightDecay: toml.training?.weight_decay ?? 0.01,

    // GRPO parameters
    groupSize: toml.grpo?.group_size ?? 8,
    clipEpsilon: toml.grpo?.clip_epsilon ?? 0.2,
    klCoef: toml.grpo?.kl_coef ?? 0.0,
    advantageNormalization: toml.grpo?.advantage_normalization ?? true,
    lossType: toml.grpo?.loss_type ?? 'grpo',

    // Generation parameters
    maxNewTokens: toml.generation?.max_new_tokens ?? 256,
    temperature: toml.generation?.temperature ?? 0.7,
    topP: toml.generation?.top_p ?? 0.95,
    topK: toml.generation?.top_k ?? 50,

    // Reward configuration
    rewardType: toml.reward?.type ?? 'function',
    rewardFunction: createRewardFunction(toml),
    rewardModelPath: toml.reward?.model_path,

    // Optimization
    gradientClipNorm: toml.training?.gradient_clip_norm,
    gradientClipValue: toml.training?.gradient_clip_value,

    // Logging and checkpointing
    logInterval: toml.logging?.log_interval ?? 10,
    saveInterval: toml.logging?.save_interval ?? 50,
    evalInterval: toml.logging?.eval_interval ?? 25,
    outputDir: toml.logging?.output_dir ?? './outputs/grpo',
    logConsole: toml.logging?.console ?? true,
    logJsonl: toml.logging?.jsonl ?? true,
    runName: toml.logging?.run_name,

    // Device
    device: toml.advanced?.device ?? 'metal',
  };

  return { grpoConfig, datasetConfig };
}

/**
 * Print configuration summary
 */
export function printConfigSummary(grpoConfig: Partial<GRPOConfig>, datasetConfig: DatasetConfig): void {
  console.log('Configuration Summary:');
  console.log('═══════════════════════════════════════════════════════');
  console.log('\nModel:');
  console.log(`  Config: ${grpoConfig.modelConfig}`);
  console.log('\nDataset:');
  console.log(`  Split: ${datasetConfig.split}`);
  console.log(`  Max samples: ${datasetConfig.maxTrainSamples ?? 'all'}`);
  console.log(`  One-shot: ${datasetConfig.includeOneShot ? 'Yes' : 'No'}`);
  console.log('\nTraining:');
  console.log(`  Learning rate: ${grpoConfig.learningRate}`);
  console.log(`  Epochs: ${grpoConfig.numEpochs}`);
  console.log(`  Batch size: ${grpoConfig.batchSize}`);
  console.log(`  Gradient accumulation: ${grpoConfig.gradientAccumulationSteps}`);
  console.log(`  Effective batch size: ${(grpoConfig.batchSize ?? 1) * (grpoConfig.gradientAccumulationSteps ?? 1)}`);
  console.log('\nGRPO:');
  console.log(`  Group size: ${grpoConfig.groupSize} generations per prompt`);
  console.log(`  Clip epsilon: ${grpoConfig.clipEpsilon}`);
  console.log(`  KL coefficient: ${grpoConfig.klCoef}`);
  console.log(`  Loss type: ${grpoConfig.lossType}`);
  console.log('\nGeneration:');
  console.log(`  Max new tokens: ${grpoConfig.maxNewTokens}`);
  console.log(`  Temperature: ${grpoConfig.temperature}`);
  console.log(`  Top-p: ${grpoConfig.topP}`);
  console.log(`  Top-k: ${grpoConfig.topK}`);
  console.log('\nLogging:');
  console.log(`  Output directory: ${grpoConfig.outputDir}`);
  console.log(`  Run name: ${grpoConfig.runName ?? 'default'}`);
  console.log(`  Log interval: ${grpoConfig.logInterval} steps`);
  console.log(`  Save interval: ${grpoConfig.saveInterval} steps`);
  console.log('═══════════════════════════════════════════════════════\n');
}
