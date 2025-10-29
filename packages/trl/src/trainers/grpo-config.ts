import { parse as parseToml } from '@std/toml';
import { readFileSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';

export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigError';
  }
}

export interface MLXGRPOConfig {
  model_name: string;
  output_dir: string;
  run_name: string;
  learning_rate: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  num_epochs: number;
  max_train_samples: number;
  warmup_ratio: number;
  max_grad_norm: number;
  logging_steps: number;
  num_generations: number;
  max_prompt_length: number;
  max_completion_length: number;
  max_new_tokens: number;
  temperature: number;
  clip_eps: number;
  kl_coeff: number;
  adam_beta1: number;
  adam_beta2: number;
  weight_decay: number;
  lr_scheduler_type: string;
  save_steps: number;
  eval_steps: number;
  eval_samples: number;
  seed: number;
  use_compile: boolean;
  quantize_for_rollouts: boolean;
  eval_every_updates: number;
  eval_subset_size: number;
  eval_max_new_tokens: number;
  log_jsonl: boolean;
}

const DEFAULT_CONFIG: MLXGRPOConfig = Object.freeze({
  model_name: 'Qwen/Qwen2.5-1.5B-Instruct',
  output_dir: 'outputs/Qwen-1.5B-MLX-GRPO',
  run_name: 'Qwen-1.5B-MLX-GRPO-gsm8k',
  learning_rate: 1e-6,
  batch_size: 1,
  gradient_accumulation_steps: 4,
  num_epochs: 1,
  max_train_samples: 0,
  warmup_ratio: 0.1,
  max_grad_norm: 0.1,
  logging_steps: 1,
  num_generations: 64,
  max_prompt_length: 512,
  max_completion_length: 1024,
  max_new_tokens: 512,
  temperature: 0.7,
  clip_eps: 0.2,
  kl_coeff: 0.0,
  adam_beta1: 0.9,
  adam_beta2: 0.999,
  weight_decay: 0.0,
  lr_scheduler_type: 'cosine',
  save_steps: 100,
  eval_steps: 50,
  eval_samples: 200,
  seed: 0,
  use_compile: true,
  quantize_for_rollouts: true,
  eval_every_updates: 25,
  eval_subset_size: 200,
  eval_max_new_tokens: 128,
  log_jsonl: true,
});

type ConfigKey = keyof MLXGRPOConfig;

const CONFIG_VALUE_TYPES: Record<ConfigKey, 'number' | 'boolean' | 'string'> = {
  model_name: 'string',
  output_dir: 'string',
  run_name: 'string',
  learning_rate: 'number',
  batch_size: 'number',
  gradient_accumulation_steps: 'number',
  num_epochs: 'number',
  max_train_samples: 'number',
  warmup_ratio: 'number',
  max_grad_norm: 'number',
  logging_steps: 'number',
  num_generations: 'number',
  max_prompt_length: 'number',
  max_completion_length: 'number',
  max_new_tokens: 'number',
  temperature: 'number',
  clip_eps: 'number',
  kl_coeff: 'number',
  adam_beta1: 'number',
  adam_beta2: 'number',
  weight_decay: 'number',
  lr_scheduler_type: 'string',
  save_steps: 'number',
  eval_steps: 'number',
  eval_samples: 'number',
  seed: 'number',
  use_compile: 'boolean',
  quantize_for_rollouts: 'boolean',
  eval_every_updates: 'number',
  eval_subset_size: 'number',
  eval_max_new_tokens: 'number',
  log_jsonl: 'boolean',
};

const INTEGER_KEYS: ReadonlySet<ConfigKey> = new Set([
  'batch_size',
  'gradient_accumulation_steps',
  'num_epochs',
  'max_train_samples',
  'logging_steps',
  'num_generations',
  'max_prompt_length',
  'max_completion_length',
  'max_new_tokens',
  'save_steps',
  'eval_steps',
  'eval_samples',
  'seed',
  'eval_every_updates',
  'eval_subset_size',
  'eval_max_new_tokens',
]);

function cloneDefaults(): MLXGRPOConfig {
  return { ...DEFAULT_CONFIG };
}

function isConfigKey(value: string): value is ConfigKey {
  return Object.prototype.hasOwnProperty.call(CONFIG_VALUE_TYPES, value);
}

function coerceBoolean(value: unknown, key: ConfigKey): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes', 'on'].includes(normalized)) return true;
    if (['false', '0', 'no', 'off'].includes(normalized)) return false;
  }
  throw new ConfigError(`Invalid boolean for ${key}: ${String(value)}`);
}

function coerceNumber(value: unknown, key: ConfigKey): number {
  if (typeof value === 'string' && value.trim() === '') {
    throw new ConfigError(`Invalid number for ${key}: empty string`);
  }
  const parsed = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(parsed)) {
    throw new ConfigError(`Invalid number for ${key}: ${String(value)}`);
  }
  if (INTEGER_KEYS.has(key) && !Number.isInteger(parsed)) {
    throw new ConfigError(`Expected integer for ${key}, received ${parsed}`);
  }
  return parsed;
}

function coerceString(value: unknown, key: ConfigKey): string {
  if (typeof value === 'string') return value;
  throw new ConfigError(`Invalid string for ${key}: ${String(value)}`);
}

function coerceValue(key: ConfigKey, value: unknown): MLXGRPOConfig[ConfigKey] {
  const expected = CONFIG_VALUE_TYPES[key];
  if (expected === 'boolean') {
    return coerceBoolean(value, key) as MLXGRPOConfig[ConfigKey];
  }
  if (expected === 'number') {
    return coerceNumber(value, key) as MLXGRPOConfig[ConfigKey];
  }
  return coerceString(value, key) as MLXGRPOConfig[ConfigKey];
}

/**
 * Type-safe setter for config values.
 * Uses a controlled type assertion to safely assign dynamically-typed values.
 */
function setConfigValue<T extends Partial<MLXGRPOConfig>>(
  config: T,
  key: ConfigKey,
  value: MLXGRPOConfig[ConfigKey],
): void {
  (config as Record<ConfigKey, MLXGRPOConfig[ConfigKey]>)[key] = value;
}

function normalizeTomlRecord(record: Record<string, unknown>): Partial<MLXGRPOConfig> {
  const normalized: Partial<MLXGRPOConfig> = {};
  for (const [rawKey, rawValue] of Object.entries(record)) {
    if (!isConfigKey(rawKey)) {
      continue;
    }
    setConfigValue(normalized, rawKey, coerceValue(rawKey, rawValue));
  }
  return normalized;
}

export function getDefaultConfig(): MLXGRPOConfig {
  return cloneDefaults();
}

export function mergeConfig(base: MLXGRPOConfig, update: Partial<MLXGRPOConfig>): MLXGRPOConfig {
  if (!update) {
    return { ...base };
  }
  const result: MLXGRPOConfig = { ...base };
  for (const [key, value] of Object.entries(update) as [ConfigKey, MLXGRPOConfig[ConfigKey]][]) {
    if (value === undefined) continue;
    if (!isConfigKey(key)) {
      throw new ConfigError(`Unknown configuration key: ${key}`);
    }
    setConfigValue(result, key, value);
  }
  return result;
}

export function loadTomlConfig(filePath: string): MLXGRPOConfig {
  const absolutePath = resolvePath(filePath);
  let fileContents: string;
  try {
    fileContents = readFileSync(absolutePath, 'utf8');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new ConfigError(`Failed to read config at ${absolutePath}: ${message}`);
  }
  let parsedRaw: unknown;
  try {
    parsedRaw = parseToml(fileContents);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new ConfigError(`Failed to parse TOML at ${absolutePath}: ${message}`);
  }
  if (parsedRaw === null || typeof parsedRaw !== 'object' || Array.isArray(parsedRaw)) {
    throw new ConfigError(`Expected table at ${absolutePath}`);
  }
  const parsed = parsedRaw as Record<string, unknown>;
  const normalized = normalizeTomlRecord(parsed);
  return mergeConfig(getDefaultConfig(), normalized);
}

export function applyOverrides(config: MLXGRPOConfig, overrides: string[]): MLXGRPOConfig {
  if (!overrides.length) {
    return { ...config };
  }
  const accumulated: Partial<MLXGRPOConfig> = {};
  for (const entry of overrides) {
    const idx = entry.indexOf('=');
    if (idx === -1) {
      throw new ConfigError(`Invalid override "${entry}", expected key=value format`);
    }
    const key = entry.slice(0, idx).trim();
    const rawValue = entry.slice(idx + 1).trim();
    if (!isConfigKey(key)) {
      throw new ConfigError(`Unknown configuration key in override: ${key}`);
    }
    setConfigValue(accumulated, key, coerceValue(key, rawValue));
  }
  return mergeConfig(config, accumulated);
}
