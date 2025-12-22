import { parse as parseToml } from '@std/toml';
import { readFileSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';

export class SFTConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SFTConfigError';
  }
}

export interface SFTTrainerConfig {
  // Model
  model_name: string;
  output_dir: string;
  run_name: string;

  // Training hyperparameters
  learning_rate: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  num_epochs: number;
  max_train_samples: number;
  max_grad_norm: number;
  weight_decay: number;

  // SFT-specific
  max_seq_length: number;
  completion_only: boolean;
  label_smoothing: number;

  // Logging & checkpointing
  logging_steps: number;
  save_steps: number;
  max_checkpoints: number;
  log_jsonl: boolean;
  tui_mode: boolean;

  // Misc
  seed: number;
  resume_from_checkpoint: string;
}

const DEFAULT_SFT_CONFIG: SFTTrainerConfig = Object.freeze({
  model_name: 'Qwen/Qwen3-0.6B',
  output_dir: 'outputs/sft',
  run_name: 'sft-run',

  learning_rate: 2e-5,
  batch_size: 4,
  gradient_accumulation_steps: 1,
  num_epochs: 3,
  max_train_samples: 0,
  max_grad_norm: 1.0,
  weight_decay: 0.01,

  max_seq_length: 2048,
  completion_only: false, // Changed to false for TRL parity
  label_smoothing: 0.0,

  logging_steps: 10,
  save_steps: 100,
  max_checkpoints: 3,
  log_jsonl: true,
  tui_mode: false,

  seed: 42,
  resume_from_checkpoint: '',
});

type SFTConfigKey = keyof SFTTrainerConfig;

const SFT_CONFIG_VALUE_TYPES: Record<SFTConfigKey, 'number' | 'boolean' | 'string'> = {
  model_name: 'string',
  output_dir: 'string',
  run_name: 'string',
  learning_rate: 'number',
  batch_size: 'number',
  gradient_accumulation_steps: 'number',
  num_epochs: 'number',
  max_train_samples: 'number',
  max_grad_norm: 'number',
  weight_decay: 'number',
  max_seq_length: 'number',
  completion_only: 'boolean',
  label_smoothing: 'number',
  logging_steps: 'number',
  save_steps: 'number',
  max_checkpoints: 'number',
  log_jsonl: 'boolean',
  tui_mode: 'boolean',
  seed: 'number',
  resume_from_checkpoint: 'string',
};

const SFT_INTEGER_KEYS: ReadonlySet<SFTConfigKey> = new Set([
  'batch_size',
  'gradient_accumulation_steps',
  'num_epochs',
  'max_train_samples',
  'max_seq_length',
  'logging_steps',
  'save_steps',
  'max_checkpoints',
  'seed',
]);

function cloneDefaults(): SFTTrainerConfig {
  return { ...DEFAULT_SFT_CONFIG };
}

function isConfigKey(value: string): value is SFTConfigKey {
  return Object.prototype.hasOwnProperty.call(SFT_CONFIG_VALUE_TYPES, value);
}

function coerceBoolean(value: unknown, key: SFTConfigKey): boolean {
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (['true', '1', 'yes', 'on'].includes(normalized)) return true;
    if (['false', '0', 'no', 'off'].includes(normalized)) return false;
  }
  throw new SFTConfigError(`Invalid boolean for ${key}: ${String(value)}`);
}

function coerceNumber(value: unknown, key: SFTConfigKey): number {
  if (typeof value === 'string' && value.trim() === '') {
    throw new SFTConfigError(`Invalid number for ${key}: empty string`);
  }
  const parsed = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(parsed)) {
    throw new SFTConfigError(`Invalid number for ${key}: ${String(value)}`);
  }
  if (SFT_INTEGER_KEYS.has(key) && !Number.isInteger(parsed)) {
    throw new SFTConfigError(`Expected integer for ${key}, received ${parsed}`);
  }
  return parsed;
}

function coerceString(value: unknown, key: SFTConfigKey): string {
  if (typeof value === 'string') return value;
  throw new SFTConfigError(`Invalid string for ${key}: ${String(value)}`);
}

function coerceValue(key: SFTConfigKey, value: unknown): SFTTrainerConfig[SFTConfigKey] {
  const expected = SFT_CONFIG_VALUE_TYPES[key];
  if (expected === 'boolean') {
    return coerceBoolean(value, key) as SFTTrainerConfig[SFTConfigKey];
  }
  if (expected === 'number') {
    return coerceNumber(value, key) as SFTTrainerConfig[SFTConfigKey];
  }
  return coerceString(value, key) as SFTTrainerConfig[SFTConfigKey];
}

function setConfigValue<T extends Partial<SFTTrainerConfig>>(
  config: T,
  key: SFTConfigKey,
  value: SFTTrainerConfig[SFTConfigKey],
): void {
  (config as Record<SFTConfigKey, SFTTrainerConfig[SFTConfigKey]>)[key] = value;
}

function normalizeTomlRecord(record: Record<string, unknown>): Partial<SFTTrainerConfig> {
  const normalized: Partial<SFTTrainerConfig> = {};
  for (const [rawKey, rawValue] of Object.entries(record)) {
    if (!isConfigKey(rawKey)) {
      continue;
    }
    setConfigValue(normalized, rawKey, coerceValue(rawKey, rawValue));
  }
  return normalized;
}

export function getDefaultSFTConfig(): SFTTrainerConfig {
  return cloneDefaults();
}

export function mergeSFTConfig(base: SFTTrainerConfig, update: Partial<SFTTrainerConfig>): SFTTrainerConfig {
  if (!update) {
    return { ...base };
  }
  const result: SFTTrainerConfig = { ...base };
  for (const [key, value] of Object.entries(update) as [SFTConfigKey, SFTTrainerConfig[SFTConfigKey]][]) {
    if (value === undefined) continue;
    if (!isConfigKey(key)) {
      throw new SFTConfigError(`Unknown configuration key: ${key}`);
    }
    setConfigValue(result, key, value);
  }
  return result;
}

export function loadSFTTomlConfig(filePath: string): SFTTrainerConfig {
  const absolutePath = resolvePath(filePath);
  let fileContents: string;
  try {
    fileContents = readFileSync(absolutePath, 'utf8');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new SFTConfigError(`Failed to read config at ${absolutePath}: ${message}`);
  }
  let parsedRaw: unknown;
  try {
    parsedRaw = parseToml(fileContents);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new SFTConfigError(`Failed to parse TOML at ${absolutePath}: ${message}`);
  }
  if (parsedRaw === null || typeof parsedRaw !== 'object' || Array.isArray(parsedRaw)) {
    throw new SFTConfigError(`Expected table at ${absolutePath}`);
  }
  const parsed = parsedRaw as Record<string, unknown>;
  const normalized = normalizeTomlRecord(parsed);
  return mergeSFTConfig(getDefaultSFTConfig(), normalized);
}

export function applySFTOverrides(config: SFTTrainerConfig, overrides: string[]): SFTTrainerConfig {
  if (!overrides.length) {
    return { ...config };
  }
  const accumulated: Partial<SFTTrainerConfig> = {};
  for (const entry of overrides) {
    const idx = entry.indexOf('=');
    if (idx === -1) {
      throw new SFTConfigError(`Invalid override "${entry}", expected key=value format`);
    }
    const key = entry.slice(0, idx).trim();
    const rawValue = entry.slice(idx + 1).trim();
    if (!isConfigKey(key)) {
      throw new SFTConfigError(`Unknown configuration key in override: ${key}`);
    }
    setConfigValue(accumulated, key, coerceValue(key, rawValue));
  }
  return mergeSFTConfig(config, accumulated);
}

export { DEFAULT_SFT_CONFIG };
