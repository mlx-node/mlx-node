import { camelCase } from 'change-case';
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
  modelName: string;
  outputDir: string;
  runName: string;

  // Training hyperparameters
  learningRate: number;
  batchSize: number;
  gradientAccumulationSteps: number;
  numEpochs: number;
  maxTrainSamples: number;
  maxGradNorm: number;
  weightDecay: number;

  // SFT-specific
  maxSeqLength: number;
  completionOnly: boolean;
  labelSmoothing: number;

  // Logging & checkpointing
  loggingSteps: number;
  saveSteps: number;
  maxCheckpoints: number;
  logJsonl: boolean;
  tuiMode: boolean;

  // Memory optimization
  gradientCheckpointing: boolean;

  // Misc
  seed: number;
  resumeFromCheckpoint: string;
}

const DEFAULT_SFT_CONFIG: SFTTrainerConfig = Object.freeze({
  modelName: 'Qwen/Qwen3-0.6B',
  outputDir: 'outputs/sft',
  runName: 'sft-run',

  learningRate: 2e-5,
  batchSize: 4,
  gradientAccumulationSteps: 1,
  numEpochs: 3,
  maxTrainSamples: 0,
  maxGradNorm: 1.0,
  weightDecay: 0.01,

  maxSeqLength: 2048,
  completionOnly: false, // Changed to false for TRL parity
  labelSmoothing: 0.0,

  loggingSteps: 10,
  saveSteps: 100,
  maxCheckpoints: 3,
  logJsonl: true,
  tuiMode: false,

  gradientCheckpointing: true,

  seed: 42,
  resumeFromCheckpoint: '',
});

type SFTConfigKey = keyof SFTTrainerConfig;

const SFT_CONFIG_VALUE_TYPES: Record<SFTConfigKey, 'number' | 'boolean' | 'string'> = {
  modelName: 'string',
  outputDir: 'string',
  runName: 'string',
  learningRate: 'number',
  batchSize: 'number',
  gradientAccumulationSteps: 'number',
  numEpochs: 'number',
  maxTrainSamples: 'number',
  maxGradNorm: 'number',
  weightDecay: 'number',
  maxSeqLength: 'number',
  completionOnly: 'boolean',
  labelSmoothing: 'number',
  loggingSteps: 'number',
  saveSteps: 'number',
  maxCheckpoints: 'number',
  logJsonl: 'boolean',
  tuiMode: 'boolean',
  seed: 'number',
  gradientCheckpointing: 'boolean',
  resumeFromCheckpoint: 'string',
};

const SFT_INTEGER_KEYS: ReadonlySet<SFTConfigKey> = new Set([
  'batchSize',
  'gradientAccumulationSteps',
  'numEpochs',
  'maxTrainSamples',
  'maxSeqLength',
  'loggingSteps',
  'saveSteps',
  'maxCheckpoints',
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
    // TOML uses snake_case, config uses camelCase
    const key = camelCase(rawKey);
    if (!isConfigKey(key)) {
      continue;
    }
    setConfigValue(normalized, key, coerceValue(key, rawValue));
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
      throw new SFTConfigError(`Unknown configuration key: ${key as string}`);
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
    const rawKey = entry.slice(0, idx).trim();
    const rawValue = entry.slice(idx + 1).trim();
    // Accept both snake_case and camelCase overrides
    const key = camelCase(rawKey);
    if (!isConfigKey(key)) {
      throw new SFTConfigError(`Unknown configuration key in override: ${rawKey}`);
    }
    setConfigValue(accumulated, key, coerceValue(key, rawValue));
  }
  return mergeSFTConfig(config, accumulated);
}

export { DEFAULT_SFT_CONFIG };
