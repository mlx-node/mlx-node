/**
 * SFT Dataset handling for Supervised Fine-Tuning
 *
 * Supports two data formats (auto-detected):
 * 1. Prompt-Completion: { prompt: ChatMessage[], completion: ChatMessage }
 * 2. Full Conversation: { messages: ChatMessage[] }
 *
 * Both formats produce tokenized batches with labels masked appropriately.
 */

import { readFileSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';
import type { Qwen3Tokenizer } from '@mlx-node/core';
import type { ChatMessage } from '../types';

// -100 is the standard ignore index for cross-entropy loss
const IGNORE_INDEX = -100;

/**
 * Prompt-Completion format for tool-use training
 */
export interface SFTPromptCompletionExample {
  prompt: ChatMessage[];
  completion: ChatMessage;
}

/**
 * Full conversation format for multi-turn dialogue
 */
export interface SFTConversationExample {
  messages: ChatMessage[];
}

/**
 * Union type for SFT examples
 */
export type SFTExample = SFTPromptCompletionExample | SFTConversationExample;

/**
 * A tokenized batch ready for SFT training
 */
export interface SFTBatch {
  inputIds: Int32Array;
  labels: Int32Array;
  shape: [number, number]; // [batch_size, seq_len]
}

/**
 * Configuration for SFT dataset
 */
export interface SFTDatasetConfig {
  maxSeqLength?: number;
  completionOnly?: boolean; // If true, only train on completion tokens (default: false for TRL parity)
  enableThinking?: boolean; // Enable thinking mode for tokenizer
  seed?: number; // Random seed for reproducible shuffling (default: 42)
}

/**
 * Detect the format of an SFT example
 */
function detectFormat(example: SFTExample): 'prompt-completion' | 'conversation' {
  if ('prompt' in example && 'completion' in example) {
    return 'prompt-completion';
  }
  if ('messages' in example) {
    return 'conversation';
  }
  throw new Error('Invalid SFT example format. Expected either {prompt, completion} or {messages}');
}

/**
 * Check if a message is from an assistant
 */
function isAssistantMessage(message: ChatMessage): boolean {
  return message.role === 'assistant';
}

/**
 * SFT Dataset class for handling SFT training data
 */
export class SFTDataset {
  private examples: SFTExample[];
  private tokenizer: Qwen3Tokenizer;
  private config: Required<Omit<SFTDatasetConfig, 'seed'>> & { seed: number };
  private format: 'prompt-completion' | 'conversation';
  private shuffledIndices: number[];
  private rng: () => number;

  constructor(examples: SFTExample[], tokenizer: Qwen3Tokenizer, config: SFTDatasetConfig = {}) {
    if (examples.length === 0) {
      throw new Error('SFT dataset must contain at least one example');
    }

    this.examples = examples;
    this.tokenizer = tokenizer;
    this.config = {
      maxSeqLength: config.maxSeqLength ?? 2048,
      completionOnly: config.completionOnly ?? false, // Changed to false for TRL parity
      enableThinking: config.enableThinking ?? false,
      seed: config.seed ?? 42,
    };
    this.rng = this.createSeededRandom(this.config.seed);

    // Detect format from first example
    this.format = detectFormat(examples[0]);

    // Validate all examples have the same format
    for (let i = 1; i < examples.length; i++) {
      const fmt = detectFormat(examples[i]);
      if (fmt !== this.format) {
        throw new Error(`Inconsistent SFT data format: example 0 is ${this.format}, example ${i} is ${fmt}`);
      }
    }

    // Initialize indices
    this.shuffledIndices = Array.from({ length: examples.length }, (_, i) => i);
  }

  /**
   * Get the number of examples in the dataset
   */
  get length(): number {
    return this.examples.length;
  }

  /**
   * Shuffle dataset for a specific epoch using epoch-based seeding.
   * This ensures reproducible shuffles across training resumes.
   * Each epoch gets a deterministic shuffle based on (baseSeed + epoch).
   *
   * @param epoch - The epoch number (used as seed offset)
   */
  shuffleForEpoch(epoch: number): void {
    // Reset RNG with epoch-specific seed for reproducibility
    this.rng = this.createSeededRandom(this.config.seed + epoch);
    // Reset indices to original order
    this.shuffledIndices = Array.from({ length: this.examples.length }, (_, i) => i);
    // Fisher-Yates shuffle
    for (let i = this.shuffledIndices.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [this.shuffledIndices[i], this.shuffledIndices[j]] = [this.shuffledIndices[j], this.shuffledIndices[i]];
    }
  }

  /**
   * Create a seeded pseudo-random number generator (Linear Congruential Generator)
   */
  private createSeededRandom(seed: number): () => number {
    let s = seed;
    return () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
  }

  /**
   * Find length of common prefix between two token arrays
   * Handles chat template quirks where prompt tokens may not be exact prefix of full tokens
   */
  private findCommonPrefixLength(prompt: number[], full: number[]): number {
    let i = 0;
    const maxLen = Math.min(prompt.length, full.length);
    while (i < maxLen && prompt[i] === full[i]) {
      i++;
    }
    return i;
  }

  /**
   * Tokenize a prompt-completion example
   */
  private async tokenizePromptCompletion(
    example: SFTPromptCompletionExample,
  ): Promise<{ inputIds: number[]; labels: number[] }> {
    // Tokenize prompt with generation prompt (so the model learns to continue)
    const promptTokens = await this.tokenizer.applyChatTemplate(
      example.prompt,
      true, // add generation prompt
      null,
      this.config.enableThinking,
    );

    // Create full messages for tokenization
    const fullMessages = [...example.prompt, example.completion];
    const fullTokens = await this.tokenizer.applyChatTemplate(
      fullMessages,
      false, // no generation prompt at the end
      null,
      this.config.enableThinking,
    );

    // Convert to regular arrays for manipulation
    const promptArr = Array.from(promptTokens).map((t) => Number(t));
    const inputIds = Array.from(fullTokens).map((t) => Number(t));

    // Use common prefix detection to handle chat template quirks
    // (some templates may not produce prompt tokens as exact prefix of full tokens)
    const promptLen = this.findCommonPrefixLength(promptArr, inputIds);

    if (promptLen !== promptArr.length) {
      console.warn(
        `[SFT Dataset] Prompt tokens differ from prefix of full sequence ` +
          `(${promptArr.length} vs ${promptLen}). Using common prefix for masking.`,
      );
    }

    // Create labels: -100 for prompt tokens, actual tokens for completion
    const labels = inputIds.map((id, i) => {
      if (this.config.completionOnly && i < promptLen) {
        return IGNORE_INDEX;
      }
      return id;
    });

    return { inputIds, labels };
  }

  /**
   * Tokenize a conversation example
   *
   * For conversations, we train on all assistant turns.
   * Non-assistant tokens (system, user) are masked with -100.
   *
   * Complexity: O(k) where k is the number of assistant messages,
   * instead of O(n) for all messages.
   */
  private async tokenizeConversation(
    example: SFTConversationExample,
  ): Promise<{ inputIds: number[]; labels: number[] }> {
    const messages = example.messages;

    // Tokenize full conversation once
    const fullTokens = await this.tokenizer.applyChatTemplate(
      messages,
      false, // no generation prompt
      null,
      this.config.enableThinking,
    );

    const inputIds = Array.from(fullTokens).map((t) => Number(t));

    // If not masking prompts, all tokens are trainable
    if (!this.config.completionOnly) {
      return { inputIds, labels: [...inputIds] };
    }

    // Find assistant message indices
    const assistantIndices = messages.map((m, i) => (isAssistantMessage(m) ? i : -1)).filter((i) => i >= 0);

    if (assistantIndices.length === 0) {
      // No assistant messages - mask everything
      const labels = inputIds.map(() => IGNORE_INDEX);
      return { inputIds, labels };
    }

    // Only tokenize at assistant boundaries - O(k) where k = # assistant messages
    // Instead of O(n) for every message
    const boundaries: Array<{ start: number; end: number; isAssistant: boolean }> = [];
    let prevEnd = 0;

    for (const assistantIdx of assistantIndices) {
      // Tokenize up to but not including this assistant message
      if (assistantIdx > 0) {
        const beforeAssistant = await this.tokenizer.applyChatTemplate(
          messages.slice(0, assistantIdx),
          false,
          null,
          this.config.enableThinking,
        );
        const start = beforeAssistant.length;
        if (start > prevEnd) {
          boundaries.push({ start: prevEnd, end: start, isAssistant: false });
        }
        prevEnd = start;
      }

      // Tokenize including this assistant message
      const afterAssistant = await this.tokenizer.applyChatTemplate(
        messages.slice(0, assistantIdx + 1),
        false,
        null,
        this.config.enableThinking,
      );
      const end = afterAssistant.length;
      if (end > prevEnd) {
        boundaries.push({ start: prevEnd, end, isAssistant: true });
      }
      prevEnd = end;
    }

    // Handle any trailing non-assistant messages
    if (prevEnd < fullTokens.length) {
      boundaries.push({ start: prevEnd, end: fullTokens.length, isAssistant: false });
    }

    // Build labels based on boundaries
    const labels = Array.from({ length: inputIds.length }, () => IGNORE_INDEX);
    for (const { start, end, isAssistant } of boundaries) {
      if (isAssistant) {
        for (let i = start; i < end; i++) {
          labels[i] = inputIds[i];
        }
      }
    }

    return { inputIds, labels };
  }

  /**
   * Tokenize a single example based on its format
   */
  private async tokenizeExample(example: SFTExample): Promise<{ inputIds: number[]; labels: number[] }> {
    if (this.format === 'prompt-completion') {
      return this.tokenizePromptCompletion(example as SFTPromptCompletionExample);
    } else {
      return this.tokenizeConversation(example as SFTConversationExample);
    }
  }

  /**
   * Collate multiple examples into a padded batch
   */
  async collateBatch(indices: number[]): Promise<SFTBatch> {
    const examples = indices.map((i) => this.examples[this.shuffledIndices[i]]);

    // Tokenize all examples
    const tokenized = await Promise.all(examples.map((example) => this.tokenizeExample(example)));

    // Find max length (capped at maxSeqLength)
    const maxLen = Math.min(this.config.maxSeqLength, Math.max(...tokenized.map((t) => t.inputIds.length)));

    // Pad and truncate
    const batchSize = examples.length;
    const paddedInputIds = new Int32Array(batchSize * maxLen);
    const paddedLabels = new Int32Array(batchSize * maxLen);

    const padTokenId = this.tokenizer.getPadTokenId();

    for (let b = 0; b < batchSize; b++) {
      const { inputIds, labels } = tokenized[b];
      const seqLen = Math.min(inputIds.length, maxLen);

      // Truncate from the left if necessary (keep the end of the sequence)
      const startIdx = Math.max(0, inputIds.length - maxLen);

      for (let s = 0; s < maxLen; s++) {
        const offset = b * maxLen + s;
        if (s < seqLen) {
          paddedInputIds[offset] = inputIds[startIdx + s];
          paddedLabels[offset] = labels[startIdx + s];
        } else {
          // Pad
          paddedInputIds[offset] = padTokenId;
          paddedLabels[offset] = IGNORE_INDEX;
        }
      }
    }

    return {
      inputIds: paddedInputIds,
      labels: paddedLabels,
      shape: [batchSize, maxLen],
    };
  }

  /**
   * Generate batches for training
   */
  async *batches(batchSize: number): AsyncGenerator<SFTBatch> {
    for (let i = 0; i < this.examples.length; i += batchSize) {
      const end = Math.min(i + batchSize, this.examples.length);
      const indices = Array.from({ length: end - i }, (_, j) => i + j);
      yield await this.collateBatch(indices);
    }
  }

  /**
   * Get total number of batches for a given batch size
   */
  numBatches(batchSize: number): number {
    return Math.ceil(this.examples.length / batchSize);
  }
}

/**
 * Read JSONL file and parse into records
 */
function readJsonl<T>(path: string, limit?: number): T[] {
  let fileContents: string;
  try {
    fileContents = readFileSync(path, 'utf8');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to read SFT dataset at ${path}: ${message}`);
  }

  const lines = fileContents.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const records: T[] = [];
  const max = typeof limit === 'number' && limit > 0 ? limit : Number.POSITIVE_INFINITY;

  for (let i = 0; i < lines.length && records.length < max; i++) {
    const line = lines[i];
    try {
      const parsed = JSON.parse(line) as T;
      records.push(parsed);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to parse JSONL at ${path}:${i + 1} - ${message}`);
    }
  }

  return records;
}

/**
 * Validate an SFT example
 */
function validateSFTExample(example: unknown, index: number): SFTExample {
  if (typeof example !== 'object' || example === null) {
    throw new Error(`SFT example ${index} must be an object`);
  }

  const obj = example as Record<string, unknown>;

  // Check for prompt-completion format
  if ('prompt' in obj && 'completion' in obj) {
    if (!Array.isArray(obj.prompt)) {
      throw new Error(`SFT example ${index}: prompt must be an array of messages`);
    }
    if (typeof obj.completion !== 'object' || obj.completion === null) {
      throw new Error(`SFT example ${index}: completion must be a message object`);
    }
    const completion = obj.completion as Record<string, unknown>;
    if (completion.role !== 'assistant') {
      throw new Error(`SFT example ${index}: completion.role must be 'assistant'`);
    }
    if (typeof completion.content !== 'string') {
      throw new Error(`SFT example ${index}: completion.content must be a string`);
    }
    return {
      prompt: obj.prompt as ChatMessage[],
      completion: obj.completion as ChatMessage,
    };
  }

  // Check for conversation format
  if ('messages' in obj) {
    if (!Array.isArray(obj.messages)) {
      throw new Error(`SFT example ${index}: messages must be an array`);
    }
    if (obj.messages.length === 0) {
      throw new Error(`SFT example ${index}: messages cannot be empty`);
    }
    // Check that at least one message is from assistant
    const hasAssistant = obj.messages.some(
      (m: unknown) => typeof m === 'object' && m !== null && (m as Record<string, unknown>).role === 'assistant',
    );
    if (!hasAssistant) {
      throw new Error(`SFT example ${index}: messages must contain at least one assistant message`);
    }
    return { messages: obj.messages as ChatMessage[] };
  }

  throw new Error(`SFT example ${index}: must have either {prompt, completion} or {messages}`);
}

/**
 * Load SFT dataset from a JSONL file
 *
 * Supports two formats:
 * 1. Prompt-Completion: {"prompt": [...], "completion": {...}}
 * 2. Conversation: {"messages": [...]}
 */
export async function loadSFTDataset(
  path: string,
  tokenizer: Qwen3Tokenizer,
  config?: SFTDatasetConfig & { limit?: number },
): Promise<SFTDataset> {
  const absolutePath = resolvePath(path);
  const rawRecords = readJsonl<unknown>(absolutePath, config?.limit);

  // Validate and convert
  const examples: SFTExample[] = rawRecords.map((record, i) => validateSFTExample(record, i));

  return new SFTDataset(examples, tokenizer, config);
}

/**
 * Create SFT dataset from examples directly
 */
export function createSFTDataset(
  examples: SFTExample[],
  tokenizer: Qwen3Tokenizer,
  config?: SFTDatasetConfig,
): SFTDataset {
  return new SFTDataset(examples, tokenizer, config);
}
