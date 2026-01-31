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
import { validatePathContainment, getAllowedRoot, type PathValidationOptions } from '../utils/path-security';

// -100 is the standard ignore index for cross-entropy loss
const IGNORE_INDEX = -100;

/**
 * Special token IDs for SFT label masking
 *
 * These are used to detect assistant message boundaries in tokenized conversations.
 * The IDs can be derived from the tokenizer or provided explicitly.
 */
export interface SpecialTokenIds {
  /** Token ID for <|im_start|> */
  imStart: number;
  /** Token ID for <|im_end|> */
  imEnd: number;
  /** Token IDs that represent newlines (for detecting end of role header) */
  newlineTokens: number[];
}

/**
 * Get special token IDs from a tokenizer
 *
 * Queries the tokenizer to get the actual token IDs for special tokens.
 * This ensures portability across different tokenizers/vocabularies.
 *
 * @param tokenizer - The tokenizer instance
 * @returns Special token IDs derived from the tokenizer
 * @throws Error if required special tokens are not found
 */
function getSpecialTokenIds(tokenizer: Qwen3Tokenizer): SpecialTokenIds {
  // Get im_start and im_end tokens using the tokenizer's special token getters
  const imStartToken = tokenizer.getImStartToken(); // "<|im_start|>"
  const imEndToken = tokenizer.getImEndToken(); // "<|im_end|>"

  const imStart = tokenizer.tokenToId(imStartToken);
  const imEnd = tokenizer.tokenToId(imEndToken);

  // Validate that we got valid IDs (tokenToId returns null for unknown tokens)
  if (imStart === null || imEnd === null) {
    throw new Error(
      `Tokenizer does not have required special tokens for ChatML format. ` +
        `Got im_start=${imStart}, im_end=${imEnd}. ` +
        `This tokenizer may not be compatible with ChatML format.`,
    );
  }

  // Get newline token IDs - these vary by tokenizer
  // Try common newline representations
  const newlineTokens: number[] = [];
  const potentialNewlines = ['\n', ' \n', '\r\n', '\n\n'];
  for (const nl of potentialNewlines) {
    const id = tokenizer.tokenToId(nl);
    if (id !== null && !newlineTokens.includes(id)) {
      newlineTokens.push(id);
    }
  }

  // If no newline tokens found, we'll rely on the fallback in tokenizeConversation
  return {
    imStart,
    imEnd,
    newlineTokens,
  };
}

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

  /**
   * Special token IDs for label masking.
   *
   * If not provided, these are automatically derived from the tokenizer.
   * This option allows explicit overriding for custom tokenizers or
   * non-standard vocabularies.
   */
  specialTokenIds?: Partial<SpecialTokenIds>;
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
 * SFT Dataset class for handling SFT training data
 */
export class SFTDataset {
  private examples: SFTExample[];
  private tokenizer: Qwen3Tokenizer;
  private config: Required<Omit<SFTDatasetConfig, 'seed' | 'specialTokenIds'>> & { seed: number };
  private format: 'prompt-completion' | 'conversation';
  private shuffledIndices: number[];
  private rng: () => number;
  /** Cached special token IDs for label masking */
  private specialTokenIds: SpecialTokenIds;

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

    // Get special token IDs from tokenizer, with optional overrides
    const derivedTokenIds = getSpecialTokenIds(tokenizer);
    this.specialTokenIds = {
      imStart: config.specialTokenIds?.imStart ?? derivedTokenIds.imStart,
      imEnd: config.specialTokenIds?.imEnd ?? derivedTokenIds.imEnd,
      newlineTokens: config.specialTokenIds?.newlineTokens ?? derivedTokenIds.newlineTokens,
    };

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
    const promptArr = Array.from(promptTokens, Number);
    const inputIds = Array.from(fullTokens, Number);

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
   * Check if a token ID is a newline token
   */
  private isNewlineToken(tokenId: number): boolean {
    return this.specialTokenIds.newlineTokens.includes(tokenId);
  }

  /**
   * Tokenize a conversation example
   *
   * For conversations, we train on all assistant turns.
   * Non-assistant tokens (system, user) are masked with -100.
   *
   * Uses single-pass tokenization with token-based boundary detection.
   * Token IDs are derived from the tokenizer for portability across models.
   */
  private async tokenizeConversation(
    example: SFTConversationExample,
  ): Promise<{ inputIds: number[]; labels: number[] }> {
    const messages = example.messages;

    // Single tokenization pass
    const fullTokens = await this.tokenizer.applyChatTemplate(messages, false, null, this.config.enableThinking);

    const inputIds = Array.from(fullTokens, Number);

    // If not masking prompts, all tokens are trainable
    if (!this.config.completionOnly) {
      return { inputIds, labels: inputIds.slice() };
    }

    // Token-based boundary detection using special tokens (derived from tokenizer)
    const { imStart, imEnd } = this.specialTokenIds;

    // Get "assistant" token ID (it's a single token in Qwen3)
    const assistantTokenId = this.tokenizer.tokenToId('assistant');

    const labels = Array.from({ length: inputIds.length }, () => IGNORE_INDEX);
    let inAssistant = false;

    for (let i = 0; i < inputIds.length; i++) {
      // Detect assistant region: <|im_start|> followed by "assistant" token
      if (inputIds[i] === imStart && i + 1 < inputIds.length && inputIds[i + 1] === assistantTokenId) {
        // Skip the <|im_start|>assistant\n header, start training from content
        // Find the newline after "assistant"
        let j = i + 2;
        while (j < inputIds.length && inputIds[j] !== imEnd) {
          // Look for newline token (dynamically derived from tokenizer)
          if (this.isNewlineToken(inputIds[j])) {
            inAssistant = true;
            i = j; // Skip to after header
            break;
          }
          j++;
        }
        if (!inAssistant) {
          // Fallback: just start after assistant token
          inAssistant = true;
          i = i + 1;
        }
        continue;
      }

      if (inAssistant && inputIds[i] !== imEnd) {
        labels[i] = inputIds[i];
      }

      if (inputIds[i] === imEnd) {
        inAssistant = false;
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
    const tokenized: Array<{ inputIds: number[]; labels: number[] }> = [];
    for (const example of examples) {
      tokenized.push(await this.tokenizeExample(example));
    }

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
 *
 * @param path - Path to the JSONL file (relative to cwd or allowedRoot)
 * @param tokenizer - Qwen3 tokenizer instance
 * @param config - Optional configuration including path validation options
 */
export async function loadSFTDataset(
  path: string,
  tokenizer: Qwen3Tokenizer,
  config?: SFTDatasetConfig & { limit?: number } & PathValidationOptions,
): Promise<SFTDataset> {
  const allowedRoot = getAllowedRoot(config);
  const absolutePath = resolvePath(allowedRoot, path);

  // Validate the path stays within allowed root to prevent directory traversal
  validatePathContainment(absolutePath, allowedRoot);

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
