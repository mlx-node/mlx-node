// Re-export types from @mlx-node/core
export type { CompletionInfo, RewardOutput } from '@mlx-node/core';
import type { RewardOutput } from '@mlx-node/core';

export type ChatRole = 'system' | 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface CompletionMessage extends ChatMessage {}

export type Completion = CompletionMessage[];

export type DatasetSplit = 'train' | 'test' | (string & {});

export interface DatasetExample {
  prompt: ChatMessage[];
  answer: string | null;
  metadata?: Record<string, unknown>;
}

export interface XmlParseResult {
  reasoning: string | null;
  answer: string | null;
  isStrictMatch: boolean;
  isSoftMatch: boolean;
  errors: string[];
}

export interface RewardComputationInput {
  prompts: ChatMessage[][];
  completions: Completion[];
  answers: (string | null)[];
}

/**
 * Unified reward function type for GRPO training.
 *
 * Takes an array of RewardOutput objects containing structured completion data.
 * Returns rewards for each completion (one per output).
 */
export type RewardFunction = (outputs: RewardOutput[]) => number[] | Float32Array | Promise<number[] | Float32Array>;

export interface PromptFormatterOptions {
  includeOneShot?: boolean;
  oneShotExample?: {
    question: string;
    reasoning: string;
    answer: string;
  };
}

export type PromptTemplate = (question: string, options?: PromptFormatterOptions) => ChatMessage[];

/**
 * Converts a ChatMessage array to a string for reward function input
 *
 * This allows customization of how prompts are formatted as strings
 * for different model architectures (Qwen3, Llama, etc.)
 */
export type PromptFormatter = (messages: ChatMessage[]) => string;

export interface DatasetLoader {
  load(split: DatasetSplit, limit?: number): Promise<DatasetExample[]>;
}
