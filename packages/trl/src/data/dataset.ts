import { readFileSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';
import type {
  DatasetExample,
  ChatMessage,
  ChatRole,
  DatasetSplit,
  PromptFormatterOptions,
  PromptTemplate,
  DatasetLoader,
} from '../types';
import { extractHashAnswer } from '../utils/xml-parser';

export interface LocalDatasetOptions extends PromptFormatterOptions {
  basePath?: string;
  promptTemplate?: PromptTemplate;
  metadata?: Record<string, unknown>;
}

interface Gsm8kRecord {
  question: string;
  answer: string;
}

const DEFAULT_BASE_PATH = resolvePath(process.cwd(), 'data/gsm8k');
const VALID_SPLITS = new Set(['train', 'test']);

export const SYSTEM_PROMPT = `
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
`.trim();

export const XML_COT_FORMAT = `<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>`;

const SYSTEM_MESSAGE: ChatMessage = {
  role: 'system',
  content: SYSTEM_PROMPT,
};

function createMessage(role: ChatRole, content: string): ChatMessage {
  return { role, content };
}

export const defaultPromptTemplate: PromptTemplate = (question, options) => {
  const messages: ChatMessage[] = [SYSTEM_MESSAGE];
  if (options?.includeOneShot && options.oneShotExample) {
    const { question: exampleQuestion, reasoning, answer } = options.oneShotExample;

    messages.push(
      createMessage('user', exampleQuestion),
      createMessage('assistant', XML_COT_FORMAT.replace('{reasoning}', reasoning).replace('{answer}', answer)),
    );
  }
  messages.push(createMessage('user', question));
  return messages;
};

export function createDatasetExample(
  prompt: ChatMessage[],
  answer: string | null,
  metadata?: Record<string, unknown>,
): DatasetExample {
  return {
    prompt: prompt.map((message) => ({ ...message })), // defensive copy
    answer,
    metadata: metadata ? { ...metadata } : undefined,
  };
}

export function extractGsm8kAnswer(raw: string): string | null {
  return extractHashAnswer(raw);
}

export function validateDatasetExample(example: DatasetExample): void {
  if (!Array.isArray(example.prompt) || example.prompt.length === 0) {
    throw new Error('Dataset example must contain at least one prompt message.');
  }
  for (const message of example.prompt) {
    if (!message || typeof message.content !== 'string' || message.content.trim() === '') {
      throw new Error('Prompt messages must include non-empty textual content.');
    }
    if (message.role !== 'system' && message.role !== 'user' && message.role !== 'assistant') {
      throw new Error(`Unsupported chat role: ${String(message.role)}`);
    }
  }
}

function resolveBasePath(optionPath?: string): string {
  return optionPath ? resolvePath(process.cwd(), optionPath) : DEFAULT_BASE_PATH;
}

function datasetFileForSplit(split: DatasetSplit): string {
  if (!VALID_SPLITS.has(split)) {
    throw new Error(`Unsupported GSM8K split "${split}". Expected one of: ${Array.from(VALID_SPLITS).join(', ')}`);
  }
  return `${split}.jsonl`;
}

function readDatasetFile(filePath: string): string {
  try {
    return readFileSync(filePath, 'utf8');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to read dataset file at ${filePath}: ${message}`);
  }
}

function readJsonl(path: string, limit?: number): Gsm8kRecord[] {
  const fileContents = readDatasetFile(path);
  const lines = fileContents.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const records: Gsm8kRecord[] = [];
  const max = typeof limit === 'number' && limit >= 0 ? limit : Number.POSITIVE_INFINITY;

  for (let i = 0; i < lines.length && records.length < max; i += 1) {
    const line = lines[i];
    try {
      const parsed = JSON.parse(line) as Partial<Gsm8kRecord>;
      if (typeof parsed.question !== 'string' || typeof parsed.answer !== 'string') {
        throw new Error('Record must include string "question" and "answer" fields.');
      }
      records.push({ question: parsed.question, answer: parsed.answer });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to parse JSONL record at ${path}:${i + 1} - ${message}`);
    }
  }

  return records;
}

export async function loadLocalGsm8kDataset(
  split: DatasetSplit,
  options: LocalDatasetOptions & { limit?: number } = {},
): Promise<DatasetExample[]> {
  const basePath = resolveBasePath(options.basePath);
  const fileName = datasetFileForSplit(split);
  const filePath = resolvePath(basePath, fileName);

  const promptTemplate = options.promptTemplate ?? defaultPromptTemplate;
  const records = readJsonl(filePath, options.limit);

  const examples: DatasetExample[] = records.map((record, index) => {
    const prompt = promptTemplate(record.question, {
      includeOneShot: options.includeOneShot,
      oneShotExample: options.oneShotExample,
    });
    const answer = extractGsm8kAnswer(record.answer);
    const example = createDatasetExample(prompt, answer, {
      split,
      index,
      raw_answer: record.answer,
      ...options.metadata,
    });
    validateDatasetExample(example);
    return example;
  });

  return examples;
}

export class LocalGsm8kDatasetLoader implements DatasetLoader {
  private readonly options: LocalDatasetOptions;

  constructor(options: LocalDatasetOptions = {}) {
    this.options = { ...options };
  }

  async load(split: DatasetSplit, limit?: number): Promise<DatasetExample[]> {
    return loadLocalGsm8kDataset(split, { ...this.options, limit });
  }
}
