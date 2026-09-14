/** Addon-free client for the desktop's global instruction installation checks. */
import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { expandPiAgentDir } from './paths.js';

export const DELEGATION_PROMPT =
  'Delegate GitHub investigation to `mlx delegate github --caller-approved --repo OWNER/REPO "TASK"`. Approve the bounded task and its tool execution before invoking. Include the PR, issue, or run number. Use its findings and evidence for implementation; request more detail when needed. Add `--allow-write` only for GitHub changes already authorized by the user. If delegation fails or reports incomplete work, continue from its handoff.';

/** Absolute paths avoid dependence on each coding agent's shell startup/PATH. */
export function delegationCommand(path: string): string {
  if (/[\0\r\n`]/.test(path)) throw new Error('The delegation command path cannot be represented in instructions.');
  return `'${path.replaceAll("'", "'\\''")}'`;
}

export function delegationPrompt(path: string): string {
  return DELEGATION_PROMPT.replace('`mlx delegate', `\`${delegationCommand(path)} delegate`);
}

export interface LocalInferenceConnection {
  url: string;
  token?: string;
  model: string;
}

export interface LocalMessage {
  role: 'user' | 'assistant';
  content: string;
}

/** Shared by setup requests and their verdict cache key. Includes thinking and final output. */
export const INSTALL_CHECK_GENERATION = Object.freeze({
  reasoning: Object.freeze({ effort: 'medium' as const }),
  max_output_tokens: 16384,
  temperature: 0,
});
export const INSTALL_CHECK_TIMEOUT_MS = 600_000;

export function expandHome(path: string, home = homedir()): string {
  return path === '~' ? home : path.startsWith('~/') ? join(home, path.slice(2)) : path;
}

/** Match the persisted default used by `mlx agent`; never select a cloud provider. */
export async function preferredLocalModel(home = homedir(), env = process.env): Promise<string | undefined> {
  try {
    const dir = env.PI_CODING_AGENT_DIR
      ? expandPiAgentDir(env.PI_CODING_AGENT_DIR, home)
      : join(home, '.mlx-node', 'agent');
    const value = JSON.parse(await readFile(join(dir, 'settings.json'), 'utf8'));
    if (value.defaultProvider === 'mlx' && typeof value.defaultModel === 'string') {
      return value.defaultModel.replace(/^mlx\//, '');
    }
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
      throw new Error('Could not read the default local model. Open mlx agent and select a model again.');
    }
  }
  return undefined;
}

/** Refuse remote endpoints even if a caller accidentally supplies a cloud API URL. */
export async function localCompletion(
  connection: LocalInferenceConnection,
  system: string,
  messages: LocalMessage[],
  signal?: AbortSignal,
  maxTokens: number = INSTALL_CHECK_GENERATION.max_output_tokens,
): Promise<string> {
  const url = new URL(connection.url);
  if (url.protocol !== 'http:' || !['127.0.0.1', '[::1]', 'localhost'].includes(url.hostname)) {
    throw new Error('Delegation requires a local inference server.');
  }
  const controller = new AbortController();
  const cancel = (): void => controller.abort(signal?.reason);
  if (signal?.aborted) cancel();
  else signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(
    () => controller.abort(new Error('The local model request timed out.')),
    INSTALL_CHECK_TIMEOUT_MS,
  );
  try {
    const headers = {
      'content-type': 'application/json',
      ...(connection.token ? { 'x-api-key': connection.token } : {}),
    };
    const catalog = await fetch(new URL('/v1/models', url), { headers, signal: controller.signal, redirect: 'error' });
    if (!catalog.ok) throw new Error('The local model service is unavailable. Restart it and try again.');
    const models = (await catalog.json()) as { data?: { id: string }[] };
    if (!models.data?.some((model) => model.id === connection.model)) {
      throw new Error(
        'The default model is not available in the running service. Restart the local model service and try again.',
      );
    }
    const response = await fetch(new URL('/v1/responses', url), {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: connection.model,
        instructions: system,
        input: messages,
        ...INSTALL_CHECK_GENERATION,
        max_output_tokens: maxTokens,
        store: false,
        stream: false,
      }),
      signal: controller.signal,
      redirect: 'error',
    });
    if (!response.ok) throw new Error(`The local model could not complete the request (HTTP ${response.status}).`);
    const body = (await response.json()) as {
      status?: string;
      output_text?: string;
      incomplete_details?: { reason?: string } | null;
    };
    if (body.incomplete_details?.reason === 'max_output_tokens')
      throw new Error('The local model ran out of output space. Try checking again.');
    if (body.status !== 'completed')
      throw new Error('The local model did not complete its answer. Try checking again.');
    const text = body.output_text?.trim();
    if (!text) throw new Error('The local model returned no answer. Try checking again.');
    return text;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}

export function parseLocalJson(text: string): unknown {
  return JSON.parse(
    text
      .trim()
      .replace(/^```(?:json)?\s*\n?/, '')
      .replace(/\n?```$/, ''),
  );
}
