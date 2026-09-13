/** Addon-free client for the desktop's global instruction installation checks. */
import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

export const DELEGATION_PROMPT =
  'Delegate GitHub investigation to `mlx delegate github --repo OWNER/REPO "TASK"`. Include the PR, issue, or run number. Use its findings and evidence for implementation; request more detail when needed. Add `--allow-write` only for GitHub changes already authorized by the user. If delegation fails or reports incomplete work, continue from its handoff.';

export interface LocalInferenceConnection {
  url: string;
  token?: string;
  model: string;
}

export interface LocalMessage {
  role: 'user' | 'assistant';
  content: string;
}

export function expandHome(path: string, home = homedir()): string {
  return path === '~' ? home : path.startsWith('~/') ? join(home, path.slice(2)) : path;
}

/** Match the persisted default used by `mlx agent`; never select a cloud provider. */
export async function preferredLocalModel(home = homedir(), env = process.env): Promise<string | undefined> {
  const dir = env.PI_CODING_AGENT_DIR ? expandHome(env.PI_CODING_AGENT_DIR, home) : join(home, '.mlx-node', 'agent');
  try {
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
  maxTokens = 1024,
): Promise<string> {
  const url = new URL(connection.url);
  if (url.protocol !== 'http:' || !['127.0.0.1', '[::1]', 'localhost'].includes(url.hostname)) {
    throw new Error('Delegation requires a local inference server.');
  }
  const controller = new AbortController();
  const cancel = (): void => controller.abort(signal?.reason);
  if (signal?.aborted) cancel();
  else signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(() => controller.abort(new Error('The local model request timed out.')), 120_000);
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
    const response = await fetch(new URL('/v1/messages', url), {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model: connection.model,
        system,
        messages,
        max_tokens: maxTokens,
        temperature: 0,
        stream: false,
      }),
      signal: controller.signal,
      redirect: 'error',
    });
    if (!response.ok) throw new Error(`The local model could not complete the request (HTTP ${response.status}).`);
    const body = (await response.json()) as { content?: { type: string; text?: string }[]; stop_reason?: string };
    if (body.stop_reason === 'max_tokens')
      throw new Error('The local model ran out of output space. Try checking again.');
    const text = body.content
      ?.filter((block) => block.type === 'text')
      .map((block) => block.text ?? '')
      .join('\n')
      .trim();
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
