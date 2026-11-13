/**
 * Spawn environments of the two agent processes. Chat is built from nothing: no variable is inherited, so no key,
 * token or user config of the Control Panel reaches it. Work is the Control Panel's own environment, like `mlx agent`
 * in a terminal, minus the app's internal variables and every provider key that is not selected.
 */

import type { ChatPaths } from '../paths.js';

export interface InferenceAccess {
  /** Loopback origin of the inference sidecar, without a path. */
  url: string;
  token: string;
}

/** Variables a provider key is set under in Work. */
export const PROVIDER_KEY_ENV = {
  exa: ['EXA_API_KEY'],
  deepseek: ['DEEPSEEK_API_KEY'],
  brave: ['BRAVE_SEARCH_API_KEY', 'BRAVE_API_KEY'],
  firecrawl: ['FIRECRAWL_API_KEY'],
} as const;

export const ALL_PROVIDER_KEY_ENV: readonly string[] = Object.values(PROVIDER_KEY_ENV).flat();

/** Removed from the Work environment: the app's own internals. */
export const WORK_ENV_REMOVED = [
  'ELECTRON_RUN_AS_NODE',
  'MLX_DESKTOP_CLI',
  'NAPI_RS_NATIVE_LIBRARY_PATH',
  'NODE_OPTIONS',
];

export const CHAT_MAX_COMPLETION_TOKENS = 8192;
export const WORK_MAX_COMPLETION_TOKENS = 16384;

function baseUrls(endpoint: InferenceAccess): Record<string, string> {
  const v1 = `${endpoint.url}/v1`;
  return { GROK_MODELS_BASE_URL: v1, GROK_CLI_CHAT_PROXY_BASE_URL: v1, GROK_XAI_API_BASE_URL: v1 };
}

export interface ChatEnvInput {
  paths: ChatPaths;
  endpoint: InferenceAccess;
  username: string;
}

export function chatAgentEnv(input: ChatEnvInput): Record<string, string> {
  const env: Record<string, string> = {
    PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
    HOME: input.paths.chatHome,
    USER: input.username,
    LOGNAME: input.username,
    SHELL: '/bin/sh',
    LANG: 'en_US.UTF-8',
    TERM: 'dumb',
    TMPDIR: input.paths.tmp,
    GROK_HOME: input.paths.chatAgent,
    ...baseUrls(input.endpoint),
    XAI_API_KEY: input.endpoint.token,
    GROK_DISABLE_AUTOUPDATER: '1',
    GROK_IMAGE_GEN: '0',
    GROK_IMAGE_EDIT: '0',
    GROK_VIDEO_GEN: '0',
    DISABLE_TELEMETRY: '1',
    GROK_TELEMETRY_ENABLED: 'false',
    GROK_TELEMETRY_TRACE_UPLOAD: 'false',
  };
  for (const vendor of ['CLAUDE', 'CURSOR', 'CODEX']) {
    env[`GROK_${vendor}_HOOKS_ENABLED`] = 'false';
    env[`GROK_${vendor}_MCPS_ENABLED`] = 'false';
  }
  env.GROK_CLAUDE_SKILLS_ENABLED = 'false';
  env.GROK_CURSOR_SKILLS_ENABLED = 'false';
  env.GROK_TURN_SUMMARY = '0';
  env.GROK_SESSION_SEARCH = '0';
  env.GROK_CONFIG = JSON.stringify({ models: { max_completion_tokens: CHAT_MAX_COMPLETION_TOKENS } });
  return env;
}

export interface WorkEnvInput {
  /** The Control Panel's own environment. */
  base: NodeJS.ProcessEnv;
  endpoint: InferenceAccess;
  /** Set only when the runtime's agent home is not the default, so reader and agent agree. */
  agentHome: string | null;
  /** The selected providers' key variables (name → key). Every other provider key variable is removed. */
  providerKeys: Readonly<Record<string, string>>;
}

function plainObject(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/**
 * `GROK_CONFIG` of Work: the user's own overlay object keeps every key and gets the output cap in `models` (a lower cap
 * of theirs stays). A value that is not a JSON object is replaced, so the cap always holds. The overlay cannot carry
 * permission rules (the agent drops `[permission]` from it); the Work guard plugin does that job.
 */
export function workOverlay(base: string | undefined): string {
  let overlay: Record<string, unknown> = {};
  if (base !== undefined) {
    try {
      overlay = plainObject(JSON.parse(base)) ?? {};
    } catch {
      overlay = {};
    }
  }
  const models = plainObject(overlay.models) ?? {};
  const own = models.max_completion_tokens;
  const cap =
    typeof own === 'number' && Number.isInteger(own) && own > 0 && own < WORK_MAX_COMPLETION_TOKENS
      ? own
      : WORK_MAX_COMPLETION_TOKENS;
  return JSON.stringify({ ...overlay, models: { ...models, max_completion_tokens: cap } });
}

export function workAgentEnv(input: WorkEnvInput): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [name, value] of Object.entries(input.base)) if (value !== undefined) env[name] = value;
  for (const name of WORK_ENV_REMOVED) delete env[name];
  for (const name of ALL_PROVIDER_KEY_ENV) delete env[name];
  Object.assign(env, baseUrls(input.endpoint));
  env.XAI_API_KEY = input.endpoint.token;
  env.GROK_DISABLE_AUTOUPDATER = '1';
  for (const name of ['GROK_IMAGE_GEN', 'GROK_IMAGE_EDIT', 'GROK_VIDEO_GEN']) env[name] ??= '0';
  env.GROK_RESUME_COMMAND ??= 'mlx agent';
  env.GROK_TURN_SUMMARY = '0';
  // Set, so the agent does not fill it from the login shell: the Auto check places a relative `cd` in the folder.
  env.CDPATH = '';
  env.GROK_CONFIG = workOverlay(env.GROK_CONFIG);
  if (input.agentHome !== null) env.GROK_HOME = input.agentHome;
  for (const [name, value] of Object.entries(input.providerKeys)) {
    if (ALL_PROVIDER_KEY_ENV.includes(name)) env[name] = value;
  }
  env.FIRECRAWL_NO_SEARCH_FEEDBACK = '1';
  env.FIRECRAWL_DISABLE_SEARCH_FEEDBACK = '1';
  env.FIRECRAWL_NO_ENDPOINT_FEEDBACK = '1';
  return env;
}
