/**
 * Spawn environments of the two agent processes. Chat is built from nothing: no variable is inherited, so no key,
 * token or user config of the Control Panel reaches it. Work is the Control Panel's own environment, like `mlx agent`
 * in a terminal, minus the app's internal variables, every provider key that is not selected and the Firecrawl
 * address, which the app sets itself.
 */

import { AGENT_HOST_THINKING } from '@mlx-node/server/host/agent-thinking';

import type { ChatPaths } from '../paths.js';
import { PROVIDER_ORIGINS } from '../web/http.js';

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

/**
 * `FIRECRAWL_API_KEY` of Work while the `firecrawl` CLI uses Firecrawl on this Mac. The CLI takes this variable before
 * the key saved by `firecrawl login`, so that key is never sent to the saved address. Firecrawl Local passes a search
 * on to Firecrawl Cloud with its own key (its `FIRECRAWL_API_KEY`, or the `firecrawl login` key it reads itself)
 * before a caller's, so only a server with no key of its own passes this one on, and it is no account's key.
 */
export const FIRECRAWL_LOCAL_PLACEHOLDER_KEY = 'fc-local-no-key';

/** Removed from the Work environment: the app's own internals. */
export const WORK_ENV_REMOVED = [
  'ELECTRON_RUN_AS_NODE',
  'MLX_DESKTOP_CLI',
  'NAPI_RS_NATIVE_LIBRARY_PATH',
  'NODE_OPTIONS',
];

/** Room for the answer after the thought. */
export const CHAT_ANSWER_TOKENS = 8192;
export const WORK_ANSWER_TOKENS = 16384;
/** Each cap is the answer room plus the engine's thinking budget, so the budget fires before the cap. */
export const CHAT_MAX_COMPLETION_TOKENS = CHAT_ANSWER_TOKENS + AGENT_HOST_THINKING.tokenBudget;
export const WORK_MAX_COMPLETION_TOKENS = WORK_ANSWER_TOKENS + AGENT_HOST_THINKING.tokenBudget;
/**
 * Work's stall limit for one model stream, in seconds. A local compaction reads the conversation for many minutes with
 * no event the agent counts, and a workflow child queued behind it dies at the agent's 600 s default. `0` would mean
 * 10 s in the agent, never "off".
 */
export const AGENT_IDLE_TIMEOUT_SECS = 3600;

/**
 * Set over any inherited value. Two-pass compaction starts a second summary early, which the local engine serves only
 * after the main step, and the 300 s compaction clock cuts every local summary before it ends.
 */
function compactionEnv(env: Record<string, string>): void {
  env.GROK_TWO_PASS_COMPACTION = '0';
  env.GROK_COMPACTION_WALL_CLOCK_SECS = '0';
}

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
  env.GROK_WORKFLOWS = '0';
  compactionEnv(env);
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
  /**
   * Firecrawl on this Mac, when the Work `firecrawl` CLI uses it: its address, with {@link FIRECRAWL_LOCAL_PLACEHOLDER_KEY}
   * in place of a Firecrawl key.
   */
  firecrawlUrl: string | null;
}

function plainObject(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/**
 * `GROK_CONFIG` of Work: the user's own overlay object keeps every key and gets the output cap and the stall limit in
 * `models` (a lower cap and a longer integer limit of theirs stay). A value that is not a JSON object is replaced, so the cap always holds. The overlay cannot carry
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
  const ownIdle = models.inference_idle_timeout_secs;
  const idle =
    typeof ownIdle === 'number' && Number.isInteger(ownIdle) && ownIdle > AGENT_IDLE_TIMEOUT_SECS
      ? ownIdle
      : AGENT_IDLE_TIMEOUT_SECS;
  return JSON.stringify({
    ...overlay,
    models: { ...models, max_completion_tokens: cap, inference_idle_timeout_secs: idle },
  });
}

/**
 * What Work inherits from the Control Panel: every variable minus the app's internals, every provider key and the
 * address the `firecrawl` CLI sends its key to.
 */
export function workInherited(base: NodeJS.ProcessEnv): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [name, value] of Object.entries(base)) if (value !== undefined) env[name] = value;
  for (const name of WORK_ENV_REMOVED) delete env[name];
  for (const name of ALL_PROVIDER_KEY_ENV) delete env[name];
  delete env.FIRECRAWL_API_URL;
  return env;
}

export function workAgentEnv(input: WorkEnvInput): Record<string, string> {
  const env = workInherited(input.base);
  Object.assign(env, baseUrls(input.endpoint));
  env.XAI_API_KEY = input.endpoint.token;
  env.GROK_DISABLE_AUTOUPDATER = '1';
  for (const name of ['GROK_IMAGE_GEN', 'GROK_IMAGE_EDIT', 'GROK_VIDEO_GEN']) env[name] ??= '0';
  env.GROK_RESUME_COMMAND ??= 'mlx agent';
  env.GROK_TURN_SUMMARY = '0';
  compactionEnv(env);
  // Set, so the agent does not fill it from the login shell: the Auto check places a relative `cd` in the folder.
  env.CDPATH = '';
  env.GROK_CONFIG = workOverlay(env.GROK_CONFIG);
  if (input.agentHome !== null) env.GROK_HOME = input.agentHome;
  const firecrawlKeys: readonly string[] = PROVIDER_KEY_ENV.firecrawl;
  for (const [name, value] of Object.entries(input.providerKeys)) {
    if (!ALL_PROVIDER_KEY_ENV.includes(name)) continue;
    // Firecrawl on this Mac passes a caller's key on to Firecrawl Cloud.
    if (input.firecrawlUrl !== null && firecrawlKeys.includes(name)) continue;
    env[name] = value;
  }
  // Set over the CLI's own saved address, which it would otherwise send the key to.
  if (input.firecrawlUrl !== null) {
    env.FIRECRAWL_API_URL = input.firecrawlUrl;
    env.FIRECRAWL_API_KEY = FIRECRAWL_LOCAL_PLACEHOLDER_KEY;
  } else if (env.FIRECRAWL_API_KEY !== undefined) env.FIRECRAWL_API_URL = PROVIDER_ORIGINS.firecrawl;
  env.FIRECRAWL_NO_SEARCH_FEEDBACK = '1';
  env.FIRECRAWL_DISABLE_SEARCH_FEEDBACK = '1';
  env.FIRECRAWL_NO_ENDPOINT_FEEDBACK = '1';
  return env;
}
