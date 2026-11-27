/**
 * Spawn environments of the two agent processes. Chat is built from nothing: no variable is inherited, so no key,
 * token or user config of the Control Panel reaches it. Work is the Control Panel's own environment, like `mlx agent`
 * in a terminal, minus the app's internal variables and every provider key that is not selected; in the app its `PATH`
 * gets the login shell's entries first, as a terminal has them.
 */

import { GROK_BUILD_ENV_REMOVED, LOOPBACK_CONNECT_ATTEMPTS } from '@mlx-node/agent/paths';
import { AGENT_HOST_THINKING } from '@mlx-node/server/host/agent-thinking';

import type { ChatPaths } from '../paths.js';
import { PROVIDER_ORIGINS } from '../web/http.js';
import { withLoginPath } from './login-path.js';

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

/**
 * Work with the user's Claude Code setup turned off, set over any inherited value: `~/.claude/settings.json` hooks,
 * `~/.claude/skills` and `commands`, `~/.claude/CLAUDE.md` (the AGENTS switch, not subagents), `~/.claude/rules`,
 * `~/.claude.json` MCP servers, and every Claude plugin source plus `.claude/agents` subagents (PLUGINS, v0.3.6). The
 * app's own `_meta.pluginDirs` guard still loads.
 */
export const CLAUDE_CODE_OFF_ENV: Readonly<Record<string, string>> = {
  GROK_CLAUDE_HOOKS_ENABLED: 'false',
  GROK_CLAUDE_SKILLS_ENABLED: 'false',
  GROK_CLAUDE_AGENTS_ENABLED: 'false',
  GROK_CLAUDE_RULES_ENABLED: 'false',
  GROK_CLAUDE_MCPS_ENABLED: 'false',
  GROK_CLAUDE_PLUGINS_ENABLED: 'false',
};

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
  platform?: NodeJS.Platform;
  systemRoot?: string;
}

export function chatAgentEnv(input: ChatEnvInput): Record<string, string> {
  const platform = input.platform ?? process.platform;
  const baseEnv: Record<string, string> =
    platform === 'win32'
      ? (() => {
          const rawSysRoot = input.systemRoot || process.env.SystemRoot || process.env.SYSTEMROOT || 'C:\\Windows';
          const sysRoot = rawSysRoot.replace(/[\\/]+$/, '') || 'C:\\Windows';
          const driveMatch = /^[A-Za-z]:/.exec(sysRoot);
          const systemDrive = driveMatch ? driveMatch[0] : 'C:';
          return {
            SystemRoot: sysRoot,
            WINDIR: sysRoot,
            SystemDrive: systemDrive,
            ComSpec: `${sysRoot}\\System32\\cmd.exe`,
            PATHEXT: '.COM;.EXE;.BAT;.CMD',
            PATH: `${sysRoot}\\System32;${sysRoot}`,
            HOME: input.paths.chatHome,
            USERPROFILE: input.paths.chatHome,
            APPDATA: input.paths.chatHome,
            LOCALAPPDATA: input.paths.chatHome,
            USER: input.username,
            USERNAME: input.username,
            LOGNAME: input.username,
            LANG: 'en_US.UTF-8',
            TERM: 'dumb',
            TMPDIR: input.paths.tmp,
            TEMP: input.paths.tmp,
            TMP: input.paths.tmp,
          };
        })()
      : {
          PATH: '/usr/bin:/bin:/usr/sbin:/sbin',
          HOME: input.paths.chatHome,
          USER: input.username,
          LOGNAME: input.username,
          SHELL: '/bin/sh',
          LANG: 'en_US.UTF-8',
          TERM: 'dumb',
          TMPDIR: input.paths.tmp,
        };
  const env: Record<string, string> = {
    ...baseEnv,
    GROK_HOME: input.paths.chatAgent,
    ...baseUrls(input.endpoint),
    XAI_API_KEY: input.endpoint.token,
    GROK_LOOPBACK_CONNECT_MAX_ATTEMPTS: LOOPBACK_CONNECT_ATTEMPTS,
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
  env.GROK_CLAUDE_PLUGINS_ENABLED = 'false';
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
  /** `GROK_HOME` of the agent, over any inherited value: the runtime's agent home, so reader and agent agree. */
  agentHome: string;
  /** The selected providers' key variables (name → key). Every other provider key variable is removed. */
  providerKeys: Readonly<Record<string, string>>;
  /** The user's login-shell `PATH` (`login-path.ts`): its entries go first, then the inherited ones it lacks. */
  loginPath?: string | null;
  /** The user's Claude Code setup loads (on when absent); off sets {@link CLAUDE_CODE_OFF_ENV}. */
  claudeCode?: boolean;
  platform?: NodeJS.Platform;
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
 * What Work inherits from the Control Panel: every variable minus the app's internals, grok-build's own files and every
 * provider key.
 */
export function workInherited(
  base: NodeJS.ProcessEnv,
  platform: NodeJS.Platform = process.platform,
): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [name, value] of Object.entries(base)) if (value !== undefined) env[name] = value;
  const removed = [...WORK_ENV_REMOVED, ...GROK_BUILD_ENV_REMOVED, ...ALL_PROVIDER_KEY_ENV];
  if (platform === 'win32') {
    const blocked = new Set(removed.map((name) => name.toLowerCase()));
    for (const key of Object.keys(env)) {
      if (blocked.has(key.toLowerCase())) delete env[key];
    }
  } else {
    for (const name of removed) delete env[name];
  }
  return env;
}

export function workAgentEnv(input: WorkEnvInput): Record<string, string> {
  const env = workInherited(withLoginPath(input.base, input.loginPath), input.platform);
  Object.assign(env, baseUrls(input.endpoint));
  env.XAI_API_KEY = input.endpoint.token;
  env.GROK_LOOPBACK_CONNECT_MAX_ATTEMPTS ??= LOOPBACK_CONNECT_ATTEMPTS;
  env.GROK_DISABLE_AUTOUPDATER = '1';
  for (const name of ['GROK_IMAGE_GEN', 'GROK_IMAGE_EDIT', 'GROK_VIDEO_GEN']) env[name] ??= '0';
  env.GROK_RESUME_COMMAND ??= 'mlx agent';
  env.GROK_TURN_SUMMARY = '0';
  // The Activity rail's Message reaches a running subagent only with it; it also gives the model `send_subagent_message`.
  env.GROK_ACTIVE_AGENT_MESSAGES = '1';
  compactionEnv(env);
  // Set, so the agent does not fill it from the login shell: the Auto check places a relative `cd` in the folder.
  env.CDPATH = '';
  env.GROK_CONFIG = workOverlay(env.GROK_CONFIG);
  env.GROK_HOME = input.agentHome;
  for (const [name, value] of Object.entries(input.providerKeys)) {
    if (ALL_PROVIDER_KEY_ENV.includes(name)) env[name] = value;
  }
  // The CLI sends the key to `FIRECRAWL_API_URL` before an address it saved itself (`firecrawl config --api-url`).
  if (env.FIRECRAWL_API_KEY !== undefined) env.FIRECRAWL_API_URL = PROVIDER_ORIGINS.firecrawl;
  env.FIRECRAWL_NO_SEARCH_FEEDBACK = '1';
  env.FIRECRAWL_DISABLE_SEARCH_FEEDBACK = '1';
  env.FIRECRAWL_NO_ENDPOINT_FEEDBACK = '1';
  if (input.claudeCode === false) Object.assign(env, CLAUDE_CODE_OFF_ENV);
  return env;
}
