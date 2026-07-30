/**
 * What the menubar shows for a given supervisor snapshot — the whole decision,
 * with no Electron in it.
 *
 * Split from `tray.ts` for the reason `www.ts` is split from `protocol.ts`:
 * `import { Tray, Menu } from 'electron'` cannot resolve outside an Electron
 * process, so anything in that file is unreachable from a plain Node test. The
 * mapping is the part that can be quietly wrong, so it lives here.
 *
 * The rule that this module exists to enforce: **`lying` is not `running`.** A
 * class-(b) failure leaves `/health` answering `ok` while the sidecar produces
 * wrong output; if the tray renders that as a green "running" the user is told
 * the one thing that is not true. It gets its own indicator, its own menubar
 * text and its own status line — because the menubar icon is a template image
 * that macOS recolours itself, so colour is not available as a signal and the
 * distinction has to be carried by text.
 */

import type { SupervisorSnapshot } from './supervisor/types.js';

/**
 * Severity, not colour. The tray icon is a template image (black + alpha, tinted
 * by macOS for light/dark/active) and cannot carry a hue, so this drives the
 * menubar text and the status line instead.
 */
export type TrayIndicator = 'idle' | 'busy' | 'ok' | 'warn' | 'error';

export interface TrayPresentation {
  indicator: TrayIndicator;
  /**
   * Text drawn next to the menubar icon. Empty whenever nothing is wrong — a
   * menubar app that permanently occupies width it does not need is the first
   * thing users uninstall — and non-empty exactly when the user needs to look.
   */
  title: string;
  tooltip: string;
  /** First menu row, always disabled. The one line that says what is going on. */
  statusLabel: string;
  /** Second disabled row: the URL, the crash reason, or the swallowed native error. */
  detail: string | null;
  canStart: boolean;
  canStop: boolean;
  canRestart: boolean;
  /**
   * Whether there is a live endpoint to hand another client.
   *
   * Only in `running`: the token belongs to one child and is cleared on every
   * respawn, so offering the copy while restarting would put a command on the
   * clipboard that is already stale. `starting` has a URL but no token yet.
   */
  canCopyConnect: boolean;
  /**
   * Always true, and stated here rather than hard-coded in `tray.ts` so the rule
   * is one a test can hold: Control Panel is where the crash reason, the trace file and
   * the logs are, so the moments it is most needed are exactly the ones where
   * inference is not running.
   */
  canOpenControlPanel: boolean;
}

/**
 * `starting`, `running`, `restarting` and `lying` all mean "there is a child, or
 * there is about to be one". `restarting` is included on purpose: the child is
 * already dead and the supervisor is sitting in its backoff, and Stop is what
 * cancels that — without it the only way out of a crash loop is Quit.
 */
function isLive(state: SupervisorSnapshot['state']): boolean {
  return state === 'starting' || state === 'running' || state === 'restarting' || state === 'lying';
}

/**
 * A native menu row does not wrap and does not scroll. `e.what()` from a Metal
 * failure runs to hundreds of characters and would stretch the menu past the
 * screen, taking the Quit item with it.
 */
const MAX_DETAIL = 72;

function clip(text: string): string {
  // Newlines in a menu row render as a box glyph, and both a crash reason and a
  // swallowed `e.what()` can contain them.
  const flat = text.replaceAll(/\s+/gu, ' ').trim();
  return flat.length <= MAX_DETAIL ? flat : `${flat.slice(0, MAX_DETAIL - 1)}…`;
}

export function presentTray(snapshot: SupervisorSnapshot): TrayPresentation {
  const described = describe(snapshot);
  const { indicator, title, statusLabel } = described;
  const detail = described.detail === null ? null : clip(described.detail);
  return {
    indicator,
    title,
    statusLabel,
    detail,
    canStart: !isLive(snapshot.state),
    canStop: isLive(snapshot.state),
    canRestart: isLive(snapshot.state),
    canOpenControlPanel: true,
    canCopyConnect: snapshot.state === 'running' && snapshot.url !== null,
    tooltip: detail === null ? `mlx-node — ${statusLabel}` : `mlx-node — ${statusLabel}\n${detail}`,
  };
}

interface Described {
  indicator: TrayIndicator;
  title: string;
  statusLabel: string;
  detail: string | null;
}

function describe(snapshot: SupervisorSnapshot): Described {
  switch (snapshot.state) {
    case 'stopped':
      return {
        indicator: 'idle',
        title: '',
        statusLabel: 'Inference: stopped',
        detail: snapshot.lastExit === null ? null : `Last exit: ${snapshot.lastExit.reason}`,
      };

    case 'starting':
      return {
        indicator: 'busy',
        title: '',
        // Not a hang: a cold 9B off a slow mmap legitimately takes a minute, and
        // the supervisor's own readiness budget is 60 s.
        statusLabel: 'Inference: starting…',
        detail: snapshot.url,
      };

    case 'running': {
      // `/health` distinguishes `ok` from `degraded` (answering, but saturated or
      // contended) and `loading` (a load holds the writer slot). All three are
      // "running" to the supervisor and only one of them is what the user
      // assumes, so the rung is named whenever it is not `ok`.
      const status = snapshot.health?.status;
      return {
        indicator: 'ok',
        title: '',
        statusLabel: status === undefined || status === 'ok' ? 'Inference: running' : `Inference: running (${status})`,
        detail: snapshot.url,
      };
    }

    case 'restarting':
      return {
        indicator: 'busy',
        title: '',
        statusLabel: `Inference: restarting (crash ${snapshot.consecutiveCrashes})`,
        detail: snapshot.lastExit === null ? null : snapshot.lastExit.reason,
      };

    case 'failed':
      return {
        indicator: 'error',
        title: '✕',
        statusLabel: `Inference: failed after ${snapshot.consecutiveCrashes} crashes`,
        detail: snapshot.lastExit === null ? null : snapshot.lastExit.reason,
      };

    case 'lying': {
      // The FIRST error, not the most recent: once `mlx_array_eval` has swallowed
      // one C++ exception every array downstream of it is suspect, so later
      // entries are cascade and the first is the one that explains it.
      // `.at` rather than `[0]`: without `noUncheckedIndexedAccess` an index
      // read is typed as present, and the empty case here is reachable.
      const first = snapshot.nativeErrors.at(0);
      return {
        indicator: 'warn',
        title: '⚠',
        // Deliberately not "running". The process is up and the output cannot be
        // trusted, and that is what the user has to be told.
        statusLabel: 'Inference: OUTPUT IS NOT TRUSTWORTHY',
        detail:
          first === undefined
            ? 'A native error was swallowed; results since then may be wrong.'
            : `Swallowed in ${first.context}: ${first.detail}`,
      };
    }
  }
}

/**
 * Single-quoted with embedded quotes escaped, so a URL or token containing
 * shell metacharacters cannot turn a paste into something else. base64url
 * tokens never contain a quote today; relying on that would make both client
 * launchers fragile the day the token format changes.
 */
function shellQuote(value: string): string {
  return `'${value.replaceAll("'", String.raw`'\''`)}'`;
}

/**
 * A ready-to-paste Claude Code command pointing at this server.
 *
 * An inherited `ANTHROPIC_API_KEY` must be removed, not emptied: Claude Code
 * sends it as `x-api-key` alongside the bearer token, and the server
 * intentionally checks `x-api-key` first. `env -u` scopes the removal to this
 * command instead of mutating the user's shell.
 */
export function claudeConnectCommand(url: string, token: string): string {
  return `env -u ANTHROPIC_API_KEY ANTHROPIC_BASE_URL=${shellQuote(url)} ANTHROPIC_AUTH_TOKEN=${shellQuote(token)} claude`;
}

/**
 * A ready-to-paste Codex command pointing at this server's Responses API.
 *
 * Codex does not use the Anthropic environment variables. Its supported
 * connection surface is a custom model provider: `base_url` selects the
 * OpenAI-compatible `/v1` root and `env_key` names the variable whose value
 * Codex sends as bearer authentication. Command-line config overrides keep the
 * mlx-node provider scoped to this invocation instead of modifying the user's
 * `~/.codex/config.toml`.
 */
export function codexConnectCommand(url: string, token: string): string {
  const baseUrl = `${url.replace(/\/+$/u, '')}/v1`;
  const config = (key: string, value: string): string => `-c ${shellQuote(`${key}=${JSON.stringify(value)}`)}`;

  return [
    `MLX_NODE_API_KEY=${shellQuote(token)}`,
    'codex',
    '-m mlx-node',
    config('model_provider', 'mlx-node'),
    config('model_providers.mlx-node.name', 'MLX-Node'),
    config('model_providers.mlx-node.base_url', baseUrl),
    config('model_providers.mlx-node.env_key', 'MLX_NODE_API_KEY'),
    config('model_providers.mlx-node.wire_api', 'responses'),
  ].join(' ');
}
