/**
 * The menubar is the only place most users ever look, so the mapping from
 * supervisor state to what it shows is the whole user-facing contract of the
 * supervisor.
 *
 * The test this file exists for is `lying` vs `running`: a swallowed C++
 * exception leaves `/health` answering `ok` and the output wrong, and a tray
 * that renders that as "running" tells the user the one thing that is not true.
 */

import { execFileSync } from 'node:child_process';

import { describe, expect, it } from 'vite-plus/test';

import type { SupervisorSnapshot, SupervisorState } from '../src/main/supervisor/types.js';
import { claudeConnectCommand, codexConnectCommand, presentTray } from '../src/main/tray-view.js';

function snapshot(state: SupervisorState, over: Partial<SupervisorSnapshot> = {}): SupervisorSnapshot {
  return {
    state,
    pid: 4242,
    url: 'http://127.0.0.1:51423',
    generation: 1,
    consecutiveCrashes: 0,
    lastExit: null,
    health: null,
    nativeErrors: [],
    traceFile: '/tmp/trace.log',
    ...over,
  };
}

const NATIVE_ERROR = { context: 'array_eval', detail: 'Metal command buffer failed', observedAtMs: 1 };

const ALL_STATES: SupervisorState[] = ['stopped', 'starting', 'running', 'restarting', 'failed', 'lying'];

describe('presentTray', () => {
  it('says something specific for every state the supervisor can report', () => {
    const labels = ALL_STATES.map((state) => presentTray(snapshot(state)).statusLabel);
    // Six states, six distinct lines. A mapping that collapsed any pair — the
    // likeliest being `lying` folded into `running` — shows up here as a
    // duplicate.
    expect(new Set(labels).size).toBe(ALL_STATES.length);
    for (const label of labels) expect(label).not.toBe('');
  });

  // The one that matters. `lying` IS `running` underneath — same process, same
  // port, same `/health` — and every visible signal must still separate them.
  it('never lets `lying` look like `running`', () => {
    const running = presentTray(snapshot('running'));
    const lying = presentTray(snapshot('lying', { nativeErrors: [NATIVE_ERROR] }));

    expect(running.indicator).toBe('ok');
    expect(lying.indicator).toBe('warn');
    expect(lying.statusLabel).not.toBe(running.statusLabel);
    // The tray icon is a template image that macOS recolours itself, so colour
    // cannot carry this. The menubar text is what is left, and it is empty while
    // things are fine — so a non-empty title IS the signal.
    expect(running.title).toBe('');
    expect(lying.title).not.toBe('');
    expect(lying.tooltip).not.toBe(running.tooltip);
  });

  it('names the swallowed native error rather than saying something went wrong', () => {
    const lying = presentTray(snapshot('lying', { nativeErrors: [NATIVE_ERROR] }));
    expect(lying.detail).toContain('array_eval');
    expect(lying.detail).toContain('Metal command buffer failed');
  });

  // The FIRST error, not the newest: once `mlx_array_eval` has swallowed one
  // exception every array downstream is suspect, so later entries are cascade.
  it('reports the first native error, not the latest', () => {
    const later = { context: 'clear_cache', detail: 'downstream', observedAtMs: 2 };
    expect(presentTray(snapshot('lying', { nativeErrors: [NATIVE_ERROR, later] })).detail).toContain('array_eval');
  });

  it('still warns when the trace gave no detail', () => {
    const lying = presentTray(snapshot('lying'));
    expect(lying.indicator).toBe('warn');
    expect(lying.detail).not.toBeNull();
  });

  describe('menu items', () => {
    // `restarting` means the child is already dead and the supervisor is sitting
    // in its backoff. Stop is what cancels that; without it the only way out of
    // a crash loop is Quit.
    it('offers stop and restart exactly while there is (or is about to be) a child', () => {
      for (const state of ['starting', 'running', 'restarting', 'lying'] as const) {
        expect(presentTray(snapshot(state)), state).toMatchObject({
          canStart: false,
          canStop: true,
          canRestart: true,
        });
      }
    });

    it('offers start exactly when there is not one', () => {
      for (const state of ['stopped', 'failed'] as const) {
        expect(presentTray(snapshot(state)), state).toMatchObject({
          canStart: true,
          canStop: false,
          canRestart: false,
        });
      }
    });

    // Control Panel is where the crash reason, the trace file and the logs are. The
    // moments it is most needed are precisely the ones where inference is dead,
    // so it is never disabled.
    it('keeps Control Panel reachable in every state', () => {
      for (const state of ALL_STATES) {
        expect(presentTray(snapshot(state)).canOpenControlPanel, state).toBe(true);
      }
    });
  });

  describe('status text', () => {
    it('shows the url once the sidecar has announced one', () => {
      expect(presentTray(snapshot('running')).detail).toBe('http://127.0.0.1:51423');
    });

    // `/health` distinguishes `ok` from `degraded` (answering, but saturated)
    // and `loading` (a load holds the writer slot). All three are `running` to
    // the supervisor and only one is what the user assumes.
    it('names the health rung whenever it is not ok', () => {
      expect(presentTray(snapshot('running', { health: { status: 'ok' } })).statusLabel).toBe('Inference: running');
      expect(presentTray(snapshot('running', { health: { status: 'loading' } })).statusLabel).toContain('loading');
      expect(presentTray(snapshot('running', { health: { status: 'degraded' } })).statusLabel).toContain('degraded');
    });

    it('counts the crashes while restarting and after giving up', () => {
      const exit = {
        verdict: 'crash' as const,
        reason: 'exited 0 without being asked to',
        code: 0,
        signal: null,
        atMs: 1,
        stderrTail: [],
      };
      expect(presentTray(snapshot('restarting', { consecutiveCrashes: 2, lastExit: exit })).statusLabel).toContain('2');
      expect(presentTray(snapshot('failed', { consecutiveCrashes: 5, lastExit: exit })).statusLabel).toContain('5');
      expect(presentTray(snapshot('failed', { consecutiveCrashes: 5, lastExit: exit })).detail).toBe(exit.reason);
    });

    it('explains a stopped sidecar that stopped by itself', () => {
      const stopped = presentTray(
        snapshot('stopped', {
          lastExit: {
            verdict: 'clean',
            reason: 'stopped on request (code 0)',
            code: 0,
            signal: null,
            atMs: 1,
            stderrTail: [],
          },
        }),
      );
      expect(stopped.detail).toContain('stopped on request');
      // Nothing is wrong, so nothing takes up menubar width.
      expect(stopped.title).toBe('');
      expect(presentTray(snapshot('stopped')).detail).toBeNull();
    });
  });

  // A native menu row does not wrap and does not scroll: an unclipped `e.what()`
  // stretches the menu past the screen edge, taking Quit with it.
  describe('detail is menu-safe', () => {
    it('clips a long reason', () => {
      const detail = presentTray(
        snapshot('lying', { nativeErrors: [{ ...NATIVE_ERROR, detail: 'x'.repeat(400) }] }),
      ).detail;
      expect(detail).not.toBeNull();
      expect(String(detail).length).toBeLessThanOrEqual(72);
      expect(String(detail).endsWith('…')).toBe(true);
    });

    it('flattens newlines, which render as a box glyph in a native menu', () => {
      const detail = presentTray(
        snapshot('lying', { nativeErrors: [{ ...NATIVE_ERROR, detail: 'first\nsecond\n\tthird' }] }),
      ).detail;
      expect(detail).not.toContain('\n');
      expect(detail).toContain('first second third');
    });
  });
});

/**
 * Handing another client the endpoint.
 *
 * Every inference route is gated, so the URL on its own is not a capability — a
 * client given only the URL gets 401 for everything. The token reaches MAIN with
 * the ready handshake and leaves it only here, as a whole command, so the secret
 * is never drawn into a menu row where a screenshot would catch it.
 */
describe('connect commands', () => {
  it('is offered only while a child is actually serving', () => {
    // Not `starting`: the URL exists there but the token has not arrived yet, so
    // the command would be built from a null. Not `restarting`: the token is
    // cleared on every respawn, so it would already be stale.
    expect(presentTray(snapshot('running')).canCopyConnect).toBe(true);
    for (const state of ALL_STATES.filter((s) => s !== 'running')) {
      expect(presentTray(snapshot(state)).canCopyConnect).toBe(false);
    }
    expect(presentTray(snapshot('running', { url: null })).canCopyConnect).toBe(false);
  });

  it('builds a Claude Code command with the credential in the environment, never in the URL', () => {
    const command = claudeConnectCommand('http://127.0.0.1:51423', 'sekrit-token');
    expect(command).toContain("ANTHROPIC_BASE_URL='http://127.0.0.1:51423'");
    expect(command).toContain("ANTHROPIC_AUTH_TOKEN='sekrit-token'");
    expect(command).toMatch(/ claude$/u);
    // A token in a query string lands in referrers and proxy logs.
    expect(command).not.toContain('?');
    expect(command).not.toContain('sekrit-token@');
  });

  it('does not let an inherited ANTHROPIC_API_KEY override the bearer token', () => {
    const command = claudeConnectCommand('http://127.0.0.1:51423', 'sekrit-token');
    const script = command.replace(
      / claude$/u,
      ` sh -c 'printf "%s|%s" "\${ANTHROPIC_API_KEY+x}" "$ANTHROPIC_AUTH_TOKEN"'`,
    );
    const out = execFileSync('bash', ['-c', script], {
      env: { ...process.env, ANTHROPIC_API_KEY: 'sk-ant-the-users-own-key' },
    }).toString();

    // Claude Code sends both `x-api-key` and `authorization` when both
    // variables are present. The server intentionally reads `x-api-key` first,
    // so the key must be absent rather than merely set to an empty string.
    expect(out).toBe('|sekrit-token');
  });

  it('quotes a token that would otherwise break out of the command', () => {
    const evil = String.raw`a'; echo PWNED; b`;
    const command = claudeConnectCommand('http://127.0.0.1:1', evil);
    // Run the real command with `claude` swapped for something that just prints
    // the variable it was handed. If the quoting were wrong, bash would either
    // fail to parse or execute the injected `echo`.
    const script = command.replace(/ claude$/u, String.raw` sh -c 'printf %s "$ANTHROPIC_AUTH_TOKEN"'`);
    const out = execFileSync('bash', ['-c', script]).toString();

    // Exact round-trip IS the proof. Had the quoting let the payload out, bash
    // would have run the injected `echo` and stdout would carry its output on
    // its own line instead of the literal token. (Asserting the absence of
    // "PWNED" would be wrong here — the token itself contains that text.)
    expect(out).toBe(evil);
    expect(out.split('\n')).toHaveLength(1);
  });

  it('builds a Codex custom Responses provider at the server v1 root', () => {
    const command = codexConnectCommand('http://127.0.0.1:51423', 'sekrit-token');
    expect(command).toContain("MLX_NODE_API_KEY='sekrit-token' codex -m mlx-node");
    expect(command).toContain(`-c 'model_provider="mlx-node"'`);
    expect(command).toContain(`-c 'model_providers.mlx-node.name="MLX-Node"'`);
    expect(command).toContain(`-c 'model_providers.mlx-node.base_url="http://127.0.0.1:51423/v1"'`);
    expect(command).toContain(`-c 'model_providers.mlx-node.env_key="MLX_NODE_API_KEY"'`);
    expect(command).toContain(`-c 'model_providers.mlx-node.wire_api="responses"'`);
    expect(command).not.toContain('ANTHROPIC_');
    expect(command).not.toContain('claude');
  });

  it('round-trips the Codex credential and config as separate shell arguments', () => {
    const evil = String.raw`a'; echo PWNED; b`;
    const command = codexConnectCommand('http://127.0.0.1:51423', evil);
    const script = command.replace(
      / codex /u,
      String.raw` sh -c 'printf "%s\n" "$MLX_NODE_API_KEY"; printf "%s\n" "$@"' -- `,
    );
    const lines = execFileSync('bash', ['-c', script]).toString().trimEnd().split('\n');

    expect(lines).toEqual([
      evil,
      '-m',
      'mlx-node',
      '-c',
      'model_provider="mlx-node"',
      '-c',
      'model_providers.mlx-node.name="MLX-Node"',
      '-c',
      'model_providers.mlx-node.base_url="http://127.0.0.1:51423/v1"',
      '-c',
      'model_providers.mlx-node.env_key="MLX_NODE_API_KEY"',
      '-c',
      'model_providers.mlx-node.wire_api="responses"',
    ]);
  });
});
