/**
 * The menubar is the only place most users ever look, so the mapping from
 * supervisor state to what it shows is the whole user-facing contract of the
 * supervisor.
 *
 * The test this file exists for is `lying` vs `running`: a swallowed C++
 * exception leaves `/health` answering `ok` and the output wrong, and a tray
 * that renders that as "running" tells the user the one thing that is not true.
 */

import { describe, expect, it } from 'vite-plus/test';

import type { SupervisorSnapshot, SupervisorState } from '../src/main/supervisor/types.js';
import { presentTray } from '../src/main/tray-view.js';

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

    // Admin is where the crash reason, the trace file and the logs are. The
    // moments it is most needed are precisely the ones where inference is dead,
    // so it is never disabled.
    it('keeps Admin reachable in every state', () => {
      for (const state of ALL_STATES) {
        expect(presentTray(snapshot(state)).canOpenAdmin, state).toBe(true);
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
