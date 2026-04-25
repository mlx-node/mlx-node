import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { computeExitCode, makeChildKillEscalation } from '../../../packages/cli/src/commands/launch-claude/index.js';

describe('makeChildKillEscalation', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('does NOT escalate to SIGKILL if the child exits before the timeout', () => {
    const kill = vi.fn();
    let exited = false;
    const forward = makeChildKillEscalation({
      child: { kill },
      isShuttingDown: () => false,
      hasChildExited: () => exited,
      escalateAfterMs: 5000,
    });

    forward('SIGINT');
    // First call: forwarded signal.
    expect(kill).toHaveBeenCalledTimes(1);
    expect(kill).toHaveBeenLastCalledWith('SIGINT');

    // Child terminates before the 5s window elapses.
    exited = true;
    vi.advanceTimersByTime(10_000);

    // No SIGKILL escalation because hasChildExited() returned true at the deadline.
    expect(kill).toHaveBeenCalledTimes(1);
    expect(kill).not.toHaveBeenCalledWith('SIGKILL');
  });

  it('escalates to SIGKILL if the child has not exited by the timeout', () => {
    const kill = vi.fn();
    const forward = makeChildKillEscalation({
      child: { kill },
      isShuttingDown: () => false,
      hasChildExited: () => false, // never exits
      escalateAfterMs: 5000,
    });

    forward('SIGTERM');
    expect(kill).toHaveBeenCalledTimes(1);
    expect(kill).toHaveBeenLastCalledWith('SIGTERM');

    vi.advanceTimersByTime(5000);

    // Now the timer has fired and the child still hasn't exited → SIGKILL.
    expect(kill).toHaveBeenCalledTimes(2);
    expect(kill).toHaveBeenLastCalledWith('SIGKILL');
  });

  it('skips signal forwarding entirely while shutting down', () => {
    const kill = vi.fn();
    const forward = makeChildKillEscalation({
      child: { kill },
      isShuttingDown: () => true,
      hasChildExited: () => false,
      escalateAfterMs: 5000,
    });

    forward('SIGINT');
    vi.advanceTimersByTime(10_000);

    expect(kill).not.toHaveBeenCalled();
  });

  it('does not escalate even if the timer fires after the child exited', () => {
    // Regression: previously the escalation check used `child.killed`, which
    // flips to true the moment kill() *sends* the signal, so the !child.killed
    // guard was always false and SIGKILL never fired. Here we model a child
    // that exits "between" forward('SIGINT') and the timer firing.
    const kill = vi.fn();
    let exited = false;
    const forward = makeChildKillEscalation({
      child: { kill },
      isShuttingDown: () => false,
      hasChildExited: () => exited,
      escalateAfterMs: 5000,
    });

    forward('SIGINT');
    // Halfway through the window, the child finally exits.
    vi.advanceTimersByTime(2500);
    exited = true;
    vi.advanceTimersByTime(2500);

    expect(kill).toHaveBeenCalledTimes(1);
    expect(kill).not.toHaveBeenCalledWith('SIGKILL');
  });
});

describe('computeExitCode', () => {
  // Regression: previously `child.on('exit', (code) => shutdown(code ?? 0))`
  // ignored the `signal` arg, so a SIGINT/SIGTERM/SIGKILL'd `claude` reported
  // success (exit 0). CI jobs reading exit code as "did the run pass" got a
  // false green. POSIX convention: signal-killed processes exit with
  // `128 + signal_number`; Node exposes the numbers via `os.constants.signals`.

  it('passes through a normal numeric exit code', () => {
    expect(computeExitCode(0, null)).toBe(0);
    expect(computeExitCode(1, null)).toBe(1);
    expect(computeExitCode(2, null)).toBe(2);
    expect(computeExitCode(127, null)).toBe(127);
  });

  it('maps SIGTERM (signal 15) to 143', () => {
    expect(computeExitCode(null, 'SIGTERM')).toBe(143);
  });

  it('maps SIGINT (signal 2) to 130', () => {
    expect(computeExitCode(null, 'SIGINT')).toBe(130);
  });

  it('maps SIGKILL (signal 9) to 137', () => {
    expect(computeExitCode(null, 'SIGKILL')).toBe(137);
  });

  it('returns 1 for the (null, null) defensive case', () => {
    expect(computeExitCode(null, null)).toBe(1);
  });

  it('returns 1 for a signal name not present in os.constants.signals', () => {
    // Cast to NodeJS.Signals to satisfy TS; in practice the runtime would only
    // ever see standard signal names, but the helper must not blow up if a
    // platform-specific or unknown name slips through.
    expect(computeExitCode(null, 'SIGUNKNOWN' as NodeJS.Signals)).toBe(1);
  });

  it('prefers the numeric exit code over the signal when both are present', () => {
    // Node's docs guarantee that exactly one of (code, signal) is non-null,
    // but defensively the helper picks `code` if both happen to arrive.
    expect(computeExitCode(0, 'SIGTERM' as NodeJS.Signals)).toBe(0);
    expect(computeExitCode(7, 'SIGINT' as NodeJS.Signals)).toBe(7);
  });
});
