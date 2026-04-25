import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { makeChildKillEscalation } from '../../../packages/cli/src/commands/launch-claude/index.js';

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
