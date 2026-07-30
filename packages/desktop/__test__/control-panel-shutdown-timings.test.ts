import { describe, expect, it } from 'vite-plus/test';

import {
  CONTROL_PANEL_BROKER_KILL_GRACE_MS,
  CONTROL_PANEL_PROCESS_EXIT_CAP_MS,
  CONTROL_PANEL_WORKER_SHUTDOWN_STEP_MS,
  DESKTOP_QUIT_DEADLINE_MS,
} from '../src/control-panel/shutdown-timings.js';

describe('CONTROL PANEL shutdown budget nesting', () => {
  it('budgets both sequential worker phases before the process cap', () => {
    const workerPhasesMs = CONTROL_PANEL_WORKER_SHUTDOWN_STEP_MS * 2;

    expect(CONTROL_PANEL_PROCESS_EXIT_CAP_MS - workerPhasesMs).toBeGreaterThanOrEqual(2_000);
  });

  it('leaves cleanup margin between every enclosing deadline', () => {
    expect(CONTROL_PANEL_BROKER_KILL_GRACE_MS - CONTROL_PANEL_PROCESS_EXIT_CAP_MS).toBeGreaterThanOrEqual(2_000);
    expect(DESKTOP_QUIT_DEADLINE_MS - CONTROL_PANEL_BROKER_KILL_GRACE_MS).toBeGreaterThanOrEqual(2_000);
  });
});
