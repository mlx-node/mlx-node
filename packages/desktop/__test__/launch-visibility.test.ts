import { describe, expect, it, vi } from 'vite-plus/test';

import { createLaunchVisibility } from '../src/main/launch-visibility.js';

describe('launch visibility', () => {
  it('opens the Control Panel when a new app process becomes ready', () => {
    const show = vi.fn();
    createLaunchVisibility().ready({ show });
    expect(show).toHaveBeenCalledOnce();
  });

  it('does not lose an activation that arrives before async bootstrap', () => {
    const visibility = createLaunchVisibility();
    const show = vi.fn();
    visibility.activate();
    visibility.activate();
    visibility.ready({ show });
    expect(show).toHaveBeenCalledOnce();
  });

  it('surfaces every activation after bootstrap', () => {
    const visibility = createLaunchVisibility();
    const show = vi.fn();
    visibility.ready({ show });
    show.mockClear();

    visibility.activate();
    visibility.activate();
    expect(show).toHaveBeenCalledTimes(2);
  });

  it('carries an open request across graceful shutdown', () => {
    const visibility = createLaunchVisibility();
    const show = vi.fn();
    visibility.ready({ show });
    show.mockClear();

    visibility.beginShutdown();
    visibility.activate();
    expect(show).not.toHaveBeenCalled();
    expect(visibility.takeRelaunchRequest()).toBe(true);
    expect(visibility.takeRelaunchRequest()).toBe(false);
  });
});
