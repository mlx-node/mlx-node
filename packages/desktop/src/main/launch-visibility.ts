/**
 * Turn macOS launch/activation events into a visible Control Panel.
 *
 * LaunchServices normally re-activates an already-running accessory app instead
 * of starting a second process. The target is installed only after async
 * bootstrap, though, so an activation that arrives first has to be remembered
 * rather than dropped. A process launch is itself an open request: the app may
 * rest in the tray after the user closes the window, but explicitly opening it
 * must never produce an invisible process.
 */

export interface LaunchSurface {
  show(): void;
}

export interface LaunchVisibility {
  /** Record a Finder/Dock/second-instance request and surface it when possible. */
  activate(): void;
  /** Install the Control Panel created by bootstrap and satisfy any pending launch. */
  ready(surface: LaunchSurface): void;
  /** Stop surfacing windows; a later activation now means "open again after quit". */
  beginShutdown(): void;
  /** Consume the relaunch request, once, immediately before the process exits. */
  takeRelaunchRequest(): boolean;
}

export function createLaunchVisibility(): LaunchVisibility {
  let surface: LaunchSurface | null = null;
  // Starting the process means the user opened the app. Do not infer intent from
  // the presence of a settings file: another build may already have created it.
  let pending = true;
  let shuttingDown = false;
  let relaunchRequested = false;

  return {
    activate(): void {
      if (shuttingDown) {
        // The new process loses the singleton race and exits. Carry its intent
        // across our graceful shutdown instead of leaving no process behind.
        relaunchRequested = true;
        return;
      }
      if (surface === null) {
        pending = true;
        return;
      }
      surface.show();
    },
    ready(next): void {
      surface = next;
      if (!pending) return;
      pending = false;
      next.show();
    },
    beginShutdown(): void {
      shuttingDown = true;
    },
    takeRelaunchRequest(): boolean {
      const requested = relaunchRequested;
      relaunchRequested = false;
      return requested;
    },
  };
}
