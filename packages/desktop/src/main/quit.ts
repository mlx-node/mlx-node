/** Drain app resources before either a normal exit or Squirrel's install/relaunch. */
export function createQuitHandler(options: {
  beginShutdown(): void;
  shutdown(): Promise<void>;
  deadlineMs: number;
  installUpdate(relaunchRequested: boolean): boolean;
  shouldRelaunch(): boolean;
  relaunch(): void;
  quit(): void;
  report(error: unknown): void;
}): (event: { preventDefault(): void }) => void {
  let started = false;
  let completed = false;

  async function finish(): Promise<void> {
    let timer: NodeJS.Timeout | undefined;
    try {
      await Promise.race([
        options.shutdown(),
        new Promise<void>((resolve) => {
          timer = setTimeout(() => {
            options.report(new Error(`shutdown did not finish within ${options.deadlineMs}ms; exiting anyway`));
            resolve();
          }, options.deadlineMs);
          timer.unref();
        }),
      ]);
    } catch (error) {
      options.report(error);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      completed = true;
      // Squirrel owns the relaunch after installation. A separate app.relaunch()
      // races the old executable against replacement of its bundle.
      const relaunchRequested = options.shouldRelaunch();
      if (!options.installUpdate(relaunchRequested)) {
        if (relaunchRequested) options.relaunch();
        options.quit();
      }
    }
  }

  return (event): void => {
    // Repeated Cmd+Q / tray quit requests must not bypass an in-flight drain.
    // Only the final quit initiated below (or by Squirrel) can proceed.
    if (completed) return;
    event.preventDefault();
    if (started) return;
    started = true;
    options.beginShutdown();
    void finish();
  };
}
