/** Finish quitting, optionally letting the updater own installation and relaunch. */
export type CompleteQuit = (install?: (relaunchRequested: boolean) => void) => void;

/** Drain app resources before either a normal exit or Squirrel's install/relaunch. */
export function createQuitHandler(options: {
  beginShutdown(): void;
  shutdown(): Promise<void>;
  deadlineMs: number;
  installUpdate(completeQuit: CompleteQuit): boolean;
  shouldRelaunch(): boolean;
  relaunch(): void;
  quit(): void;
  report(error: unknown): void;
}): (event: { preventDefault(): void }) => void {
  let started = false;
  let completed = false;
  let relaunchRequested = false;

  const completeQuit: CompleteQuit = (install) => {
    // This runs after native staging, not just after the child/settings drain.
    // Retain consumed intent if installation later fails and falls back to quit.
    relaunchRequested = options.shouldRelaunch() || relaunchRequested;
    completed = true;
    if (install) {
      install(relaunchRequested);
    } else {
      if (relaunchRequested) options.relaunch();
      options.quit();
    }
  };

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
      // Keep the completion decision live while the updater finishes staging.
      if (!options.installUpdate(completeQuit)) completeQuit();
    }
  }

  return (event): void => {
    // Repeated Cmd+Q / tray quit requests must not bypass either the resource
    // drain or native staging. The updater releases this gate before final quit.
    if (completed) return;
    event.preventDefault();
    if (started) return;
    started = true;
    options.beginShutdown();
    void finish();
  };
}
