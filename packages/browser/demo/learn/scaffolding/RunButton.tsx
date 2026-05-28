import { Loader2Icon, PlayIcon } from 'lucide-react';
import * as React from 'react';

import { Button } from '../../components/ui/button';
import { cn } from '../../lib/utils';

/**
 * Shared "Run" button for every chapter's Try-it-now panel.
 *
 * Why this exists: a plain `<Button>` with `disabled={running}` and
 * `{running ? 'Running…' : 'Run'}` was too subtle — first-time users
 * clicked Run and assumed nothing happened, especially when a generation
 * finishes in ~200ms.
 *
 * What it adds:
 *  - A spinning Loader icon while `running` is true.
 *  - A short press-flash (scale + ring) on every click, so the user
 *    *feels* the click land even before the inspector reports back.
 *  - A play-icon prefix when idle so the affordance is obvious.
 *
 * The component owns its own press-pulse state via a generation counter
 * keyed on the click event, so back-to-back clicks each re-trigger the
 * animation.
 */
export function RunButton({
  onClick,
  running,
  label = 'Run',
  runningLabel = 'Running…',
  className,
  disabled,
}: {
  onClick: () => void;
  running: boolean;
  label?: string;
  runningLabel?: string;
  className?: string;
  /** Extra disable signal — e.g. model not ready. `running` already disables. */
  disabled?: boolean;
}) {
  const [pressKey, setPressKey] = React.useState(0);

  function handleClick() {
    setPressKey((k) => k + 1);
    onClick();
  }

  return (
    <Button
      onClick={handleClick}
      disabled={running || disabled}
      className={cn(
        // The pulse animation is keyed on pressKey so back-to-back clicks
        // each re-trigger it. The `key` swap remounts the element so the
        // CSS animation restarts.
        'run-button-press relative gap-1.5 transition-transform active:scale-95',
        className,
      )}
      key={pressKey}
    >
      {running ? (
        <Loader2Icon className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
      ) : (
        <PlayIcon className="h-3.5 w-3.5" aria-hidden="true" />
      )}
      <span>{running ? runningLabel : label}</span>
    </Button>
  );
}
