import { useEffect, useState, type ReactElement } from 'react';

import { formatElapsed } from '../lib/chat-turns';
import { SwapText } from './arc-x/swap-text/swap-text';

/**
 * The time since `since` (epoch ms), `m:ss`, on a 1 s clock that reads no data. Hidden from assistive tech like the
 * load card's clock: the wait line's text says what waits.
 */
export function WaitClock({ since }: { since: number }): ReactElement {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [since]);
  return (
    <span className="shrink-0 tabular-nums text-caption" aria-hidden="true" data-wait-clock="">
      <SwapText text={formatElapsed(now - since)} granularity="char" />
    </span>
  );
}
