import { FoldVertical } from 'lucide-react';
import type { ReactElement } from 'react';

import { compactionText } from '../lib/chat-turns';
import { formatClock } from '../lib/format';
import type { TranscriptEntry } from '../lib/types';
import { RuleLabel } from './arc-x/event-row/event-row';

/** Where the agent compacted the conversation: a quiet rule with the counts and the time. */
export function CompactionMark({ entry }: { entry: Extract<TranscriptEntry, { kind: 'compaction' }> }): ReactElement {
  return (
    <RuleLabel tone={entry.outcome === 'failed' ? 'danger' : 'neutral'}>
      <FoldVertical size={12} strokeWidth={1.75} aria-hidden="true" />
      <span data-compaction-mark={entry.outcome}>
        {entry.ts > 0 ? `${compactionText(entry)} · ${formatClock(entry.ts)}` : compactionText(entry)}
      </span>
    </RuleLabel>
  );
}
