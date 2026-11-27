import { NotebookPen } from 'lucide-react';
import type { ReactElement } from 'react';

import { CHAT } from '../lib/chat-strings';
import type { TranscriptEntry } from '../lib/types';
import { RuleLabel } from './arc-x/event-row/event-row';

export type PlanMode = Extract<TranscriptEntry, { kind: 'plan-mode' }>['mode'];

const LINE: Record<PlanMode, string> = { plan: CHAT.planMode.on, default: CHAT.planMode.off, ask: CHAT.planMode.ask };

const STATUS: Record<PlanMode, string> = {
  plan: CHAT.planMode.status.on,
  default: CHAT.planMode.status.off,
  ask: CHAT.planMode.status.ask,
};

/** What a screen reader hears when a mode line arrives. */
export function planModeStatus(mode: PlanMode): string {
  return STATUS[mode];
}

/** The thin line where the agent's mode moved: "Plan mode on", "Plan mode off" or "Ask mode on". */
export function PlanModeLine({ mode }: { mode: PlanMode }): ReactElement {
  return (
    <RuleLabel>
      <NotebookPen size={12} strokeWidth={1.75} aria-hidden="true" />
      {LINE[mode]}
    </RuleLabel>
  );
}
