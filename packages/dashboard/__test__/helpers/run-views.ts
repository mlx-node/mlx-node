import type { RunView } from '../../src/chat/types.js';

/** A host run view: by default the fake agent's live `deep-research` run. */
export function runView(runId: string, over: Partial<Omit<RunView, 'runId'>> = {}): RunView {
  return {
    runId,
    name: 'deep-research',
    status: 'active',
    live: true,
    stopping: false,
    by: null,
    waiting: false,
    wakeDue: false,
    ...over,
  };
}
