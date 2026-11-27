/**
 * The dock strip above the goal strip while the activity rail is a drawer: the live runs the host holds, one line
 * each for up to two, one summary line for three or four (the agent runs four at most), and the request one of their
 * agents waits on.
 */

import { type Activity, type AskRow, type EntryAnchor, type RunRailRow, runStepWords } from './activity';
import { CHAT, fill } from './chat-strings';
import { formatCount } from './format';

/** A strip shows a line per run up to this many; more read as one line. */
export const STRIP_RUN_LINES = 2;

export type StripLine =
  | {
      kind: 'run';
      id: string;
      name: string;
      /** `Step 2 of 3: Verify`; null with no steps. The strip drops it second when narrow. */
      step: string | null;
      /** `1 running · 5 done`. The strip drops them first when narrow. */
      counts: string[];
      /** Its agents that run: its Stop confirm counts them. */
      running: number;
      startTs: number;
      stopping: boolean;
      stoppable: boolean;
      anchor: EntryAnchor;
    }
  | { kind: 'summary'; text: string }
  | {
      kind: 'ask';
      /** The request's id: Review brings its card to the front. */
      id: string;
      name: string;
      text: string;
    };

function countsOf(row: RunRailRow): string[] {
  const words = CHAT.run.counts;
  const out: string[] = [];
  if (row.agentsRunning > 0) out.push(fill(words.running, { n: formatCount(row.agentsRunning) }));
  if (row.done > 0) out.push(fill(words.done, { n: formatCount(row.done) }));
  return out;
}

/** The request of an agent of `run` that waits first; null with none. */
function askOf(needs: readonly AskRow[], run: RunRailRow): StripLine | null {
  const ask = needs.find((row) => row.from.who === 'workflow' && row.from.run === run.title);
  if (ask === undefined || ask.from.who !== 'workflow') return null;
  const words = CHAT.run.asks[ask.access ?? 'other'];
  return { kind: 'ask', id: ask.id, name: run.title, text: fill(words, { label: ask.from.label }) };
}

/** The strip's lines; none while the host holds no live run. */
export function runStripOf(activity: Pick<Activity, 'running' | 'needs'>): StripLine[] {
  const runs = activity.running.filter((row): row is RunRailRow => row.kind === 'workflow' && row.live);
  if (runs.length === 0) return [];
  const asks = runs.map((run) => askOf(activity.needs, run)).filter((line): line is StripLine => line !== null);
  if (runs.length > STRIP_RUN_LINES) {
    let agents = 0;
    for (const run of runs) agents += run.agentsRunning;
    const s = CHAT.run.strip;
    const text = fill(s.summary, {
      runs: formatCount(runs.length),
      agents: agents === 1 ? s.agentsOne : fill(s.agents, { n: formatCount(agents) }),
    });
    return [{ kind: 'summary', text }, ...asks.slice(0, 1)];
  }
  const out: StripLine[] = [];
  for (const run of runs) {
    out.push({
      kind: 'run',
      id: run.id,
      name: run.title,
      step: runStepWords(run),
      counts: countsOf(run),
      running: run.agentsRunning,
      startTs: run.startTs,
      stopping: run.stopping,
      stoppable: run.stoppable,
      anchor: run.anchor,
    });
    const ask = askOf(activity.needs, run);
    if (ask !== null) out.push(ask);
  }
  return out;
}
