import type { ReactElement } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import type { TaskEvent, TaskLive } from '../../../lib/chat-types';
import { formatClock } from '../../../lib/format';
import { visibleText } from '../../../lib/work-tools';
import { CodePanel } from '../code-panel/code-panel';
import { Disclosure } from '../disclosure/disclosure';
import styles from './task-events.module.css';

/**
 * A monitor's kept events as one text, newest first: each event's clock with seconds, then its output, the lines after
 * its first under the output column. Hidden characters made visible (a line's closing carriage return dropped); never
 * Markdown or HTML.
 */
export function eventsText(recent: readonly TaskEvent[]): string {
  const blocks: string[] = [];
  for (let i = recent.length - 1; i >= 0; i--) {
    const event = recent[i]!;
    const clock = `${formatClock(event.at, true)}  `;
    const pad = ' '.repeat(clock.length);
    const lines = event.text.split('\n').map((line) => visibleText(line.endsWith('\r') ? line.slice(0, -1) : line));
    blocks.push(lines.map((line, n) => (n === 0 ? clock : pad) + line).join('\n'));
  }
  return blocks.join('\n');
}

/**
 * The events of a monitor the running agent holds, under its thread row: closed, "Events (12)", then the last five as
 * plain text. Not a live region. Nothing while it has none.
 */
export function TaskEvents({ task }: { task: TaskLive }): ReactElement | null {
  if (task.events === 0 || task.recent.length === 0) return null;
  const title = fill(CHAT.task.eventsTitle, { n: task.events });
  return (
    <Disclosure title={title}>
      <div className={styles.root} data-task-events="">
        <CodePanel code={eventsText(task.recent)} label={title} wrap />
        {task.events > task.recent.length ? <p className={styles.kept}>{CHAT.task.eventsKept}</p> : null}
      </div>
    </Disclosure>
  );
}
