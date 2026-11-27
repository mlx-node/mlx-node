import { useId, useState, type CSSProperties, type ReactElement } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { formatCount } from '../../../lib/format';
import type { TranscriptEntry } from '../../../lib/types';
import { relativeTo, visibleText } from '../../../lib/work-tools';
import { CopyButton } from '../../arc/copy-button/copy-button';
import { Disclosure } from '../disclosure/disclosure';
import { PathText } from '../path-text/path-text';
import styles from './diff-view.module.css';

export type ToolEdit = NonNullable<Extract<TranscriptEntry, { kind: 'tool' }>['edits']>[number];
type Hunk = ToolEdit['hunks'][number];

/** `pending`: asked for, not run yet; `applied`: the call completed; `not-applied`: it failed or was refused. */
export type DiffState = 'pending' | 'applied' | 'not-applied';

/** The diff's state from the tool call's status. */
export function diffStateOf(status: string): DiffState {
  return status === 'completed' ? 'applied' : status === 'failed' ? 'not-applied' : 'pending';
}

export interface DiffViewProps {
  /** As the user reads it (relative to the Work folder when inside it). */
  path: string;
  hunks: readonly Hunk[];
  added: number;
  removed: number;
  state: DiffState;
  /** A new file: one number column. */
  create?: boolean;
  /** Changed lines the host left out. */
  cut?: number;
  endsOnly?: boolean;
  /** Rows shown before "Show n more lines". Default 14. */
  maxLines?: number;
  /** Name the file above the rows: one diff of several. */
  showPath?: boolean;
}

type LineRow = { kind: 'line'; mark: ' ' | '-' | '+'; old: number | null; new: number | null; text: string };
type Row = LineRow | { kind: 'gap'; n: number };

/** The rows of every hunk; between two hunks with numbers, the unchanged lines they skip as one row. */
export function diffRows(hunks: readonly Hunk[]): Row[] {
  const rows: Row[] = [];
  let nextNew: number | null = null;
  for (const hunk of hunks) {
    const start = hunk.start;
    if (start !== null && nextNew !== null && start[1] > nextNew) rows.push({ kind: 'gap', n: start[1] - nextNew });
    let o = start?.[0] ?? null;
    let n = start?.[1] ?? null;
    for (const line of hunk.body.split('\n')) {
      const head = line.charAt(0);
      const mark = head === '-' || head === '+' ? head : ' ';
      const row: LineRow = { kind: 'line', mark, old: null, new: null, text: line.slice(1) };
      if (mark !== '+' && o !== null) row.old = o++;
      if (mark !== '-' && n !== null) row.new = n++;
      rows.push(row);
    }
    nextNew = n;
  }
  return rows;
}

/** The hunks as unified-diff text, for the copy button. */
export function unifiedText(path: string, hunks: readonly Hunk[], create: boolean): string {
  const parts = [`--- ${create ? '/dev/null' : `a/${path}`}`, `+++ b/${path}`];
  for (const hunk of hunks) {
    const lines = hunk.body.split('\n');
    if (hunk.start === null) parts.push('@@');
    else {
      const olds = lines.filter((line) => !line.startsWith('+')).length;
      const news = lines.filter((line) => !line.startsWith('-')).length;
      const from = olds === 0 ? Math.max(0, hunk.start[0] - 1) : hunk.start[0];
      const to = news === 0 ? Math.max(0, hunk.start[1] - 1) : hunk.start[1];
      parts.push(`@@ -${from},${olds} +${to},${news} @@`);
    }
    parts.push(...lines);
  }
  return `${parts.join('\n')}\n`;
}

function countText(one: string, many: string, n: number): string {
  return n === 1 ? one : fill(many, { n: formatCount(n) });
}

/** One file's change as a unified diff: two number columns (one for a new file or a narrow column), tinted rows. */
export function DiffView(props: DiffViewProps): ReactElement {
  const maxLines = props.maxLines ?? 14;
  const [expanded, setExpanded] = useState(false);
  const scrollerId = useId();
  const rows = diffRows(props.hunks);
  const clamps = rows.length > maxLines;
  const shown = clamps && !expanded ? rows.slice(0, maxLines) : rows;
  const counts = fill(CHAT.diff.countsSr, { added: formatCount(props.added), removed: formatCount(props.removed) });
  const name = `${fill(CHAT.diff.label, { path: props.path })}: ${counts}`;
  let widest = 1;
  for (const row of rows) if (row.kind === 'line') widest = Math.max(widest, row.old ?? 0, row.new ?? 0);
  const header =
    props.state === 'pending' ? CHAT.diff.pending : props.state === 'not-applied' ? CHAT.diff.notApplied : null;
  return (
    <figure
      className={styles.root}
      aria-label={name}
      data-state={props.state}
      data-create={props.create === true ? '' : undefined}
      style={{ '--digits': String(widest).length } as CSSProperties}
    >
      {header !== null || props.showPath === true ? (
        <figcaption className={styles.caption}>
          {props.showPath === true ? <PathText path={props.path} keepEnd className={styles.path} /> : null}
          {header !== null ? <span className={styles.state}>{header}</span> : null}
        </figcaption>
      ) : null}
      {rows.length > 0 ? (
        <div
          id={scrollerId}
          className={styles.scroller}
          data-expanded={expanded ? '' : undefined}
          {...(expanded ? { role: 'region', 'aria-label': name, tabIndex: 0 } : {})}
        >
          {shown.map((row, i) =>
            row.kind === 'gap' ? (
              <div key={i} className={styles.gap}>
                <span className={styles.gapMark} aria-hidden="true">
                  ⋯
                </span>
                {countText(CHAT.diff.gapOne, CHAT.diff.gap, row.n)}
              </div>
            ) : (
              <div
                key={i}
                className={styles.row}
                data-mark={row.mark === '-' ? 'removed' : row.mark === '+' ? 'added' : undefined}
              >
                <span className={styles.old} aria-hidden="true">
                  {row.old ?? ''}
                </span>
                <span className={styles.new} aria-hidden="true">
                  {row.new ?? ''}
                </span>
                <span className={styles.mark} aria-hidden="true">
                  {row.mark === '-' ? '−' : row.mark}
                </span>
                <span className={styles.text}>
                  {row.mark === ' ' ? null : (
                    <span
                      className={styles.srOnly}
                    >{`${row.mark === '-' ? CHAT.diff.removed : CHAT.diff.added} `}</span>
                  )}
                  {visibleText(row.text)}
                </span>
              </div>
            ),
          )}
        </div>
      ) : null}
      {clamps ? (
        <button
          type="button"
          className={styles.more}
          aria-expanded={expanded}
          aria-controls={scrollerId}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? CHAT.diff.less : countText(CHAT.diff.moreOne, CHAT.diff.more, rows.length - maxLines)}
        </button>
      ) : null}
      {props.cut !== undefined && props.cut > 0 ? (
        <p className={styles.note}>{countText(CHAT.diff.cutOne, CHAT.diff.cut, props.cut)}</p>
      ) : null}
      {props.endsOnly === true ? <p className={styles.note}>{CHAT.diff.endsOnly}</p> : null}
      {props.hunks.length > 0 ? (
        <CopyButton
          value={unifiedText(props.path, props.hunks, props.create === true)}
          label={CHAT.diff.copy}
          iconOnly
          variant="plain"
          className={styles.copy}
        />
      ) : null}
    </figure>
  );
}

/** Diffs shown before the rest of a call's files fold behind "n more files". */
const FILES_SHOWN = 3;

/** Every file of one call, in its order: the first three open, the rest behind "n more files". */
export function EditDiffs(props: {
  edits: readonly ToolEdit[];
  state: DiffState;
  /** Work: the conversation's folder; paths inside it read relative. */
  folder: string | null;
  maxLines?: number;
}): ReactElement {
  const several = props.edits.length > 1;
  const view = (edit: ToolEdit, i: number): ReactElement => (
    <DiffView
      key={`${i}:${edit.path}`}
      path={relativeTo(edit.path, props.folder)}
      hunks={edit.hunks}
      added={edit.added}
      removed={edit.removed}
      state={props.state}
      create={edit.create === true}
      cut={edit.cut}
      endsOnly={edit.endsOnly === true}
      maxLines={props.maxLines}
      showPath={several}
    />
  );
  const rest = props.edits.slice(FILES_SHOWN);
  return (
    <div className={styles.files}>
      {props.edits.slice(0, FILES_SHOWN).map(view)}
      {rest.length > 0 ? (
        <Disclosure title={countText(CHAT.diff.moreFilesOne, CHAT.diff.moreFiles, rest.length)}>
          <div className={styles.files}>{rest.map((edit, i) => view(edit, i + FILES_SHOWN))}</div>
        </Disclosure>
      ) : null}
    </div>
  );
}

/** Lines added and removed over every file of a call. */
export function editCounts(edits: readonly ToolEdit[]): { added: number; removed: number } {
  let added = 0;
  let removed = 0;
  for (const edit of edits) {
    added += edit.added;
    removed += edit.removed;
  }
  return { added, removed };
}
