import { GitCompareArrows, RotateCcw, Trash2 } from 'lucide-react';
import { type ReactElement, useMemo, useState } from 'react';

import { Rule } from '../../components/arc-x/rule/rule';
import { Badge, type BadgeTone } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { Drawer, DrawerContent } from '../../components/arc/drawer/drawer';
import { EmptyState } from '../../components/arc/empty-state/empty-state';
import { formatCount, formatDuration, formatPercent, formatRelativeTime } from '../../lib/format';
import { formatAnswerSummary, isPlainObject } from '../../lib/system-one';
import type { SystemOneHistoryEntry } from '../../lib/types';
import styles from './system-one.module.css';

export interface HistoryDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  history: readonly SystemOneHistoryEntry[];
  clearing: boolean;
  onClear: () => void;
  onRestore: (entry: SystemOneHistoryEntry) => void;
}

function statusTone(status: SystemOneHistoryEntry['status']): BadgeTone {
  if (status === 'done') return 'success';
  if (status === 'error') return 'danger';
  return 'neutral';
}

function answerConfidence(answer: unknown): number | null {
  if (!isPlainObject(answer)) return null;
  if (typeof answer.confidence === 'number') return answer.confidence;
  if (answer.type === 'noul' && typeof answer.noul === 'number') return Math.max(answer.noul, 1 - answer.noul);
  return null;
}

function answerPNull(answer: unknown): number | null {
  if (!isPlainObject(answer)) return null;
  return typeof answer.p_null === 'number' ? answer.p_null : null;
}

/** Run history & side-by-side A/B comparison drawer. */
export function HistoryDrawer({
  open,
  onOpenChange,
  history,
  clearing,
  onClear,
  onRestore,
}: HistoryDrawerProps): ReactElement {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  const toggleSelect = (id: string): void => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((item) => item !== id);
      if (prev.length >= 2) return [prev[1]!, id];
      return [...prev, id];
    });
  };

  const compared = useMemo(() => {
    if (selectedIds.length !== 2) return null;
    const a = history.find((item) => item.id === selectedIds[0]);
    const b = history.find((item) => item.id === selectedIds[1]);
    if (a === undefined || b === undefined) return null;
    return [a, b] as const;
  }, [history, selectedIds]);

  const diffRows = useMemo(() => {
    if (compared === null) return [];
    const [a, b] = compared;
    const ansA = isPlainObject(a.result) && isPlainObject(a.result.answers) ? a.result.answers : {};
    const ansB = isPlainObject(b.result) && isPlainObject(b.result.answers) ? b.result.answers : {};
    const fieldNames = [...new Set([...Object.keys(ansA), ...Object.keys(ansB)])];
    return fieldNames.map((field) => {
      const valA = formatAnswerSummary(ansA[field]);
      const valB = formatAnswerSummary(ansB[field]);
      const confA = answerConfidence(ansA[field]);
      const confB = answerConfidence(ansB[field]);
      const pNullA = answerPNull(ansA[field]);
      const pNullB = answerPNull(ansB[field]);
      return {
        field,
        valA,
        valB,
        changed: valA !== valB,
        confA,
        confB,
        confDelta: confA !== null && confB !== null ? confB - confA : null,
        pNullA,
        pNullB,
      };
    });
  }, [compared]);

  return (
    <Drawer open={open} onOpenChange={onOpenChange}>
      <DrawerContent
        side="right"
        title="Run history & compare"
        description="Restore any previous run or select 2 runs to diff field values, confidence, and latency."
      >
        <div className="grid min-w-0 gap-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="text-caption text-secondary">
              {selectedIds.length === 2
                ? 'Comparing 2 selected runs'
                : `${String(history.length)} ${history.length === 1 ? 'run' : 'runs'} · pick 2 to compare`}
            </span>
            <div className="flex items-center gap-2">
              {selectedIds.length > 0 ? (
                <Button type="button" variant="ghost" size="sm" onClick={() => setSelectedIds([])}>
                  Clear selection
                </Button>
              ) : null}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                loading={clearing}
                disabled={history.length === 0}
                onClick={onClear}
              >
                <Trash2 size={14} strokeWidth={1.75} aria-hidden="true" />
                Clear history
              </Button>
            </div>
          </div>

          {compared !== null ? (
            <section className={styles.compareBox} aria-label="Side-by-side comparison">
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-1.5 text-body font-medium text-foreground">
                  <GitCompareArrows size={15} strokeWidth={1.75} aria-hidden="true" />
                  A/B Comparison
                </span>
                <span className="text-caption text-secondary tabular-nums">
                  {`Latency: ${formatDuration(compared[0].elapsedMs)} vs ${formatDuration(compared[1].elapsedMs)}`}
                </span>
              </div>
              <div className={styles.compareHeaders}>
                <div className={styles.compareRunTag}>
                  <Badge size="sm" color="sky">
                    Run A
                  </Badge>
                  <span className="truncate font-mono text-caption">{compared[0].model}</span>
                </div>
                <div className={styles.compareRunTag}>
                  <Badge size="sm" color="violet">
                    Run B
                  </Badge>
                  <span className="truncate font-mono text-caption">{compared[1].model}</span>
                </div>
              </div>
              {diffRows.length === 0 ? (
                <p className="m-0 text-caption text-secondary">No extracted answers to compare.</p>
              ) : (
                <div className={styles.miniTableWrap}>
                  <table className={styles.miniTable}>
                    <thead>
                      <tr>
                        <th>Field</th>
                        <th>Run A</th>
                        <th>Run B</th>
                        <th>Δ Conf</th>
                      </tr>
                    </thead>
                    <tbody>
                      {diffRows.map((row) => (
                        <tr key={row.field} data-diff={row.changed ? 'changed' : undefined}>
                          <td className="font-mono text-foreground">{row.field}</td>
                          <td>
                            <div className="grid gap-0.5">
                              <span>{row.valA}</span>
                              <span className="text-[11px] text-muted tabular-nums">
                                {row.confA !== null ? `conf ${formatPercent(row.confA)}` : ''}
                                {row.pNullA !== null ? ` · p_null ${formatPercent(row.pNullA)}` : ''}
                              </span>
                            </div>
                          </td>
                          <td>
                            <div className="grid gap-0.5">
                              <span>{row.valB}</span>
                              <span className="text-[11px] text-muted tabular-nums">
                                {row.confB !== null ? `conf ${formatPercent(row.confB)}` : ''}
                                {row.pNullB !== null ? ` · p_null ${formatPercent(row.pNullB)}` : ''}
                              </span>
                            </div>
                          </td>
                          <td className="tabular-nums">
                            {row.confDelta === null ? (
                              '—'
                            ) : (
                              <span
                                className={
                                  row.confDelta > 0.005
                                    ? 'text-success'
                                    : row.confDelta < -0.005
                                      ? 'text-danger'
                                      : 'text-muted'
                                }
                              >
                                {`${row.confDelta >= 0 ? '+' : ''}${Math.round(row.confDelta * 100)}%`}
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <Rule />
            </section>
          ) : null}

          {history.length === 0 ? (
            <EmptyState
              title="No runs recorded yet"
              description="Completed extraction and decision runs will appear here for quick restore and comparison."
            />
          ) : (
            <div className={styles.historyList}>
              {history.map((entry) => {
                const checked = selectedIds.includes(entry.id);
                const slot = selectedIds.indexOf(entry.id);
                return (
                  <div key={entry.id} className={styles.historyRow} data-selected={checked ? 'true' : undefined}>
                    <label className={styles.historyCheck}>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleSelect(entry.id)}
                        aria-label={`Select run ${entry.id} for comparison`}
                      />
                    </label>
                    <div className={styles.historyInfo}>
                      <div className={styles.historyHead}>
                        <span className="truncate font-mono text-body font-medium text-foreground">{entry.model}</span>
                        <Badge size="sm" outline>
                          {entry.mode}
                        </Badge>
                        <Badge size="sm" tone={statusTone(entry.status)}>
                          {entry.status}
                        </Badge>
                        {slot === 0 ? (
                          <Badge size="sm" color="sky">
                            A
                          </Badge>
                        ) : slot === 1 ? (
                          <Badge size="sm" color="violet">
                            B
                          </Badge>
                        ) : null}
                      </div>
                      <div className={styles.historyMeta}>
                        <span>{formatDuration(entry.elapsedMs)}</span>
                        <span>·</span>
                        <span>{`${formatCount(entry.fieldCount)} ${entry.fieldCount === 1 ? 'field' : 'fields'}`}</span>
                        <span>·</span>
                        <span>{`${formatCount(entry.inputTokens)} in / ${formatCount(entry.outputTokens)} out`}</span>
                        <span>·</span>
                        <span>{formatRelativeTime(entry.createdAt)}</span>
                      </div>
                      {entry.error !== undefined ? <p className="m-0 text-caption text-danger">{entry.error}</p> : null}
                    </div>
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      onClick={() => {
                        onRestore(entry);
                        onOpenChange(false);
                      }}
                    >
                      <RotateCcw size={13} strokeWidth={1.75} aria-hidden="true" />
                      Restore
                    </Button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </DrawerContent>
    </Drawer>
  );
}
