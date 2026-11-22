import { AlertTriangle, ArrowDown, ArrowUp, BookmarkPlus, ChevronRight, Plus, Trash2 } from 'lucide-react';
import { type ReactElement, useEffect, useRef, useState } from 'react';

import { CompactField } from '../../components/arc-x/compact-field/compact-field';
import { IconButton } from '../../components/arc-x/icon-button/icon-button';
import { Button } from '../../components/arc/button/button';
import { EmptyState } from '../../components/arc/empty-state/empty-state';
import { Input } from '../../components/arc/input/input';
import { Select } from '../../components/arc/select/select';
import {
  type ClefQuestionDraft,
  type ClefQuestionType,
  createEmptyClefQuestion,
  fieldAccentColor,
  nextDraftId,
} from '../../lib/system-one';
import styles from './system-one.module.css';

export interface ClefBuilderProps {
  questions: readonly ClefQuestionDraft[];
  warnings: Readonly<Record<string, string[]>>;
  onChange: (next: ClefQuestionDraft[]) => void;
  disabled?: boolean;
  /** Increment to auto-expand the first question with validation warnings. */
  focusInvalidNonce?: number;
  /** Save the current rubric to the reusable Specs library. */
  onSaveSpec?: () => void;
}

const QUESTION_TYPE_OPTIONS: Array<{ value: ClefQuestionType; label: string }> = [
  { value: 'noul', label: 'Yes / No' },
  { value: 'choice', label: 'Pick one' },
  { value: 'score', label: 'Rating scale' },
];

function compactQuestionType(q: ClefQuestionDraft): string {
  if (q.type === 'noul') return 'yes / no';
  if (q.type === 'choice') return `${String(q.choices.length)} choices`;
  return `${String(q.scoreLevels.length)}-level rating`;
}

/** Flat hairline accordion builder for CLEF decision questions (`noul`, `choice`, `score`). */
export function ClefBuilder({
  questions,
  warnings,
  onChange,
  disabled = false,
  focusInvalidNonce = 0,
  onSaveSpec,
}: ClefBuilderProps): ReactElement {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const lastFocusNonceRef = useRef(0);

  useEffect(() => {
    if (focusInvalidNonce <= 0 || focusInvalidNonce === lastFocusNonceRef.current) return;
    lastFocusNonceRef.current = focusInvalidNonce;
    const firstInvalid = questions.find((q) => (warnings[q.id]?.length ?? 0) > 0);
    if (firstInvalid !== undefined) setExpandedId(firstInvalid.id);
  }, [focusInvalidNonce, questions, warnings]);

  const update = (id: string, patch: Partial<ClefQuestionDraft>): void => {
    onChange(questions.map((q) => (q.id === id ? { ...q, ...patch } : q)));
  };

  const addQuestion = (type: ClefQuestionType = 'noul'): void => {
    const created = createEmptyClefQuestion(type, `question_${String(questions.length + 1)}`);
    onChange([...questions, created]);
    setExpandedId(created.id);
  };

  const removeQuestion = (id: string): void => {
    if (expandedId === id) setExpandedId(null);
    onChange(questions.filter((q) => q.id !== id));
  };

  const moveQuestion = (index: number, delta: -1 | 1): void => {
    const target = index + delta;
    if (target < 0 || target >= questions.length) return;
    const copy = [...questions];
    const [item] = copy.splice(index, 1);
    copy.splice(target, 0, item!);
    onChange(copy);
  };

  return (
    <div className={styles.fieldsSection}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionHeadingGroup}>
          <span className={styles.sectionLabel}>
            2. Questions to ask <span className={styles.sectionCount}>({questions.length})</span>
          </span>
          <span className={styles.sectionHint}>
            Each question returns calibrated probabilities. Click any row to edit.
          </span>
        </div>
        <div className={styles.sectionHeaderActions}>
          {onSaveSpec !== undefined && questions.length > 0 ? (
            <Button type="button" variant="ghost" size="sm" disabled={disabled} onClick={onSaveSpec}>
              <BookmarkPlus size={13} strokeWidth={1.75} aria-hidden="true" />
              Save
            </Button>
          ) : null}
          <Button type="button" variant="ghost" size="sm" disabled={disabled} onClick={() => addQuestion('noul')}>
            <Plus size={14} strokeWidth={1.75} aria-hidden="true" />
            Add question
          </Button>
        </div>
      </div>

      {questions.length === 0 ? (
        <div className={styles.emptySection}>
          <EmptyState
            title="No questions yet"
            description="Add a yes/no, pick-one, or rating scale question."
            action={
              <Button
                type="button"
                variant="secondary"
                size="sm"
                disabled={disabled}
                onClick={() => addQuestion('noul')}
              >
                <Plus size={14} strokeWidth={1.75} aria-hidden="true" />
                Add question
              </Button>
            }
          />
        </div>
      ) : (
        <div className={styles.accordionList}>
          {questions.map((q, index) => {
            const expanded = expandedId === q.id;
            const qWarnings = warnings[q.id] ?? [];
            const hasWarnings = qWarnings.length > 0;
            const color = fieldAccentColor(index);

            return (
              <div
                key={q.id}
                className={styles.accordionRow}
                data-open={expanded ? 'true' : undefined}
                data-invalid={hasWarnings ? 'true' : undefined}
              >
                <div className={styles.rowHeaderWrap}>
                  <button
                    type="button"
                    className={styles.rowTrigger}
                    aria-expanded={expanded}
                    onClick={() => setExpandedId(expanded ? null : q.id)}
                  >
                    <span className={styles.rowDot} style={{ background: color }} aria-hidden="true" />
                    <span className={q.name.trim() === '' ? styles.rowNameEmpty : styles.rowName}>
                      {q.name.trim() || 'untitled'}
                    </span>
                    <span className={styles.rowType}>{compactQuestionType(q)}</span>
                    {q.instructions.trim() !== '' ? (
                      <span className={styles.rowDesc}>{q.instructions.trim()}</span>
                    ) : null}
                    {hasWarnings ? (
                      <AlertTriangle
                        size={13}
                        strokeWidth={1.75}
                        className={styles.rowWarnIcon}
                        aria-label={qWarnings[0]}
                      />
                    ) : null}
                    <ChevronRight size={14} strokeWidth={1.75} className={styles.rowChevron} aria-hidden="true" />
                  </button>
                  <div className={styles.rowHoverAction}>
                    <IconButton
                      label={`Remove question ${q.name || String(index + 1)}`}
                      icon={<Trash2 size={13} strokeWidth={1.75} aria-hidden="true" />}
                      disabled={disabled}
                      onClick={() => removeQuestion(q.id)}
                    />
                  </div>
                </div>

                {expanded ? (
                  <div className={styles.inlineEditor}>
                    <div className={styles.editorTopGrid}>
                      <Input
                        label="Question ID"
                        placeholder="question_id"
                        value={q.name}
                        disabled={disabled}
                        className="font-mono"
                        onChange={(event) => update(q.id, { name: event.target.value })}
                      />
                      <Select
                        label="Type"
                        options={QUESTION_TYPE_OPTIONS}
                        value={q.type}
                        disabled={disabled}
                        onValueChange={(value) => update(q.id, { type: value as ClefQuestionType })}
                      />
                    </div>

                    <Input
                      label="Question prompt"
                      placeholder="Question instructions…"
                      value={q.instructions}
                      disabled={disabled}
                      onChange={(event) => update(q.id, { instructions: event.target.value })}
                    />

                    {q.type === 'noul' ? (
                      <div className={styles.twoCol}>
                        <Input
                          label="When to answer Yes (optional)"
                          placeholder="Criteria for a Yes answer"
                          value={q.noulTrue}
                          disabled={disabled}
                          onChange={(event) => update(q.id, { noulTrue: event.target.value })}
                        />
                        <Input
                          label="When to answer No (optional)"
                          placeholder="Criteria for a No answer"
                          value={q.noulFalse}
                          disabled={disabled}
                          onChange={(event) => update(q.id, { noulFalse: event.target.value })}
                        />
                      </div>
                    ) : null}

                    {q.type === 'choice' ? (
                      <div className={styles.subOptions}>
                        <span className={styles.smallLabel}>Options</span>
                        {q.choices.map((opt) => (
                          <div key={opt.id} className={styles.choiceRow}>
                            <CompactField>
                              <Input
                                label="Option key"
                                placeholder="option_key"
                                value={opt.key}
                                disabled={disabled}
                                className="font-mono"
                                onChange={(event) =>
                                  update(q.id, {
                                    choices: q.choices.map((c) =>
                                      c.id === opt.id ? { ...c, key: event.target.value } : c,
                                    ),
                                  })
                                }
                              />
                            </CompactField>
                            <CompactField>
                              <Input
                                label="Option criteria"
                                placeholder="Criteria description"
                                value={opt.description}
                                disabled={disabled}
                                onChange={(event) =>
                                  update(q.id, {
                                    choices: q.choices.map((c) =>
                                      c.id === opt.id ? { ...c, description: event.target.value } : c,
                                    ),
                                  })
                                }
                              />
                            </CompactField>
                            <IconButton
                              label={`Remove option ${opt.key}`}
                              icon={<Trash2 size={13} strokeWidth={1.75} aria-hidden="true" />}
                              disabled={disabled || q.choices.length <= 1}
                              onClick={() => update(q.id, { choices: q.choices.filter((c) => c.id !== opt.id) })}
                            />
                          </div>
                        ))}
                        <div>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            disabled={disabled}
                            onClick={() =>
                              update(q.id, {
                                choices: [
                                  ...q.choices,
                                  {
                                    id: nextDraftId('opt'),
                                    key: `option_${String(q.choices.length + 1)}`,
                                    description: '',
                                  },
                                ],
                              })
                            }
                          >
                            <Plus size={13} strokeWidth={1.75} aria-hidden="true" />
                            Add option
                          </Button>
                        </div>
                      </div>
                    ) : null}

                    {q.type === 'score' ? (
                      <div className={styles.subOptions}>
                        <span className={styles.smallLabel}>Score rubric levels</span>
                        <div className={styles.rubricList}>
                          {q.scoreLevels.map((level, levelIndex) => (
                            <div key={levelIndex} className={styles.rubricRow}>
                              <span className={styles.rubricIndex}>{levelIndex}</span>
                              <CompactField>
                                <Input
                                  label={`Level ${String(levelIndex)} criteria`}
                                  placeholder={`Criteria for score ${String(levelIndex)}…`}
                                  value={level}
                                  disabled={disabled}
                                  onChange={(event) =>
                                    update(q.id, {
                                      scoreLevels: q.scoreLevels.map((v, idx) =>
                                        idx === levelIndex ? event.target.value : v,
                                      ),
                                    })
                                  }
                                />
                              </CompactField>
                              <IconButton
                                label={`Remove score level ${String(levelIndex)}`}
                                icon={<Trash2 size={13} strokeWidth={1.75} aria-hidden="true" />}
                                disabled={disabled || q.scoreLevels.length <= 2}
                                onClick={() =>
                                  update(q.id, {
                                    scoreLevels: q.scoreLevels.filter((_, idx) => idx !== levelIndex),
                                  })
                                }
                              />
                            </div>
                          ))}
                        </div>
                        <div className={styles.rubricFooter}>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            disabled={disabled || q.scoreLevels.length >= 10}
                            onClick={() => update(q.id, { scoreLevels: [...q.scoreLevels, ''] })}
                          >
                            <Plus size={13} strokeWidth={1.75} aria-hidden="true" />
                            Add level
                          </Button>
                          <span className={styles.sectionHint}>
                            {`0 to ${String(Math.max(0, q.scoreLevels.length - 1))} scale · lowest to highest (2–10 levels)`}
                          </span>
                        </div>
                      </div>
                    ) : null}

                    {hasWarnings ? (
                      <div className={styles.warningList} role="alert">
                        {qWarnings.map((msg) => (
                          <span key={msg} className={styles.warningLine}>
                            <AlertTriangle size={12} strokeWidth={1.75} aria-hidden="true" />
                            <span>{msg}</span>
                          </span>
                        ))}
                      </div>
                    ) : null}

                    <div className={styles.editorFooter}>
                      <div className={styles.editorMoveGroup}>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          disabled={disabled || index === 0}
                          onClick={() => moveQuestion(index, -1)}
                        >
                          <ArrowUp size={13} strokeWidth={1.75} aria-hidden="true" />
                          Move up
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          disabled={disabled || index === questions.length - 1}
                          onClick={() => moveQuestion(index, 1)}
                        >
                          <ArrowDown size={13} strokeWidth={1.75} aria-hidden="true" />
                          Move down
                        </Button>
                      </div>
                      <Button type="button" variant="ghost" size="sm" onClick={() => setExpandedId(null)}>
                        Done
                      </Button>
                    </div>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
