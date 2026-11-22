import { AlertTriangle, ArrowDown, ArrowUp, BookmarkPlus, ChevronRight, Plus, Sparkles, Trash2 } from 'lucide-react';
import { type ReactElement, useEffect, useRef, useState } from 'react';

import { IconButton } from '../../components/arc-x/icon-button/icon-button';
import { TagInput } from '../../components/arc-x/tag-input/tag-input';
import { Button } from '../../components/arc/button/button';
import { EmptyState } from '../../components/arc/empty-state/empty-state';
import { Input } from '../../components/arc/input/input';
import { Select } from '../../components/arc/select/select';
import { Switch } from '../../components/arc/switch/switch';
import {
  type ClairicDurationUnit,
  type ClairicEstimateScale,
  type ClairicFieldType,
  type ClairicForecastInterval,
  type ClairicListItemKind,
  createEmptyFieldDraft,
  DURATION_UNITS,
  fieldAccentColor,
  type FieldDraft,
  FORECAST_INTERVALS,
  supportsKnowledge,
  supportsNumericOptions,
  supportsSourceOnly,
} from '../../lib/system-one';
import styles from './system-one.module.css';

export interface SchemaBuilderProps {
  fields: readonly FieldDraft[];
  warnings: Readonly<Record<string, string[]>>;
  onChange: (next: FieldDraft[]) => void;
  disabled?: boolean;
  /** Increment to auto-expand the first field with validation warnings. */
  focusInvalidNonce?: number;
  /** Trigger 1-click Clairic auto-detection of fields from the current document/image. */
  onAutoDetect?: () => void;
  autoDetecting?: boolean;
  canAutoDetect?: boolean;
  /** Save the current field schema to the reusable Specs library. */
  onSaveSpec?: () => void;
}

const TYPE_OPTIONS: Array<{ value: ClairicFieldType; label: string }> = [
  { value: 'text', label: 'Text' },
  { value: 'number', label: 'Number' },
  { value: 'boolean', label: 'Yes / No' },
  { value: 'enum', label: 'Pick from list' },
  { value: 'object', label: 'Nested fields' },
  { value: 'list', label: 'List' },
  { value: 'date', label: 'Date' },
  { value: 'time', label: 'Time' },
  { value: 'datetime', label: 'Date & time' },
  { value: 'duration', label: 'Duration' },
  { value: 'compute', label: 'Calculate' },
  { value: 'estimate', label: 'Estimate range' },
  { value: 'forecast', label: 'Forecast' },
];

const LIST_ITEM_OPTIONS: Array<{ value: ClairicListItemKind; label: string }> = [
  { value: 'text', label: 'Text' },
  { value: 'number', label: 'Number' },
  { value: 'date', label: 'Date' },
  { value: 'object', label: 'Nested fields' },
];

const SCALE_OPTIONS: Array<{ value: ClairicEstimateScale; label: string }> = [
  { value: 'linear', label: 'Linear' },
  { value: 'log', label: 'Log' },
];

const DURATION_OPTIONS = DURATION_UNITS.map((unit) => ({ value: unit, label: unit }));
const INTERVAL_OPTIONS = FORECAST_INTERVALS.map((interval) => ({ value: interval, label: interval }));

function compactTypeSummary(field: FieldDraft): string {
  if (field.type === 'list') return `list of ${field.listItemKind}`;
  if (field.type === 'enum' && field.enumValues.length > 0) {
    return `${String(field.enumValues.length)} choices`;
  }
  if (field.type === 'boolean') return 'yes / no';
  if (field.type === 'datetime') return 'date & time';
  if (field.type === 'compute') return 'calculate';
  if (field.type === 'estimate') return 'estimate';
  if (field.type === 'forecast') {
    const h = field.horizon.trim() || '3';
    return `forecast (${h} ${field.interval}s)`;
  }
  return field.type;
}

/** Flat hairline accordion schema builder for all 13 Clairic extraction & typed-head field types. */
export function SchemaBuilder({
  fields,
  warnings,
  onChange,
  disabled = false,
  focusInvalidNonce = 0,
  onAutoDetect,
  autoDetecting = false,
  canAutoDetect = false,
  onSaveSpec,
}: SchemaBuilderProps): ReactElement {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const lastFocusNonceRef = useRef(0);

  useEffect(() => {
    if (focusInvalidNonce <= 0 || focusInvalidNonce === lastFocusNonceRef.current) return;
    lastFocusNonceRef.current = focusInvalidNonce;
    const firstInvalid = fields.find((f) => (warnings[f.id]?.length ?? 0) > 0);
    if (firstInvalid !== undefined) setExpandedId(firstInvalid.id);
  }, [focusInvalidNonce, fields, warnings]);

  const updateField = (id: string, patch: Partial<FieldDraft>): void => {
    onChange(fields.map((field) => (field.id === id ? { ...field, ...patch } : field)));
  };

  const changeType = (field: FieldDraft, nextType: ClairicFieldType): void => {
    const fresh = createEmptyFieldDraft(nextType, field.name);
    updateField(field.id, {
      type: nextType,
      enumValues: nextType === 'enum' && field.enumValues.length === 0 ? fresh.enumValues : field.enumValues,
      min: (nextType === 'estimate' || nextType === 'forecast') && field.min === '' ? fresh.min : field.min,
      max: (nextType === 'estimate' || nextType === 'forecast') && field.max === '' ? fresh.max : field.max,
      horizon: nextType === 'forecast' && field.horizon === '' ? fresh.horizon : field.horizon,
    });
  };

  const removeField = (id: string): void => {
    if (expandedId === id) setExpandedId(null);
    onChange(fields.filter((field) => field.id !== id));
  };

  const moveField = (index: number, delta: -1 | 1): void => {
    const target = index + delta;
    if (target < 0 || target >= fields.length) return;
    const copy = [...fields];
    const [item] = copy.splice(index, 1);
    copy.splice(target, 0, item!);
    onChange(copy);
  };

  const addField = (type: ClairicFieldType = 'text'): void => {
    const nextNum = fields.length + 1;
    const created = createEmptyFieldDraft(type, `field_${String(nextNum)}`);
    onChange([...fields, created]);
    setExpandedId(created.id);
  };

  return (
    <div className={styles.fieldsSection}>
      <div className={styles.sectionHeader}>
        <div className={styles.sectionHeadingGroup}>
          <span className={styles.sectionLabel}>
            2. What to extract <span className={styles.sectionCount}>({fields.length})</span>
          </span>
          <span className={styles.sectionHint}>
            {fields.length === 0
              ? 'Leave empty and press Run — or click Auto-detect to generate fields.'
              : 'Pick the fields to pull out. Click any row to edit.'}
          </span>
        </div>
        <div className={styles.sectionHeaderActions}>
          {onAutoDetect !== undefined ? (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              loading={autoDetecting}
              disabled={disabled || !canAutoDetect}
              onClick={onAutoDetect}
            >
              <Sparkles size={13} strokeWidth={1.75} aria-hidden="true" />
              Auto-detect
            </Button>
          ) : null}
          {onSaveSpec !== undefined && fields.length > 0 ? (
            <Button type="button" variant="ghost" size="sm" disabled={disabled} onClick={onSaveSpec}>
              <BookmarkPlus size={13} strokeWidth={1.75} aria-hidden="true" />
              Save
            </Button>
          ) : null}
          <Button type="button" variant="ghost" size="sm" disabled={disabled} onClick={() => addField('text')}>
            <Plus size={14} strokeWidth={1.75} aria-hidden="true" />
            Add field
          </Button>
        </div>
      </div>

      {fields.length === 0 ? (
        <div className={styles.emptySection}>
          <EmptyState
            title="Auto-detect or add fields"
            description="Press Run right away to auto-detect fields from your document or photo, or define them manually."
            action={
              <div className={styles.emptyActions}>
                {onAutoDetect !== undefined ? (
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    loading={autoDetecting}
                    disabled={disabled || !canAutoDetect}
                    onClick={onAutoDetect}
                  >
                    <Sparkles size={13} strokeWidth={1.75} aria-hidden="true" />
                    Auto-detect fields
                  </Button>
                ) : null}
                <Button type="button" variant="ghost" size="sm" disabled={disabled} onClick={() => addField('text')}>
                  <Plus size={14} strokeWidth={1.75} aria-hidden="true" />
                  Add manually
                </Button>
              </div>
            }
          />
        </div>
      ) : (
        <div className={styles.accordionList}>
          {fields.map((field, index) => {
            const expanded = expandedId === field.id;
            const fieldWarnings = warnings[field.id] ?? [];
            const hasWarnings = fieldWarnings.length > 0;
            const color = fieldAccentColor(index);

            return (
              <div
                key={field.id}
                className={styles.accordionRow}
                data-open={expanded ? 'true' : undefined}
                data-invalid={hasWarnings ? 'true' : undefined}
              >
                <div className={styles.rowHeaderWrap}>
                  <button
                    type="button"
                    className={styles.rowTrigger}
                    aria-expanded={expanded}
                    onClick={() => setExpandedId(expanded ? null : field.id)}
                  >
                    <span className={styles.rowDot} style={{ background: color }} aria-hidden="true" />
                    <span className={field.name.trim() === '' ? styles.rowNameEmpty : styles.rowName}>
                      {field.name.trim() || 'untitled'}
                    </span>
                    <span className={styles.rowType}>{compactTypeSummary(field)}</span>
                    {field.instructions.trim() !== '' ? (
                      <span className={styles.rowDesc}>{field.instructions.trim()}</span>
                    ) : null}
                    {hasWarnings ? (
                      <AlertTriangle
                        size={13}
                        strokeWidth={1.75}
                        className={styles.rowWarnIcon}
                        aria-label={fieldWarnings[0]}
                      />
                    ) : null}
                    <ChevronRight size={14} strokeWidth={1.75} className={styles.rowChevron} aria-hidden="true" />
                  </button>
                  <div className={styles.rowHoverAction}>
                    <IconButton
                      label={`Remove field ${field.name || String(index + 1)}`}
                      icon={<Trash2 size={13} strokeWidth={1.75} aria-hidden="true" />}
                      disabled={disabled}
                      onClick={() => removeField(field.id)}
                    />
                  </div>
                </div>

                {expanded ? (
                  <div className={styles.inlineEditor}>
                    <div className={styles.editorTopGrid}>
                      <Input
                        label="Field name"
                        placeholder="field_name"
                        value={field.name}
                        disabled={disabled}
                        onChange={(event) => updateField(field.id, { name: event.target.value })}
                        className="font-mono"
                      />
                      <Select
                        label="Type"
                        options={TYPE_OPTIONS}
                        value={field.type}
                        disabled={disabled}
                        onValueChange={(value) => changeType(field, value as ClairicFieldType)}
                      />
                    </div>

                    <Input
                      label="What to look for"
                      placeholder="e.g. Issuing company name or final total due…"
                      value={field.instructions}
                      disabled={disabled}
                      onChange={(event) => updateField(field.id, { instructions: event.target.value })}
                    />

                    <div className={styles.switchRow}>
                      <Switch
                        label="Allow null"
                        checked={field.nullable}
                        disabled={disabled}
                        onCheckedChange={(checked) => updateField(field.id, { nullable: checked })}
                      />
                      {supportsSourceOnly(field.type) ? (
                        <Switch
                          label="Source only"
                          checked={field.sourceOnly}
                          disabled={disabled}
                          onCheckedChange={(checked) => updateField(field.id, { sourceOnly: checked })}
                        />
                      ) : null}
                      {supportsKnowledge(field.type) ? (
                        <Switch
                          label="Prior knowledge"
                          checked={field.knowledge}
                          disabled={disabled}
                          onCheckedChange={(checked) => updateField(field.id, { knowledge: checked })}
                        />
                      ) : null}
                      {supportsNumericOptions(field.type) ? (
                        <Switch
                          label="Integer only"
                          checked={field.integer}
                          disabled={disabled}
                          onCheckedChange={(checked) => updateField(field.id, { integer: checked })}
                        />
                      ) : null}
                    </div>

                    {supportsNumericOptions(field.type) ? (
                      <div className={styles.twoCol}>
                        <Input
                          label="Decimals"
                          type="number"
                          min={0}
                          step={1}
                          placeholder="e.g. 2"
                          value={field.decimals}
                          disabled={disabled}
                          onChange={(event) => updateField(field.id, { decimals: event.target.value })}
                        />
                        {field.type === 'duration' ? (
                          <Select
                            label="Duration unit"
                            options={DURATION_OPTIONS}
                            value={field.durationUnit}
                            disabled={disabled}
                            onValueChange={(value) =>
                              updateField(field.id, { durationUnit: value as ClairicDurationUnit })
                            }
                          />
                        ) : (
                          <Input
                            label="Unit"
                            placeholder="e.g. USD, kg"
                            value={field.unit}
                            disabled={disabled}
                            onChange={(event) => updateField(field.id, { unit: event.target.value })}
                          />
                        )}
                      </div>
                    ) : null}

                    {field.type === 'enum' ? (
                      <TagInput
                        label="Allowed values"
                        values={field.enumValues}
                        onChange={(next) => updateField(field.id, { enumValues: next })}
                        placeholder="Add option and press Enter…"
                        hint="Comma or Enter adds a value."
                      />
                    ) : null}

                    {field.type === 'list' ? (
                      <div className={styles.subOptions}>
                        <div className={styles.twoCol}>
                          <Select
                            label="Item type"
                            options={LIST_ITEM_OPTIONS}
                            value={field.listItemKind}
                            disabled={disabled}
                            onValueChange={(value) =>
                              updateField(field.id, { listItemKind: value as ClairicListItemKind })
                            }
                          />
                          <Input
                            label="Max items"
                            type="number"
                            min={1}
                            step={1}
                            placeholder="e.g. 8"
                            value={field.listMaxItems}
                            disabled={disabled}
                            onChange={(event) => updateField(field.id, { listMaxItems: event.target.value })}
                            description="Leave blank for up to 32 items."
                          />
                        </div>
                        {field.listItemKind === 'object' ? (
                          <div className={styles.jsonSubField}>
                            <label htmlFor={`list-props-${field.id}`} className={styles.smallLabel}>
                              Item object properties (JSON)
                            </label>
                            <textarea
                              id={`list-props-${field.id}`}
                              aria-label="Item object properties (JSON)"
                              className={styles.codeTextarea}
                              rows={4}
                              value={field.listObjectPropertiesJson}
                              disabled={disabled}
                              spellCheck={false}
                              onChange={(event) =>
                                updateField(field.id, { listObjectPropertiesJson: event.target.value })
                              }
                            />
                          </div>
                        ) : null}
                      </div>
                    ) : null}

                    {field.type === 'object' ? (
                      <div className={styles.jsonSubField}>
                        <label htmlFor={`obj-props-${field.id}`} className={styles.smallLabel}>
                          Nested properties (JSON)
                        </label>
                        <textarea
                          id={`obj-props-${field.id}`}
                          aria-label="Nested properties (JSON)"
                          className={styles.codeTextarea}
                          rows={4}
                          value={field.objectPropertiesJson}
                          disabled={disabled}
                          spellCheck={false}
                          onChange={(event) => updateField(field.id, { objectPropertiesJson: event.target.value })}
                        />
                      </div>
                    ) : null}

                    {field.type === 'estimate' || field.type === 'forecast' ? (
                      <div className={styles.subOptions}>
                        <div className={styles.threeCol}>
                          <Input
                            label="Min"
                            placeholder="0 or YYYY-MM-DD"
                            value={field.min}
                            disabled={disabled}
                            onChange={(event) => updateField(field.id, { min: event.target.value })}
                          />
                          <Input
                            label="Max"
                            placeholder="100 or YYYY-MM-DD"
                            value={field.max}
                            disabled={disabled}
                            onChange={(event) => updateField(field.id, { max: event.target.value })}
                          />
                          <Input
                            label="Unit"
                            placeholder="e.g. orders"
                            value={field.unit}
                            disabled={disabled}
                            onChange={(event) => updateField(field.id, { unit: event.target.value })}
                          />
                        </div>
                        <div className={field.type === 'forecast' ? styles.threeCol : styles.twoCol}>
                          <Select
                            label="Scale"
                            options={SCALE_OPTIONS}
                            value={field.scale}
                            disabled={disabled}
                            onValueChange={(value) => updateField(field.id, { scale: value as ClairicEstimateScale })}
                          />
                          {field.type === 'forecast' ? (
                            <>
                              <Input
                                label="Horizon"
                                type="number"
                                min={1}
                                step={1}
                                value={field.horizon}
                                disabled={disabled}
                                onChange={(event) => updateField(field.id, { horizon: event.target.value })}
                              />
                              <Select
                                label="Interval"
                                options={INTERVAL_OPTIONS}
                                value={field.interval}
                                disabled={disabled}
                                onValueChange={(value) =>
                                  updateField(field.id, { interval: value as ClairicForecastInterval })
                                }
                              />
                            </>
                          ) : null}
                        </div>
                      </div>
                    ) : null}

                    {hasWarnings ? (
                      <div className={styles.warningList} role="alert">
                        {fieldWarnings.map((msg) => (
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
                          onClick={() => moveField(index, -1)}
                        >
                          <ArrowUp size={13} strokeWidth={1.75} aria-hidden="true" />
                          Move up
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          disabled={disabled || index === fields.length - 1}
                          onClick={() => moveField(index, 1)}
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
