import { BookOpen, ChevronDown, Trash2 } from 'lucide-react';
import { type FormEvent, type ReactElement, useMemo, useState } from 'react';

import { CompactField } from '../../components/arc-x/compact-field/compact-field';
import { IconButton } from '../../components/arc-x/icon-button/icon-button';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { Input } from '../../components/arc/input/input';
import { Popover, PopoverContent, PopoverTrigger } from '../../components/arc/popover/popover';
import type { SavedSpec, SystemOneMode, SystemOnePreset } from '../../lib/system-one';
import styles from './system-one.module.css';

export interface SpecsPopoverProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: SystemOneMode;
  presets: readonly SystemOnePreset[];
  savedSpecs: readonly SavedSpec[];
  canSaveCurrent: boolean;
  onPickPreset: (preset: SystemOnePreset) => void;
  onApplySavedSpec: (spec: SavedSpec) => void;
  onSaveCurrent: (name: string) => void;
  onDeleteSavedSpec: (id: string) => void;
}

function specCountLabel(spec: SavedSpec): string {
  if (spec.mode === 'decide') {
    const count = spec.questions !== undefined ? Object.keys(spec.questions).length : 0;
    return `${String(count)} ${count === 1 ? 'question' : 'questions'}`;
  }
  const count = spec.fields !== undefined ? Object.keys(spec.fields).length : 0;
  return `${String(count)} ${count === 1 ? 'field' : 'fields'}`;
}

/** Combined library popover for reusable saved schemas/rubrics and built-in examples. */
export function SpecsPopover({
  open,
  onOpenChange,
  mode,
  presets,
  savedSpecs,
  canSaveCurrent,
  onPickPreset,
  onApplySavedSpec,
  onSaveCurrent,
  onDeleteSavedSpec,
}: SpecsPopoverProps): ReactElement {
  const [draftName, setDraftName] = useState('');

  const { matchingPresets, otherPresets } = useMemo(() => {
    const matching: SystemOnePreset[] = [];
    const other: SystemOnePreset[] = [];
    for (const preset of presets) {
      if (preset.mode === mode) matching.push(preset);
      else other.push(preset);
    }
    return { matchingPresets: matching, otherPresets: other };
  }, [presets, mode]);

  const handleSaveSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    const trimmed = draftName.trim();
    if (trimmed === '' || !canSaveCurrent) return;
    onSaveCurrent(trimmed);
    setDraftName('');
  };

  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <PopoverTrigger asChild>
        <Button type="button" variant="secondary" size="sm">
          <BookOpen size={14} strokeWidth={1.75} aria-hidden="true" />
          {savedSpecs.length > 0 ? `Specs (${String(savedSpecs.length)})` : 'Specs'}
          <ChevronDown size={13} strokeWidth={1.75} aria-hidden="true" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className={styles.specsPopover}>
        <section className={styles.specsSection} aria-label="Saved specs">
          <div className={styles.specsSectionHead}>
            <span className={styles.specsSectionTitle}>Saved specs</span>
            <span className={styles.specsSectionHint}>Keeps your current input</span>
          </div>

          {canSaveCurrent ? (
            <form className={styles.specsSaveRow} onSubmit={handleSaveSubmit}>
              <CompactField className="min-w-0 flex-1">
                <Input
                  label={mode === 'decide' ? 'Name this rubric' : 'Name this schema'}
                  placeholder={mode === 'decide' ? 'Name this rubric…' : 'Name this schema…'}
                  value={draftName}
                  onChange={(event) => setDraftName(event.target.value)}
                />
              </CompactField>
              <Button type="submit" variant="secondary" size="sm" disabled={draftName.trim() === ''}>
                Save
              </Button>
            </form>
          ) : null}

          {savedSpecs.length === 0 ? (
            <p className={styles.specsEmpty}>
              {canSaveCurrent
                ? `Save your current ${mode === 'decide' ? 'rubric' : 'schema'} above to reuse on any input.`
                : `Add ${mode === 'decide' ? 'questions' : 'fields'} first, then save them here to reuse.`}
            </p>
          ) : (
            <div className={styles.specsList}>
              {savedSpecs.map((spec) => (
                <div key={spec.id} className={styles.specsItem}>
                  <button type="button" className={styles.specsItemBtn} onClick={() => onApplySavedSpec(spec)}>
                    <div className={styles.popoverItemTitle}>
                      <span className="truncate">{spec.name}</span>
                      <Badge size="sm" color={spec.mode === 'decide' ? 'orchid' : 'sky'} outline>
                        {spec.mode === 'decide' ? 'Decide' : 'Extract'}
                      </Badge>
                    </div>
                    <p className={styles.popoverItemDesc}>{`${specCountLabel(spec)} · Keeps your current input`}</p>
                  </button>
                  <div className={styles.specsItemDelete}>
                    <IconButton
                      label={`Delete saved spec ${spec.name}`}
                      icon={<Trash2 size={13} strokeWidth={1.75} aria-hidden="true" />}
                      onClick={(event) => {
                        event.stopPropagation();
                        onDeleteSavedSpec(spec.id);
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className={styles.specsSection} aria-label="Examples">
          <div className={styles.specsSectionHead}>
            <span className={styles.specsSectionTitle}>Examples</span>
            <span className={styles.specsSectionHint}>Loads sample input &amp; spec</span>
          </div>
          <div className={styles.specsList}>
            {matchingPresets.map((preset) => (
              <button key={preset.id} type="button" className={styles.popoverItem} onClick={() => onPickPreset(preset)}>
                <div className="min-w-0">
                  <div className={styles.popoverItemTitle}>
                    <span>{preset.label}</span>
                  </div>
                  <p className={styles.popoverItemDesc}>{preset.description}</p>
                </div>
              </button>
            ))}
            {otherPresets.map((preset) => (
              <button key={preset.id} type="button" className={styles.popoverItem} onClick={() => onPickPreset(preset)}>
                <div className="min-w-0">
                  <div className={styles.popoverItemTitle}>
                    <span>{preset.label}</span>
                    <Badge size="sm" color={preset.mode === 'decide' ? 'orchid' : 'sky'} outline>
                      {preset.mode === 'decide' ? 'Decide' : 'Extract'}
                    </Badge>
                  </div>
                  <p className={styles.popoverItemDesc}>{preset.description}</p>
                </div>
              </button>
            ))}
          </div>
        </section>
      </PopoverContent>
    </Popover>
  );
}
