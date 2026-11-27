import { type CSSProperties, type ReactElement, type ReactNode, useId, useMemo, useState } from 'react';

import { ChatMarkdown } from '../components/arc-x/chat-markdown/chat-markdown';
import { CodePanel } from '../components/arc-x/code-panel/code-panel';
import { CompactField } from '../components/arc-x/compact-field/compact-field';
import { Button } from '../components/arc/button/button';
import { Combobox } from '../components/arc/combobox/combobox';
import { NumberField } from '../components/arc/number-field/number-field';
import { fill } from '../lib/chat-strings';
import { FONTS } from '../lib/fonts-strings';
import { fontOptions, type LocalFontList, missingFamily, useLocalFonts } from '../lib/local-fonts';
import { Page, PageHeader } from '../shell/page';
import {
  FONT_DEFAULTS,
  FONT_LIMITS,
  type FontFamily,
  type FontRole,
  type FontSettings,
  fontProperties,
  isDefaultFonts,
  stockTypeProperties,
  useFonts,
} from '../theme/fonts';
import styles from './fonts.module.css';

const ZONE_STYLE = stockTypeProperties() as CSSProperties;
const READING_LINKS = { mode: 'work' } as const;

function isFontFamily(value: string): value is FontFamily {
  return value === 'default' || value === 'system' || value.startsWith('name:');
}

function rangeLine(role: FontRole): string {
  const { min, max } = FONT_LIMITS[role];
  return fill(role === 'ui' ? FONTS.rangeUi : FONTS.range, { min, max });
}

function fontNote(role: FontRole, family: FontFamily, list: LocalFontList): string | undefined {
  if (list.status === 'unavailable') return FONTS.unavailable;
  const missing = missingFamily(family, list);
  if (missing === null) return undefined;
  return fill(FONTS.missing, { family: missing, fallback: FONTS.fallbackName[role] });
}

function Row({ kind, label, children }: { kind: 'font' | 'size' | 'sample'; label: string; children: ReactNode }) {
  return (
    <div className={styles.row} data-row={kind}>
      <span className={styles.rowLabel}>{label}</span>
      <div className={styles.rowControl}>{children}</div>
    </div>
  );
}

interface GroupProps {
  role: FontRole;
  settings: FontSettings;
  list: LocalFontList;
  onChange: (role: FontRole, change: { family: FontFamily } | { size: number }) => void;
  sample: ReactNode;
}

function FontGroup({ role, settings, list, onChange, sample }: GroupProps): ReactElement {
  const titleId = useId();
  const group = FONTS.groups[role];
  const names = FONTS.controls[role];
  const { family, size } = settings[role];
  const { min, max } = FONT_LIMITS[role];
  const options = useMemo(() => fontOptions(role, list, family), [role, list, family]);
  return (
    <section className={styles.group} aria-labelledby={titleId}>
      <div className={styles.groupHead}>
        <h2 id={titleId} className={styles.groupTitle}>
          {group.title}
        </h2>
        <p className={styles.groupDescription}>{group.description}</p>
      </div>
      <Row kind="font" label={FONTS.rows.font}>
        <CompactField className={styles.font}>
          <Combobox
            label={names.font}
            options={options}
            value={family}
            placeholder={FONTS.placeholder}
            emptyMessage={FONTS.empty}
            description={fontNote(role, family, list)}
            onValueChange={(value) => {
              if (value === '') onChange(role, { family: 'default' });
              else if (isFontFamily(value)) onChange(role, { family: value });
            }}
          />
        </CompactField>
      </Row>
      <Row kind="size" label={FONTS.rows.size}>
        <CompactField className={styles.size}>
          <NumberField
            label={names.size}
            size="sm"
            min={min}
            max={max}
            step={1}
            largeStep={2}
            limitHint={false}
            suffix={FONTS.sizeSuffix}
            description={rangeLine(role)}
            value={size}
            onValueChange={(next) => onChange(role, { size: next })}
          />
        </CompactField>
      </Row>
      <Row kind="sample" label={FONTS.rows.sample}>
        <div role="group" aria-label={FONTS.samples[role]}>
          {sample}
        </div>
      </Row>
    </section>
  );
}

function InterfaceSample({ settings }: { settings: FontSettings }): ReactElement {
  const style = fontProperties({ ...FONT_DEFAULTS, ui: settings.ui }) as CSSProperties;
  return (
    <div className={styles.uiSample} style={style}>
      <span className={styles.uiSampleTitle}>{FONTS.uiSample.title}</span>
      <span className={styles.uiSampleLine}>{FONTS.uiSample.line}</span>
      <span className={styles.uiSampleTime}>{FONTS.uiSample.time}</span>
    </div>
  );
}

export default function Fonts(): ReactElement {
  const { settings, setFont, resetFonts } = useFonts();
  const list = useLocalFonts();
  const atDefaults = isDefaultFonts(settings);
  const [resetDone, setResetDone] = useState(false);

  const change = (role: FontRole, next: { family: FontFamily } | { size: number }): void => {
    setResetDone(false);
    setFont(role, next);
  };
  const reset = (): void => {
    if (atDefaults) return;
    resetFonts();
    setResetDone(true);
  };

  return (
    <div className={styles.zone} style={ZONE_STYLE}>
      <Page width="reading">
        <PageHeader
          title={FONTS.title}
          description={FONTS.description}
          actions={
            <Button
              variant="secondary"
              className={styles.reset}
              aria-disabled={atDefaults || undefined}
              onClick={reset}
            >
              {FONTS.reset}
            </Button>
          }
        />
        <div role="status" aria-live="polite" className="sr-only">
          {resetDone && atDefaults ? FONTS.resetDone : ''}
        </div>
        <FontGroup
          role="reading"
          settings={settings}
          list={list}
          onChange={change}
          sample={<ChatMarkdown text={FONTS.readingSample} links={READING_LINKS} />}
        />
        <FontGroup
          role="code"
          settings={settings}
          list={list}
          onChange={change}
          sample={<CodePanel code={FONTS.codeSample} />}
        />
        <FontGroup
          role="ui"
          settings={settings}
          list={list}
          onChange={change}
          sample={<InterfaceSample settings={settings} />}
        />
      </Page>
    </div>
  );
}
