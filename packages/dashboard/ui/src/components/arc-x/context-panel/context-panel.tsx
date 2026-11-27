import { ArrowRight } from 'lucide-react';
import { useId, type CSSProperties, type ReactElement } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import type { ContextParts } from '../../../lib/chat-types';
import { formatCount, formatDuration, formatNumber } from '../../../lib/format';
import type { SessionUsage } from '../../../lib/session-usage';
import { Button } from '../../arc/button/button';
import { ButtonLink } from '../button-link/button-link';
import { contextShare, formatTokenSize } from '../composer/context-ring';
import { KeyValue, type KeyValueItem } from '../key-value/key-value';
import { Placeholder } from '../placeholder/placeholder';
import { Meter, SplitBar, type BarSegment } from '../split-bar/split-bar';
import styles from './context-panel.module.css';

export interface ContextPanelProps {
  /** Tokens in the context now; null while unknown. */
  used: number | null;
  /** The model's window; null while unknown (no bar is drawn). */
  window: number | null;
  /** The agent's parts: undefined while they load, null when the conversation is not loaded or the read failed. */
  parts: ContextParts | null | undefined;
  usage: SessionUsage;
  compact: {
    /** Why Compact now is off (the button stays focusable and reads it); null when it may run. */
    off: string | null;
    onCompact: () => void;
  };
  /** The conversation's Sessions page. */
  inspectTo: string;
  onInspect?: () => void;
}

interface Part {
  key: 'system' | 'tools' | 'messages' | 'free';
  label: string;
  tokens: number;
  color: string;
}

const COLORS = {
  system: 'var(--model-7)',
  tools: 'var(--model-4)',
  messages: 'var(--model-1)',
  free: 'var(--surface-muted)',
} as const;

/** "62%", or "<1%" for a share above 0 that rounds to 0. */
function percentOf(share: number): string {
  return share > 0 && share < 0.01 ? CHAT.contextUnderOne : fill(CHAT.contextPercent, { n: Math.round(share * 100) });
}

/**
 * The agent's three parts, scaled to `used` when it is known (they are estimates and need not add up to it), and the
 * free rest of a known window. The bar and the list show the same numbers; scaled, the three add up to `used` exactly.
 */
export function contextParts(parts: ContextParts, used: number | null, window: number | null): Part[] {
  const raw = [
    { key: 'system' as const, label: CHAT.history.partSystem, tokens: parts.systemTokens },
    { key: 'tools' as const, label: fill(CHAT.history.partTools, { n: parts.toolCount }), tokens: parts.toolTokens },
    { key: 'messages' as const, label: CHAT.history.partMessages, tokens: parts.messageTokens },
  ];
  const sum = raw.reduce((total, part) => total + part.tokens, 0);
  const scaled = used !== null && sum > 0;
  let given = 0;
  const out: Part[] = raw.map((part, index) => {
    let tokens = part.tokens;
    if (scaled) {
      tokens = index === raw.length - 1 ? Math.max(0, used - given) : Math.round((part.tokens * used) / sum);
      given += tokens;
    }
    return { ...part, tokens, color: COLORS[part.key] };
  });
  if (window !== null && used !== null) {
    out.push({ key: 'free', label: CHAT.history.partFree, tokens: Math.max(0, window - used), color: COLORS.free });
  }
  return out;
}

/** "Compacts by itself at 85%", then how often the agent compacted in this process. */
export function autoLine(parts: ContextParts): string {
  const auto = fill(CHAT.history.contextAuto, { n: parts.autoCompactPercent });
  if (parts.compactions === 0) return auto;
  const times =
    parts.compactions === 1 ? CHAT.history.compactedOnce : fill(CHAT.history.compactedTimes, { n: parts.compactions });
  return `${auto} · ${times}`;
}

function usageItems(usage: SessionUsage): KeyValueItem[] {
  const read =
    usage.readTokens === null
      ? null
      : usage.cachedTokens !== null && usage.cachedTokens > 0
        ? fill(CHAT.history.usageReadValue, {
            n: formatCount(usage.readTokens),
            cached: formatCount(usage.cachedTokens),
          })
        : fill(CHAT.history.usageTokens, { n: formatCount(usage.readTokens) });
  return [
    { key: 'turns', label: CHAT.history.usageTurns, value: formatNumber(usage.turns), numeric: true },
    {
      key: 'calls',
      label: CHAT.history.usageCalls,
      value: usage.modelCalls === null ? null : formatNumber(usage.modelCalls),
      numeric: true,
    },
    { key: 'read', label: CHAT.history.usageRead, value: read, numeric: true },
    {
      key: 'written',
      label: CHAT.history.usageWritten,
      value:
        usage.writtenTokens === null ? null : fill(CHAT.history.usageTokens, { n: formatCount(usage.writtenTokens) }),
      numeric: true,
    },
    {
      key: 'time',
      label: CHAT.history.usageTime,
      value: usage.elapsedMs === null ? null : formatDuration(usage.elapsedMs),
      numeric: true,
    },
  ];
}

/**
 * The Context panel's body: how full the context is and what fills it (the agent's split, only while the conversation
 * is loaded), Compact now, and the conversation's totals summed from its stored turns. The parts are drawn twice, as a
 * bar whose name reads every part and as a list, so nothing depends on color.
 */
export function ContextPanel(props: ContextPanelProps): ReactElement {
  const noteId = useId();
  const { used, window, parts } = props;
  const share = used !== null && window !== null ? contextShare({ used, window }) : null;
  const head =
    used === null
      ? null
      : window === null || share === null
        ? fill(CHAT.history.usageTokens, { n: formatTokenSize(used) })
        : fill(CHAT.history.contextHead, {
            used: formatTokenSize(used),
            window: formatTokenSize(window),
            percent: percentOf(share),
          });
  const list = parts === undefined || parts === null ? null : contextParts(parts, used, window);
  const auto = parts === undefined || parts === null ? null : autoLine(parts);
  const marked =
    window !== null && used !== null && parts !== undefined && parts !== null
      ? parts.autoCompactPercent > 0 && parts.autoCompactPercent < 100
      : false;
  const segments: BarSegment[] | null =
    list === null
      ? null
      : list.map((part) => ({ key: part.key, label: part.label, value: part.tokens, color: part.color }));
  const off = props.compact.off;

  return (
    <div className={styles.root}>
      <section className={styles.section}>
        <h2 className={styles.title}>{CHAT.history.context}</h2>
        {head !== null ? <p className={styles.head}>{head}</p> : null}
        {window !== null && used !== null ? (
          segments !== null && parts !== null && parts !== undefined && auto !== null ? (
            <SplitBar
              label={CHAT.history.context}
              segments={segments}
              total={window}
              legend={false}
              formatValue={formatTokenSize}
              threshold={{ value: (window * parts.autoCompactPercent) / 100, label: auto }}
            />
          ) : (
            <Meter label={CHAT.history.context} value={used} max={window} formatValue={formatTokenSize} />
          )
        ) : null}
        {parts === undefined ? (
          <div className={styles.loading} aria-busy="true">
            {[0, 1, 2].map((step) => (
              <Placeholder key={step} width={step === 2 ? '60%' : '100%'} />
            ))}
          </div>
        ) : list === null ? (
          <p className={styles.note}>{CHAT.history.contextCold}</p>
        ) : (
          <>
            <dl className={styles.parts}>
              {list.map((part) => (
                <div key={part.key} className={styles.part}>
                  <dt className={styles.partName}>
                    <span
                      className={styles.key}
                      style={{ '--c': part.color } as CSSProperties}
                      data-free={part.key === 'free' ? '' : undefined}
                      aria-hidden="true"
                    />
                    {part.label}
                  </dt>
                  <dd className={styles.partValue}>{formatTokenSize(part.tokens)}</dd>
                </div>
              ))}
            </dl>
            {marked ? null : <p className={styles.note}>{auto}</p>}
          </>
        )}
        <div className={styles.actions}>
          {off !== null ? (
            <span id={noteId} className={styles.note}>
              {off}
            </span>
          ) : null}
          <Button
            variant="secondary"
            size="sm"
            className={styles.compact}
            aria-disabled={off !== null || undefined}
            aria-describedby={off !== null ? noteId : undefined}
            onClick={() => {
              if (off === null) props.compact.onCompact();
            }}
          >
            {CHAT.history.compact}
          </Button>
        </div>
      </section>
      <section className={styles.section}>
        <h2 className={styles.title}>{CHAT.history.usage}</h2>
        <KeyValue layout="rows" items={usageItems(props.usage)} ariaLabel={CHAT.history.usage} />
        <p className={styles.note}>{CHAT.history.usageLocal}</p>
        <ButtonLink to={props.inspectTo} variant="ghost" size="sm" className={styles.inspect} onClick={props.onInspect}>
          {CHAT.inspect}
          <ArrowRight size={14} strokeWidth={1.75} aria-hidden="true" />
        </ButtonLink>
      </section>
    </div>
  );
}
