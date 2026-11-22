import { type CSSProperties, type ReactElement, useEffect, useMemo, useRef } from 'react';

import { cn } from '../../../lib/cn';
import { formatPercent } from '../../../lib/format';
import type { ClairicSpan, MediaAttachment } from '../../../lib/system-one';
import { EmptyState } from '../../arc/empty-state/empty-state';
import { Tooltip } from '../../arc/tooltip/tooltip';
import styles from './span-highlighter.module.css';

export interface HighlightSpanEntry {
  field: string;
  color: string;
  confidence?: number;
  spans: readonly ClairicSpan[];
}

export interface SpanHighlighterProps {
  text: string;
  entries: readonly HighlightSpanEntry[];
  focusedField: string | null;
  onFocusField: (field: string | null) => void;
  attachments?: readonly MediaAttachment[];
  className?: string;
}

interface TextSegment {
  key: string;
  text: string;
  hits: HighlightSpanEntry[];
  start: number;
  end: number;
}

/**
 * Code-point safe segmentation of `text` over potentially overlapping `[start, end)` spans.
 * Uses `Array.from(text)` so multi-byte characters / emoji never shift Clairic's character offsets.
 */
function buildSegments(text: string, entries: readonly HighlightSpanEntry[]): TextSegment[] {
  const chars = Array.from(text);
  const len = chars.length;
  if (len === 0) return [];

  const cuts = new Set<number>([0, len]);
  const normalized: Array<{ entry: HighlightSpanEntry; start: number; end: number }> = [];

  for (const entry of entries) {
    for (const [rawStart, rawEnd] of entry.spans) {
      const start = Math.max(0, Math.min(len, rawStart));
      const end = Math.max(start, Math.min(len, rawEnd));
      if (end <= start) continue;
      cuts.add(start);
      cuts.add(end);
      normalized.push({ entry, start, end });
    }
  }

  const sorted = [...cuts].sort((a, b) => a - b);
  const segments: TextSegment[] = [];

  for (let i = 0; i < sorted.length - 1; i += 1) {
    const start = sorted[i]!;
    const end = sorted[i + 1]!;
    if (end <= start) continue;
    const hits: HighlightSpanEntry[] = [];
    for (const span of normalized) {
      if (span.start <= start && span.end >= end) {
        if (!hits.some((h) => h.field === span.entry.field)) {
          hits.push(span.entry);
        }
      }
    }
    segments.push({
      key: `${String(start)}-${String(end)}`,
      text: chars.slice(start, end).join(''),
      hits,
      start,
      end,
    });
  }

  return segments;
}

/** Interactive evidence view with color-coded span highlights, field filter legend, and optional attached images. */
export function SpanHighlighter({
  text,
  entries,
  focusedField,
  onFocusField,
  attachments = [],
  className,
}: SpanHighlighterProps): ReactElement {
  const containerRef = useRef<HTMLPreElement | null>(null);
  const withSpans = useMemo(() => entries.filter((e) => e.spans.length > 0), [entries]);
  const segments = useMemo(() => buildSegments(text, withSpans), [text, withSpans]);
  const imageAttachments = attachments.filter((a) => a.kind === 'image');

  useEffect(() => {
    if (focusedField === null || containerRef.current === null) return;
    const mark = containerRef.current.querySelector<HTMLElement>('[data-focused="true"]');
    mark?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [focusedField]);

  if (text.trim() === '' && imageAttachments.length === 0) {
    return (
      <EmptyState
        title="No highlights yet"
        description="Run an extraction with text input to see where each answer came from in your document."
      />
    );
  }

  return (
    <div className={cn(styles.root, className)}>
      {withSpans.length > 0 ? (
        <div className={styles.legend} role="toolbar" aria-label="Filter highlights by field">
          <button
            type="button"
            className={styles.legendBtn}
            data-active={focusedField === null ? 'true' : undefined}
            onClick={() => onFocusField(null)}
          >
            All highlights ({withSpans.length})
          </button>
          {withSpans.map((entry) => {
            const active = focusedField === entry.field;
            return (
              <button
                key={entry.field}
                type="button"
                className={styles.legendBtn}
                style={{ '--c': entry.color } as CSSProperties}
                data-active={active ? 'true' : undefined}
                onClick={() => onFocusField(active ? null : entry.field)}
              >
                <span className={styles.swatch} aria-hidden="true" />
                <span>{entry.field}</span>
              </button>
            );
          })}
        </div>
      ) : null}

      <div className={styles.split} data-has-media={imageAttachments.length > 0 ? 'true' : undefined}>
        <pre ref={containerRef} className={styles.textBox} aria-label="Source text with highlighted spans">
          {segments.map((seg) => {
            if (seg.hits.length === 0) return <span key={seg.key}>{seg.text}</span>;
            const focusedHit = focusedField !== null ? seg.hits.find((h) => h.field === focusedField) : undefined;
            const primary = focusedHit ?? seg.hits[0]!;
            const isDim = focusedField !== null && focusedHit === undefined;
            const isFocused = focusedField !== null && focusedHit !== undefined;
            const tipText = seg.hits
              .map((h) =>
                h.confidence !== undefined
                  ? `${h.field} · ${formatPercent(h.confidence)} [${String(seg.start)}, ${String(seg.end)})`
                  : `${h.field} [${String(seg.start)}, ${String(seg.end)})`,
              )
              .join(' | ');
            return (
              <Tooltip key={seg.key} content={tipText}>
                <mark
                  className={styles.mark}
                  style={{ '--c': primary.color } as CSSProperties}
                  data-dim={isDim ? 'true' : undefined}
                  data-focused={isFocused ? 'true' : undefined}
                  onClick={() => onFocusField(focusedField === primary.field ? null : primary.field)}
                >
                  {seg.text}
                </mark>
              </Tooltip>
            );
          })}
        </pre>

        {imageAttachments.length > 0 ? (
          <div className={styles.mediaCol} aria-label="Attached evidence images">
            {imageAttachments.map((img) => (
              <figure key={img.id} className={styles.mediaFrame}>
                <div className={styles.mediaImgWrap}>
                  <img
                    src={img.dataUrl}
                    alt={img.name}
                    className={styles.mediaImg}
                    style={{ transform: `rotate(${String(img.rotation)}deg)` }}
                  />
                </div>
                <figcaption className={styles.mediaCaption}>
                  {img.name}
                  {img.rotation !== 0 ? ` · ${String(img.rotation)}°` : ''}
                </figcaption>
              </figure>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
