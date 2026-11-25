/**
 * The states of the System One Output card that have no answers to show: a run in flight, a failed run, and a run
 * with no text for the Highlights view. Each one must say what it is, not show an empty box or old answers.
 */

import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vite-plus/test';

import {
  type HighlightSpanEntry,
  SpanHighlighter,
} from '../ui/src/components/arc-x/span-highlighter/span-highlighter.js';
import { NO_VALUE } from '../ui/src/lib/format.js';
import { DEFAULT_DECISION_POLICY, type MediaAttachment, SYSTEM_ONE_PRESETS } from '../ui/src/lib/system-one.js';
import { ResultsView, type ResultsViewProps } from '../ui/src/pages/system-one/results-view.js';

const FIELDS = { vendor: { type: 'text' }, total: { type: 'number' } };

function results(props: Partial<ResultsViewProps>): string {
  return renderToStaticMarkup(
    createElement(ResultsView, {
      result: undefined,
      pendingNames: ['vendor', 'total'],
      requestFields: FIELDS,
      mode: 'extract',
      presets: SYSTEM_ONE_PRESETS,
      canRun: true,
      onRun: () => {},
      onPickPreset: () => {},
      policy: DEFAULT_DECISION_POLICY,
      onPolicyChange: () => {},
      ...props,
    }),
  );
}

function count(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

describe('ResultsView with no answers', () => {
  it('shows the examples before the first run', () => {
    const html = results({});
    expect(html).toContain('Try an example');
    expect(html).not.toContain('Routing policy');
  });

  it('lists the requested fields as loading rows while a run is in flight', () => {
    const html = results({ pending: true });
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain('vendor');
    expect(html).toContain('total');
    expect(html).not.toContain(NO_VALUE);
    expect(html).not.toContain('Try an example');
  });

  it('keeps the rows with no value after a failed run', () => {
    const html = results({ failed: true });
    expect(html).toContain('vendor');
    expect(html).toContain('total');
    expect(count(html, `>${NO_VALUE}<`)).toBe(2);
    expect(html).not.toContain('aria-busy');
    expect(html).not.toContain('Try an example');
  });

  it('holds the policy control of Extract in place before the answers come, and not for Decide', () => {
    expect(results({ pending: true })).toContain('Routing policy');
    expect(results({ failed: true })).toContain('Routing policy');
    expect(results({ pending: true, mode: 'decide' })).not.toContain('Routing policy');
  });
});

const IMAGE: MediaAttachment = {
  id: 'a1',
  name: 'receipt.png',
  contentType: 'image/png',
  sizeBytes: 4,
  base64: 'AAAA',
  dataUrl: 'data:image/png;base64,AAAA',
  rotation: 0,
  kind: 'image',
};

function highlights(props: {
  text: string;
  hasResult: boolean;
  entries?: readonly HighlightSpanEntry[];
  attachments?: readonly MediaAttachment[];
}): string {
  return renderToStaticMarkup(
    createElement(SpanHighlighter, {
      entries: [],
      focusedField: null,
      onFocusField: () => {},
      ...props,
    }),
  );
}

describe('SpanHighlighter with nothing to mark', () => {
  it('shows the empty state before the first run, also when draft text exists', () => {
    const html = highlights({ text: 'draft text', hasResult: false });
    expect(html).toContain('No highlights yet');
    expect(html).not.toContain('draft text');
  });

  it('says that a run with only an image has no text, and shows the image', () => {
    const html = highlights({ text: '', hasResult: true, attachments: [IMAGE] });
    expect(html).toContain('No text to highlight.');
    expect(html).toContain('data-large="true"');
    expect(html).toContain('alt="receipt.png"');
    expect(html).not.toContain('<pre');
  });

  it('shows an empty state for a run with no text and no image to show', () => {
    const html = highlights({ text: '', hasResult: true });
    expect(html).toContain('No text to highlight');
    expect(html).not.toContain('<pre');
  });

  it('says why a text run has no highlight', () => {
    const entries: HighlightSpanEntry[] = [{ field: 'vendor', color: 'var(--model-1)', spans: [] }];
    const html = highlights({ text: 'INVOICE 42', hasResult: true, entries });
    expect(html).toContain('No answer has a highlight.');
    expect(html).toContain('INVOICE 42');
    expect(html).not.toContain('only in the image');
    expect(highlights({ text: 'INVOICE 42', hasResult: true, entries, attachments: [IMAGE] })).toContain(
      'only in the image',
    );
  });
});
