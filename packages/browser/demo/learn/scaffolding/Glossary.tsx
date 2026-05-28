import * as React from 'react';

import type { GlossaryTerm } from './learning-data';

export type GlossaryProps = {
  terms: GlossaryTerm[];
};

/**
 * Collapsible glossary built on native <details> (no shadcn accordion in the
 * project). Default closed; opens to a two-column term/definition list.
 *
 * Renders nothing when `terms` is empty so chapters with no glossary aren't
 * forced to show a dangling header.
 */
export function Glossary({ terms }: GlossaryProps) {
  if (terms.length === 0) return null;
  return (
    <details className="group rounded-lg border border-border bg-card/40 px-4 py-3 text-sm">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-foreground/90 [&::-webkit-details-marker]:hidden">
        <span className="font-medium">
          Glossary <span className="text-muted-foreground">·</span>{' '}
          <span className="text-muted-foreground">
            {terms.length} term{terms.length === 1 ? '' : 's'}
          </span>
        </span>
        <span aria-hidden="true" className="text-xs text-muted-foreground transition-transform group-open:rotate-90">
          ▸
        </span>
      </summary>
      <dl className="mt-3 grid grid-cols-[minmax(0,9rem)_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
        {terms.map((t) => (
          <React.Fragment key={t.term}>
            <dt className="font-mono text-[13px] text-foreground">{t.term}</dt>
            <dd className="text-foreground/80">{t.definition}</dd>
          </React.Fragment>
        ))}
      </dl>
    </details>
  );
}
