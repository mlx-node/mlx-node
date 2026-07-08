import { Link } from '@tanstack/react-router';
import * as React from 'react';

import { useHref, useLocale } from '../../lib/i18n-react';
import type { SectionMeta } from '../chapters';
import { localizedSection } from '../i18n/localized';
import { useUiStrings } from '../i18n/ui-react';

// SubChapterNav — an in-chapter "go deeper" card that links out to a complex
// chapter's sub-chapters, each served as its own sub-route
// (/chapters/<chapterId>/<sectionId>, locale-prefixed under /zh).
//
// Replaces the old inline collapsible (<SubChapterIndex> + <SubChapter>): the
// deep-dives are now real pages (own URL, prerendered, SEO-indexed, nested in
// the sidebar). This renders TanStack <Link>s, so it is client-side nav in the
// SPA and a real crawlable <a href> in the prerendered HTML. The chapter
// bodies pass the ENGLISH registry sections; display strings are re-resolved
// through the locale overlay here so the card reads localized even when the
// surrounding body is the English fallback.

export function SubChapterNav({
  chapterId,
  items,
  eyebrow,
}: {
  chapterId: string;
  items: SectionMeta[] | undefined;
  eyebrow?: string;
}) {
  const ui = useUiStrings();
  const locale = useLocale();
  if (!items || items.length === 0) return null;
  return (
    <nav className="not-prose my-5 rounded-md border border-border bg-background/50 p-4">
      <div className="mb-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {eyebrow ?? ui.scaffolding.goDeeper}
      </div>
      <ul className="space-y-1.5">
        {items.map((s) => {
          const display = localizedSection(locale, chapterId, s.id) ?? s;
          return <SectionLink key={s.id} chapterId={chapterId} section={display} />;
        })}
      </ul>
    </nav>
  );
}

function SectionLink({ chapterId, section }: { chapterId: string; section: SectionMeta }) {
  // Concrete locale-prefixed href; the cast is the project-standard typing
  // concession for computed `to` values (see lib/locale-navigate.ts).
  const href = useHref(`/chapters/${chapterId}/${section.id}`);
  return (
    <li>
      <Link
        to={href as never}
        search={(prev: unknown) => prev as never}
        className="group flex items-baseline gap-2 text-sm text-foreground/90 hover:text-primary"
      >
        <span aria-hidden="true" className="text-muted-foreground transition-transform group-hover:translate-x-0.5">
          ↳
        </span>
        <span>
          <span className="font-medium underline-offset-2 group-hover:underline">{section.title}</span>
          <span className="text-muted-foreground"> — {section.blurb}</span>
        </span>
      </Link>
    </li>
  );
}
