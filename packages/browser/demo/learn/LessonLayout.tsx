import { ArrowLeftIcon, CheckIcon, DownloadIcon, MessageSquareIcon, PlayIcon } from 'lucide-react';
import * as React from 'react';

import { LanguageSwitcher } from '../components/LanguageSwitcher';
import { Button } from '../components/ui/button';
import { useLocale } from '../lib/i18n-react';
import type { ChapterMeta, SectionMeta } from './chapters';
import { localizedChapters } from './i18n/localized';
import { useUiStrings } from './i18n/ui-react';

export type LessonLayoutProps = {
  /** The chapter currently being viewed. */
  current: ChapterMeta;
  /** The main reading column — typically a <Prose> with chapter text. */
  children: React.ReactNode;
  /**
   * The right-hand "Try it now" panel content. When falsy, the third column is
   * dropped entirely and the reading column widens to fill the space — used by
   * chapters whose interactive content lives inline in the body (e.g. the
   * full-width architecture poster).
   */
  tryItPanel?: React.ReactNode;
  /**
   * Only relevant for panel-less chapters. By default a panel-less chapter is a
   * pure-reading lesson, so its body is centered at a comfortable article width.
   * Set this when the body instead needs the full remaining width (e.g. the
   * architecture poster, which is ~1280px wide). Ignored when `tryItPanel` is set.
   */
  wideBody?: boolean;
  /**
   * When viewing a sub-chapter, the id of the active section. Highlights the
   * nested sidebar row and drives the header breadcrumb. Omit on a chapter page.
   */
  currentSectionId?: string;
  /** The active sub-chapter's title — appended to the header breadcrumb. */
  sectionTitle?: string;
  onOpenChapter: (chapterId: string) => void;
  /** Navigate to a sub-chapter (nested sidebar rows + in-chapter "go deeper"). */
  onOpenSection?: (chapterId: string, sectionId: string) => void;
  onBackToIndex: () => void;
  onOpenFreeChat: () => void;
  /**
   * Global "Load model" affordance shown in the header so a learner can
   * pre-load from any chapter. Passed in by the route (which owns the model
   * loader) so this layout stays presentational and model-free. When the model
   * is already loaded (`modelReady`), the control is hidden. Omit `onLoadModel`
   * to drop the control entirely.
   */
  modelReady?: boolean;
  onLoadModel?: () => void;
};

export function LessonLayout({
  current,
  children,
  tryItPanel,
  wideBody,
  currentSectionId,
  sectionTitle,
  onOpenChapter,
  onOpenSection,
  onBackToIndex,
  onOpenFreeChat,
  modelReady = false,
  onLoadModel,
}: LessonLayoutProps) {
  // Per-chapter scroll reset.
  //
  // The two scrollable panes (`<main>` prose + `<section>` try-it) live
  // inside this layout and persist across chapter swaps — TanStack Router
  // keeps the layout mounted and just changes the children when the chapter
  // id segment changes. Without this effect, scrolling halfway down ch 5
  // and clicking ch 6 leaves the new chapter scrolled half-way too, which
  // looks like a bug. Reset both panes to the top whenever `current.id`
  // changes. The sidebar is intentionally not reset — a learner may have
  // scrolled it to see a far-away chapter and we shouldn't snap them back.
  const mainRef = React.useRef<HTMLElement>(null);
  const tryItRef = React.useRef<HTMLElement>(null);
  React.useEffect(() => {
    mainRef.current?.scrollTo({ top: 0, behavior: 'instant' });
    tryItRef.current?.scrollTo({ top: 0, behavior: 'instant' });
    // Also reset when moving between a chapter and one of its sub-chapters (the
    // chapter id is unchanged across that hop, so key on the section too).
  }, [current.id, currentSectionId]);

  const hasTryIt = Boolean(tryItPanel);
  const ui = useUiStrings();

  return (
    <div className="absolute inset-0 z-10 flex flex-col bg-background">
      {/* Header bar */}
      {/* Header bar. On mobile the three groups (back · title · actions) can't
          all fit at full size, so the back-link and the action buttons collapse
          to icon-only (labels return at >=sm) and the breadcrumb truncates in a
          flexible middle — nothing clips off-screen. */}
      <div className="flex shrink-0 items-center gap-2 border-b border-border px-3 py-2.5 sm:gap-3 sm:px-6 sm:py-3">
        <button
          type="button"
          onClick={onBackToIndex}
          // Keep an accessible name at every width: the label is `sr-only` below
          // `sm` (screen readers still announce it) and visible from `sm` up. Below
          // `sm` the label leaves flow, so the icon alone would give a tiny hit box;
          // `h-8 px-2` floors it to a real ~32x32 target matching the neighboring
          // `size="sm"` Buttons (also `h-8`), and `-mx-2` outsets that padding as
          // pure hit area so the icon's position and the row height are unchanged.
          className="-mx-2 inline-flex h-11 shrink-0 items-center gap-2 px-2 text-sm text-muted-foreground hover:text-foreground sm:h-8"
        >
          <ArrowLeftIcon className="size-4" />
          <span className="sr-only sm:not-sr-only">{ui.lessonLayout.allChapters}</span>
        </button>
        <div className="min-w-0 flex-1 truncate text-center font-mono text-[0.65rem] uppercase tracking-[0.1em] text-muted-foreground sm:text-xs sm:tracking-[0.15em]">
          {sectionTitle ? (
            <>
              {/* In a sub-chapter the full "Chapter N · Title › Section" crumb
                  truncates to "Chapter N · Ti…" on a phone, dropping the section's
                  own identity. Below sm, show just the section title (its most
                  specific name); the full breadcrumb returns from sm up. */}
              <span className="hidden sm:inline">
                {ui.lessonLayout.breadcrumb(current.number, current.title)}
              </span>
              <span className="hidden text-foreground/80 sm:inline"> › </span>
              <span className="text-foreground/80">{sectionTitle}</span>
            </>
          ) : (
            ui.lessonLayout.breadcrumb(current.number, current.title)
          )}
        </div>
        <div className="flex shrink-0 items-center gap-1 sm:gap-2">
          {/* Global pre-load affordance. Hidden once the model is ready (there
              is nothing left to load); omit `onLoadModel` to drop it entirely. */}
          {/* Hidden on mobile: the interactive "Try it now" panel (and the chat
              overlay) each carry their own load CTA there, so the header stays
              uncluttered and the breadcrumb keeps its width. */}
          {onLoadModel && !modelReady ? (
            <Button variant="outline" size="sm" onClick={onLoadModel} className="hidden gap-2 sm:inline-flex">
              <DownloadIcon className="size-4" />
              <span className="sr-only sm:not-sr-only">{ui.lessonLayout.loadModel}</span>
            </Button>
          ) : null}
          <Button
            variant="ghost"
            size="sm"
            onClick={onOpenFreeChat}
            className="h-11 w-11 gap-2 sm:h-8 sm:w-auto"
          >
            <MessageSquareIcon className="size-4" />
            <span className="sr-only sm:not-sr-only">{ui.lessonLayout.freeChat}</span>
          </Button>
          <LanguageSwitcher />
        </div>
      </div>

      {/* Body: sidebar | prose | (optional) try-it-now. When the chapter has no
          try-it panel, the reading column spans the full remaining width. */}
      <div
        className={[
          'grid min-h-0 flex-1 grid-cols-1',
          hasTryIt ? 'lg:grid-cols-[220px_minmax(0,1fr)_minmax(0,1fr)]' : 'lg:grid-cols-[220px_minmax(0,1fr)]',
        ].join(' ')}
      >
        <aside className="hidden border-r border-border lg:block">
          <ChapterSidebar
            currentId={current.id}
            currentSectionId={currentSectionId}
            onOpenChapter={onOpenChapter}
            onOpenSection={onOpenSection}
          />
        </aside>

        <main ref={mainRef} className="min-h-0 overflow-y-auto px-5 py-8 sm:px-8 sm:py-10">
          {/* Chapters with a try-it panel keep their natural (narrow) reading
              column from the 3-col grid. Panel-less chapters span the whole
              remaining width, so we cap and center their body: pure-reading
              lessons (overview, post-training) sit at a comfortable article
              width, while a `wideBody` chapter (the ~1280px architecture
              poster) gets the full poster width. Without this cap a reading
              lesson's narrow left-aligned prose strands in the left third with
              a large empty right half. */}
          {hasTryIt ? (
            children
          ) : (
            <div className={['mx-auto w-full', wideBody ? 'max-w-[1400px]' : 'max-w-3xl'].join(' ')}>{children}</div>
          )}
        </main>

        {hasTryIt ? (
          <section
            ref={tryItRef}
            className="min-h-0 overflow-y-auto border-t border-border bg-card/30 p-6 lg:border-t-0 lg:border-l"
          >
            <div className="mb-4 flex items-center gap-2">
              <PlayIcon className="size-4 text-primary" />
              <h2 className="text-sm font-semibold uppercase tracking-wider text-foreground">
                {ui.lessonLayout.tryItNow}
              </h2>
            </div>
            {tryItPanel}
          </section>
        ) : null}
      </div>
    </div>
  );
}

function ChapterSidebar({
  currentId,
  currentSectionId,
  onOpenChapter,
  onOpenSection,
}: {
  currentId: string;
  currentSectionId?: string;
  onOpenChapter: (chapterId: string) => void;
  onOpenSection?: (chapterId: string, sectionId: string) => void;
}) {
  const ui = useUiStrings();
  // Locale overlay applied to the registry — English passes through untouched.
  const chapters = localizedChapters(useLocale());
  return (
    <nav className="flex h-full flex-col gap-1 overflow-y-auto p-4">
      <div className="mb-2 px-2 font-mono text-xs uppercase tracking-[0.15em] text-muted-foreground">
        {ui.lessonLayout.chaptersHeading}
      </div>
      {chapters.map((c) => {
        const active = c.id === currentId;
        return (
          <React.Fragment key={c.id}>
            <SidebarRow chapter={c} active={active} onOpen={() => (c.available ? onOpenChapter(c.id) : undefined)} />
            {/* Sub-chapters expand only under the ACTIVE chapter, so the sidebar
                stays scannable. Each row deep-links to its own sub-route. */}
            {active && c.sections && c.sections.length > 0 ? (
              <div className="mb-1 ml-[1.05rem] flex flex-col gap-0.5 border-l border-border pl-2">
                {c.sections.map((s) => (
                  <SectionRow
                    key={s.id}
                    section={s}
                    active={s.id === currentSectionId}
                    onOpen={() => onOpenSection?.(c.id, s.id)}
                  />
                ))}
              </div>
            ) : null}
          </React.Fragment>
        );
      })}
    </nav>
  );
}

function SectionRow({
  section,
  active,
  onOpen,
}: {
  section: SectionMeta;
  active: boolean;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      title={section.blurb}
      className={[
        'flex items-center gap-2 rounded-md px-2 py-1 text-left text-[0.8rem] transition-colors',
        active ? 'bg-primary/10 text-primary' : 'text-foreground/70 hover:bg-accent/40 hover:text-foreground',
      ].join(' ')}
    >
      <span aria-hidden="true" className="shrink-0 text-muted-foreground">
        ↳
      </span>
      <span className="flex-1 truncate">{section.title}</span>
    </button>
  );
}

function SidebarRow({ chapter, active, onOpen }: { chapter: ChapterMeta; active: boolean; onOpen: () => void }) {
  const ui = useUiStrings();
  const disabled = !chapter.available;
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onOpen}
      className={[
        'flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors',
        active
          ? 'bg-primary/10 text-primary'
          : disabled
            ? 'text-muted-foreground/60'
            : 'text-foreground/80 hover:bg-accent/40 hover:text-foreground',
      ].join(' ')}
    >
      <span className="w-5 shrink-0 text-center font-mono text-[0.7rem] text-muted-foreground">{chapter.number}</span>
      <span className="flex-1 truncate">{chapter.title}</span>
      {disabled ? (
        <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground/60">
          {ui.lessonLayout.soonBadge}
        </span>
      ) : active ? (
        <CheckIcon className="size-3.5 text-primary" />
      ) : null}
    </button>
  );
}
