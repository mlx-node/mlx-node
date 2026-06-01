import { ArrowLeftIcon, CheckIcon, MessageSquareIcon, PlayIcon } from 'lucide-react';
import * as React from 'react';

import { Button } from '../components/ui/button';
import { CHAPTERS, type ChapterMeta } from './chapters';

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
  onOpenChapter: (chapterId: string) => void;
  onBackToIndex: () => void;
  onOpenFreeChat: () => void;
};

export function LessonLayout({
  current,
  children,
  tryItPanel,
  wideBody,
  onOpenChapter,
  onBackToIndex,
  onOpenFreeChat,
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
  }, [current.id]);

  const hasTryIt = Boolean(tryItPanel);

  return (
    <div className="absolute inset-0 z-10 flex flex-col bg-background">
      {/* Header bar */}
      <div className="flex shrink-0 items-center justify-between border-b border-border px-6 py-3">
        <button
          type="button"
          onClick={onBackToIndex}
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeftIcon className="size-4" />
          All chapters
        </button>
        <div className="font-mono text-xs uppercase tracking-[0.15em] text-muted-foreground">
          Chapter {current.number} · {current.title}
        </div>
        <Button variant="ghost" size="sm" onClick={onOpenFreeChat} className="gap-2">
          <MessageSquareIcon className="size-4" />
          Free chat
        </Button>
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
          <ChapterSidebar currentId={current.id} onOpenChapter={onOpenChapter} />
        </aside>

        <main ref={mainRef} className="min-h-0 overflow-y-auto px-8 py-10">
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
          <section ref={tryItRef} className="min-h-0 overflow-y-auto border-l border-border bg-card/30 p-6">
            <div className="mb-4 flex items-center gap-2">
              <PlayIcon className="size-4 text-primary" />
              <h2 className="text-sm font-semibold uppercase tracking-wider text-foreground">Try it now</h2>
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
  onOpenChapter,
}: {
  currentId: string;
  onOpenChapter: (chapterId: string) => void;
}) {
  return (
    <nav className="flex h-full flex-col gap-1 overflow-y-auto p-4">
      <div className="mb-2 px-2 font-mono text-xs uppercase tracking-[0.15em] text-muted-foreground">Chapters</div>
      {CHAPTERS.map((c) => (
        <SidebarRow
          key={c.id}
          chapter={c}
          active={c.id === currentId}
          onOpen={() => (c.available ? onOpenChapter(c.id) : undefined)}
        />
      ))}
    </nav>
  );
}

function SidebarRow({ chapter, active, onOpen }: { chapter: ChapterMeta; active: boolean; onOpen: () => void }) {
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
        <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground/60">soon</span>
      ) : active ? (
        <CheckIcon className="size-3.5 text-primary" />
      ) : null}
    </button>
  );
}
