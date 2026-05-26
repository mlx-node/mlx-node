import * as React from "react";
import {
  ArrowLeftIcon,
  CheckIcon,
  MessageSquareIcon,
  PlayIcon,
} from "lucide-react";

import { Button } from "../components/ui/button";
import { CHAPTERS, type ChapterMeta } from "./chapters";

export type LessonLayoutProps = {
  /** The chapter currently being viewed. */
  current: ChapterMeta;
  /** The main reading column — typically a <Prose> with chapter text. */
  children: React.ReactNode;
  /** The right-hand "Try it now" panel content. */
  tryItPanel: React.ReactNode;
  onOpenChapter: (chapterId: string) => void;
  onBackToIndex: () => void;
  onOpenFreeChat: () => void;
};

export function LessonLayout({
  current,
  children,
  tryItPanel,
  onOpenChapter,
  onBackToIndex,
  onOpenFreeChat,
}: LessonLayoutProps) {
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
        <Button
          variant="ghost"
          size="sm"
          onClick={onOpenFreeChat}
          className="gap-2"
        >
          <MessageSquareIcon className="size-4" />
          Free chat
        </Button>
      </div>

      {/* Three-column body: sidebar | prose | try-it-now */}
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)_minmax(0,1fr)]">
        <aside className="hidden border-r border-border lg:block">
          <ChapterSidebar
            currentId={current.id}
            onOpenChapter={onOpenChapter}
          />
        </aside>

        <main className="min-h-0 overflow-y-auto px-8 py-10">{children}</main>

        <section className="min-h-0 overflow-y-auto border-l border-border bg-card/30 p-6">
          <div className="mb-4 flex items-center gap-2">
            <PlayIcon className="size-4 text-primary" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-foreground">
              Try it now
            </h2>
          </div>
          {tryItPanel}
        </section>
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
      <div className="mb-2 px-2 font-mono text-xs uppercase tracking-[0.15em] text-muted-foreground">
        Chapters
      </div>
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

function SidebarRow({
  chapter,
  active,
  onOpen,
}: {
  chapter: ChapterMeta;
  active: boolean;
  onOpen: () => void;
}) {
  const disabled = !chapter.available;
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onOpen}
      className={[
        "flex items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors",
        active
          ? "bg-primary/10 text-primary"
          : disabled
            ? "text-muted-foreground/60"
            : "text-foreground/80 hover:bg-accent/40 hover:text-foreground",
      ].join(" ")}
    >
      <span className="w-5 shrink-0 text-center font-mono text-[0.7rem] text-muted-foreground">
        {chapter.number}
      </span>
      <span className="flex-1 truncate">{chapter.title}</span>
      {disabled ? (
        <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground/60">
          soon
        </span>
      ) : active ? (
        <CheckIcon className="size-3.5 text-primary" />
      ) : null}
    </button>
  );
}
