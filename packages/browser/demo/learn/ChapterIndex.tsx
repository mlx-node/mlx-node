import { ArrowLeftIcon, MessageSquareIcon } from "lucide-react";

import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { CHAPTERS, type ChapterMeta } from "./chapters";

export type ChapterIndexProps = {
  onOpenChapter: (chapterId: string) => void;
  onBackToLanding: () => void;
  onOpenFreeChat: () => void;
};

// Curriculum phase bands. Each chapter belongs to exactly one phase; the
// CurriculumMap colours its pill by phase to communicate the conceptual
// grouping (text in → architecture → inference/deployment).
type Phase = "text-in" | "architecture" | "inference";

const PHASE_BY_NUMBER: Record<number, Phase> = {
  1: "text-in",
  2: "text-in",
  3: "architecture",
  4: "architecture",
  5: "architecture",
  6: "architecture",
  7: "architecture",
  8: "architecture",
  9: "inference",
  10: "inference",
};

// Soft tints — kept light so the chapter number/title stays readable in both
// themes. Tailwind palette indices are picked to match the Engineering
// Takeaways amber accent in chapter 9/10 territory.
const PHASE_STYLES: Record<
  Phase,
  { bg: string; border: string; text: string }
> = {
  "text-in": {
    bg: "bg-sky-100 dark:bg-sky-950/30",
    border: "border-sky-300/60 dark:border-sky-800/60",
    text: "text-sky-900 dark:text-sky-100",
  },
  architecture: {
    bg: "bg-emerald-100 dark:bg-emerald-950/30",
    border: "border-emerald-300/60 dark:border-emerald-800/60",
    text: "text-emerald-900 dark:text-emerald-100",
  },
  inference: {
    bg: "bg-amber-100 dark:bg-amber-950/30",
    border: "border-amber-300/60 dark:border-amber-800/60",
    text: "text-amber-900 dark:text-amber-100",
  },
};

export function ChapterIndex({
  onOpenChapter,
  onBackToLanding,
  onOpenFreeChat,
}: ChapterIndexProps) {
  return (
    <div className="absolute inset-0 z-10 overflow-y-auto bg-background">
      <div className="mx-auto flex w-full max-w-5xl flex-col px-6 py-10">
        {/* Top bar */}
        <div className="mb-8 flex items-center justify-between">
          <button
            type="button"
            onClick={onBackToLanding}
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
          >
            <ArrowLeftIcon className="size-4" />
            Back
          </button>
          <Button
            variant="outline"
            size="sm"
            onClick={onOpenFreeChat}
            className="gap-2"
          >
            <MessageSquareIcon className="size-4" />
            Open free chat
          </Button>
        </div>

        {/* Title */}
        <div className="mb-8">
          <div className="mb-2 font-mono text-xs uppercase tracking-[0.2em] text-primary">
            Learn LLMs · powered by Qwen3.5 in your browser
          </div>
          <h1 className="text-4xl font-semibold tracking-tight text-foreground">
            Chapters
          </h1>
          <p className="mt-3 max-w-2xl text-muted-foreground">
            Ten guided lessons that explain how a modern transformer LLM works,
            using the real model running in your browser via WebGPU. Read the
            prose on the left, then poke at the live model on the right.
          </p>
        </div>

        {/* Curriculum dependency map */}
        <CurriculumMap onOpenChapter={onOpenChapter} />

        {/* Grid of chapter cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {CHAPTERS.map((chapter) => (
            <ChapterCard
              key={chapter.id}
              chapter={chapter}
              onOpen={() =>
                chapter.available ? onOpenChapter(chapter.id) : undefined
              }
            />
          ))}
        </div>

        <p className="mt-10 text-xs text-muted-foreground">
          Each chapter is self-contained, but the suggested order builds the
          model from the bottom up — text in, attention through the middle,
          sampling out.
        </p>
      </div>
    </div>
  );
}

/**
 * Horizontal flow of 10 chapter pills connected by arrows. Each pill is a
 * button that opens that chapter. Pills are tinted by curriculum phase
 * (text-in / architecture / inference) so the user can see the conceptual
 * arc at a glance. Wraps on narrow viewports.
 */
function CurriculumMap({
  onOpenChapter,
}: {
  onOpenChapter: (chapterId: string) => void;
}) {
  return (
    <div className="mb-8">
      <div className="mb-2 font-mono text-xs uppercase tracking-[0.15em] text-muted-foreground">
        Suggested order
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {CHAPTERS.map((chapter, i) => {
          const phase = PHASE_BY_NUMBER[chapter.number] ?? "architecture";
          const style = PHASE_STYLES[phase];
          const isLast = i === CHAPTERS.length - 1;
          return (
            <div key={chapter.id} className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={() => onOpenChapter(chapter.id)}
                disabled={!chapter.available}
                className={[
                  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors",
                  style.bg,
                  style.border,
                  style.text,
                  chapter.available
                    ? "cursor-pointer hover:brightness-110"
                    : "cursor-not-allowed opacity-50",
                ].join(" ")}
                title={chapter.blurb}
                aria-label={`Open chapter ${chapter.number}: ${chapter.title}`}
              >
                <span
                  aria-hidden="true"
                  className={[
                    "inline-flex h-5 w-5 items-center justify-center rounded-full border bg-background/60 font-mono text-[10px]",
                    style.border,
                  ].join(" ")}
                >
                  {chapter.number}
                </span>
                <span className="font-medium">{chapter.title}</span>
              </button>
              {!isLast ? (
                <span
                  aria-hidden="true"
                  className="text-xs text-muted-foreground"
                >
                  →
                </span>
              ) : null}
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        All chapters are independent — open them in any order. This is the
        order they build on each other.
      </p>
    </div>
  );
}

function ChapterCard({
  chapter,
  onOpen,
}: {
  chapter: ChapterMeta;
  onOpen: () => void;
}) {
  const interactive = chapter.available;
  return (
    <Card
      onClick={interactive ? onOpen : undefined}
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : -1}
      onKeyDown={(e) => {
        if (!interactive) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen();
        }
      }}
      className={
        interactive
          ? "cursor-pointer transition-colors hover:bg-card/80 hover:border-primary/40"
          : "opacity-60"
      }
      aria-disabled={!interactive}
    >
      <CardHeader>
        <div className="mb-1 flex items-center justify-between">
          <span className="font-mono text-xs text-muted-foreground">
            Ch. {chapter.number.toString().padStart(2, "0")}
          </span>
          {chapter.available ? (
            <Badge variant="default">Ready</Badge>
          ) : (
            <Badge variant="secondary">Coming soon</Badge>
          )}
        </div>
        <CardTitle className="text-lg">{chapter.title}</CardTitle>
        <CardDescription>{chapter.blurb}</CardDescription>
      </CardHeader>
      <CardContent>
        <span
          className={
            interactive
              ? "text-sm text-primary"
              : "text-sm text-muted-foreground"
          }
        >
          {interactive ? "Open chapter →" : "Not yet authored"}
        </span>
      </CardContent>
    </Card>
  );
}
