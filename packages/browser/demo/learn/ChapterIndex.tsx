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
          More chapters are in progress. The Attention chapter ships the
          end-to-end inspector pipeline; everything else lights up after that.
        </p>
      </div>
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
