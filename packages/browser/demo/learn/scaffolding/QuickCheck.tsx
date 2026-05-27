import * as React from "react";

import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import type { QuizQuestion } from "./learning-data";

export type QuickCheckProps = {
  chapterId: string;
  questions: QuizQuestion[];
};

/** Map of questionId -> selected optionId. */
type AnswerMap = Record<string, string>;

function storageKey(chapterId: string): string {
  return `mlx:chapter:${chapterId}:quiz`;
}

function loadAnswers(
  chapterId: string,
  questions: QuizQuestion[],
): AnswerMap {
  // Defensive: SSR, privacy mode, JSON corruption all return {}.
  // Also drop any persisted question/option ids that no longer exist —
  // otherwise a content edit (new question, renamed option id) can leave
  // a "ghost answered" state where the explanation shows but no radio is
  // checked, and the question is counted wrong in the score header.
  try {
    if (typeof window === "undefined" || !window.localStorage) return {};
    const raw = window.localStorage.getItem(storageKey(chapterId));
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    // Build (questionId -> Set<optionId>) once so the loop below is O(n+m).
    const validOptionsByQ = new Map<string, Set<string>>();
    for (const q of questions) {
      validOptionsByQ.set(q.id, new Set(q.options.map((o) => o.id)));
    }
    const out: AnswerMap = {};
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof v !== "string") continue;
      const valid = validOptionsByQ.get(k);
      if (valid && valid.has(v)) out[k] = v;
    }
    return out;
  } catch {
    return {};
  }
}

function saveAnswers(chapterId: string, answers: AnswerMap): void {
  try {
    if (typeof window === "undefined" || !window.localStorage) return;
    window.localStorage.setItem(storageKey(chapterId), JSON.stringify(answers));
  } catch {
    // Privacy mode or quota — silently drop. The in-memory state still works.
  }
}

function clearAnswers(chapterId: string): void {
  try {
    if (typeof window === "undefined" || !window.localStorage) return;
    window.localStorage.removeItem(storageKey(chapterId));
  } catch {
    // Same defensive policy as saveAnswers.
  }
}

/**
 * Three-question multiple-choice card. Selections are immediately marked
 * correct/incorrect and persisted to localStorage so a returning learner
 * still sees their previous attempt. "Reset answers" wipes both storage
 * and in-memory state.
 */
export function QuickCheck({ chapterId, questions }: QuickCheckProps) {
  const [answers, setAnswers] = React.useState<AnswerMap>({});

  // Hydrate from localStorage on mount. We deliberately do this in an effect
  // (not as initial state) so SSR/streaming-render environments don't bail
  // — useState's initializer would run on the server too. The loader also
  // filters out any persisted ids that no longer exist in the current quiz
  // shape (see loadAnswers), and rewrites storage if anything was dropped so
  // the cleanup is sticky.
  React.useEffect(() => {
    const loaded = loadAnswers(chapterId, questions);
    setAnswers(loaded);
    // If the in-memory shape differs from what was persisted, sync storage
    // back to the cleaned set so we don't keep re-filtering on every load.
    try {
      const raw =
        typeof window !== "undefined" && window.localStorage
          ? window.localStorage.getItem(storageKey(chapterId))
          : null;
      const persisted =
        raw && typeof raw === "string" ? JSON.parse(raw) : null;
      const persistedKeys =
        persisted && typeof persisted === "object" && !Array.isArray(persisted)
          ? Object.keys(persisted as Record<string, unknown>).sort()
          : [];
      const cleanedKeys = Object.keys(loaded).sort();
      if (persistedKeys.join(",") !== cleanedKeys.join(",")) {
        saveAnswers(chapterId, loaded);
      }
    } catch {
      // Ignore — the in-memory state is the source of truth from here.
    }
  }, [chapterId, questions]);

  const handleSelect = React.useCallback(
    (questionId: string, optionId: string) => {
      setAnswers((prev) => {
        const next = { ...prev, [questionId]: optionId };
        saveAnswers(chapterId, next);
        return next;
      });
    },
    [chapterId],
  );

  const handleReset = React.useCallback(() => {
    clearAnswers(chapterId);
    setAnswers({});
  }, [chapterId]);

  const answeredCount = questions.reduce(
    (n, q) => (answers[q.id] ? n + 1 : n),
    0,
  );
  const correctCount = questions.reduce(
    (n, q) => (answers[q.id] === q.correctId ? n + 1 : n),
    0,
  );

  return (
    <Card className="gap-3 py-4">
      <CardHeader className="px-6 [.border-b]:pb-0">
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="text-base text-foreground">
            Quick check
          </CardTitle>
          {answeredCount > 0 ? (
            <span
              aria-live="polite"
              className="font-mono text-xs text-muted-foreground"
            >
              {correctCount} / {questions.length} correct
            </span>
          ) : null}
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {questions.map((q, qi) => (
          <QuestionRow
            key={q.id}
            chapterId={chapterId}
            question={q}
            index={qi}
            selectedId={answers[q.id]}
            onSelect={(optionId) => handleSelect(q.id, optionId)}
          />
        ))}
        {answeredCount > 0 ? (
          <div>
            <button
              type="button"
              onClick={handleReset}
              className="text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
            >
              Reset answers
            </button>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

type QuestionRowProps = {
  chapterId: string;
  question: QuizQuestion;
  index: number;
  selectedId: string | undefined;
  onSelect: (optionId: string) => void;
};

function QuestionRow({
  chapterId,
  question,
  index,
  selectedId,
  onSelect,
}: QuestionRowProps) {
  // Radio group name must be unique per (chapter, question) so multiple
  // QuickCheck instances or remounts don't bleed into each other.
  const name = `quickcheck-${chapterId}-${question.id}`;
  const answered = selectedId !== undefined;
  const isCorrect = answered && selectedId === question.correctId;

  return (
    <fieldset className="space-y-2">
      <legend className="text-sm font-medium text-foreground">
        <span className="text-muted-foreground">{index + 1}.</span> {question.prompt}
      </legend>
      <div className="space-y-1.5">
        {question.options.map((opt) => {
          const checked = selectedId === opt.id;
          const isThisCorrect = answered && opt.id === question.correctId;
          const isThisWrongPick =
            answered && checked && opt.id !== question.correctId;
          const stateClass = isThisCorrect
            ? "border-emerald-500/50 bg-emerald-500/10 text-foreground"
            : isThisWrongPick
              ? "border-destructive/50 bg-destructive/10 text-foreground"
              : checked
                ? "border-primary/50 bg-primary/5 text-foreground"
                : "border-border bg-background hover:bg-accent/40 text-foreground/85";
          return (
            <label
              key={opt.id}
              className={[
                "flex cursor-pointer items-start gap-2 rounded-md border px-3 py-2 text-sm transition-colors",
                stateClass,
              ].join(" ")}
            >
              <input
                type="radio"
                name={name}
                value={opt.id}
                checked={checked}
                onChange={() => onSelect(opt.id)}
                className="mt-0.5"
              />
              <span className="flex-1">{opt.label}</span>
              {isThisCorrect ? (
                <span
                  aria-label="correct answer"
                  className="font-mono text-xs text-emerald-700 dark:text-emerald-400"
                >
                  correct
                </span>
              ) : isThisWrongPick ? (
                <span
                  aria-label="incorrect"
                  className="font-mono text-xs text-destructive"
                >
                  not quite
                </span>
              ) : null}
            </label>
          );
        })}
      </div>
      {answered && question.explanation ? (
        <p
          className={[
            "rounded-md border px-3 py-2 text-xs",
            isCorrect
              ? "border-emerald-500/30 bg-emerald-500/5 text-foreground/85"
              : "border-amber-500/30 bg-amber-500/5 text-foreground/85",
          ].join(" ")}
        >
          {question.explanation}
        </p>
      ) : null}
    </fieldset>
  );
}
