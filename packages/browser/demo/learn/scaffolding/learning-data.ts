/**
 * Data contract for chapter scaffolding (header, glossary, takeaways,
 * exercise, quiz). Each chapter exports a `learning: ChapterLearningData`
 * object alongside its `ChapterBody` component; `ChapterFrame` composes the
 * scaffolding around the chapter's prose.
 *
 * The shape is intentionally small: this is reading-focused content, not a
 * full curriculum runtime. `chapterId` is the join key used to scope
 * localStorage for quiz state.
 */

export type GlossaryTerm = {
  term: string;
  definition: string;
};

export type QuizOption = {
  /** Stable id (e.g. "a", "b", "c") — persisted in localStorage. */
  id: string;
  /** Displayed text. */
  label: string;
};

export type QuizQuestion = {
  /** Stable key (used inside localStorage). */
  id: string;
  prompt: string;
  /** 2–4 options. */
  options: QuizOption[];
  /** Matches one option's id. */
  correctId: string;
  /** Shown after answering. */
  explanation?: string;
};

export type ChapterLearningData = {
  /** Must equal the chapter's ChapterMeta.id (used to scope localStorage). */
  chapterId: string;
  /** One short sentence: what the learner will be able to do after this chapter. */
  objective: string;
  /** One short sentence: what problem in LLMs this concept solves / why it exists. */
  problem: string;
  /** Estimated reading + interaction time, in minutes. */
  minutes: number;
  /** 6–10 terms. Start at 6, grow only if genuinely needed — the merged capstone (chapter 16) runs to 12. */
  glossary: GlossaryTerm[];
  /** Usually 3 bullets a learner could quote in an interview; the merged capstone carries more. */
  takeaways: string[];
  /** One small thing to try in the existing demo (the merged capstone uses a two-part prompt). */
  exercise: {
    prompt: string;
    answer: string;
  };
  /** Usually 3 multiple-choice questions; the merged capstone (chapter 16) carries 6. */
  quiz: QuizQuestion[];
};
