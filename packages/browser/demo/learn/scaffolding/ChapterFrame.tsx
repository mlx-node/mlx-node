import * as React from 'react';

import { ChapterHeader } from './ChapterHeader';
import { EngineeringTakeaways } from './EngineeringTakeaways';
import { Glossary } from './Glossary';
import type { ChapterLearningData } from './learning-data';
import { MiniExercise } from './MiniExercise';
import { QuickCheck } from './QuickCheck';

export type ChapterFrameProps = {
  learning: ChapterLearningData;
  /** The chapter body — typically a `<Prose>` block. */
  children: React.ReactNode;
};

/**
 * Composes the per-chapter scaffolding around the body prose:
 * header + glossary above, takeaways + exercise + quick-check below.
 *
 * The wrapper sets a uniform vertical rhythm so the chapter reads as a
 * single document rather than a stack of unrelated cards.
 */
export function ChapterFrame({ learning, children }: ChapterFrameProps) {
  return (
    <div className="space-y-8">
      <ChapterHeader objective={learning.objective} problem={learning.problem} minutes={learning.minutes} />
      <Glossary terms={learning.glossary} />
      <div>{children}</div>
      <EngineeringTakeaways takeaways={learning.takeaways} />
      <MiniExercise prompt={learning.exercise.prompt} answer={learning.exercise.answer} />
      <QuickCheck chapterId={learning.chapterId} questions={learning.quiz} />
    </div>
  );
}
