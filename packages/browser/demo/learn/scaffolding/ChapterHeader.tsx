import * as React from 'react';

import { Badge } from '../../components/ui/badge';
import { Card } from '../../components/ui/card';

export type ChapterHeaderProps = {
  /** One short sentence: what the learner will be able to do. */
  objective: string;
  /** One short sentence: the LLM problem this concept addresses. */
  problem: string;
  /** Estimated reading + interaction time, in minutes. */
  minutes: number;
};

/**
 * Small Card-styled header for a chapter. Objective + italicized problem on
 * the left, a minutes Badge on the right. Pairs with `ChapterFrame`.
 */
export function ChapterHeader({ objective, problem, minutes }: ChapterHeaderProps) {
  return (
    <Card className="gap-2 border-b py-4">
      <div className="flex items-start justify-between gap-4 px-6">
        <div className="min-w-0 space-y-1">
          <h2 className="text-lg font-semibold leading-snug tracking-tight text-foreground">{objective}</h2>
          <p className="text-sm italic text-muted-foreground">{problem}</p>
        </div>
        <Badge variant="secondary" className="shrink-0">
          {minutes} min
        </Badge>
      </div>
    </Card>
  );
}
