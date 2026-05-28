import * as React from 'react';

import { Button } from '../../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/card';

export type MiniExerciseProps = {
  prompt: string;
  answer: string;
};

/**
 * "Try this" card with a single prompt and a Show/Hide-answer toggle.
 * Intentionally small — the chapter's own demo is the real exercise; this is
 * just a nudge.
 */
export function MiniExercise({ prompt, answer }: MiniExerciseProps) {
  const [open, setOpen] = React.useState(false);
  return (
    <Card className="gap-3 py-4">
      <CardHeader className="px-6 [.border-b]:pb-0">
        <CardTitle className="text-base text-foreground">Try this</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-foreground/85">{prompt}</p>
        <Button variant="outline" size="sm" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          {open ? 'Hide answer' : 'Show answer'}
        </Button>
        {open ? (
          <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-sm text-foreground/85">
            {answer}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
