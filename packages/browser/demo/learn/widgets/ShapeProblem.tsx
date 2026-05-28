import * as React from 'react';

import { Button } from '../../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/card';

export type ShapeProblemProps = {
  /** Headline question, e.g. "How big is Qwen3's embedding matrix?" */
  question: string;
  /** Formula shown in mono, e.g. "vocab_size × hidden_dim". */
  formula: string;
  /** Named inputs to the formula. Rendered as a small two-column table. */
  values: { label: string; value: string }[];
  /** Worked-through answer revealed by the button. */
  answer: string;
};

/**
 * Click-to-reveal shape-arithmetic exercise. Used inline in chapter bodies to
 * give the reader a concrete number to compute (parameter count, KV-cache
 * footprint, MLP intermediate size, etc.) instead of just reading prose.
 */
export function ShapeProblem({ question, formula, values, answer }: ShapeProblemProps) {
  const [open, setOpen] = React.useState(false);
  return (
    <Card className="gap-3 py-4">
      <CardHeader className="px-6 [.border-b]:pb-0">
        <CardTitle className="text-base text-foreground">{question}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-sm text-foreground/85">
          {formula}
        </div>
        <table className="w-full text-sm">
          <tbody>
            {values.map((v, i) => (
              <tr key={i} className="border-b border-border last:border-b-0">
                <td className="py-1.5 pr-4 font-mono text-muted-foreground">{v.label}</td>
                <td className="py-1.5 text-right font-mono text-foreground">{v.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
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
