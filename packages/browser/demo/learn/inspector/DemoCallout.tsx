import { EyeIcon } from 'lucide-react';
import * as React from 'react';

import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/card';

export type DemoCalloutProps = {
  /** Short, demo-specific observations the reader should notice in the UI. */
  items: string[];
};

/**
 * "What to look for" callout placed near the bottom of each chapter demo.
 *
 * Uses a soft primary tint — distinct from the amber `EngineeringTakeaways`
 * card, which is reserved for the per-chapter engineering bullets. The eye
 * icon signals "now that you can see the visualization, here's what to
 * notice" rather than "remember this for an interview".
 */
export function DemoCallout({ items }: DemoCalloutProps) {
  return (
    <Card className="gap-3 border-primary/30 bg-primary/5 py-4">
      <CardHeader className="px-6 [.border-b]:pb-0">
        <CardTitle className="flex items-center gap-2 text-base text-foreground">
          <EyeIcon aria-hidden="true" className="size-4 text-primary" />
          What to look for
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-foreground/85">
          {items.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
