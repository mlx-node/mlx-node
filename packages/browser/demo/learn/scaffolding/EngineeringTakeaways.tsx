import * as React from "react";
import { LightbulbIcon } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";

export type EngineeringTakeawaysProps = {
  /** Exactly 3 short bullets the reader could quote in an interview. */
  takeaways: string[];
};

/**
 * "Remember this" callout listing the three engineering takeaways for the
 * chapter. Warm amber tint to distinguish it from the rest of the page
 * without being shouty.
 */
export function EngineeringTakeaways({ takeaways }: EngineeringTakeawaysProps) {
  return (
    <Card className="gap-3 border-amber-500/30 bg-amber-500/5 py-4">
      <CardHeader className="px-6 [.border-b]:pb-0">
        <CardTitle className="flex items-center gap-2 text-base text-foreground">
          <LightbulbIcon
            aria-hidden="true"
            className="size-4 text-amber-600 dark:text-amber-400"
          />
          Engineering takeaways
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-foreground/85">
          {takeaways.map((t, i) => (
            <li key={i}>{t}</li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
