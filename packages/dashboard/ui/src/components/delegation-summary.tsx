import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { formatCount, formatNumber } from '@/lib/format';
import type { DelegationSummary } from '@/lib/types';
import { GitBranch } from 'lucide-react';

const EXPLANATION =
  'Estimates the reduction from evidence read locally to the final summary. Repeated results and tool errors are excluded. Caller prompts, reasoning, retries and transcript rereads are not measured.';

function unavailableReason(summary: Exclude<DelegationSummary, { status: 'complete' }>): string {
  switch (summary.reason) {
    case 'no-final-handoff':
      return 'Some delegated work has no completed handoff, so savings cannot be estimated.';
    case 'no-evidence':
      return 'There is no successful tool evidence to compare with the handoff.';
    case 'unsupported-content':
      return 'This session includes content that cannot be compared as text.';
    case 'legacy':
      return 'This older delegate session has no recorded measurement boundary.';
    case 'partial-record':
      return 'The session record is incomplete or unavailable.';
    case 'tokenizer-unavailable':
      return 'The tokenizer could not be loaded or run, so savings cannot be estimated.';
  }
}

function savingsColor(saved: number): string {
  return saved > 0
    ? 'text-emerald-700 dark:text-emerald-400'
    : saved < 0
      ? 'text-amber-700 dark:text-amber-400'
      : 'text-muted-foreground';
}

export function DelegateBadge() {
  return (
    <Badge variant="outline" className="border-violet-500/20 bg-violet-500/10 text-violet-700 dark:text-violet-300">
      <GitBranch aria-hidden />
      Delegate
    </Badge>
  );
}

/** Fits under the title at every breakpoint, including when the token column is hidden. */
export function DelegationIndicator({ summary }: { summary: DelegationSummary }) {
  const text =
    summary.status === 'complete'
      ? `≈${formatCount(Math.abs(summary.savedTokens))} ${summary.savedTokens < 0 ? 'extra tokens' : 'tokens saved'}`
      : 'Savings unavailable';
  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
      <DelegateBadge />
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            tabIndex={0}
            className={`cursor-help text-xs tabular-nums ${summary.status === 'complete' ? savingsColor(summary.savedTokens) : 'text-muted-foreground'}`}
          >
            {text}
          </span>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">
          {summary.status === 'complete'
            ? `${formatNumber(summary.evidenceTokens)} evidence tokens → ${formatNumber(summary.handoffTokens)} summary tokens (${Math.abs(summary.savingsRatio * 100).toFixed(1)}% ${summary.savedTokens < 0 ? 'larger' : 'reduction'}). ${EXPLANATION}`
            : unavailableReason(summary)}
        </TooltipContent>
      </Tooltip>
    </div>
  );
}

export function DelegationSavingsCard({ summary }: { summary: DelegationSummary }) {
  return (
    <Card>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="font-medium">Token savings</h2>
          <Badge variant="secondary" className="font-normal">
            Estimated
          </Badge>
        </div>
        {summary.status === 'complete' ? (
          <div className="grid gap-5 sm:grid-cols-3">
            <div>
              <p className="text-muted-foreground text-xs">
                {summary.savedTokens < 0 ? 'Extra tokens returned' : 'Tokens saved'}
              </p>
              <p className={`mt-1 text-2xl font-semibold tabular-nums ${savingsColor(summary.savedTokens)}`}>
                {formatNumber(Math.abs(summary.savedTokens))}
              </p>
              <p className="text-muted-foreground mt-1 text-xs">
                {Math.abs(summary.savingsRatio * 100).toFixed(1)}%{' '}
                {summary.savedTokens < 0 ? 'larger summary' : 'reduction'}
              </p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Evidence read locally</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">{formatNumber(summary.evidenceTokens)}</p>
              <p className="text-muted-foreground mt-1 text-xs">tokens</p>
            </div>
            <div>
              <p className="text-muted-foreground text-xs">Summary returned</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">{formatNumber(summary.handoffTokens)}</p>
              <p className="text-muted-foreground mt-1 text-xs">tokens</p>
            </div>
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">{unavailableReason(summary)}</p>
        )}
        <p className="text-muted-foreground text-xs leading-relaxed">
          {EXPLANATION} Actual coding-agent usage may differ.
        </p>
      </CardContent>
    </Card>
  );
}
