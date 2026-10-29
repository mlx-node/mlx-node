import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { formatCount, formatRelativeTime } from '@/lib/format';
import type { TranscriptEntry } from '@/lib/types';
import { cn } from '@/lib/utils';
import {
  Bot,
  Brain,
  ChevronRight,
  Circle,
  CircleCheck,
  CircleDot,
  Cpu,
  GitBranch,
  ListChecks,
  User,
  Wrench,
} from 'lucide-react';
import Markdown, { type Components } from 'react-markdown';
import { Link } from 'react-router-dom';
import remarkGfm from 'remark-gfm';

type Entry<K extends TranscriptEntry['kind']> = Extract<TranscriptEntry, { kind: K }>;

/**
 * Compact Tailwind styling for the GitHub-flavored markdown in message prose.
 * `react-markdown` never emits raw HTML (no `rehype-raw`), so assistant/user text
 * renders as safe React elements.
 */
const MD_COMPONENTS: Components = {
  h1: ({ children }) => <h1 className="mt-3 mb-1 text-base font-semibold first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mt-3 mb-1 text-sm font-semibold first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mt-2 mb-0.5 text-sm font-semibold first:mt-0">{children}</h3>,
  h4: ({ children }) => (
    <h4 className="text-muted-foreground mt-2 mb-0.5 text-xs font-semibold tracking-wide uppercase first:mt-0">
      {children}
    </h4>
  ),
  p: ({ children }) => <p className="my-1.5 leading-relaxed first:mt-0 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="my-1.5 list-disc space-y-0.5 pl-5 first:mt-0 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="my-1.5 list-decimal space-y-0.5 pl-5 first:mt-0 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2">
      {children}
    </a>
  ),
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  blockquote: ({ children }) => (
    <blockquote className="text-muted-foreground my-2 border-l-2 pl-3 italic">{children}</blockquote>
  ),
  hr: () => <hr className="my-3" />,
  code: ({ className, children }) => {
    // Flatten only string children for the multiline check; never coerce a
    // React node to `[object …]`.
    const text = Array.isArray(children)
      ? children.map((child) => (typeof child === 'string' ? child : '')).join('')
      : typeof children === 'string'
        ? children
        : '';
    const isBlock = /language-/.test(className ?? '') || text.includes('\n');
    return isBlock ? (
      <code className="font-mono">{children}</code>
    ) : (
      <code className="bg-muted rounded px-1 py-0.5 font-mono text-[0.85em]">{children}</code>
    );
  },
  pre: ({ children }) => (
    <pre className="bg-muted/50 my-2 overflow-x-auto rounded-md border p-3 font-mono text-xs leading-relaxed">
      {children}
    </pre>
  ),
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto">
      <table className="w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  th: ({ children }) => <th className="border px-2 py-1 text-left font-semibold">{children}</th>,
  td: ({ children }) => <td className="border px-2 py-1 align-top">{children}</td>,
};

function Prose({ text }: { text: string }) {
  return (
    <div className="text-sm break-words">
      <Markdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
        {text}
      </Markdown>
    </div>
  );
}

function Timestamp({ ts }: { ts: number }) {
  if (ts <= 0) return null;
  return (
    <span className="text-muted-foreground shrink-0 text-xs font-normal" title={new Date(ts).toISOString()}>
      {formatRelativeTime(ts)}
    </span>
  );
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

function MessageEntry({ entry }: { entry: Entry<'user'> | Entry<'assistant'> }) {
  const user = entry.kind === 'user';
  const Icon = user ? User : Bot;
  const label = user ? (entry.interjection === true ? 'You · interjection' : 'You') : 'Assistant';
  return (
    <div className={cn('rounded-lg border px-4 py-3', user ? 'border-primary/25 bg-primary/5' : 'bg-card')}>
      <div className="mb-2 flex items-center gap-2">
        <span
          className={cn(
            'text-foreground flex items-center gap-1.5 rounded px-1.5 py-0.5 text-xs font-medium',
            user ? 'bg-primary/10' : 'bg-muted',
          )}
        >
          <Icon className="size-3.5 shrink-0" aria-hidden />
          {label}
        </span>
        <Timestamp ts={entry.ts} />
      </div>
      <Prose text={entry.text} />
    </div>
  );
}

function ThinkingEntry({ entry }: { entry: Entry<'thinking'> }) {
  return (
    <Collapsible className="rounded-md border border-dashed">
      <CollapsibleTrigger className="text-muted-foreground hover:text-foreground group px-3 py-1.5 text-xs font-medium transition-colors">
        <Brain className="size-3.5 shrink-0" aria-hidden />
        <span>Thinking</span>
        <ChevronRight
          className="size-3.5 shrink-0 transition-transform group-data-[state=open]:rotate-90"
          aria-hidden
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="text-muted-foreground border-t px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap">
        {entry.text}
      </CollapsibleContent>
    </Collapsible>
  );
}

function ToolEntry({ entry }: { entry: Entry<'tool'> }) {
  const failed = entry.status === 'failed';
  const hasBody = entry.input !== undefined || entry.output !== undefined;
  const header = (
    <>
      <Wrench className={cn('size-3.5 shrink-0', failed ? 'text-destructive' : 'text-muted-foreground')} aria-hidden />
      <span className="shrink-0 font-mono font-medium">{entry.name}</span>
      {entry.title !== '' && entry.title !== entry.name ? (
        <span className="text-muted-foreground min-w-0 flex-1 truncate text-left font-normal" title={entry.title}>
          {entry.title}
        </span>
      ) : (
        <span className="flex-1" />
      )}
      {entry.status !== 'completed' && (
        <span className={cn('shrink-0', failed ? 'text-destructive' : 'text-muted-foreground')}>· {entry.status}</span>
      )}
      <Timestamp ts={entry.ts} />
    </>
  );
  const container = cn('rounded-lg border', failed ? 'border-destructive/40 bg-destructive/5' : 'bg-muted/30');
  if (!hasBody) {
    return (
      <div className={container}>
        <div className="flex items-center gap-1.5 px-3 py-2 text-xs">{header}</div>
      </div>
    );
  }
  // Collapsed by default; a failed call opens so the error is never buried.
  return (
    <Collapsible defaultOpen={failed} className={container}>
      <CollapsibleTrigger className="hover:bg-muted/40 group gap-1.5 rounded-lg px-3 py-2 text-xs transition-colors data-[state=open]:rounded-b-none">
        {header}
        <ChevronRight
          className="text-muted-foreground size-3.5 shrink-0 transition-transform group-data-[state=open]:rotate-90"
          aria-hidden
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-2 border-t px-3 py-2">
        {entry.input !== undefined && <ToolField label="Input" text={entry.input} />}
        {entry.output !== undefined && <ToolField label="Output" text={entry.output} error={failed} />}
        {entry.truncated === true && (
          <p className="text-muted-foreground text-xs italic">Long fields are truncated for display.</p>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}

function ToolField({ label, text, error }: { label: string; text: string; error?: boolean }) {
  return (
    <div>
      <div className="text-muted-foreground mb-1 text-[11px] font-semibold tracking-wide uppercase">{label}</div>
      <pre
        className={cn(
          'max-h-72 overflow-auto font-mono text-xs leading-relaxed whitespace-pre-wrap',
          error === true ? 'text-destructive' : 'text-muted-foreground',
        )}
      >
        {text}
      </pre>
    </div>
  );
}

function PlanEntry({ entry }: { entry: Entry<'plan'> }) {
  return (
    <div className="bg-muted/30 rounded-lg border px-3 py-2">
      <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium">
        <ListChecks className="text-muted-foreground size-3.5 shrink-0" aria-hidden />
        Plan
        <span className="flex-1" />
        <Timestamp ts={entry.ts} />
      </div>
      <ul className="space-y-1">
        {entry.entries.map((item, i) => {
          const Icon = item.status === 'completed' ? CircleCheck : item.status === 'in_progress' ? CircleDot : Circle;
          return (
            <li key={i} className="flex items-start gap-2 text-sm" data-status={item.status}>
              <Icon
                className={cn(
                  'mt-0.5 size-4 shrink-0',
                  item.status === 'completed' ? 'text-emerald-600' : 'text-muted-foreground',
                )}
                aria-label={item.status}
              />
              <span className={cn(item.status === 'completed' && 'text-muted-foreground line-through')}>
                {item.content}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function SubagentEntry({ entry, linkable }: { entry: Entry<'subagent'>; linkable: boolean }) {
  const label = entry.description !== '' ? entry.description : entry.childId;
  const facts = [
    entry.status,
    entry.durationMs !== undefined ? formatDuration(entry.durationMs) : undefined,
    entry.tokens !== undefined ? `${formatCount(entry.tokens)} tokens` : undefined,
  ].filter((v): v is string => v !== undefined && v !== '');
  return (
    <div
      className={cn(
        'rounded-lg border px-3 py-2 text-xs',
        entry.error !== undefined ? 'border-destructive/40 bg-destructive/5' : 'bg-muted/30',
      )}
    >
      <div className="flex items-center gap-1.5">
        <GitBranch className="text-muted-foreground size-3.5 shrink-0" aria-hidden />
        <span className="shrink-0 font-medium">Subagent{entry.type !== '' ? ` · ${entry.type}` : ''}</span>
        {linkable ? (
          <Link
            to={`/sessions/${encodeURIComponent(entry.childId)}`}
            className="text-primary min-w-0 flex-1 truncate hover:underline"
            title={label}
          >
            {label}
          </Link>
        ) : (
          <span className="text-muted-foreground min-w-0 flex-1 truncate" title={label}>
            {label}
          </span>
        )}
        {facts.length > 0 && <span className="text-muted-foreground shrink-0">{facts.join(' · ')}</span>}
        <Timestamp ts={entry.ts} />
      </div>
      {entry.error !== undefined && <p className="text-destructive mt-1.5 whitespace-pre-wrap">{entry.error}</p>}
    </div>
  );
}

function Marker({ entry }: { entry: Entry<'model'> | Entry<'turn-end'> }) {
  const text =
    entry.kind === 'model'
      ? `Model switched to ${entry.modelId}`
      : `Turn ended · ${entry.stopReason}${entry.elapsedMs !== undefined ? ` · ${formatDuration(entry.elapsedMs)}` : ''}`;
  return (
    <div>
      <div className="text-muted-foreground flex items-center gap-2 text-xs">
        <span className="bg-border h-px flex-1" aria-hidden />
        {entry.kind === 'model' && <Cpu className="size-3.5 shrink-0" aria-hidden />}
        <span>{text}</span>
        <span className="bg-border h-px flex-1" aria-hidden />
      </div>
      {entry.kind === 'turn-end' && entry.error !== undefined && (
        <p className="text-destructive mt-1 text-center text-xs whitespace-pre-wrap">{entry.error}</p>
      )}
    </div>
  );
}

export interface TranscriptProps {
  entries: TranscriptEntry[];
  /** Subagent child ids whose folder is known to be gone; those rows are not linked. */
  missingChildren?: ReadonlySet<string>;
}

/**
 * Presentational transcript, entries already merged and ordered by the server:
 * message bubbles, collapsed thinking and tool calls, plan checklists, subagent
 * rows linking to the child session, and thin markers for model switches and
 * turn ends.
 */
export function Transcript({ entries, missingChildren }: TranscriptProps) {
  return (
    <div className="space-y-3">
      {entries.map((entry, i) => {
        switch (entry.kind) {
          case 'user':
          case 'assistant':
            return <MessageEntry key={i} entry={entry} />;
          case 'thinking':
            return <ThinkingEntry key={i} entry={entry} />;
          case 'tool':
            return <ToolEntry key={i} entry={entry} />;
          case 'plan':
            return <PlanEntry key={i} entry={entry} />;
          case 'subagent':
            return <SubagentEntry key={i} entry={entry} linkable={missingChildren?.has(entry.childId) !== true} />;
          case 'model':
          case 'turn-end':
            return <Marker key={i} entry={entry} />;
        }
      })}
    </div>
  );
}
