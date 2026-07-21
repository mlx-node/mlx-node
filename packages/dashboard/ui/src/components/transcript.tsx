import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { formatRelativeTime } from '@/lib/format';
import type { TranscriptEntry, TranscriptToolCall } from '@/lib/types';
import { cn } from '@/lib/utils';
import { Bot, Brain, ChevronRight, Settings2, Terminal, User, Wrench } from 'lucide-react';
import type { ComponentType } from 'react';

/** Split assistant text into plain prose and `<think>…</think>` reasoning spans. */
interface TextSegment {
  kind: 'text' | 'think';
  content: string;
}

function splitThinking(text: string): TextSegment[] {
  const segments: TextSegment[] = [];
  const re = /<think>([\s\S]*?)<\/think>/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) segments.push({ kind: 'text', content: text.slice(last, match.index) });
    segments.push({ kind: 'think', content: match[1] });
    last = re.lastIndex;
  }
  const rest = text.slice(last);
  const openIdx = rest.indexOf('<think>');
  if (openIdx !== -1) {
    // Unclosed `<think>` (e.g. a truncated stream): treat the tail as reasoning.
    if (openIdx > 0) segments.push({ kind: 'text', content: rest.slice(0, openIdx) });
    segments.push({ kind: 'think', content: rest.slice(openIdx + '<think>'.length) });
  } else if (rest.length > 0) {
    segments.push({ kind: 'text', content: rest });
  }
  return segments.filter((s) => s.content.trim() !== '');
}

function stringifyArgs(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return '[unserializable]';
  }
}

interface RoleMeta {
  label: string;
  icon: ComponentType<{ className?: string }>;
  container: string;
  chip: string;
}

function roleMeta(role: string): RoleMeta {
  switch (role) {
    case 'user':
      return {
        label: 'You',
        icon: User,
        container: 'border-primary/25 bg-primary/5',
        chip: 'bg-primary/10 text-foreground',
      };
    case 'assistant':
      return { label: 'Assistant', icon: Bot, container: 'border-border bg-card', chip: 'bg-muted text-foreground' };
    case 'system':
      return {
        label: 'System',
        icon: Settings2,
        container: 'border-border bg-muted/40',
        chip: 'bg-muted text-muted-foreground',
      };
    default:
      return {
        label: role,
        icon: Terminal,
        container: 'border-border bg-muted/30',
        chip: 'bg-muted text-muted-foreground',
      };
  }
}

function ThinkingBlock({ content }: { content: string }) {
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
        {content}
      </CollapsibleContent>
    </Collapsible>
  );
}

function ToolCallBlock({ call }: { call: TranscriptToolCall }) {
  const args = stringifyArgs(call.arguments);
  return (
    <Collapsible className="bg-muted/40 rounded-md border">
      <CollapsibleTrigger className="hover:bg-muted/60 group rounded-md px-3 py-1.5 text-xs transition-colors">
        <Wrench className="text-muted-foreground size-3.5 shrink-0" aria-hidden />
        <span className="font-mono font-medium">{call.name || 'tool'}</span>
        <ChevronRight
          className="text-muted-foreground ml-auto size-3.5 shrink-0 transition-transform group-data-[state=open]:rotate-90"
          aria-hidden
        />
      </CollapsibleTrigger>
      {args !== '' && (
        <CollapsibleContent className="border-t">
          <pre className="overflow-x-auto px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap">{args}</pre>
        </CollapsibleContent>
      )}
    </Collapsible>
  );
}

function ToolResultEntry({ entry }: { entry: TranscriptEntry }) {
  const name = entry.toolName ?? 'result';
  return (
    <div
      className={cn(
        'rounded-lg border px-3 py-2',
        entry.isError ? 'border-destructive/40 bg-destructive/5' : 'bg-muted/30',
      )}
    >
      <div className="mb-1 flex items-center gap-1.5 text-xs font-medium">
        <Terminal
          className={cn('size-3.5 shrink-0', entry.isError ? 'text-destructive' : 'text-muted-foreground')}
          aria-hidden
        />
        <span className="font-mono">{name}</span>
        {entry.isError && <span className="text-destructive">· error</span>}
        {entry.ts > 0 && (
          <span className="text-muted-foreground ml-auto font-normal">{formatRelativeTime(entry.ts)}</span>
        )}
      </div>
      {entry.text.trim() !== '' && (
        <pre
          className={cn(
            'max-h-72 overflow-auto font-mono text-xs leading-relaxed whitespace-pre-wrap',
            entry.isError ? 'text-destructive' : 'text-muted-foreground',
          )}
        >
          {entry.text}
        </pre>
      )}
    </div>
  );
}

function MessageEntry({ entry }: { entry: TranscriptEntry }) {
  const meta = roleMeta(entry.role);
  const Icon = meta.icon;
  const segments = entry.text.trim() !== '' ? splitThinking(entry.text) : [];
  return (
    <div className={cn('rounded-lg border px-4 py-3', meta.container)}>
      <div className="mb-2 flex items-center gap-2">
        <span className={cn('flex items-center gap-1.5 rounded px-1.5 py-0.5 text-xs font-medium', meta.chip)}>
          <Icon className="size-3.5 shrink-0" aria-hidden />
          {meta.label}
        </span>
        {entry.ts > 0 && <span className="text-muted-foreground text-xs">{formatRelativeTime(entry.ts)}</span>}
      </div>
      <div className="space-y-2">
        {segments.map((seg, i) =>
          seg.kind === 'think' ? (
            <ThinkingBlock key={i} content={seg.content} />
          ) : (
            <p key={i} className="text-sm leading-relaxed whitespace-pre-wrap">
              {seg.content}
            </p>
          ),
        )}
        {entry.toolCalls.map((call, i) => (
          <ToolCallBlock key={call.id || i} call={call} />
        ))}
      </div>
    </div>
  );
}

export interface TranscriptProps {
  entries: TranscriptEntry[];
}

/**
 * Presentational transcript: role-styled message bubbles, tool calls and tool
 * results as collapsibles (collapsed by default), and `<think>` reasoning spans
 * split out into their own collapsed "Thinking" blocks. Entries are already
 * flattened and ordered by the server.
 */
export function Transcript({ entries }: TranscriptProps) {
  return (
    <div className="space-y-3">
      {entries.map((entry, i) =>
        entry.role === 'toolResult' ? (
          <ToolResultEntry key={i} entry={entry} />
        ) : (
          <MessageEntry key={i} entry={entry} />
        ),
      )}
    </div>
  );
}
