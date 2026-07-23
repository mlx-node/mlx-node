import { ModelLogo, prettyModelName } from '@/components/model-logos';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { formatRelativeTime } from '@/lib/format';
import type { TranscriptEntry, TranscriptToolCall } from '@/lib/types';
import { cn } from '@/lib/utils';
import { Binary, Bot, Brain, ChevronRight, Settings2, Terminal, User, Wrench } from 'lucide-react';
import type { ComponentType } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';

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

/**
 * Compact Tailwind styling for the GitHub-flavored markdown in message prose.
 * `react-markdown` never emits raw HTML (no `rehype-raw`), so assistant/user text
 * renders as safe React elements — headings, lists, tables, inline/fenced code —
 * instead of literal `##`/`**`/`` ` `` characters.
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
    // Fenced/indented code (has a `language-*` class, or spans multiple lines) is
    // laid out by the `pre` container; inline code gets a chip. Flatten only string
    // children for the multiline check — never coerce a React node to `[object …]`.
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

/** Render message prose as GitHub-flavored markdown. */
function Prose({ text }: { text: string }) {
  return (
    <div className="text-sm">
      <Markdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
        {text}
      </Markdown>
    </div>
  );
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
      <CollapsibleTrigger className="hover:bg-muted/60 group rounded-md px-3 py-1.5 text-xs transition-colors data-[state=open]:rounded-b-none">
        <Wrench className="text-muted-foreground size-3.5 shrink-0" aria-hidden />
        <span className="shrink-0 font-mono font-medium">{call.name || 'tool'}</span>
        {call.summary ? (
          <span className="text-muted-foreground min-w-0 flex-1 truncate text-left font-normal">{call.summary}</span>
        ) : (
          <span className="flex-1" />
        )}
        <ChevronRight
          className="text-muted-foreground size-3.5 shrink-0 transition-transform group-data-[state=open]:rotate-90"
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

/**
 * Inline image thumbnails and binary-blob chips for an entry. A `read` of an
 * image yields a base64 block (shown as a click-to-open thumbnail); a `read` of
 * raw binary (HEIC/PDF bytes returned verbatim) yields a chip label instead of
 * dumping kilobytes of garbage into the transcript.
 */
function Attachments({ entry }: { entry: TranscriptEntry }) {
  const images = entry.images ?? [];
  const notes = entry.binaryNotes ?? [];
  if (images.length === 0 && notes.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      {images.map((img, i) => {
        const src = `data:${img.mimeType};base64,${img.data}`;
        return (
          <a
            key={`img-${i}`}
            href={src}
            target="_blank"
            rel="noreferrer"
            className="hover:border-primary/50 block overflow-hidden rounded-md border transition-colors"
            title="Open image in a new tab"
          >
            <img src={src} alt="attachment" loading="lazy" className="max-h-48 max-w-full object-contain" />
          </a>
        );
      })}
      {notes.map((note, i) => (
        <span
          key={`bin-${i}`}
          className="text-muted-foreground bg-muted/60 inline-flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-xs"
        >
          <Binary className="size-3.5 shrink-0" aria-hidden />
          {note}
        </span>
      ))}
    </div>
  );
}

/**
 * Coordinate-mapping image-read notes, collapsed by default. Right beside the
 * rendered thumbnail these are model plumbing, not something a reader wants in
 * their face — but kept one click away for anyone who needs the exact dimensions.
 */
function ImageDetails({ notes }: { notes: string[] }) {
  return (
    <Collapsible>
      <CollapsibleTrigger className="text-muted-foreground hover:text-foreground group inline-flex items-center gap-1 text-xs font-medium transition-colors">
        <ChevronRight className="size-3 shrink-0 transition-transform group-data-[state=open]:rotate-90" aria-hidden />
        Image details
      </CollapsibleTrigger>
      <CollapsibleContent className="text-muted-foreground mt-1 font-mono text-xs leading-relaxed whitespace-pre-wrap">
        {notes.join('\n')}
      </CollapsibleContent>
    </Collapsible>
  );
}

/**
 * The icon + tool name + a one-line arg digest (the file path read, the command
 * run) + error flag + timestamp shown on every result header. The digest fills
 * the middle and truncates, so a collapsed row still says what it acted on.
 */
function ResultHeaderInner({ entry, name }: { entry: TranscriptEntry; name: string }) {
  return (
    <>
      <Terminal
        className={cn('size-3.5 shrink-0', entry.isError ? 'text-destructive' : 'text-muted-foreground')}
        aria-hidden
      />
      <span className="shrink-0 font-mono">{name}</span>
      {entry.title ? (
        <span className="text-muted-foreground min-w-0 flex-1 truncate text-left font-normal">{entry.title}</span>
      ) : (
        <span className="flex-1" />
      )}
      {entry.isError && <span className="text-destructive shrink-0">· error</span>}
      {entry.ts > 0 && (
        <span className="text-muted-foreground shrink-0 font-normal">{formatRelativeTime(entry.ts)}</span>
      )}
    </>
  );
}

function ToolResultEntry({ entry }: { entry: TranscriptEntry }) {
  const name = entry.toolName ?? 'result';
  // Beside a rendered image the result's own text ("Read image file […]") is
  // plumbing too — fold it and the coordinate note into one collapsed "Image
  // details" disclosure so the default view is just the image. This holds whether
  // the image inlines as a thumbnail or (too large) surfaces as an `image · N MB`
  // chip.
  const hasInlineImage = (entry.images?.length ?? 0) > 0;
  const hasImageChip = (entry.binaryNotes ?? []).some((n) => n.startsWith('image'));
  const isImageRead = hasInlineImage || hasImageChip || (entry.imageNotes?.length ?? 0) > 0;
  const hasText = entry.text.trim() !== '';
  const details = isImageRead
    ? [...(hasText ? [entry.text] : []), ...(entry.imageNotes ?? [])]
    : (entry.imageNotes ?? []);
  const hasAttachments = hasInlineImage || (entry.binaryNotes?.length ?? 0) > 0;
  const hasContent = hasText || hasAttachments || details.length > 0;
  const container = cn('rounded-lg border', entry.isError ? 'border-destructive/40 bg-destructive/5' : 'bg-muted/30');

  // An empty result (a command with no output) has nothing to expand — a plain
  // header, no chevron.
  if (!hasContent) {
    return (
      <div className={container}>
        <div className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium">
          <ResultHeaderInner entry={entry} name={name} />
        </div>
      </div>
    );
  }

  // The body collapses under the header. Text results (bash output, file reads)
  // start collapsed — just the `>_ name` header + a chevron; image results start
  // open so the picture shows without a click, and errors start open so a failure
  // is never buried.
  return (
    <Collapsible defaultOpen={isImageRead || entry.isError === true} className={container}>
      <CollapsibleTrigger className="hover:bg-muted/40 group flex w-full items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium transition-colors data-[state=open]:rounded-b-none">
        <ResultHeaderInner entry={entry} name={name} />
        <ChevronRight
          className={cn(
            'text-muted-foreground size-3.5 shrink-0 transition-transform group-data-[state=open]:rotate-90',
            entry.ts > 0 ? 'ml-1.5' : 'ml-auto',
          )}
          aria-hidden
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-2 px-3 pb-2">
        {hasText && !isImageRead && (
          <pre
            className={cn(
              'max-h-72 overflow-auto font-mono text-xs leading-relaxed whitespace-pre-wrap',
              entry.isError ? 'text-destructive' : 'text-muted-foreground',
            )}
          >
            {entry.text}
          </pre>
        )}
        <Attachments entry={entry} />
        {details.length > 0 && <ImageDetails notes={details} />}
      </CollapsibleContent>
    </Collapsible>
  );
}

function MessageEntry({ entry }: { entry: TranscriptEntry }) {
  const meta = roleMeta(entry.role);
  const Icon = meta.icon;
  const segments = entry.text.trim() !== '' ? splitThinking(entry.text) : [];
  // An assistant turn shows the logo + name of the model that produced it; every
  // other role keeps its generic role icon + label.
  const model = entry.role === 'assistant' && entry.model ? entry.model : undefined;
  return (
    <div className={cn('rounded-lg border px-4 py-3', meta.container)}>
      <div className="mb-2 flex items-center gap-2">
        <span className={cn('flex items-center gap-1.5 rounded px-1.5 py-0.5 text-xs font-medium', meta.chip)}>
          {model !== undefined ? (
            <>
              <ModelLogo model={model} className="size-3.5 shrink-0" />
              {prettyModelName(model)}
            </>
          ) : (
            <>
              <Icon className="size-3.5 shrink-0" aria-hidden />
              {meta.label}
            </>
          )}
        </span>
        {entry.ts > 0 && <span className="text-muted-foreground text-xs">{formatRelativeTime(entry.ts)}</span>}
      </div>
      <div className="space-y-2">
        {segments.map((seg, i) =>
          seg.kind === 'think' ? <ThinkingBlock key={i} content={seg.content} /> : <Prose key={i} text={seg.content} />,
        )}
        {entry.toolCalls.map((call, i) => (
          <ToolCallBlock key={call.id || i} call={call} />
        ))}
        <Attachments entry={entry} />
        {entry.imageNotes && entry.imageNotes.length > 0 && <ImageDetails notes={entry.imageNotes} />}
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
