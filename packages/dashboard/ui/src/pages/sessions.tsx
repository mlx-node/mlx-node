import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { mutate } from '@/lib/api';
import { formatCount, formatNumber, formatRelativeTime } from '@/lib/format';
import { shQuote } from '@/lib/shell';
import { stableNow } from '@/lib/stable-now';
import type { SessionRenameResponse, SessionRow, SessionsResponse } from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { AlertCircle, Copy, Inbox, Loader2, Pencil, Search, Trash2 } from 'lucide-react';
import { type ReactNode, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { toast } from 'sonner';

const DAY_MS = 24 * 60 * 60 * 1000;

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function copyResume(session: SessionRow): void {
  const command = `mlx agent --session ${shQuote(session.path)}`;
  void navigator.clipboard.writeText(command).then(
    () => toast.success('Resume command copied', { description: command }),
    () => toast.error('Failed to copy to clipboard'),
  );
}

const SINCE_OPTIONS: Array<{ value: string; label: string; days: number | null }> = [
  { value: 'all', label: 'Any time', days: null },
  { value: '1', label: 'Last 24 hours', days: 1 },
  { value: '7', label: 'Last 7 days', days: 7 },
  { value: '30', label: 'Last 30 days', days: 30 },
];

/**
 * The request path for a set of filters — and therefore the cache key.
 *
 * Exported and pure so the clock in it is testable. Inline in the component it
 * was reachable only by driving a radix `Select`, which is not something a
 * happy-dom test can do; the consequence was that the only guard on the `from`
 * timestamp was one that never ran with a date filter set, and so passed just as
 * happily against a raw `Date.now()`.
 */
export function sessionsPath({ query, cwd, since }: { query: string; cwd: string; since: string }): string {
  const params = new URLSearchParams();
  if (query !== '') params.set('q', query);
  if (cwd !== 'all') params.set('cwd', cwd);
  const days = SINCE_OPTIONS.find((o) => o.value === since)?.days ?? null;
  // Quantized, for the reason spelled out in `stable-now.ts`. The default filter
  // is "Any time", which adds no `from` at all, so the miss-every-visit bug was
  // latent here rather than absent: it appeared the moment a date was picked.
  if (days !== null) params.set('from', String(stableNow() - days * DAY_MS));
  const qs = params.toString();
  return qs === '' ? '/sessions' : `/sessions?${qs}`;
}

export default function Sessions() {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [cwd, setCwd] = useState('all');
  const [since, setSince] = useState('all');

  const [renameTarget, setRenameTarget] = useState<SessionRow | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [renaming, setRenaming] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<SessionRow | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Debounce the free-text search so typing does not fire a request per keystroke.
  useEffect(() => {
    const handle = setTimeout(() => setDebouncedQuery(query.trim()), 250);
    return () => clearTimeout(handle);
  }, [query]);

  const path = useMemo(() => sessionsPath({ query: debouncedQuery, cwd, since }), [debouncedQuery, cwd, since]);

  const sessions = useJson<SessionsResponse>(path);

  // Served, not derived from the rows: the list is capped at 500, so a directory
  // whose sessions are all older than the cap has no row to be read off — and it
  // would vanish from the filter exactly when the footnote below tells the user
  // to reach older sessions with it. The served list is also unfiltered, so
  // picking a directory never hides the others.
  const cwdOptions = useMemo(
    () => [...(sessions.data?.cwds ?? [])].sort((a, b) => a.localeCompare(b)),
    [sessions.data],
  );

  const refresh = (): void => {
    sessions.reload();
  };

  const submitRename = async (): Promise<void> => {
    if (renameTarget === null) return;
    const name = renameValue.trim();
    if (name === '') return;
    setRenaming(true);
    try {
      await mutate<SessionRenameResponse>('PATCH', `/sessions/${encodeURIComponent(renameTarget.id)}`, { name });
      toast.success('Session renamed', { description: name });
      setRenameTarget(null);
      refresh();
    } catch (err) {
      toast.error('Failed to rename session', { description: errMessage(err) });
    } finally {
      setRenaming(false);
    }
  };

  const confirmDelete = async (): Promise<void> => {
    if (deleteTarget === null) return;
    setDeleting(true);
    try {
      await mutate('DELETE', `/sessions/${encodeURIComponent(deleteTarget.id)}`);
      toast.success('Session deleted', { description: sessionTitle(deleteTarget) });
      setDeleteTarget(null);
      refresh();
    } catch (err) {
      toast.error('Failed to delete session', { description: errMessage(err) });
    } finally {
      setDeleting(false);
    }
  };

  const rows = sessions.data?.sessions ?? [];
  const filtersActive = debouncedQuery !== '' || cwd !== 'all' || since !== 'all';

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Sessions</h1>
        <p className="text-muted-foreground text-sm">Browse, rename, delete, and resume agent sessions.</p>
      </div>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <Search
            className="text-muted-foreground pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2"
            aria-hidden
          />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search name or first message…"
            className="pl-9"
            aria-label="Search sessions"
          />
        </div>
        <Select value={cwd} onValueChange={setCwd}>
          <SelectTrigger className="w-full sm:w-64" aria-label="Filter by directory">
            <SelectValue placeholder="All directories" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All directories</SelectItem>
            {cwdOptions.map((dir) => (
              <SelectItem key={dir} value={dir}>
                <span className="truncate font-mono text-xs" title={dir}>
                  {dir}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={since} onValueChange={setSince}>
          <SelectTrigger className="w-full sm:w-40" aria-label="Filter by date">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SINCE_OPTIONS.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Card>
        <CardContent className="px-0">
          {sessions.error ? (
            <div className="text-destructive flex items-center gap-2 px-6 py-6 text-sm">
              <AlertCircle className="size-4 shrink-0" aria-hidden />
              {sessions.error.message}
            </div>
          ) : sessions.loading ? (
            <Table className="table-fixed">
              <SessionsTableHead />
              <SessionsSkeletonRows />
            </Table>
          ) : rows.length === 0 ? (
            <div className="text-muted-foreground flex flex-col items-center gap-2 px-6 py-12 text-sm">
              <Inbox className="size-6" aria-hidden />
              {filtersActive ? 'No sessions match these filters.' : 'No sessions recorded yet.'}
            </div>
          ) : (
            // Fixed layout with proportional columns: every cell truncates instead of
            // widening the table past the card, which used to hide the four right-hand
            // columns behind a horizontal scroll. The outer cells carry the card's own
            // 24px gutter so full-bleed row separators still line up with the footnote.
            <Table className="table-fixed">
              <SessionsTableHead />
              <TableBody>
                {rows.map((session) => (
                  <SessionTableRow
                    key={session.id}
                    session={session}
                    onRename={() => {
                      setRenameValue(session.name ?? '');
                      setRenameTarget(session);
                    }}
                    onDelete={() => setDeleteTarget(session)}
                  />
                ))}
              </TableBody>
            </Table>
          )}
          {sessions.data && sessions.data.total > rows.length && (
            <p className="text-muted-foreground px-6 pt-3 text-xs">
              Showing the newest {rows.length.toLocaleString()} of {sessions.data.total.toLocaleString()} matching
              sessions. Narrow with search or the directory / date filters to reach older ones.
            </p>
          )}
        </CardContent>
      </Card>

      <Dialog open={renameTarget !== null} onOpenChange={(open) => !open && setRenameTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename session</DialogTitle>
            <DialogDescription>
              Sets a display name, appended to the session file as a new info entry.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            placeholder="Session name"
            aria-label="Session name"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && renameValue.trim() !== '') void submitRename();
            }}
          />
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" disabled={renaming}>
                Cancel
              </Button>
            </DialogClose>
            <Button onClick={() => void submitRename()} disabled={renaming || renameValue.trim() === ''}>
              {renaming && <Loader2 className="size-4 animate-spin" />}
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteTarget !== null} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete session?</DialogTitle>
            <DialogDescription>
              This permanently removes{' '}
              <span className="text-foreground font-medium">
                {deleteTarget !== null ? sessionTitle(deleteTarget) : ''}
              </span>{' '}
              and deletes its file from disk. This cannot be undone. If an agent is currently using this session,
              deleting it may orphan in-progress turns.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" disabled={deleting}>
                Cancel
              </Button>
            </DialogClose>
            <Button variant="destructive" onClick={() => void confirmDelete()} disabled={deleting}>
              {deleting && <Loader2 className="size-4 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function sessionTitle(session: SessionRow): string {
  return session.name ?? session.firstMessage ?? session.id;
}

/**
 * The column set, rendered by the loading branch and the loaded branch from this
 * one definition.
 *
 * Both branches need it and they must never disagree. A header only the loaded
 * branch renders arrives with the response and shoves every row down by its own
 * 40px; worse here than elsewhere, because these percentage widths are what the
 * fixed layout solves the whole grid from — without them the loading rows would
 * be laid out by a different algorithm and every column would jump sideways as
 * well. Sharing the markup is what stops the two from drifting apart again.
 */
function SessionsTableHead() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead className="w-[50%] pl-6 lg:w-[40%] xl:w-[34%] 2xl:w-[30%]">Session</TableHead>
        <TableHead className="hidden xl:table-cell xl:w-[17%] 2xl:w-[14%]">Directory</TableHead>
        <TableHead className="hidden lg:table-cell lg:w-[21%] xl:w-[20%] 2xl:w-[14%]">Models</TableHead>
        <TableHead className="w-[24%] text-right lg:w-[17%] xl:w-[13%] 2xl:w-[11%]">Modified</TableHead>
        <TableHead className="hidden text-right 2xl:table-cell 2xl:w-[9%]">Messages</TableHead>
        <TableHead className="hidden text-right 2xl:table-cell 2xl:w-[8%]">Tokens</TableHead>
        <TableHead className="w-[26%] pr-6 lg:w-[22%] xl:w-[16%] 2xl:w-[14%]" />
      </TableRow>
    </TableHeader>
  );
}

/**
 * Six placeholder rows in the real table, under the real header.
 *
 * Six because this page browses every recorded session and the server serves up
 * to 500 of them, so a short list is the exception — under-reserving is the safe
 * direction, and six rows is about what the card shows above the fold once the
 * filter bar is subtracted. Claiming more would leave a gap on the one reading
 * that legitimately comes back short.
 *
 * Each cell repeats its loaded counterpart's classes, responsive visibility
 * included, so the same columns exist in both states. The bars are `h-[1lh]`,
 * one line box of the text of the cell they sit in — the `text-xs` on the
 * directory bar is what makes that the mono line box rather than the row's. Only
 * the last cell is measured rather than approximated: three `size-9` icon
 * buttons are the tallest thing in a row and are what set its 52px, so the model
 * badges (22px) can be plain bars without moving anything.
 */
function SessionsSkeletonRows() {
  return (
    <TableBody>
      {Array.from({ length: 6 }).map((_, i) => (
        <TableRow key={i}>
          <TableCell className="pl-6">
            <Skeleton className="h-[1lh] w-3/4" />
          </TableCell>
          <TableCell className="text-muted-foreground hidden xl:table-cell">
            <Skeleton className="h-[1lh] w-full text-xs" />
          </TableCell>
          <TableCell className="hidden lg:table-cell">
            <Skeleton className="h-[1lh] w-24" />
          </TableCell>
          <TableCell className="text-muted-foreground text-right">
            <Skeleton className="ml-auto h-[1lh] w-16" />
          </TableCell>
          <TableCell className="hidden text-right tabular-nums 2xl:table-cell">
            <Skeleton className="ml-auto h-[1lh] w-8" />
          </TableCell>
          <TableCell className="hidden text-right tabular-nums 2xl:table-cell">
            <Skeleton className="ml-auto h-[1lh] w-10" />
          </TableCell>
          <TableCell className="pr-6 text-right">
            <div className="flex justify-end gap-0.5">
              <Skeleton className="size-9" />
              <Skeleton className="size-9" />
              <Skeleton className="size-9" />
            </div>
          </TableCell>
        </TableRow>
      ))}
    </TableBody>
  );
}

interface SessionTableRowProps {
  session: SessionRow;
  onRename: () => void;
  onDelete: () => void;
}

function SessionTableRow({ session, onRename, onDelete }: SessionTableRowProps) {
  const tokens = session.inputTokens + session.outputTokens;
  const shownModels = session.models.slice(0, 2);
  const extraModels = session.models.length - shownModels.length;

  return (
    <TableRow>
      <TableCell className="pl-6">
        <Link
          to={`/sessions/${encodeURIComponent(session.id)}`}
          className="block truncate font-medium hover:underline"
          title={sessionTitle(session)}
        >
          {sessionTitle(session)}
        </Link>
      </TableCell>
      <TableCell className="text-muted-foreground hidden xl:table-cell">
        <span className="block truncate font-mono text-xs" title={session.cwd}>
          {session.cwd}
        </span>
      </TableCell>
      <TableCell className="hidden lg:table-cell">
        <div className="flex items-center gap-1">
          {shownModels.map((model) => (
            // `shrink` overrides the badge's own `shrink-0`: a model name is long and
            // this column is proportional, so the badge has to give before the cell does.
            <Badge key={model} variant="secondary" className="min-w-0 shrink font-normal" title={model}>
              <span className="truncate">{model}</span>
            </Badge>
          ))}
          {extraModels > 0 && <span className="text-muted-foreground shrink-0 text-xs">+{extraModels}</span>}
          {session.models.length === 0 && <span className="text-muted-foreground text-xs">—</span>}
        </div>
      </TableCell>
      <TableCell className="text-muted-foreground truncate text-right">
        {formatRelativeTime(session.modified)}
      </TableCell>
      <TableCell className="hidden text-right tabular-nums 2xl:table-cell">
        {formatNumber(session.messageCount)}
      </TableCell>
      <TableCell className="hidden text-right tabular-nums 2xl:table-cell">
        {tokens > 0 ? formatCount(tokens) : '—'}
      </TableCell>
      <TableCell className="pr-6 text-right">
        <div className="flex justify-end gap-0.5">
          <RowAction label="Copy resume command" onClick={() => copyResume(session)}>
            <Copy className="size-4" />
          </RowAction>
          <RowAction label="Rename" onClick={onRename}>
            <Pencil className="size-4" />
          </RowAction>
          <RowAction label="Delete" destructive onClick={onDelete}>
            <Trash2 className="size-4" />
          </RowAction>
        </div>
      </TableCell>
    </TableRow>
  );
}

function RowAction({
  label,
  onClick,
  destructive,
  children,
}: {
  label: string;
  onClick: () => void;
  destructive?: boolean;
  children: ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={label}
          onClick={onClick}
          className={destructive === true ? 'text-muted-foreground hover:text-destructive' : 'text-muted-foreground'}
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}
