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
import { getJson, mutate } from '@/lib/api';
import { formatCount, formatNumber, formatRelativeTime } from '@/lib/format';
import { copyResumeCommand, errMessage, isConflict, isoMs, LIVE_HINT, sessionTitle } from '@/lib/sessions';
import type {
  SessionDeleteResponse,
  SessionDetailResponse,
  SessionListResponse,
  SessionRenameResponse,
  SessionRow,
} from '@/lib/types';
import { useJson } from '@/lib/use-api';
import { AlertCircle, Copy, Inbox, Loader2, Pencil, Search, Trash2 } from 'lucide-react';
import { type ReactNode, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { toast } from 'sonner';

/** The server's cap; asked for explicitly so the footnote below is the only truncation. */
export const SESSIONS_LIST_LIMIT = 500;

/** The request path for a set of filters, and therefore the `useJson` cache key. */
export function sessionsPath({ query, cwd }: { query: string; cwd: string }): string {
  const params = new URLSearchParams();
  if (query !== '') params.set('q', query);
  if (cwd !== 'all') params.set('cwd', cwd);
  params.set('limit', String(SESSIONS_LIST_LIMIT));
  return `/sessions?${params.toString()}`;
}

function sessionUrl(id: string): string {
  return `/sessions/${encodeURIComponent(id)}`;
}

async function copyResume(session: SessionRow): Promise<void> {
  // Rows do not carry the command: it depends on the agent home, which only the
  // server knows, so it is read from the detail route.
  try {
    const detail = await getJson<SessionDetailResponse>(sessionUrl(session.id));
    copyResumeCommand(detail.resumeCommand);
  } catch (err) {
    toast.error('Failed to read the resume command', { description: errMessage(err) });
  }
}

export default function Sessions() {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [cwd, setCwd] = useState('all');

  const [renameTarget, setRenameTarget] = useState<SessionRow | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [deleteTarget, setDeleteTarget] = useState<SessionRow | null>(null);
  const [busy, setBusy] = useState(false);
  /** The server's 409 message for the open dialog, shown inside it. */
  const [conflict, setConflict] = useState<string | null>(null);

  useEffect(() => {
    const handle = setTimeout(() => setDebouncedQuery(query.trim()), 250);
    return () => clearTimeout(handle);
  }, [query]);

  const path = useMemo(() => sessionsPath({ query: debouncedQuery, cwd }), [debouncedQuery, cwd]);
  const sessions = useJson<SessionListResponse>(path);

  // Served, not derived from the rows: a directory whose sessions all fell past
  // the cap has no row to read it from, and the served list ignores the filters.
  const cwdOptions = sessions.data?.cwds ?? [];

  const openRename = (session: SessionRow): void => {
    setConflict(null);
    setRenameValue(session.title);
    setRenameTarget(session);
  };

  const openDelete = (session: SessionRow): void => {
    setConflict(null);
    setDeleteTarget(session);
  };

  const fail = (action: string, err: unknown): void => {
    if (isConflict(err)) {
      // Keep the dialog open with the reason, and refresh so the row's live
      // badge catches up with what the server just saw.
      setConflict(err.message);
      sessions.reload();
      return;
    }
    toast.error(`Failed to ${action} session`, { description: errMessage(err) });
  };

  const submitRename = async (): Promise<void> => {
    if (renameTarget === null) return;
    const title = renameValue.trim();
    if (title === '') return;
    setBusy(true);
    try {
      const result = await mutate<SessionRenameResponse>('PATCH', sessionUrl(renameTarget.id), { title });
      toast.success('Session renamed', { description: result.title });
      setRenameTarget(null);
      sessions.reload();
    } catch (err) {
      fail('rename', err);
    } finally {
      setBusy(false);
    }
  };

  const confirmDelete = async (): Promise<void> => {
    if (deleteTarget === null) return;
    setBusy(true);
    try {
      const result = await mutate<SessionDeleteResponse>('DELETE', sessionUrl(deleteTarget.id));
      const children = result.deleted.length - 1;
      toast.success('Session deleted', {
        description:
          children > 0
            ? `${sessionTitle(deleteTarget)} and ${children} subagent session${children === 1 ? '' : 's'}`
            : sessionTitle(deleteTarget),
      });
      setDeleteTarget(null);
      sessions.reload();
    } catch (err) {
      fail('delete', err);
    } finally {
      setBusy(false);
    }
  };

  const rows = sessions.data?.sessions ?? [];
  const filtersActive = debouncedQuery !== '' || cwd !== 'all';

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Sessions</h1>
        <p className="text-muted-foreground text-sm">Browse, rename, delete, and resume mlx agent sessions.</p>
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
            placeholder="Search title or directory…"
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
              {filtersActive ? 'No sessions match these filters.' : 'No agent sessions yet.'}
            </div>
          ) : (
            // Fixed layout with proportional columns: every cell truncates instead
            // of widening the table past the card.
            <Table className="table-fixed">
              <SessionsTableHead />
              <TableBody>
                {rows.map((session) => (
                  <SessionTableRow
                    key={session.id}
                    session={session}
                    onRename={() => openRename(session)}
                    onDelete={() => openDelete(session)}
                  />
                ))}
              </TableBody>
            </Table>
          )}
          {sessions.data && sessions.data.total > rows.length && (
            <p className="text-muted-foreground px-6 pt-3 text-xs">
              Showing the newest {rows.length.toLocaleString()} of {sessions.data.total.toLocaleString()} matching
              sessions. Narrow with search or the directory filter to reach older ones.
            </p>
          )}
        </CardContent>
      </Card>

      <Dialog open={renameTarget !== null} onOpenChange={(open) => !open && !busy && setRenameTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename session</DialogTitle>
            <DialogDescription>
              Sets the title the agent shows in its session picker. The agent keeps a manual title and stops generating
              one.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            placeholder="Session title"
            aria-label="Session title"
            disabled={renameTarget?.live === true}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && renameValue.trim() !== '' && renameTarget?.live !== true) void submitRename();
            }}
          />
          <DialogNotice live={renameTarget?.live === true} conflict={conflict} />
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" disabled={busy}>
                Cancel
              </Button>
            </DialogClose>
            <Button
              onClick={() => void submitRename()}
              disabled={busy || renameValue.trim() === '' || renameTarget?.live === true}
            >
              {busy && <Loader2 className="size-4 animate-spin" />}
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteTarget !== null} onOpenChange={(open) => !open && !busy && setDeleteTarget(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete session?</DialogTitle>
            <DialogDescription>
              This permanently removes{' '}
              <span className="text-foreground font-medium">
                {deleteTarget !== null ? sessionTitle(deleteTarget) : ''}
              </span>
              {deleteTarget !== null && deleteTarget.subagentCount > 0 && (
                <>
                  {' '}
                  and its {formatNumber(deleteTarget.subagentCount)} subagent session
                  {deleteTarget.subagentCount === 1 ? '' : 's'}
                </>
              )}{' '}
              from disk. This cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogNotice live={deleteTarget?.live === true} conflict={conflict} />
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" disabled={busy}>
                Cancel
              </Button>
            </DialogClose>
            <Button
              variant="destructive"
              onClick={() => void confirmDelete()}
              disabled={busy || deleteTarget?.live === true}
            >
              {busy && <Loader2 className="size-4 animate-spin" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function DialogNotice({ live, conflict }: { live: boolean; conflict: string | null }) {
  const message = conflict ?? (live ? LIVE_HINT : null);
  if (message === null) return null;
  return (
    <p role="alert" className="text-destructive flex items-start gap-2 text-sm">
      <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden />
      {message}
    </p>
  );
}

/**
 * The column set, shared by the loading and loaded branches. The percentage
 * widths are what the fixed layout solves the grid from, so both branches must
 * render the same header or every column jumps when the rows arrive.
 */
function SessionsTableHead() {
  return (
    <TableHeader>
      <TableRow>
        <TableHead className="w-[50%] pl-6 lg:w-[40%] xl:w-[34%] 2xl:w-[26%]">Session</TableHead>
        <TableHead className="hidden xl:table-cell xl:w-[17%] 2xl:w-[13%]">Directory</TableHead>
        <TableHead className="hidden lg:table-cell lg:w-[21%] xl:w-[20%] 2xl:w-[13%]">Model</TableHead>
        <TableHead className="hidden 2xl:table-cell 2xl:w-[10%]">Branch</TableHead>
        <TableHead className="w-[24%] text-right lg:w-[17%] xl:w-[13%] 2xl:w-[10%]">Last active</TableHead>
        <TableHead className="hidden text-right 2xl:table-cell 2xl:w-[7%]">Messages</TableHead>
        <TableHead className="hidden text-right 2xl:table-cell 2xl:w-[7%]">Tokens</TableHead>
        <TableHead className="w-[26%] pr-6 lg:w-[22%] xl:w-[16%] 2xl:w-[14%]" />
      </TableRow>
    </TableHeader>
  );
}

/**
 * Six placeholder rows under the real header. Each cell repeats its loaded
 * counterpart's classes, responsive visibility included; the bars are `h-[1lh]`
 * of the cell's own text, and the three `size-9` action buttons set the row
 * height exactly as the loaded row's buttons do.
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
          <TableCell className="text-muted-foreground hidden 2xl:table-cell">
            <Skeleton className="h-[1lh] w-16 text-xs" />
          </TableCell>
          <TableCell className="text-muted-foreground truncate text-right">
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
  const title = sessionTitle(session);
  const tokens = session.tokens === null ? null : session.tokens.input + session.tokens.output;
  const lastActive = isoMs(session.lastActiveAt);
  const writeHint = session.live ? LIVE_HINT : undefined;

  return (
    <TableRow data-session-id={session.id}>
      <TableCell className="pl-6">
        <div className="flex min-w-0 items-center gap-1.5">
          <Link to={sessionUrl(session.id)} className="min-w-0 truncate font-medium hover:underline" title={title}>
            {title}
          </Link>
          {session.live && (
            <Badge className="bg-emerald-600 text-white" title={LIVE_HINT}>
              Live
            </Badge>
          )}
          {session.kind === 'fork' && (
            <Badge variant="outline" className="font-normal">
              fork
            </Badge>
          )}
          {session.subagentCount > 0 && (
            <Badge variant="secondary" className="font-normal" title="Subagent sessions">
              +{formatNumber(session.subagentCount)}
            </Badge>
          )}
        </div>
      </TableCell>
      <TableCell className="text-muted-foreground hidden xl:table-cell">
        <span className="block truncate font-mono text-xs" title={session.cwd}>
          {session.cwd}
        </span>
      </TableCell>
      <TableCell className="hidden lg:table-cell">
        {session.model !== null ? (
          // `shrink` overrides the badge's own `shrink-0`: model ids are long and
          // this column is proportional.
          <Badge variant="secondary" className="max-w-full min-w-0 shrink font-normal" title={session.model}>
            <span className="truncate">{session.model}</span>
          </Badge>
        ) : (
          <span className="text-muted-foreground text-xs">—</span>
        )}
      </TableCell>
      <TableCell className="text-muted-foreground hidden 2xl:table-cell">
        <span className="block truncate font-mono text-xs" title={session.branch ?? undefined}>
          {session.branch ?? '—'}
        </span>
      </TableCell>
      <TableCell className="text-muted-foreground truncate text-right" title={session.lastActiveAt}>
        {formatRelativeTime(lastActive)}
      </TableCell>
      <TableCell className="hidden text-right tabular-nums 2xl:table-cell">
        {formatNumber(session.messageCount)}
      </TableCell>
      <TableCell className="hidden text-right tabular-nums 2xl:table-cell">
        {tokens !== null && tokens > 0 ? formatCount(tokens) : '—'}
      </TableCell>
      <TableCell className="pr-6 text-right">
        <div className="flex justify-end gap-0.5">
          <RowAction label="Copy resume command" onClick={() => void copyResume(session)}>
            <Copy className="size-4" />
          </RowAction>
          <RowAction label="Rename" disabledHint={writeHint} onClick={onRename}>
            <Pencil className="size-4" />
          </RowAction>
          <RowAction label="Delete" destructive disabledHint={writeHint} onClick={onDelete}>
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
  disabledHint,
  children,
}: {
  label: string;
  onClick: () => void;
  destructive?: boolean;
  /** Set to disable the action; shown as the hover hint. */
  disabledHint?: string;
  children: ReactNode;
}) {
  const disabled = disabledHint !== undefined;
  return (
    // The hint sits on a wrapper: a disabled button receives no pointer events,
    // so a `title` on the button itself would never show.
    <span title={disabledHint ?? label} className="inline-flex">
      <Button
        variant="ghost"
        size="icon"
        aria-label={label}
        disabled={disabled}
        onClick={onClick}
        className={destructive === true ? 'text-muted-foreground hover:text-destructive' : 'text-muted-foreground'}
      >
        {children}
      </Button>
    </span>
  );
}
