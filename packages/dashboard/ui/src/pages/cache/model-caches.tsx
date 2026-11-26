import { Trash2 } from 'lucide-react';
import { useEffect, useRef, useState, type ReactElement, type ReactNode } from 'react';

import { ConfirmPopover, type ConfirmContent } from '../../components/arc-x/confirm-popover/confirm-popover';
import { DataTable, type DataColumn } from '../../components/arc-x/data-table/data-table';
import { MiddleText } from '../../components/arc-x/middle-text/middle-text';
import { NumberValue } from '../../components/arc-x/number-value/number-value';
import { Panel } from '../../components/arc-x/panel/panel';
import { PathText } from '../../components/arc-x/path-text/path-text';
import { StatGroup, StatTile } from '../../components/arc-x/stat-tile/stat-tile';
import { Alert } from '../../components/arc/alert/alert';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { CopyButton } from '../../components/arc/copy-button/copy-button';
import { HOST_WORD } from '../../lib/engine-view';
import {
  formatBytes,
  formatBytesParts,
  formatClock,
  formatCount,
  formatNumber,
  formatPercent,
  formatRelativeTime,
} from '../../lib/format';
import { deleteNamespace, lockWord, prefixTotals } from '../../lib/prefix-cache';
import { toast } from '../../lib/toast';
import type {
  PrefixCacheDeleteResult,
  PrefixCacheNamespace,
  PrefixCacheResponse,
  PrefixCacheRoot,
} from '../../lib/types';
import { useJson, type AsyncState } from '../../lib/use-api';

const TITLE = 'Model caches';
const DESCRIPTION =
  'One folder per model identity (Qwen3.5, Flash Next). The engine removes one unused for 14 days, and the oldest idle ones when the folder is over its limit.';
const EMPTY = {
  title: 'No model cache yet',
  description: 'A Qwen3.5 or Flash Next model writes here when it runs with the disk cache on.',
};
const HELD_REASON = 'In use by a running engine. Unload the model or stop the engine first.';
const UNKNOWN_REASON = 'The lock cannot be checked on this system.';

function plural(n: number, one: string, many: string): string {
  return n === 1 ? one : many;
}

function folderOf(root: string, namespace: string): string {
  return `${root.endsWith('/') ? root.slice(0, -1) : root}/${namespace}`;
}

function shortName(namespace: string): string {
  return `${namespace.slice(0, 8)}…${namespace.slice(-8)}`;
}

function Strong({ children }: { children: ReactNode }): ReactElement {
  return <span className="font-medium text-foreground">{children}</span>;
}

/** A value split into the tile's value and unit; null when nothing was measured. */
function bytesTile(bytes: number | null, lowerBound: boolean): { value: string | null; unit?: string } {
  if (bytes === null) return { value: null };
  const parts = formatBytesParts(bytes);
  return { value: `${lowerBound ? '≥ ' : ''}${parts.value}`, unit: parts.unit };
}

function StateBadge({ state }: { state: PrefixCacheNamespace['lock'] }): ReactElement {
  if (state === 'held') {
    return (
      <Badge size="sm" tone="success" dot>
        {lockWord(state)}
      </Badge>
    );
  }
  return (
    <Badge size="sm" outline={state === 'unknown'}>
      {lockWord(state)}
    </Badge>
  );
}

function CacheName({ root, ns }: { root: string; ns: PrefixCacheNamespace }): ReactElement {
  return (
    <span className="flex min-w-0 items-center gap-1">
      {ns.model === null ? (
        <MiddleText text={ns.namespace} className="font-mono" />
      ) : (
        <>
          <MiddleText text={ns.model} />
          <span className="shrink-0 font-mono text-secondary" title={ns.namespace}>
            {ns.namespace.slice(0, 8)}
          </span>
        </>
      )}
      <CopyButton value={folderOf(root, ns.namespace)} iconOnly variant="plain" label="Copy cache folder" />
    </span>
  );
}

interface RootPanelProps {
  root: PrefixCacheRoot | null;
  canLock: boolean;
  prefix: AsyncState<PrefixCacheResponse>;
  receivedAt: number | null;
  results: Map<string, PrefixCacheDeleteResult>;
}

function deleteContent(root: string, ns: PrefixCacheNamespace, props: RootPanelProps): ConfirmContent {
  const key = folderOf(root, ns.namespace);
  const what = `${formatNumber(ns.files)} ${plural(ns.files, 'file', 'files')} · ${formatBytes(ns.bytes)}`;
  return {
    title: 'Delete this model cache?',
    description: (
      <>
        Removes {ns.complete ? '' : 'at least '}
        <Strong>{what}</Strong> from <span className="font-mono">{shortName(ns.namespace)}</span>. The next turn of its
        model prefills its whole prompt again. This cannot be undone.
      </>
    ),
    confirmLabel: 'Delete',
    workingLabel: 'Deleting…',
    successLabel: 'Deleted',
    tone: 'danger',
    errorTitle: 'Could not delete the cache',
    onConfirm: async () => {
      try {
        props.results.set(key, await deleteNamespace(root, ns.namespace, ns.openedMs));
      } catch (err) {
        props.prefix.reload();
        throw err;
      }
    },
    onDone: () => {
      const result = props.results.get(key);
      props.results.delete(key);
      props.prefix.patch((data) => ({
        ...data,
        roots: data.roots.map((r) =>
          r.root === root ? { ...r, namespaces: r.namespaces.filter((n) => n.namespace !== ns.namespace) } : r,
        ),
      }));
      if (result === undefined) return;
      const parts = [
        `${formatNumber(result.removed)} ${plural(result.removed, 'file', 'files')} removed`,
        `${formatBytes(result.freedBytes)} freed`,
      ];
      if (result.left > 0)
        parts.push(`${formatNumber(result.left)} ${plural(result.left, 'folder', 'folders')} inside was left`);
      toast.success('Deleted model cache', { description: parts.join(' · ') });
    },
  };
}

function RootPanel(props: RootPanelProps): ReactElement {
  const { root, prefix } = props;
  const loading = root === null && prefix.error === undefined;
  const failed = root === null && prefix.error !== undefined;
  const rows = root?.namespaces ?? [];
  const totals = root === null ? null : prefixTotals(root);
  const exists = root?.exists === true && root.readError === null;
  const shown = totals !== null && exists;
  const engine = root?.engines[0];

  const size = bytesTile(shown ? totals.bytes : null, totals?.complete === false);
  const free = bytesTile(root?.volume?.availableBytes ?? null, false);
  let sizeHint = 'no running engine reports a limit';
  if (totals !== null && totals.limitBytes !== null) {
    sizeHint = `${formatPercent(totals.bytes / totals.limitBytes)} of ${formatBytes(totals.limitBytes)} limit`;
    if (totals.limitEngines > 1) sizeHint += ` · the smallest of ${formatNumber(totals.limitEngines)} engines`;
  } else if (totals !== null && !totals.complete) {
    sizeHint = 'counting files…';
  }
  const cachesHint =
    totals === null
      ? '—'
      : `${formatNumber(totals.held)} in use · ${formatNumber(totals.idle)} idle${totals.unknown > 0 ? ` · ${formatNumber(totals.unknown)} unknown` : ''}`;

  const columns: DataColumn<PrefixCacheNamespace>[] = [
    {
      id: 'cache',
      header: 'Cache',
      flex: true,
      truncate: true,
      minWidth: 140,
      sortValue: (ns) => ns.model ?? ns.namespace,
      cell: (ns) => <CacheName root={root?.root ?? ''} ns={ns} />,
    },
    {
      id: 'state',
      header: 'State',
      width: 96,
      sortValue: (ns) => (ns.lock === 'held' ? 0 : ns.lock === 'free' ? 1 : 2),
      cell: (ns) => <StateBadge state={ns.lock} />,
    },
    {
      id: 'size',
      header: 'Size',
      align: 'end',
      width: 96,
      sortValue: (ns) => ns.bytes,
      cell: (ns) => (
        <span title={ns.complete ? `as of ${formatClock(ns.measuredAtMs)}` : 'Still counting'}>
          {ns.complete ? '' : '≥ '}
          <NumberValue text={formatBytes(ns.bytes)} countUp={false} />
        </span>
      ),
    },
    {
      id: 'files',
      header: 'Files',
      align: 'end',
      width: 80,
      minTableWidth: 560,
      sortValue: (ns) => ns.files,
      cell: (ns) => formatCount(ns.files),
    },
    {
      id: 'lastUsed',
      header: 'Last used',
      width: 120,
      minTableWidth: 480,
      sortValue: (ns) => ns.lastUsedMs,
      cell: (ns) =>
        ns.lock === 'held' ? (
          'in use now'
        ) : ns.lastUsedMs === null ? (
          '—'
        ) : (
          <span title={new Date(ns.lastUsedMs).toISOString()}>{formatRelativeTime(ns.lastUsedMs)}</span>
        ),
    },
    {
      id: 'actions',
      header: '',
      align: 'end',
      width: 68,
      cell: (ns) => (
        <ConfirmPopover
          trigger={{
            kind: 'icon',
            label: `Delete cache ${ns.namespace.slice(0, 8)}…`,
            icon: <Trash2 size={16} strokeWidth={1.75} aria-hidden="true" />,
            size: 'sm',
          }}
          align="end"
          disabled={ns.lock !== 'free' || !props.canLock}
          disabledReason={
            ns.lock === 'held' ? HELD_REASON : ns.lock === 'unknown' || !props.canLock ? UNKNOWN_REASON : undefined
          }
          {...deleteContent(root?.root ?? '', ns, props)}
        />
      ),
    },
  ];

  return (
    <Panel
      title={TITLE}
      titleSuffix={
        root !== null && !root.isDefault && engine !== undefined
          ? ` · ${HOST_WORD[engine.host]} engine, pid ${String(engine.pid)}`
          : undefined
      }
      description={DESCRIPTION}
      padding="none"
      actions={
        root === null ? undefined : (
          <span className="flex min-w-0 items-center gap-1">
            <span className="min-w-0 truncate text-caption text-secondary">
              <PathText path={root.root} maxChars={44} />
              {root.readError !== null ? ' · could not be read' : root.exists ? '' : ' · not created yet'}
            </span>
            <CopyButton value={root.root} iconOnly variant="plain" label="Copy model caches folder" />
          </span>
        )
      }
    >
      <div className="grid min-w-0 gap-3">
        {prefix.error !== undefined && root !== null ? (
          <div className="px-4">
            <Alert tone="warning" title="Could not refresh">
              <span className="grid justify-items-start gap-2">
                <span>{`Showing data from ${formatClock(props.receivedAt)}.`}</span>
                <Button variant="secondary" size="sm" onClick={prefix.reload}>
                  Try again
                </Button>
              </span>
            </Alert>
          </div>
        ) : null}
        <div className="px-4">
          <StatGroup label={TITLE}>
            <StatTile
              label="Size on disk"
              value={size.value}
              unit={size.unit}
              countUp={false}
              loading={loading}
              hint={sizeHint}
            />
            <StatTile
              label="Caches"
              value={shown ? formatCount(totals.caches) : null}
              countUp={false}
              loading={loading}
              hint={cachesHint}
            />
            <StatTile
              label="Files"
              value={shown ? formatCount(totals.files) : null}
              countUp={false}
              loading={loading}
              hint={
                totals === null ? '—' : `in ${formatNumber(totals.caches)} ${plural(totals.caches, 'cache', 'caches')}`
              }
            />
            <StatTile
              label="Free on volume"
              value={free.value}
              unit={free.unit}
              countUp={false}
              loading={loading}
              hint={
                root?.volume != null
                  ? `of ${formatBytes(root.volume.sizeBytes)}`
                  : exists
                    ? 'could not be measured'
                    : '—'
              }
            />
          </StatGroup>
        </div>
        <DataTable
          caption={TITLE}
          density="compact"
          columns={columns}
          rows={rows}
          rowKey={(ns) => ns.namespace}
          rowAttrs={(ns) => ({ 'data-namespace': ns.namespace, 'data-lock': ns.lock })}
          defaultSort={{ id: 'size', direction: 'desc' }}
          loading={loading}
          loadingRows={3}
          error={failed ? { message: prefix.error?.message ?? '', retry: prefix.reload } : undefined}
          empty={
            root?.readError != null
              ? {
                  title: 'Could not read this folder',
                  description: `${root.readError === 'ENOTDIR' ? 'It is not a folder' : `The system answered ${root.readError}`}. Its caches, if any, are not listed.`,
                }
              : EMPTY
          }
          footer={
            root?.truncated === true
              ? `Showing ${formatNumber(rows.length)} caches. More exist in this folder.`
              : undefined
          }
        />
      </div>
    </Panel>
  );
}

/** The model caches under `prefix-cache/`: one panel per root the runtime lists. */
export function ModelCaches(): ReactElement {
  const prefix = useJson<PrefixCacheResponse>('/cache/prefix', { live: true });
  const [receivedAt, setReceivedAt] = useState<number | null>(null);
  const results = useRef(new Map<string, PrefixCacheDeleteResult>());
  const data = prefix.data;

  useEffect(() => {
    if (data === undefined) return;
    setReceivedAt(Date.now());
    if (data.counting) prefix.reload();
  }, [data]);

  const shared = { canLock: data?.canLock ?? true, prefix, receivedAt, results: results.current };
  if (data === undefined) return <RootPanel root={null} {...shared} />;
  return (
    <>
      {data.roots.map((root) => (
        <RootPanel key={root.root} root={root} {...shared} />
      ))}
    </>
  );
}
