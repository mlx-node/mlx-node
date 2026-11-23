import { ExternalLink, TriangleAlert } from 'lucide-react';
import { type ReactElement, useEffect, useRef, useState } from 'react';

import { ConfirmPopover } from '../../components/arc-x/confirm-popover/confirm-popover';
import { LiveLabel } from '../../components/arc-x/live-label/live-label';
import { MorphLoader } from '../../components/arc-x/morph-loader/morph-loader';
import { SwapCell } from '../../components/arc-x/swap-cell/swap-cell';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { CopyButton } from '../../components/arc/copy-button/copy-button';
import { Input } from '../../components/arc/input/input';
import { PROVIDERS } from '../../lib/providers-strings';
import type { LocalFirecrawl as LocalState } from '../../lib/providers-types';
import { LOCAL_DEFAULT_URL, LOCAL_GUIDE, localBadge, localLine } from '../../lib/providers-view';

export interface LocalFirecrawlProps {
  local: LocalState;
  /** Firecrawl reads pages on this Mac now. */
  inUse: boolean;
  /** This block's own request in flight. */
  busy: 'save' | 'check' | 'remove' | null;
  /** Another change of the page is in flight. */
  locked: boolean;
  /** Check and save a typed address. Resolves to null when it was saved, else to the field error to show. */
  onSave: (raw: string) => Promise<string | null>;
  onCheck: () => void;
  /** Rejects with an Error whose message is fixed text. */
  onRemove: () => Promise<void>;
}

/** The label column of the block, as in the key rows of the card. */
const LABEL_GRID = 'grid items-start gap-x-4 gap-y-2 grid-cols-[6rem_minmax(0,1fr)]';

function Command({ text }: { text: string }): ReactElement {
  return (
    <span className="flex flex-wrap items-center gap-2">
      <code className="font-mono text-caption text-foreground">{text}</code>
      <CopyButton value={text} label="Copy command" iconOnly variant="plain" />
    </span>
  );
}

function StateBadge({ local, checking }: { local: LocalState; checking: boolean }): ReactElement {
  if (checking) {
    return (
      <span className="inline-flex items-center gap-1.5 text-caption font-medium text-secondary">
        <MorphLoader size={14} status="loading" />
        <LiveLabel active>{PROVIDERS.badgeChecking}</LiveLabel>
      </span>
    );
  }
  if (local.url === null) return <Badge outline>{PROVIDERS.badgeNotSetUp}</Badge>;
  const badge = localBadge(local);
  if (badge === 'running')
    return (
      <Badge tone="success" dot>
        {PROVIDERS.badgeRunning}
      </Badge>
    );
  if (badge === 'not-running')
    return (
      <Badge tone="warning" dot>
        {PROVIDERS.badgeNotRunning}
      </Badge>
    );
  if (badge === 'check-failed')
    return (
      <Badge tone="danger" dot>
        {PROVIDERS.badgeCheckFailed}
      </Badge>
    );
  return (
    <Badge tone="warning" dot>
      {PROVIDERS.badgeUnverified}
    </Badge>
  );
}

/**
 * Firecrawl on this Mac, in the Firecrawl card: the address of a `firecrawl-local serve` the user runs, checked before
 * it is saved, and how to install and start one. The app never installs or starts it.
 */
export function LocalFirecrawl(props: LocalFirecrawlProps): ReactElement {
  const { local } = props;
  const [draft, setDraft] = useState(LOCAL_DEFAULT_URL);
  const [replacing, setReplacing] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const saving = props.busy === 'save';
  const entering = local.url === null || replacing;
  const off = props.locked || props.busy !== null;
  const root = useRef<HTMLDivElement>(null);
  /** Where focus goes when the form and the saved row swap: the control that had it is hidden by the swap. */
  const refocus = useRef<'saved' | 'field' | null>(null);
  /** A Remove went through: once its popover has closed, focus goes to the address field. */
  const removed = useRef(false);

  useEffect(() => {
    const target = refocus.current;
    refocus.current = null;
    if (target === null) return;
    root.current
      ?.querySelector<HTMLElement>(target === 'field' ? '[data-local-form] input' : '[data-local-row] button')
      ?.focus();
  }, [entering]);

  const save = async (): Promise<void> => {
    if (off) return;
    refocus.current = root.current?.contains(document.activeElement) === true ? 'saved' : null;
    setError(undefined);
    const failure = await props.onSave(draft);
    if (failure === null) {
      setReplacing(false);
      setDraft(LOCAL_DEFAULT_URL);
    } else {
      refocus.current = null;
      setError(failure);
    }
  };

  const line = localLine(local);
  const form = (
    <div className="grid gap-2" data-local-form="">
      {local.url === null ? (
        <div className="grid gap-1 text-caption text-secondary">
          <span>{PROVIDERS.localText}</span>
          <span className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span>{PROVIDERS.localNeeds}</span>
            <a className="link inline-flex items-center gap-1" href={LOCAL_GUIDE} target="_blank" rel="noreferrer">
              {PROVIDERS.localGuide}
              <ExternalLink size={12} aria-hidden="true" />
            </a>
          </span>
          <Command text={PROVIDERS.localInstall} />
          <Command text={PROVIDERS.localStart} />
          <span>{PROVIDERS.localStartNote}</span>
        </div>
      ) : null}
      <Input
        label={PROVIDERS.localAddress}
        value={draft}
        description={error === undefined ? PROVIDERS.localAddressHelp : undefined}
        error={error}
        readOnly={saving}
        disabled={props.locked}
        autoComplete="off"
        spellCheck={false}
        autoCapitalize="off"
        autoCorrect="off"
        className="font-mono"
        onChange={(event) => {
          setDraft(event.target.value);
          if (error !== undefined) setError(undefined);
        }}
        onKeyDown={(event) => {
          if (event.key !== 'Enter' || event.nativeEvent.isComposing) return;
          event.preventDefault();
          void save();
        }}
      />
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="primary"
          size="sm"
          loading={saving}
          disabled={props.locked || (!saving && (props.busy !== null || draft.trim() === ''))}
          onClick={() => void save()}
        >
          {PROVIDERS.localSave}
        </Button>
        {replacing ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={saving}
            onClick={() => {
              refocus.current = 'saved';
              setDraft(LOCAL_DEFAULT_URL);
              setError(undefined);
              setReplacing(false);
            }}
          >
            {PROVIDERS.cancel}
          </Button>
        ) : null}
      </div>
    </div>
  );

  const saved = (
    <div className="grid gap-2" data-local-row="">
      <div className="grid gap-0.5">
        <span className="font-mono text-body text-foreground">{local.url ?? ''}</span>
        {line !== null ? (
          <span
            className={
              line.tone === 'danger'
                ? 'text-caption text-danger'
                : line.tone === 'warning'
                  ? 'text-caption text-warning'
                  : 'text-caption text-secondary'
            }
          >
            {line.text}
          </span>
        ) : null}
        {local.check === 'ok' && local.cloud === 'fallback' ? (
          <span className="grid gap-0.5 text-caption">
            <span className="flex items-start gap-1.5 text-warning">
              <TriangleAlert size={12} aria-hidden="true" className="mt-0.5 shrink-0" />
              <span>{PROVIDERS.localCloudFallback}</span>
            </span>
            <Command text={PROVIDERS.localStart} />
          </span>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="secondary"
          size="sm"
          disabled={props.locked || (props.busy !== null && props.busy !== 'check')}
          loading={props.busy === 'check'}
          onClick={props.onCheck}
        >
          {PROVIDERS.checkAgain}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={off}
          onClick={() => {
            setError(undefined);
            setDraft(local.url ?? LOCAL_DEFAULT_URL);
            refocus.current = 'field';
            setReplacing(true);
          }}
        >
          {PROVIDERS.replace}
        </Button>
        <ConfirmPopover
          trigger={{ kind: 'button', label: PROVIDERS.remove, size: 'sm', variant: 'ghost' }}
          disabled={off}
          title={PROVIDERS.localRemoveTitle}
          description={
            props.inUse ? `${PROVIDERS.localRemoveText} ${PROVIDERS.localRemoveSelected}` : PROVIDERS.localRemoveText
          }
          confirmLabel={PROVIDERS.removeConfirm}
          workingLabel={PROVIDERS.removing2}
          successLabel={PROVIDERS.removedDone}
          tone="danger"
          align="start"
          onConfirm={async () => {
            await props.onRemove();
            removed.current = true;
          }}
          onDone={() => {
            if (!removed.current) return;
            removed.current = false;
            root.current?.querySelector<HTMLElement>('[data-local-form] input')?.focus();
          }}
        />
      </div>
    </div>
  );

  return (
    <div ref={root} className={LABEL_GRID} data-local="">
      <span className="text-body text-secondary">{PROVIDERS.localRow}</span>
      <div className="grid gap-2">
        <span className="justify-self-start">
          <StateBadge local={local} checking={saving || props.busy === 'check'} />
        </span>
        <SwapCell active={entering ? 'form' : 'saved'} items={{ form, saved }} />
        <p className="m-0 text-caption text-secondary">{PROVIDERS.localLimits}</p>
      </div>
    </div>
  );
}
