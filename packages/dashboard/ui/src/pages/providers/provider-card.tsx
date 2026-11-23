import { ExternalLink, Info, TriangleAlert } from 'lucide-react';
import { type ReactElement, useEffect, useRef, useState } from 'react';

import { ConfirmPopover } from '../../components/arc-x/confirm-popover/confirm-popover';
import { LiveLabel } from '../../components/arc-x/live-label/live-label';
import { MorphLoader } from '../../components/arc-x/morph-loader/morph-loader';
import { Panel } from '../../components/arc-x/panel/panel';
import { Rule } from '../../components/arc-x/rule/rule';
import { SecretField } from '../../components/arc-x/secret-field/secret-field';
import { SiteMark } from '../../components/arc-x/site-mark/site-mark';
import { SwapCell } from '../../components/arc-x/swap-cell/swap-cell';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { CopyButton } from '../../components/arc/copy-button/copy-button';
import { Tooltip } from '../../components/arc/tooltip/tooltip';
import { PROVIDERS } from '../../lib/providers-strings';
import type { ProviderState } from '../../lib/providers-types';
import {
  badgeState,
  BILLING_PAGES,
  checkedLine,
  factsLine,
  fill,
  KEY_PAGES,
  keyReachesWork,
  providerName,
  skillSource,
} from '../../lib/providers-view';

export interface ProviderCardProps {
  provider: ProviderState;
  /** This provider's key is being saved or checked. */
  checking: boolean;
  /** Another change is in flight, or the Keychain is down. */
  locked: boolean;
  /** Why the controls are off, when the Keychain is down. */
  lockedReason?: string;
  /**
   * Save a typed key. Resolves to null when it was saved, else to the field error to show. The caller sends the key in
   * one request and keeps no copy.
   */
  onSave: (key: string) => Promise<string | null>;
  onCheck: () => void;
  /** Rejects with an Error whose message is fixed text. */
  onRemove: () => Promise<void>;
  onRemoved: () => void;
}

function StatusBadge({ provider, checking }: { provider: ProviderState; checking: boolean }): ReactElement {
  const state = badgeState(provider, checking);
  if (state === 'checking') {
    return (
      <span className="inline-flex items-center gap-1.5 text-caption font-medium text-secondary">
        <MorphLoader size={14} status="loading" />
        <LiveLabel active>{PROVIDERS.badgeChecking}</LiveLabel>
      </span>
    );
  }
  if (state === 'no-key') return <Badge outline>{PROVIDERS.badgeNoKey}</Badge>;
  if (state === 'valid')
    return (
      <Badge tone="success" dot>
        {PROVIDERS.badgeValid}
      </Badge>
    );
  if (state === 'rejected')
    return (
      <Badge tone="danger" dot>
        {PROVIDERS.badgeRejected}
      </Badge>
    );
  if (state === 'no-credit')
    return (
      <Badge tone="warning" dot>
        {PROVIDERS.badgeNoCredit}
      </Badge>
    );
  return (
    <Badge tone="warning" dot>
      {PROVIDERS.badgeUnverified}
    </Badge>
  );
}

/** The label column of a card: the key form, the key row and the skills row line up on it at every width. */
const LABEL_GRID = 'grid items-start gap-x-4 gap-y-2 grid-cols-[6rem_minmax(0,1fr)]';

/** One provider: its key (entered, saved, checked, replaced, removed), its skills and what reaches Work. */
export function ProviderCard(props: ProviderCardProps): ReactElement {
  const p = props.provider;
  const name = providerName(p.id);
  const [draft, setDraft] = useState('');
  const [replacing, setReplacing] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const [saving, setSaving] = useState(false);
  const entering = !p.configured || replacing;
  const off = props.locked || props.checking;
  const root = useRef<HTMLDivElement>(null);
  /** Where focus goes when the form and the key row swap: the control that had it is hidden by the swap. */
  const refocus = useRef<'key-row' | 'replace' | null>(null);
  /** A Remove went through: once its popover has closed, focus goes to the new key field, not to the hidden trigger. */
  const removed = useRef(false);

  useEffect(() => {
    const target = refocus.current;
    refocus.current = null;
    if (target === null) return;
    root.current
      ?.querySelector<HTMLElement>(target === 'replace' ? '[data-replace]' : '[data-key-row] button')
      ?.focus();
  }, [entering]);

  const save = async (): Promise<void> => {
    if (saving || off) return;
    refocus.current = root.current?.contains(document.activeElement) === true ? 'key-row' : null;
    setSaving(true);
    setError(undefined);
    try {
      const failure = await props.onSave(draft);
      if (failure === null) {
        setDraft('');
        setReplacing(false);
      } else {
        refocus.current = null;
        setError(failure);
      }
    } finally {
      setSaving(false);
    }
  };

  const help =
    error !== undefined
      ? undefined
      : replacing
        ? PROVIDERS.replaceHelp
        : p.id === 'brave'
          ? PROVIDERS.saveHelpBrave
          : fill(PROVIDERS.saveHelp, { provider: name });

  const check = (label: string, variant: 'secondary' | 'ghost' | 'primary'): ReactElement => {
    const button = (
      <Button
        type="button"
        variant={variant}
        size="sm"
        disabled={props.locked}
        loading={props.checking && !saving}
        onClick={props.onCheck}
      >
        {label}
      </Button>
    );
    return p.id === 'brave' ? <Tooltip content={PROVIDERS.checkBraveTip}>{button}</Tooltip> : button;
  };
  const replace = (variant: 'ghost' | 'primary'): ReactElement => (
    <Button
      type="button"
      variant={variant}
      size="sm"
      disabled={off}
      data-replace=""
      onClick={() => {
        setError(undefined);
        setReplacing(true);
      }}
    >
      {PROVIDERS.replace}
    </Button>
  );
  const remove = (
    <ConfirmPopover
      trigger={{ kind: 'button', label: PROVIDERS.remove, size: 'sm', variant: 'ghost' }}
      disabled={off}
      disabledReason={props.lockedReason}
      title={fill(PROVIDERS.removeTitle, { provider: name })}
      description={
        p.selected
          ? `${PROVIDERS.removeText} ${fill(PROVIDERS.removeSelected, { provider: name })}`
          : PROVIDERS.removeText
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
        props.onRemoved();
        if (!removed.current) return;
        removed.current = false;
        root.current?.querySelector<HTMLElement>('[data-key-form] input')?.focus();
      }}
    />
  );

  let line: ReactElement | null = null;
  let buttons: ReactElement;
  if (p.check === 'ok') {
    const facts = factsLine(p);
    line = (
      <span className="grid gap-0.5 text-caption text-secondary">
        {checkedLine(p) !== null ? <span>{checkedLine(p)}</span> : null}
        {facts !== null ? <span>{facts}</span> : null}
      </span>
    );
    buttons = (
      <>
        {check(PROVIDERS.check, 'secondary')}
        {replace('ghost')}
        {remove}
      </>
    );
  } else if (p.check === 'rejected') {
    line = <span className="text-caption text-danger">{fill(PROVIDERS.rejectedLater, { provider: name })}</span>;
    buttons = (
      <>
        {replace('primary')}
        {remove}
        {check(PROVIDERS.checkAgain, 'ghost')}
      </>
    );
  } else if (p.check === 'no-credit') {
    line = (
      <span className="text-caption text-warning">
        {fill(PROVIDERS.noCredit, { provider: name })}
        <a className="link" href={BILLING_PAGES[p.id]} target="_blank" rel="noreferrer">
          {BILLING_PAGES[p.id]}
        </a>
      </span>
    );
    buttons = (
      <>
        {check(PROVIDERS.checkAgain, 'secondary')}
        {replace('ghost')}
        {remove}
      </>
    );
  } else {
    line =
      p.check === 'unverified' ? (
        <span className="text-caption text-secondary">{fill(PROVIDERS.unverified, { provider: name })}</span>
      ) : null;
    buttons = (
      <>
        {check(PROVIDERS.checkAgain, 'secondary')}
        {replace('ghost')}
        {remove}
      </>
    );
  }

  const form = (
    <div key={entering ? 'shown' : 'hidden'} className="grid gap-2" data-key-form="">
      <SecretField
        label={PROVIDERS.keyLabel}
        context={name}
        value={draft}
        onValueChange={(value) => {
          setDraft(value);
          if (error !== undefined) setError(undefined);
        }}
        placeholder={fill(PROVIDERS.keyPlaceholder, { provider: name })}
        description={help}
        error={error}
        readOnly={saving}
        disabled={props.locked}
        autoFocus={replacing}
        onSubmit={() => void save()}
      />
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="primary"
          size="sm"
          loading={saving}
          disabled={props.locked || (!saving && draft.trim() === '')}
          onClick={() => void save()}
        >
          {PROVIDERS.save}
          <span className="sr-only"> for {name}</span>
        </Button>
        {replacing ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            disabled={saving}
            onClick={() => {
              refocus.current = 'replace';
              setDraft('');
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

  const keyRow = (
    <div className={LABEL_GRID}>
      <span className="text-body text-secondary">{PROVIDERS.keyRow}</span>
      <div className="grid gap-2" data-key-row="">
        <div className="grid gap-0.5">
          <span className="text-body font-medium text-foreground">{PROVIDERS.keySaved}</span>
          {line}
        </div>
        <div className="flex flex-wrap items-center gap-2">{buttons}</div>
      </div>
    </div>
  );

  return (
    <Panel ariaLabel={name}>
      <div ref={root} className="grid gap-3">
        <div className="grid gap-1">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <SiteMark host={p.id} letter={name} size={20} />
            <h3 className="m-0 text-strong font-medium text-foreground">{name}</h3>
            {PROVIDERS.roles[p.id].map((role) => (
              <Badge key={role} size="sm" outline>
                {role}
              </Badge>
            ))}
            <span className="ms-auto">
              <StatusBadge provider={p} checking={props.checking || saving} />
            </span>
          </div>
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <p className="m-0 text-body text-secondary">{PROVIDERS.lines[p.id]}</p>
            <a
              className="link inline-flex items-center gap-1 text-caption"
              href={KEY_PAGES[p.id]}
              target="_blank"
              rel="noreferrer"
            >
              {PROVIDERS.getKey}
              <ExternalLink size={12} aria-hidden="true" />
            </a>
          </div>
        </div>
        <Rule />
        <div className={LABEL_GRID}>
          <SwapCell active={entering ? 'form' : 'key'} items={{ form, key: keyRow }} className="col-span-2" />
          {keyReachesWork(p) ? (
            <p className="m-0 flex items-start gap-1.5 text-caption text-secondary col-start-2">
              <Info size={12} aria-hidden="true" className="mt-0.5 shrink-0" />
              <span>{PROVIDERS.workKeyNotice}</span>
            </p>
          ) : null}
          {p.skill.installed ? (
            <>
              <span className="text-body text-secondary">{PROVIDERS.skills}</span>
              <span className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5">
                <span className="font-mono text-caption text-foreground">{p.skill.names.join(' · ')}</span>
                <span className="text-caption text-secondary">{skillSource(p)}</span>
              </span>
            </>
          ) : null}
          {p.skill.installed && p.work.missing === 'firecrawl-cli' ? (
            <div className="grid gap-1 text-caption col-start-2">
              <p className="m-0 flex items-start gap-1.5 text-warning">
                <TriangleAlert size={12} aria-hidden="true" className="mt-0.5 shrink-0" />
                <span>{PROVIDERS.firecrawlCli}</span>
              </p>
              <span className="flex flex-wrap items-center gap-2">
                <code className="font-mono text-foreground">{PROVIDERS.firecrawlCliCommand}</code>
                <CopyButton value={PROVIDERS.firecrawlCliCommand} label="Copy command" iconOnly variant="plain" />
              </span>
              <span className="text-secondary">{PROVIDERS.firecrawlCliChat}</span>
            </div>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}
