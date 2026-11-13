import { type ReactElement, useId } from 'react';

import { LiveLabel } from '../../components/arc-x/live-label/live-label';
import { OptionList, type OptionItem } from '../../components/arc-x/option-list/option-list';
import { Panel } from '../../components/arc-x/panel/panel';
import { ProofLine } from '../../components/arc-x/proof-line/proof-line';
import { Rule } from '../../components/arc-x/rule/rule';
import { SwapText } from '../../components/arc-x/swap-text/swap-text';
import { Badge } from '../../components/arc/badge/badge';
import { Switch } from '../../components/arc/switch/switch';
import { PROVIDERS } from '../../lib/providers-strings';
import type { ProviderSelectionInput, ProvidersResponse, SearchProviderId } from '../../lib/providers-types';
import {
  fill,
  type Progress,
  providerById,
  providerName,
  roleWords,
  SEARCH_IDS,
  skillsForWork,
} from '../../lib/providers-view';

const OFF = 'off';

export type Selecting = { search?: SearchProviderId | null; firecrawl?: boolean } | null;

export interface UsePanelProps {
  data: ProvidersResponse;
  /** The selection change in flight, if any. */
  selecting: Selecting;
  /** Another change is in flight, or the Keychain is down: nothing can be chosen. */
  locked: boolean;
  lockedReason?: string;
  progress: Progress | null;
  onSelect: (input: ProviderSelectionInput) => void;
}

function checkBadge(check: string | null): ReactElement | null {
  if (check === 'rejected')
    return (
      <Badge size="sm" tone="danger">
        {PROVIDERS.badgeRejected}
      </Badge>
    );
  if (check === 'no-credit')
    return (
      <Badge size="sm" tone="warning">
        {PROVIDERS.badgeNoCredit}
      </Badge>
    );
  if (check === 'unverified' || check === null)
    return (
      <Badge size="sm" tone="warning">
        {PROVIDERS.badgeUnverified}
      </Badge>
    );
  return null;
}

/** The exclusive search provider choice and the independent Firecrawl switch. */
export function UsePanel({ data, selecting, locked, lockedReason, progress, onSelect }: UsePanelProps): ReactElement {
  const switchId = useId();
  const descriptionId = useId();
  const configured = SEARCH_IDS.map((id) => providerById(data, id)).filter((p) => p !== undefined && p.configured);
  const firecrawl = providerById(data, 'firecrawl');
  const busySearch = selecting !== null && selecting.search !== undefined;
  const busyFire = selecting !== null && selecting.firecrawl !== undefined;
  const blocked = locked || selecting !== null;

  const options: OptionItem[] = configured.map((p) => {
    const working = busySearch && selecting.search === p!.id;
    return {
      value: p!.id,
      title: providerName(p!.id),
      note: roleWords(p!.id),
      trailing: (
        <>
          {checkBadge(p!.check)}
          <span>{skillsForWork(p!)}</span>
        </>
      ),
      working,
      workingLabel: PROVIDERS.installing,
      disabledReason: lockedReason,
    };
  });
  if (options.length > 0) {
    options.push({
      value: OFF,
      title: PROVIDERS.off,
      note: PROVIDERS.offNote,
      working: busySearch && selecting.search === null,
      workingLabel: PROVIDERS.removing,
      disabledReason: lockedReason,
    });
  }

  const fireConfigured = firecrawl?.configured === true;
  const fireCount = firecrawl?.skill.names.length ?? 0;
  let fireText: string;
  if (busyFire) fireText = selecting.firecrawl === true ? PROVIDERS.installing : PROVIDERS.removing;
  else if (!fireConfigured) fireText = PROVIDERS.firecrawlNoKey;
  else if (!data.firecrawl) fireText = PROVIDERS.firecrawlOff;
  else if (firecrawl?.work.missing === 'firecrawl-cli') fireText = PROVIDERS.firecrawlOnNoCli;
  else fireText = fill(PROVIDERS.firecrawlOn, { n: fireCount });

  const proof = (area: Progress['area']): ReactElement | null =>
    progress !== null && progress.area === area ? (
      <ProofLine
        key={progress.text}
        kind={progress.names.length > 0 ? 'installed' : 'removed'}
        text={progress.text}
        chips={progress.names}
      />
    ) : null;

  return (
    <Panel title={PROVIDERS.use}>
      <div className="grid gap-2">
        <p className="m-0 text-caption font-medium text-secondary uppercase">
          {PROVIDERS.searchProvider} · {PROVIDERS.oneAtATime}
        </p>
        {options.length === 0 ? (
          <p className="m-0 text-body text-secondary">{PROVIDERS.noneConfigured}</p>
        ) : (
          <OptionList
            label={PROVIDERS.searchProvider}
            value={data.search ?? OFF}
            options={options}
            disabled={blocked}
            onValueChange={(value) => onSelect({ search: value === OFF ? null : (value as SearchProviderId) })}
          />
        )}
        {proof('search')}
        <Rule className="my-2" />
        <div className="flex items-start justify-between gap-4">
          <div className="grid min-w-0 gap-0.5">
            <label htmlFor={switchId} className="text-body font-medium text-foreground">
              {PROVIDERS.firecrawlSwitch}
            </label>
            <span id={descriptionId} className="text-caption text-secondary">
              {busyFire ? <LiveLabel active>{fireText}</LiveLabel> : <SwapText text={fireText} />}
            </span>
          </div>
          <Switch
            id={switchId}
            aria-describedby={descriptionId}
            checked={data.firecrawl}
            disabled={(blocked && !busyFire) || !fireConfigured}
            aria-disabled={busyFire ? true : undefined}
            onCheckedChange={(next) => {
              if (selecting === null) onSelect({ firecrawl: next });
            }}
          />
        </div>
        {proof('firecrawl')}
      </div>
    </Panel>
  );
}
