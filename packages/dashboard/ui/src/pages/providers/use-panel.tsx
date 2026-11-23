import type { ReactElement } from 'react';

import { LiveLabel } from '../../components/arc-x/live-label/live-label';
import { OptionList, type OptionItem } from '../../components/arc-x/option-list/option-list';
import { Panel } from '../../components/arc-x/panel/panel';
import { ProofLine } from '../../components/arc-x/proof-line/proof-line';
import { Rule } from '../../components/arc-x/rule/rule';
import { SwapText } from '../../components/arc-x/swap-text/swap-text';
import { Badge } from '../../components/arc/badge/badge';
import { PROVIDERS } from '../../lib/providers-strings';
import type { ProviderSelectionInput, ProvidersResponse, SearchProviderId } from '../../lib/providers-types';
import {
  CRAWLER_BODIES,
  crawlerOffNote,
  type CrawlerValue,
  crawlerValue,
  fill,
  localBadge,
  type Progress,
  providerById,
  providerName,
  roleWords,
  SEARCH_IDS,
  skillsForWork,
} from '../../lib/providers-view';

const OFF = 'off';

export type Selecting = ProviderSelectionInput | null;

/** The Crawler option a selection body picks, or null for a body that does not touch the crawler. */
function crawlerTarget(input: ProviderSelectionInput): CrawlerValue | null {
  if (input.firecrawl === undefined) return null;
  if (!input.firecrawl) return 'off';
  return input.firecrawlLocal === true ? 'local' : 'cloud';
}

export interface UsePanelProps {
  data: ProvidersResponse;
  /** The selection change in flight, if any. */
  selecting: Selecting;
  /** Another change is in flight: nothing can be chosen. */
  locked: boolean;
  /**
   * Set while the Keychain is down: the search list is locked with this reason. The Crawler list is not: Firecrawl
   * Cloud is listed only with its key in the cache, and this Mac needs no key.
   */
  keychainReason?: string;
  progress: Progress | null;
  onSelect: (input: ProviderSelectionInput) => void;
}

function localTrailing(data: ProvidersResponse): ReactElement | null {
  const badge = localBadge(data.local);
  if (badge === 'not-running')
    return (
      <Badge size="sm" tone="warning">
        {PROVIDERS.badgeNotRunning}
      </Badge>
    );
  if (badge === 'check-failed')
    return (
      <Badge size="sm" tone="danger">
        {PROVIDERS.badgeCheckFailed}
      </Badge>
    );
  if (badge === 'unchecked')
    return (
      <Badge size="sm" tone="warning">
        {PROVIDERS.badgeUnverified}
      </Badge>
    );
  return null;
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

/** The exclusive search provider choice and the crawler that reads pages: Firecrawl Cloud, on this Mac, or off. */
export function UsePanel({ data, selecting, locked, keychainReason, progress, onSelect }: UsePanelProps): ReactElement {
  const configured = SEARCH_IDS.map((id) => providerById(data, id)).filter((p) => p !== undefined && p.configured);
  const firecrawl = providerById(data, 'firecrawl');
  const busySearch = selecting !== null && selecting.search !== undefined;
  const target = selecting === null ? null : crawlerTarget(selecting);
  const blocked = locked || selecting !== null;
  const searchBlocked = blocked || keychainReason !== undefined;

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
      disabledReason: keychainReason,
    };
  });
  if (options.length > 0) {
    options.push({
      value: OFF,
      title: PROVIDERS.off,
      note: PROVIDERS.offNote,
      working: busySearch && selecting.search === null,
      workingLabel: PROVIDERS.removing,
      disabledReason: keychainReason,
    });
  }

  const current = crawlerValue(data);
  const fireCount = firecrawl?.skill.names.length ?? 0;
  const workingLabel = (value: CrawlerValue): string =>
    value === 'off' ? PROVIDERS.removing : current === 'off' ? PROVIDERS.installing : PROVIDERS.switching;
  const crawler: OptionItem[] = [];
  if (firecrawl?.configured === true) {
    crawler.push({
      value: 'cloud',
      title: PROVIDERS.crawlerCloud,
      note: roleWords('firecrawl'),
      trailing: (
        <>
          {checkBadge(firecrawl.check)}
          <span>{skillsForWork(firecrawl)}</span>
        </>
      ),
      working: target === 'cloud',
      workingLabel: workingLabel('cloud'),
    });
  }
  if (data.local.url !== null) {
    crawler.push({
      value: 'local',
      title: PROVIDERS.crawlerLocal,
      note: data.local.url,
      trailing: localTrailing(data),
      working: target === 'local',
      workingLabel: workingLabel('local'),
    });
  }
  if (crawler.length > 0) {
    crawler.push({
      value: 'off',
      title: PROVIDERS.off,
      note: crawlerOffNote(data.search),
      working: target === 'off',
      workingLabel: workingLabel('off'),
    });
  }

  let fireText: string | null = null;
  if (target !== null) fireText = workingLabel(target);
  else if (current !== 'off') {
    const noCli = firecrawl?.work.missing === 'firecrawl-cli';
    if (current === 'local') fireText = noCli ? PROVIDERS.localOnNoCli : fill(PROVIDERS.localOn, { n: fireCount });
    else fireText = noCli ? PROVIDERS.firecrawlOnNoCli : fill(PROVIDERS.firecrawlOn, { n: fireCount });
  }

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
            disabled={searchBlocked}
            onValueChange={(value) => onSelect({ search: value === OFF ? null : (value as SearchProviderId) })}
          />
        )}
        {proof('search')}
        <Rule className="my-2" />
        <p className="m-0 text-caption font-medium text-secondary uppercase">
          {PROVIDERS.crawler} · {PROVIDERS.oneAtATime}
        </p>
        {crawler.length === 0 ? (
          <p className="m-0 text-body text-secondary">{PROVIDERS.crawlerNone}</p>
        ) : (
          <OptionList
            label={PROVIDERS.crawler}
            value={current}
            options={crawler}
            disabled={blocked}
            wrap
            onValueChange={(value) => onSelect(CRAWLER_BODIES[value as CrawlerValue])}
          />
        )}
        {fireText !== null ? (
          <p className="m-0 text-caption text-secondary">
            {target !== null ? <LiveLabel active>{fireText}</LiveLabel> : <SwapText text={fireText} />}
          </p>
        ) : null}
        {proof('firecrawl')}
      </div>
    </Panel>
  );
}
