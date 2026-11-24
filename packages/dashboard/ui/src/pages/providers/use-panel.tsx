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
  return input.firecrawl === undefined ? null : input.firecrawl ? 'firecrawl' : 'off';
}

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

/** The exclusive search provider choice and the crawler that reads pages: Firecrawl or off. */
export function UsePanel({ data, selecting, locked, lockedReason, progress, onSelect }: UsePanelProps): ReactElement {
  const configured = SEARCH_IDS.map((id) => providerById(data, id)).filter((p) => p !== undefined && p.configured);
  const firecrawl = providerById(data, 'firecrawl');
  const busySearch = selecting !== null && selecting.search !== undefined;
  const target = selecting === null ? null : crawlerTarget(selecting);
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

  const current = crawlerValue(data);
  const fireCount = firecrawl?.skill.names.length ?? 0;
  const crawler: OptionItem[] = [];
  if (firecrawl?.configured === true) {
    crawler.push({
      value: 'firecrawl',
      title: providerName('firecrawl'),
      note: roleWords('firecrawl'),
      trailing: (
        <>
          {checkBadge(firecrawl.check)}
          <span>{skillsForWork(firecrawl)}</span>
        </>
      ),
      working: target === 'firecrawl',
      workingLabel: PROVIDERS.installing,
      disabledReason: lockedReason,
    });
    crawler.push({
      value: 'off',
      title: PROVIDERS.off,
      note: crawlerOffNote(data.search),
      working: target === 'off',
      workingLabel: PROVIDERS.removing,
      disabledReason: lockedReason,
    });
  }

  let fireText: string | null = null;
  if (target !== null) fireText = target === 'off' ? PROVIDERS.removing : PROVIDERS.installing;
  else if (current !== 'off') {
    fireText =
      firecrawl?.work.missing === 'firecrawl-cli'
        ? PROVIDERS.firecrawlOnNoCli
        : fill(PROVIDERS.firecrawlOn, { n: fireCount });
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
            disabled={blocked}
            onValueChange={(value) => onSelect({ search: value === OFF ? null : (value as SearchProviderId) })}
          />
        )}
        {proof('search')}
        <Rule className="my-2" />
        <p className="m-0 text-caption font-medium text-secondary uppercase">
          {PROVIDERS.crawler} · {PROVIDERS.oneAtATime}
        </p>
        {crawler.length === 0 ? (
          <p className="m-0 text-body text-secondary">{PROVIDERS.firecrawlNoKey}</p>
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
