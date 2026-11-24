import { FileText, Globe, Paperclip, Plus } from 'lucide-react';
import { type ReactElement, useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import {
  Menu,
  MenuGroup,
  MenuItem,
  MenuList,
  MenuRadio,
  MenuSeparator,
  MenuSub,
} from '../../components/arc-x/menu/menu';
import type { ModelBaseOption, ModelVariantOption } from '../../components/arc-x/model-picker/model-picker';
import { ToolChip } from '../../components/arc-x/tool-chip/tool-chip';
import { DashboardApiError, getJson, mutate } from '../../lib/api';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatHostState, ChatMode } from '../../lib/chat-types';
import { formatBytes } from '../../lib/format';
import { PROVIDERS } from '../../lib/providers-strings';
import type { ProviderId, ProvidersResponse, ProviderState, SearchProviderId } from '../../lib/providers-types';
import { CRAWLER_BODIES, type CrawlerValue, crawlerValue, SEARCH_IDS } from '../../lib/providers-view';
import { toast } from '../../lib/toast';
import { providerName } from '../../lib/web-tools';
import { useHotkeys } from '../../shell/hotkeys';

/** Context below this is "small for Work". */
const WORK_CONTEXT = 32_768;

type HostModel = ChatHostState['models'][number];
type State = ModelBaseOption['state'];

const STATE_RANK: Record<State, number> = { loaded: 0, loading: 1, idle: 2 };

function contextText(tokens: number): string {
  return fill(CHAT.modelContext, { n: Math.round(tokens / 1024) });
}

/** Smallest on disk first, unknown sizes last, equal sizes by label. */
function bySize(a: HostModel, b: HostModel): number {
  if (a.sizeBytes !== b.sizeBytes) {
    if (a.sizeBytes === null) return 1;
    if (b.sizeBytes === null) return -1;
    return a.sizeBytes - b.sizeBytes;
  }
  return a.variant < b.variant ? -1 : a.variant > b.variant ? 1 : 0;
}

/**
 * The picker rows: one per base, each with its quantizations. Both lists put the model in memory first, then the model
 * in load, then the rest (bases by name without case, quantizations by size). `value` is the conversation's model;
 * `picked` is the last model picked per base (keys in lower case).
 */
export function modelBases(
  host: ChatHostState,
  resident: ReadonlySet<string>,
  mode: ChatMode,
  value: string | null,
  picked: Readonly<Record<string, string>>,
): ModelBaseOption[] {
  const stateOf = (id: string): State =>
    host.loading.some((load) => load.modelId === id) ? 'loading' : resident.has(id) ? 'loaded' : 'idle';
  const small = (model: HostModel): boolean =>
    mode === 'work' && model.contextTokens !== null && model.contextTokens < WORK_CONTEXT;
  const groups = new Map<string, HostModel[]>();
  for (const model of host.models) {
    const group = groups.get(model.base);
    if (group === undefined) groups.set(model.base, [model]);
    else group.push(model);
  }
  const bases: Array<{ option: ModelBaseOption; rank: number }> = [];
  for (const [base, members] of groups) {
    const sized = [...members].sort(bySize);
    const ordered = [...sized].sort((a, b) => STATE_RANK[stateOf(a.id)] - STATE_RANK[stateOf(b.id)]);
    const contexts = members.flatMap((model) => (model.contextTokens === null ? [] : [model.contextTokens]));
    const differ = contexts.some((tokens) => tokens !== contexts[0]);
    const variants = ordered.map((model): ModelVariantOption => {
      const facts = [model.format === 'gguf' ? 'GGUF' : 'MLX'];
      if (model.sizeBytes !== null) facts.push(formatBytes(model.sizeBytes));
      if (differ && model.contextTokens !== null) facts.push(contextText(model.contextTokens));
      const option: ModelVariantOption = { id: model.id, label: model.variant, state: stateOf(model.id), facts };
      return small(model) ? { ...option, warning: CHAT.modelSmall } : option;
    });
    const loading = variants.find((variant) => variant.state === 'loading');
    const loaded = variants.find((variant) => variant.state === 'loaded');
    const stateVariant = loading ?? loaded;
    const most = contexts.length === 0 ? null : Math.max(...contexts);
    const context =
      most === null ? [] : [differ ? fill(CHAT.modelContextUpTo, { n: Math.round(most / 1024) }) : contextText(most)];
    const facts =
      variants.length === 1
        ? [...context, ...variants[0]!.facts]
        : [fill(CHAT.modelQuantCount, { n: variants.length }), ...context];
    const lastId = picked[base.toLowerCase()];
    const preselect =
      [value, loaded?.id, loading?.id, lastId].find(
        (id) => id !== undefined && id !== null && members.some((model) => model.id === id),
      ) ?? sized[0]!.id;
    const option: ModelBaseOption = {
      base,
      state: stateVariant?.state ?? 'idle',
      facts,
      variants,
      preselect,
    };
    if (stateVariant !== undefined && variants.length > 1) option.stateVariant = stateVariant.label;
    if (members.every(small)) option.warning = CHAT.modelSmall;
    bases.push({ option, rank: loaded !== undefined ? 0 : loading !== undefined ? 1 : 2 });
  }
  const name = (option: ModelBaseOption): string => option.base.toLowerCase();
  return bases
    .sort((a, b) => a.rank - b.rank || (name(a.option) < name(b.option) ? -1 : name(a.option) > name(b.option) ? 1 : 0))
    .map((entry) => entry.option);
}

/**
 * The model a new conversation starts on: the model in memory of the app's engine, else the model in load, else the
 * stored choice, else the default, else the first.
 */
export function startModel(host: ChatHostState | null, stored: string | null): string | null {
  if (host === null || host.models.length === 0) return null;
  const served = (id: string | null): id is string => id !== null && host.models.some((model) => model.id === id);
  const inMemory = host.resident.find((entry) => served(entry.modelId))?.modelId ?? null;
  for (const id of [inMemory, host.loading[0]?.modelId ?? null, stored, host.defaultModelId]) if (served(id)) return id;
  return host.models[0]!.id;
}

export interface WebRowView {
  /** Right side of the menu row: the provider, or what is wrong. */
  end: string;
  dot?: 'warning' | 'danger';
  /** Accessible name of the row: the dot has its words here. */
  name: string;
  /** Words for the dot on the `+`, or null when there is none. */
  problem: string | null;
}

/** The web search row of the composer's `+` menu, from the selected providers. */
export function webRow(providers: ChatHostState['providers']): WebRowView {
  const search = providers.search;
  if (search !== null) {
    const provider = providerName(search);
    if (providers.searchCheck === 'rejected') {
      return {
        end: CHAT.webEndRejected,
        dot: 'danger',
        name: fill(CHAT.webNameRejected, { provider }),
        problem: CHAT.webProblemRejected,
      };
    }
    if (providers.searchCheck === 'no-credit') {
      return {
        end: CHAT.webEndNoCredit,
        dot: 'warning',
        name: fill(CHAT.webNameNoCredit, { provider }),
        problem: CHAT.webProblemNoCredit,
      };
    }
    return { end: provider, name: fill(CHAT.webNameOn, { provider }), problem: null };
  }
  return { end: CHAT.webEndOff, name: CHAT.webNameOff, problem: null };
}

function firecrawlState(body: ProvidersResponse | null): ProviderState | undefined {
  return body?.providers.find((p) => p.id === 'firecrawl');
}

/**
 * The crawler row of the `+` menu. On or off comes from the host summary; what is wrong needs the `/providers` body,
 * so those words show from the menu's first read on. The dot on the `+` stays search only.
 */
export function crawlerRow(providers: ChatHostState['providers'], body: ProvidersResponse | null): WebRowView {
  if (crawlerValue(providers) === 'off') return { end: CHAT.webEndOff, name: CHAT.crawlerNameOff, problem: null };
  const check = firecrawlState(body)?.check ?? null;
  if (check === 'rejected') {
    return { end: CHAT.webEndRejected, dot: 'danger', name: CHAT.crawlerNameRejected, problem: null };
  }
  if (check === 'no-credit') {
    return { end: CHAT.webEndNoCredit, dot: 'warning', name: CHAT.crawlerNameNoCredit, problem: null };
  }
  return { end: providerName('firecrawl'), name: CHAT.crawlerNameOn, problem: null };
}

type Read = 'reading' | 'failed' | 'done';
type Caption = { text: string; tone: 'danger' | 'warning' | 'secondary' } | null;

/** What the web search panel knows besides the host summary. */
export interface WebPanelState {
  /** The last `GET /providers` body or selection reply; null before the first. */
  body: ProvidersResponse | null;
  /** The latest read. */
  read: Read;
  /** The row picked, from the click until the host state after the reply, or until the reply fails. */
  busy: { search: SearchProviderId | null } | null;
  /** The last pick failed, while the menu stays open. */
  error: string | null;
  /** The provider whose key was gone at app start, as the server reports it once. */
  vanished: ProviderId | null;
}

export interface WebChoiceRow {
  /** The provider the row picks; null is Off. */
  search: SearchProviderId | null;
  label: string;
  name: string;
  checked: boolean;
  disabled: boolean;
  end?: string;
  endTone?: 'warning' | 'danger';
}

export interface WebChoices {
  /** Exa, DeepSeek, Brave, Off. */
  rows: WebChoiceRow[];
  /** Second line of the settings row. */
  note: string;
  caption: Caption;
}

/** The captions both panels share, after their own: Keychain down, the read, a change that waits (`pending`). */
function readCaption(
  providers: ChatHostState['providers'],
  body: ProvidersResponse | null,
  read: Read,
  pending: string,
): Caption {
  if (body?.keychain === 'unavailable') return { text: CHAT.webKeychain, tone: 'warning' };
  if (body === null && read === 'reading') return { text: CHAT.webReading, tone: 'secondary' };
  if (body === null && read === 'failed') return { text: CHAT.webReadFailed, tone: 'secondary' };
  if (providers.applyPending) return { text: pending, tone: 'secondary' };
  return null;
}

/**
 * The rows of the web search panel. The check mark follows the host summary only. A provider row is disabled when it
 * has no key, when its key state is not known (no body yet, or a Keychain that does not answer) unless it is the
 * current one, and while another row's write runs.
 */
export function webChoices(providers: ChatHostState['providers'], panel: WebPanelState): WebChoices {
  const { body, busy } = panel;
  const missing: string[] = [];
  const rows = SEARCH_IDS.map((id): WebChoiceRow => {
    const provider = providerName(id);
    const checked = providers.search === id;
    const state = body?.providers.find((p) => p.id === id);
    const known = state !== undefined && (state.configured || body?.keychain === 'ok');
    const noKey = known && !state.configured;
    const unknown = !known && !checked;
    if (noKey) missing.push(provider);
    const check = state !== undefined ? state.check : checked ? providers.searchCheck : null;
    const row: WebChoiceRow = {
      search: id,
      label: provider,
      name: provider,
      checked,
      disabled: noKey || unknown || (busy !== null && busy.search !== id),
    };
    if (busy !== null && busy.search === id) row.end = CHAT.webSwitching;
    else if (noKey) row.end = CHAT.webNoKey;
    else if (check === 'rejected') {
      row.end = CHAT.webEndRejected;
      row.endTone = 'danger';
    } else if (check === 'no-credit') {
      row.end = CHAT.webEndNoCredit;
      row.endTone = 'warning';
    }
    if (noKey) row.name = fill(CHAT.webRowNoKey, { provider });
    else if (unknown) row.name = fill(CHAT.webRowUnread, { provider });
    else if (check === 'rejected') row.name = fill(CHAT.webRowRejected, { provider });
    else if (check === 'no-credit') row.name = fill(CHAT.webRowNoCredit, { provider });
    return row;
  });
  const off: WebChoiceRow = {
    search: null,
    label: CHAT.webEndOff,
    name: CHAT.webEndOff,
    checked: providers.search === null,
    disabled: busy !== null && busy.search !== null,
  };
  if (busy !== null && busy.search === null) off.end = CHAT.webSwitching;
  const [a, b, c] = missing;
  const note =
    a === undefined
      ? CHAT.webSettingsNote
      : b === undefined
        ? fill(CHAT.webAddKeyOne, { a })
        : c === undefined
          ? fill(CHAT.webAddKeyTwo, { a, b })
          : fill(CHAT.webAddKeyThree, { a, b, c });
  let caption: Caption;
  if (panel.error !== null) caption = { text: panel.error, tone: 'danger' };
  else if (panel.vanished !== null) {
    caption = { text: fill(CHAT.webVanished, { provider: providerName(panel.vanished) }), tone: 'warning' };
  } else caption = readCaption(providers, body, panel.read, CHAT.webPending);
  return { rows: [...rows, off], note, caption };
}

/** What the crawler panel knows besides the host summary. */
export interface CrawlerPanelState {
  /** The last `/providers` body or reply; null before the first. */
  body: ProvidersResponse | null;
  read: Read;
  /** The row picked, from the click until the host summary shows it, or until the write fails. */
  busy: CrawlerValue | null;
  /** The last pick failed, while the menu stays open. */
  error: string | null;
}

export interface CrawlerChoiceRow {
  value: CrawlerValue;
  label: string;
  name: string;
  checked: boolean;
  disabled: boolean;
  end?: string;
  endTone?: 'warning' | 'danger';
  note?: string;
}

export interface CrawlerChoices {
  /** Firecrawl, Off. */
  rows: CrawlerChoiceRow[];
  /** Second line of the settings row. */
  note: string;
  caption: Caption;
}

/** Who reads pages for Chat when the crawler is off. */
function offNote(search: SearchProviderId | null): string {
  if (search === null) return CHAT.crawlerOffNone;
  if (search === 'brave') return CHAT.crawlerOffExcerpts;
  return fill(CHAT.crawlerOffReads, { provider: providerName(search) });
}

/**
 * The rows of the crawler panel, by the same rules as the web search panel: the check mark follows the host summary
 * only; Firecrawl with no key is disabled, and so is a key not read yet unless Firecrawl is the current one, and every
 * other row while a write runs. A rejected key can still be picked.
 */
export function crawlerChoices(providers: ChatHostState['providers'], panel: CrawlerPanelState): CrawlerChoices {
  const { body, busy } = panel;
  const current = crawlerValue(providers);
  const blocked = (value: CrawlerValue): boolean => busy !== null && busy !== value;
  const provider = providerName('firecrawl');

  const state = firecrawlState(body);
  const keyKnown = state !== undefined && (state.configured || body?.keychain === 'ok');
  const noKey = keyKnown && !state.configured;
  const keyUnknown = !keyKnown && current !== 'firecrawl';
  const keyCheck = state?.check ?? null;
  const firecrawl: CrawlerChoiceRow = {
    value: 'firecrawl',
    label: provider,
    name: provider,
    checked: current === 'firecrawl',
    disabled: noKey || keyUnknown || blocked('firecrawl'),
  };
  if (busy === 'firecrawl') firecrawl.end = CHAT.webSwitching;
  else if (noKey) firecrawl.end = CHAT.webNoKey;
  else if (keyCheck === 'rejected') {
    firecrawl.end = CHAT.webEndRejected;
    firecrawl.endTone = 'danger';
  } else if (keyCheck === 'no-credit') {
    firecrawl.end = CHAT.webEndNoCredit;
    firecrawl.endTone = 'warning';
  }
  if (noKey) firecrawl.name = fill(CHAT.webRowNoKey, { provider });
  else if (keyUnknown) firecrawl.name = fill(CHAT.webRowUnread, { provider });
  else if (keyCheck === 'rejected') firecrawl.name = fill(CHAT.webRowRejected, { provider });
  else if (keyCheck === 'no-credit') firecrawl.name = fill(CHAT.webRowNoCredit, { provider });

  const off: CrawlerChoiceRow = {
    value: 'off',
    label: CHAT.webEndOff,
    name: CHAT.webEndOff,
    note: offNote(providers.search),
    checked: current === 'off',
    disabled: blocked('off'),
  };
  if (busy === 'off') off.end = CHAT.webSwitching;

  const note = noKey ? CHAT.crawlerAddKey : CHAT.webSettingsNote;
  const caption: Caption =
    panel.error !== null
      ? { text: panel.error, tone: 'danger' }
      : readCaption(providers, body, panel.read, CHAT.crawlerPending);
  return { rows: [firecrawl, off], note, caption };
}

/** Work: the search provider's skills. */
function searchWork(data: ProvidersResponse): string {
  const provider = data.search === null ? undefined : data.providers.find((p) => p.id === data.search);
  if (provider === undefined || !provider.skill.installed || provider.skill.names.length === 0) return CHAT.webWorkNone;
  const n = provider.skill.names.length;
  return fill(CHAT.webWork, { skills: `${n} ${providerName(provider.id)} ${n === 1 ? 'skill' : 'skills'}` });
}

/** Work: the Firecrawl skills, which are installed but not passed while the CLI is missing. */
function crawlerWork(data: ProvidersResponse): string {
  if (!data.firecrawl) return CHAT.crawlerWorkOff;
  const state = firecrawlState(data);
  if (state === undefined || !state.skill.installed || state.skill.names.length === 0) return CHAT.crawlerWorkMissing;
  if (state.work.missing !== null) return CHAT.crawlerWorkNoCli;
  return fill(CHAT.crawlerWorkKey, { n: state.skill.names.length });
}

function useMounted(): { readonly current: boolean } {
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  return mounted;
}

/** The menu's open state and the host summary as of the last render, for replies that land later. */
function useLive(open: boolean, providers: ChatHostState['providers'] | null) {
  const live = useRef({ open, providers });
  useEffect(() => {
    live.current = { open, providers };
  });
  return live;
}

/** The `/providers` read both panels use. */
interface ProvidersRead {
  body: ProvidersResponse | null;
  read: Read;
  /** The provider whose key was gone at app start, until `clearVanished`. */
  vanished: ProviderId | null;
  clearVanished: () => void;
  load: () => void;
  /** A write starts: a read already in flight is dropped when it lands. */
  begin: () => void;
  /** A write's reply: the newest body. */
  take: (reply: ProvidersResponse) => void;
}

/** `GET /providers` when the menu opens and when the host summary changes while it is open, once for both panels. */
function useProvidersRead(open: boolean, providers: ChatHostState['providers'] | null): ProvidersRead {
  const [body, setBody] = useState<ProvidersResponse | null>(null);
  const [read, setRead] = useState<Read>('reading');
  const [vanished, setVanished] = useState<ProviderId | null>(null);
  const epoch = useRef(0);
  const mounted = useMounted();

  const load = useCallback((): void => {
    const token = ++epoch.current;
    setRead('reading');
    getJson<ProvidersResponse>('/providers').then(
      (next) => {
        if (!mounted.current) return;
        // The one-time report is used up by this read even when its rows are dropped.
        const gone = next.vanished;
        if (gone !== null) setVanished((was) => was ?? gone);
        if (token !== epoch.current) return;
        setBody(next);
        setRead('done');
      },
      () => {
        if (mounted.current && token === epoch.current) setRead('failed');
      },
    );
  }, [mounted]);

  const summary =
    providers === null
      ? null
      : `${providers.search}|${providers.searchCheck}|${providers.firecrawl}|${providers.fetch}`;
  useEffect(() => {
    if (open) load();
  }, [open, summary, load]);

  const begin = useCallback((): void => {
    epoch.current += 1;
  }, []);
  const take = useCallback(
    (reply: ProvidersResponse): void => {
      if (!mounted.current) return;
      epoch.current += 1;
      setBody(reply);
      setRead('done');
    },
    [mounted],
  );
  const clearVanished = useCallback((): void => setVanished(null), []);
  return { body, read, vanished, clearVanished, load, begin, take };
}

function switchFailure(error: unknown, search: SearchProviderId | null): string {
  if (error instanceof DashboardApiError) {
    if (error.code === 'E_CONFLICT' && search !== null)
      return fill(CHAT.webNeedKey, { provider: providerName(search) });
    if (error.reason === 'skill-bundle-corrupt') return CHAT.webCorrupt;
  }
  return CHAT.webSwitchFailed;
}

interface WebPanel {
  choices: WebChoices | null;
  /** Work: the skills Work gets, from the same body. */
  work: string | null;
  pick: (search: SearchProviderId | null) => void;
  setPanelOpen: (open: boolean) => void;
}

/** The web search panel on top of the shared read: the pick, its busy row, its error and the vanished key. */
function useWebPanel(
  providersRead: ProvidersRead,
  open: boolean,
  providers: ChatHostState['providers'] | null,
  mode: ChatMode,
): WebPanel {
  const { body, read, vanished, clearVanished, load, begin, take } = providersRead;
  /** `seen`: the host summary when the reply landed; absent while the write runs. */
  const [busy, setBusy] = useState<{
    search: SearchProviderId | null;
    seen?: ChatHostState['providers'] | null;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [vanishedShown, setVanishedShown] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const mounted = useMounted();
  const live = useLive(open, providers);

  useEffect(() => {
    if (busy?.seen === undefined) return;
    if (providers !== busy.seen || providers?.search === busy.search) setBusy(null);
  }, [busy, providers]);

  useEffect(() => {
    if (panelOpen && vanished !== null) setVanishedShown(true);
  }, [panelOpen, vanished]);

  useEffect(() => {
    if (open) return;
    setError(null);
    if (vanished === null) return;
    if (!vanishedShown) {
      toast.warning(fill(PROVIDERS.vanishedTitle, { provider: providerName(vanished) }), {
        description: PROVIDERS.vanishedText,
      });
    }
    clearVanished();
    setVanishedShown(false);
  }, [open, vanished, vanishedShown, clearVanished]);

  const pick = (search: SearchProviderId | null): void => {
    if (busy !== null || providers === null || providers.search === search) return;
    begin();
    setError(null);
    setBusy({ search });
    mutate<ProvidersResponse>('PUT', '/providers/selection', { search }).then(
      (reply) => {
        if (!mounted.current) return;
        take(reply);
        setBusy({ search, seen: live.current.providers });
      },
      (failure: unknown) => {
        if (!mounted.current) return;
        setBusy(null);
        const text = switchFailure(failure, search);
        if (live.current.open) {
          setError(text);
          load();
        } else {
          toast.error(text);
        }
      },
    );
  };

  const choices =
    providers === null
      ? null
      : webChoices(providers, {
          body,
          read,
          busy: busy === null ? null : { search: busy.search },
          error,
          vanished,
        });
  return { choices, work: mode === 'work' && body !== null ? searchWork(body) : null, pick, setPanelOpen };
}

function crawlerFailure(error: unknown, target: CrawlerValue): string {
  if (error instanceof DashboardApiError) {
    if (error.code === 'E_CONFLICT' && target === 'firecrawl') return CHAT.crawlerNeedKey;
    if (error.reason === 'skill-bundle-corrupt') return CHAT.webCorrupt;
  }
  return CHAT.crawlerSwitchFailed;
}

interface CrawlerPanel {
  choices: CrawlerChoices | null;
  /** Work: the Firecrawl skills Work gets, from the same body. */
  work: string | null;
  pick: (target: CrawlerValue) => void;
}

/**
 * The crawler panel on top of the shared read. A pick writes `firecrawl`, never `search`, and is busy until the host
 * summary shows it, because the reply can wait seconds for the Work PATH.
 */
function useCrawlerPanel(
  providersRead: ProvidersRead,
  open: boolean,
  providers: ChatHostState['providers'] | null,
  mode: ChatMode,
): CrawlerPanel {
  const { body, read, load, begin, take } = providersRead;
  const [busy, setBusy] = useState<CrawlerValue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const mounted = useMounted();
  const live = useLive(open, providers);

  useEffect(() => {
    if (busy !== null && providers !== null && crawlerValue(providers) === busy) setBusy(null);
  }, [busy, providers]);

  useEffect(() => {
    if (!open) setError(null);
  }, [open]);

  const pick = (target: CrawlerValue): void => {
    if (busy !== null || providers === null || crawlerValue(providers) === target) return;
    begin();
    setError(null);
    setBusy(target);
    mutate<ProvidersResponse>('PUT', '/providers/selection', CRAWLER_BODIES[target]).then(take, (failure: unknown) => {
      if (!mounted.current) return;
      setBusy(null);
      const text = crawlerFailure(failure, target);
      if (live.current.open) {
        setError(text);
        load();
      } else {
        toast.error(text);
      }
    });
  };

  const choices = providers === null ? null : crawlerChoices(providers, { body, read, busy, error });
  return { choices, work: mode === 'work' && body !== null ? crawlerWork(body) : null, pick };
}

const CAPTION_TONE = { danger: 'text-danger', warning: 'text-warning', secondary: 'text-secondary' } as const;

/** Marks the panel open while it is drawn. */
function usePanelOpen(setPanelOpen: (open: boolean) => void): void {
  useEffect(() => {
    setPanelOpen(true);
    return () => setPanelOpen(false);
  }, [setPanelOpen]);
}

/** The live caption above a panel's list: the node stays, the text comes and goes. */
function PanelCaption({ caption }: { caption: Caption }): ReactElement {
  return (
    <div role="status">
      {caption !== null ? (
        <p className={`m-0 px-2.5 py-1.5 text-caption ${CAPTION_TONE[caption.tone]}`}>{caption.text}</p>
      ) : null}
    </div>
  );
}

/** Work: what Work gets, under a panel's list. */
function WorkLine({ text }: { text: string | null }): ReactElement | null {
  return text !== null ? <p className="m-0 px-2.5 py-1.5 text-caption text-secondary">{text}</p> : null;
}

/** The web search panel: a caption, the provider list with the settings row, and in Work the skills line. */
function WebSearchPanel({ choices, work, pick, setPanelOpen }: WebPanel & { choices: WebChoices }): ReactElement {
  const navigate = useNavigate();
  usePanelOpen(setPanelOpen);
  return (
    <>
      <PanelCaption caption={choices.caption} />
      <MenuList>
        <MenuGroup label={CHAT.webProvider}>
          {choices.rows.map((row) => (
            <MenuRadio
              key={row.search ?? 'off'}
              label={row.label}
              name={row.name}
              checked={row.checked}
              disabled={row.disabled}
              end={row.end}
              endTone={row.endTone}
              onSelect={() => pick(row.search)}
            />
          ))}
        </MenuGroup>
        <MenuSeparator />
        <MenuItem label={CHAT.webSettings} note={choices.note} onSelect={() => navigate('/providers')} />
      </MenuList>
      <WorkLine text={work} />
    </>
  );
}

/** The crawler panel: a caption, Firecrawl and Off with the settings row, and in Work the Firecrawl line. */
function CrawlerPanelView({ choices, work, pick }: CrawlerPanel & { choices: CrawlerChoices }): ReactElement {
  const navigate = useNavigate();
  return (
    <>
      <PanelCaption caption={choices.caption} />
      <MenuList>
        <MenuGroup label={CHAT.crawlerGroup}>
          {choices.rows.map((row) => (
            <MenuRadio
              key={row.value}
              label={row.label}
              name={row.name}
              note={row.note}
              checked={row.checked}
              disabled={row.disabled}
              end={row.end}
              endTone={row.endTone}
              onSelect={() => pick(row.value)}
            />
          ))}
        </MenuGroup>
        <MenuSeparator />
        <MenuItem label={CHAT.webSettings} note={choices.note} onSelect={() => navigate('/providers')} />
      </MenuList>
      <WorkLine text={work} />
    </>
  );
}

/** The "Add files or photos" row: `open` clicks the page's file input; `full` holds the most a message can carry. */
export interface AttachRow {
  open: () => void;
  note?: string;
  full: boolean;
}

/**
 * A modal dialog is open (Rename, Confirm, the palette, a permission ask). A popover or a tooltip is not: Radix wraps
 * those in a popper wrapper, and the web search and crawler panels' `dialog` role sits inside one.
 */
function modalOpen(): boolean {
  for (const node of document.querySelectorAll('[role="dialog"], [role="alertdialog"]')) {
    if (node.closest('[data-radix-popper-content-wrapper]') === null) return true;
  }
  return false;
}

/**
 * The `+` at the start of the composer's tools row and its menu: Add files or photos, then web search and the crawler,
 * each with its panel.
 */
export function ComposerMenu({
  providers,
  mode,
  attach,
}: {
  providers: ChatHostState['providers'] | null;
  mode: ChatMode;
  attach: AttachRow;
}): ReactElement {
  const [open, setOpen] = useState(false);
  const view = providers === null ? null : webRow(providers);
  const shown = open && view !== null;
  const providersRead = useProvidersRead(shown, providers);
  const web = useWebPanel(providersRead, shown, providers, mode);
  const crawler = useCrawlerPanel(providersRead, shown, providers, mode);
  const crawl = providers === null ? null : crawlerRow(providers, providersRead.body);
  useHotkeys([
    {
      key: 'u',
      meta: true,
      allowInInput: true,
      run: () => {
        if (view === null || attach.full || modalOpen()) return;
        setOpen(false);
        attach.open();
      },
    },
  ]);
  return (
    <Menu
      label={CHAT.tools}
      open={shown}
      onOpenChange={setOpen}
      trigger={
        <ToolChip
          compact
          icon={<Plus size={14} strokeWidth={1.75} />}
          label={
            view !== null && view.problem !== null ? fill(CHAT.toolsProblem, { problem: view.problem }) : CHAT.tools
          }
          dot={view?.dot}
          disabled={view === null}
        />
      }
    >
      {view !== null && crawl !== null ? (
        <>
          <MenuItem
            icon={<Paperclip size={14} strokeWidth={1.75} />}
            label={CHAT.attach.row}
            end={CHAT.attach.keys}
            keys="Meta+U"
            note={attach.note}
            disabled={attach.full}
            onSelect={attach.open}
          />
          <MenuSub
            icon={<Globe size={14} strokeWidth={1.75} />}
            label={CHAT.webSearch}
            end={view.end}
            endTone={view.dot}
            name={view.name}
          >
            {web.choices !== null ? <WebSearchPanel {...web} choices={web.choices} /> : null}
          </MenuSub>
          <MenuSub
            icon={<FileText size={14} strokeWidth={1.75} />}
            label={CHAT.crawler}
            end={crawl.end}
            endTone={crawl.dot}
            name={crawl.name}
          >
            {crawler.choices !== null ? <CrawlerPanelView {...crawler} choices={crawler.choices} /> : null}
          </MenuSub>
        </>
      ) : null}
    </Menu>
  );
}
