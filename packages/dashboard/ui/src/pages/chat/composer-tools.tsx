import { Globe } from 'lucide-react';
import { type ReactElement, useEffect, useState } from 'react';

import { ButtonLink } from '../../components/arc-x/button-link/button-link';
import type { ModelBaseOption, ModelVariantOption } from '../../components/arc-x/model-picker/model-picker';
import { ToolChip } from '../../components/arc-x/tool-chip/tool-chip';
import { Popover, PopoverContent, PopoverTrigger } from '../../components/arc/popover/popover';
import { getJson } from '../../lib/api';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatHostState, ChatMode } from '../../lib/chat-types';
import { formatBytes } from '../../lib/format';
import type { ProviderId, ProvidersResponse } from '../../lib/providers-types';
import { providerName } from '../../lib/web-tools';

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

export interface WebChipView {
  label: string;
  state: 'on' | 'off';
  dot?: 'warning' | 'danger';
  /** Accessible name: the dot has its words here. */
  name: string;
  /** Sentences of the popover, Chat form. */
  lines: string[];
  /** The first line warns (a rejected key, no credit). */
  alert: 'warning' | 'danger' | null;
  action: { label: string; to: string };
  pending: boolean;
}

/** The web chip of the composer, from the selected providers. */
export function webChip(providers: ChatHostState['providers']): WebChipView {
  const search = providers.search;
  const lines: string[] = [];
  let alert: WebChipView['alert'] = null;
  let dot: WebChipView['dot'];
  let extra = '';
  if (search !== null && providers.searchCheck === 'rejected') {
    lines.push(fill(CHAT.webRejected, { provider: providerName(search) }));
    alert = 'danger';
    dot = 'danger';
    extra = ', key rejected';
  } else if (search !== null && providers.searchCheck === 'no-credit') {
    lines.push(fill(CHAT.webNoCredit, { provider: providerName(search) }));
    alert = 'warning';
    dot = 'warning';
    extra = ', no credit';
  }
  let view: Pick<WebChipView, 'label' | 'state' | 'name' | 'action'>;
  const settings = { label: CHAT.webSettings, to: '/providers' };
  if (search !== null) {
    const name = providerName(search);
    const fetch = providers.fetch;
    lines.push(
      fetch === null || fetch === search
        ? fill(search === 'brave' ? CHAT.webChatExcerpts : CHAT.webChatSame, { search: name })
        : fill(CHAT.webChatBoth, { search: name, fetch: providerName(fetch) }),
    );
    view = {
      label: fill(CHAT.webOn, { provider: name }),
      state: 'on',
      name: `Web, ${name}${extra}`,
      action: alert === 'danger' ? { label: CHAT.webFix, to: '/providers' } : settings,
    };
  } else if (providers.firecrawl) {
    lines.push(CHAT.webChatPagesOnly);
    view = { label: CHAT.webPagesOnly, state: 'on', name: CHAT.webPagesOnly, action: settings };
  } else {
    lines.push(CHAT.webChatOff);
    view = { label: CHAT.webOff, state: 'off', name: CHAT.webOff, action: { label: CHAT.webSetUp, to: '/providers' } };
  }
  return { ...view, dot, lines, alert, pending: providers.applyPending };
}

function skillsText(data: ProvidersResponse): string {
  const parts: string[] = [];
  const add = (id: ProviderId): void => {
    const provider = data.providers.find((p) => p.id === id);
    if (provider === undefined || !provider.skill.installed || provider.skill.names.length === 0) return;
    const n = provider.skill.names.length;
    parts.push(`${n} ${providerName(id)} ${n === 1 ? 'skill' : 'skills'}`);
  };
  if (data.search !== null) add(data.search);
  if (data.firecrawl) add('firecrawl');
  return parts.length === 0 ? CHAT.webWorkNone : fill(CHAT.webWork, { skills: parts.join(' and ') });
}

/** The Work form of the popover counts the installed skills, read when it opens. */
function useWorkSkills(open: boolean, mode: ChatMode): string | null {
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    if (!open || mode !== 'work') return;
    let active = true;
    getJson<ProvidersResponse>('/providers').then(
      (body) => {
        if (active) setText(skillsText(body));
      },
      () => {
        if (active) setText(null);
      },
    );
    return () => {
      active = false;
    };
  }, [open, mode]);
  return text;
}

export function WebChip({ providers, mode }: { providers: ChatHostState['providers']; mode: ChatMode }): ReactElement {
  const [open, setOpen] = useState(false);
  const view = webChip(providers);
  const work = useWorkSkills(open, mode);
  const lines =
    mode === 'work' ? [...(view.alert !== null ? [view.lines[0]!] : []), ...(work !== null ? [work] : [])] : view.lines;
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <ToolChip
          icon={<Globe size={14} strokeWidth={1.75} />}
          label={view.label}
          give={2}
          state={view.state}
          dot={view.dot}
          aria-label={view.name}
        />
      </PopoverTrigger>
      <PopoverContent side="top" align="start" aria-label={CHAT.webSettings}>
        <div className="grid gap-2">
          {lines.map((line, index) => (
            <p
              key={line}
              className={
                index === 0 && view.alert !== null
                  ? `m-0 text-body ${view.alert === 'danger' ? 'text-danger' : 'text-warning'}`
                  : 'm-0 text-body text-foreground'
              }
            >
              {line}
            </p>
          ))}
          {view.pending ? <p className="m-0 text-caption text-secondary">{CHAT.webPending}</p> : null}
          <span>
            <ButtonLink to={view.action.to} size="sm" variant={view.state === 'off' ? 'secondary' : 'ghost'}>
              {view.action.label}
            </ButtonLink>
          </span>
        </div>
      </PopoverContent>
    </Popover>
  );
}
