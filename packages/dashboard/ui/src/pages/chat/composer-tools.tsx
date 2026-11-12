import { Globe } from 'lucide-react';
import { type ReactElement, useEffect, useState } from 'react';

import { ButtonLink } from '../../components/arc-x/button-link/button-link';
import type { ModelOption } from '../../components/arc-x/model-picker/model-picker';
import { ToolChip } from '../../components/arc-x/tool-chip/tool-chip';
import { Popover, PopoverContent, PopoverTrigger } from '../../components/arc/popover/popover';
import { getJson } from '../../lib/api';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatHostState, ChatMode } from '../../lib/chat-types';
import { formatBytes } from '../../lib/format';
import type { ProviderId, ProvidersResponse } from '../../lib/providers-types';
import type { LocalModel } from '../../lib/types';
import { providerName } from '../../lib/web-tools';

/** Context below this is "small for Work". */
const WORK_CONTEXT = 32_768;

function localFor(models: readonly LocalModel[] | undefined, id: string): LocalModel | undefined {
  return models?.find((model) => model.servedIds.includes(id) || (model.servedIds.length === 0 && model.name === id));
}

/** Size on disk of a served model, or null. */
export function modelSizeOf(models: readonly LocalModel[] | undefined, id: string): number | null {
  return localFor(models, id)?.sizeBytes ?? null;
}

/** The picker rows: each model's state, context, quant and size. */
export function modelOptions(
  host: ChatHostState,
  resident: ReadonlySet<string>,
  local: readonly LocalModel[] | undefined,
  mode: ChatMode,
): ModelOption[] {
  const options = host.models.map((model): ModelOption => {
    const facts: string[] = [];
    if (model.contextTokens !== null) facts.push(`${Math.round(model.contextTokens / 1024)}K context`);
    const disk = localFor(local, model.id);
    if (disk !== undefined) {
      if (disk.quant !== null) facts.push(disk.quant);
      facts.push(formatBytes(disk.sizeBytes));
    }
    const state: ModelOption['state'] =
      host.loading?.modelId === model.id ? 'loading' : resident.has(model.id) ? 'loaded' : 'idle';
    const small = mode === 'work' && model.contextTokens !== null && model.contextTokens < WORK_CONTEXT;
    return small ? { id: model.id, state, facts, warning: CHAT.modelSmall } : { id: model.id, state, facts };
  });
  return [
    ...options.filter((option) => option.state === 'loaded'),
    ...options.filter((option) => option.state === 'loading'),
    ...options.filter((option) => option.state === 'idle'),
  ];
}

/**
 * The model a new conversation starts on: the model in memory of the app's engine, else the model in load, else the
 * stored choice, else the default, else the first.
 */
export function startModel(host: ChatHostState | null, stored: string | null): string | null {
  if (host === null || host.models.length === 0) return null;
  const served = (id: string | null): id is string => id !== null && host.models.some((model) => model.id === id);
  const inMemory = host.resident.find((entry) => served(entry.modelId))?.modelId ?? null;
  for (const id of [inMemory, host.loading?.modelId ?? null, stored, host.defaultModelId]) if (served(id)) return id;
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
