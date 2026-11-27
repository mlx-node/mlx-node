import { Boxes, Download, Power, TriangleAlert, Unplug } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { MorphLoader } from '../../components/arc-x/morph-loader/morph-loader';
import { SwapCell } from '../../components/arc-x/swap-cell/swap-cell';
import { Button } from '../../components/arc/button/button';
import { EmptyState } from '../../components/arc/empty-state/empty-state';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatHostState, ChatMode } from '../../lib/chat-types';
import { providerName } from '../../lib/web-tools';
import type { BlockAction, BlockIcon, Blocker } from './blockers';

const ICONS: Record<BlockIcon, ReactNode> = {
  power: <Power size={20} strokeWidth={1.75} />,
  unplug: <Unplug size={20} strokeWidth={1.75} />,
  boxes: <Boxes size={20} strokeWidth={1.75} />,
  download: <Download size={20} strokeWidth={1.75} />,
  loader: <MorphLoader size={20} status="loading" />,
  alert: <TriangleAlert size={20} strokeWidth={1.75} />,
};

export interface NewScreenProps {
  mode: ChatMode;
  model: string | null;
  providers: ChatHostState['providers'] | null;
  /** A mode-wide state: its panel takes the hero's place. */
  blocker: Blocker | null;
  onStarter: (text: string) => void;
  onAction: (action: BlockAction) => void;
  /** Work: the next conversation starts without a folder, in a new one under `root` (as people read it). */
  noFolder?: { root: string } | null;
}

function BlockedPanel({
  panel,
  onAction,
}: {
  panel: NonNullable<Blocker['panel']>;
  onAction: NewScreenProps['onAction'];
}) {
  const buttons = [panel.action, panel.secondary].filter((action): action is BlockAction => action !== undefined);
  return (
    <EmptyState
      title={panel.title}
      description={panel.description}
      icon={ICONS[panel.icon]}
      action={
        buttons.length === 0 ? undefined : (
          <span className="flex flex-wrap justify-center gap-2">
            {buttons.map((action, index) => (
              <Button
                key={action.kind}
                type="button"
                size="sm"
                variant={index === 0 ? 'primary' : 'secondary'}
                data-hero-action=""
                onClick={() => onAction(action)}
              >
                {action.label}
              </Button>
            ))}
          </span>
        )
      }
    />
  );
}

function Starters({ items, onStarter }: { items: readonly string[]; onStarter: (text: string) => void }) {
  return (
    <div className="flex w-full max-w-[42.5rem] flex-wrap justify-center gap-2 [@media(max-height:559px)]:hidden">
      {items.map((text) => (
        <Button key={text} type="button" variant="secondary" size="sm" onClick={() => onStarter(text)}>
          {text}
        </Button>
      ))}
    </div>
  );
}

function ChatHero({ model, providers, onStarter }: Pick<NewScreenProps, 'model' | 'providers' | 'onStarter'>) {
  const search = providers?.search ?? null;
  const starters = [
    CHAT.startersChat[0],
    CHAT.startersChat[1],
    search !== null ? CHAT.startersChat[2] : CHAT.starterChatNoWeb,
  ];
  return (
    <div className="grid justify-items-center gap-4 text-center">
      <div className="grid gap-1">
        <p className="m-0 font-heading text-title font-medium text-foreground">{CHAT.heroChatTitle}</p>
        <p className="m-0 text-body text-secondary">
          {model !== null ? `${fill(CHAT.heroChatModel, { model })} ` : ''}
          {search !== null ? (
            fill(CHAT.heroChatWeb, { provider: providerName(search) })
          ) : (
            <>
              {`${CHAT.heroChatWebOff} `}
              <Link className="link" to="/providers">
                {CHAT.heroChatSetUp}
              </Link>
            </>
          )}
        </p>
      </div>
      <Starters items={starters} onStarter={onStarter} />
    </div>
  );
}

function WorkHero({ providers, onStarter, noFolder }: Pick<NewScreenProps, 'providers' | 'onStarter' | 'noFolder'>) {
  const keyShared = providers !== null && (providers.search !== null || providers.firecrawl);
  const none = noFolder ?? null;
  return (
    <div className="grid justify-items-center gap-4 text-center">
      <div className="grid max-w-[34rem] gap-1">
        <p className="m-0 font-heading text-title font-medium text-foreground">
          {none === null ? CHAT.heroWorkTitle : CHAT.heroWorkNoneTitle}
        </p>
        <p className="m-0 text-body text-secondary">
          {none === null ? CHAT.heroWorkText : fill(CHAT.heroWorkNoneText, { root: none.root })}
        </p>
      </div>
      <Starters items={none === null ? CHAT.startersWork : CHAT.startersWorkNone} onStarter={onStarter} />
      {keyShared ? <p className="m-0 max-w-[34rem] text-caption text-secondary">{CHAT.workKeyNotice}</p> : null}
    </div>
  );
}

/** The New screen: the hero of the mode, or the panel of a state that stops every message. */
export function NewScreen(props: NewScreenProps): ReactElement {
  const panel = props.blocker?.panel;
  const key = panel !== undefined ? `blocked-${props.mode}` : props.mode;
  const body =
    panel !== undefined ? (
      <BlockedPanel panel={panel} onAction={props.onAction} />
    ) : props.mode === 'chat' ? (
      <ChatHero model={props.model} providers={props.providers} onStarter={props.onStarter} />
    ) : (
      <WorkHero providers={props.providers} onStarter={props.onStarter} noFolder={props.noFolder} />
    );
  return (
    <div className="grid h-full min-h-0 content-center py-6">
      <SwapCell active={key} items={{ [key]: body }} />
    </div>
  );
}
