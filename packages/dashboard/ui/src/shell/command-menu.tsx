import * as DialogPrimitive from '@radix-ui/react-dialog';
import { AnimatePresence, motion, type Transition } from 'motion/react';
import { type ReactElement, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';

import { motionX } from '../components/arc-x/motion/motion';
import { useStaticMotion } from '../components/arc-x/motion/use-static-motion';
import { type CommandItem, CommandPalette } from '../components/arc/command-palette/command-palette';
import { motionTokens } from '../components/arc/motion-tokens';
import { getJson } from '../lib/api';
import type { ConversationRow } from '../lib/chat-types';
import { useHomeDir } from '../lib/home-dir';
import type { ModelsResponse, SessionListResponse, SessionRow } from '../lib/types';
import { chatHostSnapshot } from '../pages/chat/use-chat-host';
import styles from './command-menu.module.css';
import { type AppCommand, conversationCommands, modelCommands, sessionCommands, usePageCommands } from './commands';

export interface CommandMenuProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The shell's Go to and Actions groups. */
  commands: readonly AppCommand[];
  navigate: (to: string) => void;
}

const still: Transition = { duration: 0 };
const leave: Transition = { duration: motionTokens.duration.fast, ease: [...motionTokens.ease.standard] };
const scrimIn: Transition = { duration: motionTokens.duration.standard, ease: [...motionTokens.ease.enter] };
const panelIn: Transition = {
  default: motionTokens.spring.smooth,
  opacity: { duration: motionTokens.duration.fast, ease: [...motionTokens.ease.enter] },
};

/** Models, Sessions and Chat conversations, fetched each time the palette opens; absent while they load and on failure. */
function useOpenTimeCommands(open: boolean, navigate: (to: string) => void): AppCommand[] {
  const [models, setModels] = useState<AppCommand[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const home = useHomeDir();
  useEffect(() => {
    if (!open) return;
    let active = true;
    getJson<ModelsResponse>('/models').then(
      (body) => {
        if (active) setModels(modelCommands(body.models, navigate));
      },
      () => {
        if (active) setModels([]);
      },
    );
    getJson<SessionListResponse>('/sessions?limit=20').then(
      (body) => {
        if (active) setSessions(body.sessions.slice(0, 20));
      },
      () => {
        if (active) setSessions([]);
      },
    );
    Promise.all(
      (['chat', 'work'] as const).map((mode) =>
        getJson<{ conversations: ConversationRow[] }>(`/chat/conversations?mode=${mode}`).then(
          (body) => body.conversations.filter((row) => row.mode === mode),
          () => [] as ConversationRow[],
        ),
      ),
    ).then(
      (lists) => {
        if (active) setConversations(lists.flat());
      },
      () => undefined,
    );
    return () => {
      active = false;
    };
  }, [open, navigate]);
  return useMemo(
    () => [
      ...conversationCommands(conversations, chatHostSnapshot().state?.active ?? [], navigate, home),
      ...models,
      ...sessionCommands(sessions, navigate, home),
    ],
    [conversations, models, sessions, navigate, home],
  );
}

/** Arc's CommandPalette block in a modal layer that moves like `arc/dialog`. */
export function CommandMenu({ open, onOpenChange, commands, navigate }: CommandMenuProps): ReactElement {
  const staticMotion = useStaticMotion();
  const { in: enter, out } = motionX.layer;
  const pageCommands = usePageCommands();
  const fetched = useOpenTimeCommands(open, navigate);
  const all = useMemo(() => [...pageCommands, ...commands, ...fetched], [pageCommands, commands, fetched]);
  const byId = useMemo(() => new Map(all.map((command) => [command.id, command])), [all]);
  const items = useMemo<CommandItem[]>(
    () =>
      all.map((command) => ({
        id: command.id,
        label: command.label,
        description: command.description,
        group: command.group,
        keywords: command.keywords,
        icon: command.icon,
        shortcut: command.shortcut,
      })),
    [all],
  );

  const returnTo = useRef<HTMLElement | null>(null);
  useLayoutEffect(() => {
    if (open) returnTo.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  }, [open]);
  // Radix's modal Dialog returns focus only to a Dialog.Trigger, and this layer has none.
  const restoreFocus = (event: Event): void => {
    event.preventDefault();
    const target = returnTo.current;
    returnTo.current = null;
    const active = document.activeElement;
    if (active !== null && active !== document.body) return;
    const fallback = document.querySelector<HTMLElement>('[data-scroll-root]');
    (target?.isConnected === true ? target : fallback)?.focus({ preventScroll: true });
  };

  const select = (item: CommandItem): void => {
    const command = byId.get(item.id);
    onOpenChange(false);
    if (command !== undefined) void command.run();
  };

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <AnimatePresence>
        {open ? (
          <DialogPrimitive.Portal key="command-menu" forceMount>
            <DialogPrimitive.Overlay asChild forceMount>
              <motion.div
                className={styles.scrim}
                initial={staticMotion ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: staticMotion ? still : leave }}
                transition={staticMotion ? still : scrimIn}
              />
            </DialogPrimitive.Overlay>
            <DialogPrimitive.Content
              asChild
              forceMount
              aria-label="Command palette"
              aria-describedby={undefined}
              onCloseAutoFocus={restoreFocus}
              onEscapeKeyDown={(event) => {
                // Esc first clears a typed query (the block does that); only an empty field closes the layer.
                const target = event.target;
                if (target instanceof HTMLInputElement && target.value !== '') event.preventDefault();
              }}
            >
              <motion.div
                className={styles.panel}
                initial={staticMotion ? false : { opacity: 0, ...enter }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, ...(staticMotion ? { transition: still } : { ...out, transition: leave }) }}
                transition={staticMotion ? still : panelIn}
              >
                <DialogPrimitive.Title className={styles.title}>Command palette</DialogPrimitive.Title>
                <CommandPalette
                  items={items}
                  placeholder="Search or jump to…"
                  onSelect={select}
                  onClose={() => onOpenChange(false)}
                />
              </motion.div>
            </DialogPrimitive.Content>
          </DialogPrimitive.Portal>
        ) : null}
      </AnimatePresence>
    </DialogPrimitive.Root>
  );
}
