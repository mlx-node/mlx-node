import { ClipboardCheck, Shield, ShieldCheck, ShieldOff, TriangleAlert } from 'lucide-react';
import {
  type MouseEvent,
  type ReactElement,
  type ReactNode,
  type RefObject,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react';

import { ChatMarkdown } from '../../../components/arc-x/chat-markdown/chat-markdown';
import {
  CardTextField,
  ExactText,
  InteractionCard,
  Well,
} from '../../../components/arc-x/interaction-card/interaction-card';
import { MenuItem } from '../../../components/arc-x/menu/menu';
import { OptionList } from '../../../components/arc-x/option-list/option-list';
import { PathText } from '../../../components/arc-x/path-text/path-text';
import { SplitButton } from '../../../components/arc-x/split-button/split-button';
import { Button } from '../../../components/arc/button/button';
import { CHAT, fill } from '../../../lib/chat-strings';
import type { Interaction, PermissionInteraction, PermissionMode, PlanInteraction } from '../../../lib/chat-types';
import { autoAskText, cardDescription, cardWarnings } from '../../../lib/permission-text';
import { oneLine } from '../../../lib/web-tools';

/**
 * What the page sends for an answer. `note`: a deny with the user's note; the agent reads it and goes on. `never`: "Never
 * allow" with the scope of words the card offered; the call is denied with the rule's note. `answers`: a
 * question card's answer by option index, null for a question left open; `reply: 'skip'`: no answer; `reply:
 * 'discuss' | 'plan-now'` (plan mode): the answers so far, and the agent stops asking or plans now. `mcp`: an MCP
 * form's values, one per shown field (text, a switch, an option index, option indexes, null for none), a link's accept
 * with none (the page was opened), or Decline.
 */
export type InteractionAnswer =
  | { optionId: string; note?: string }
  | { never: 'short' | 'exact' }
  | { decision: 'approved' | 'rejected'; feedback?: string }
  | { answers: Array<{ picks: number[]; other?: string } | null> }
  | { reply: 'skip' }
  | { reply: 'discuss' | 'plan-now'; answers: Array<{ picks: number[]; other?: string } | null> }
  | { mcp: 'accept'; values?: Array<string | boolean | number | number[] | null> }
  | { mcp: 'decline' };

type Access = PermissionInteraction['access'];
type Option = PermissionInteraction['options'][number];

/** Characters of the folder in a card's meta; the whole path is its hover title and part of the card's description. */
const META_PATH_CHARS = 36;
/** Time after a card is first shown before a press of a button that allows can count: a click meant for the card before. */
export const CARD_ARM_MS = 600;
/** The plan feedback and deny note fields' cap, as the answer route's. */
export const FEEDBACK_MAX_CHARS = 4_000;

const GAP_WORDS = { spaces: CHAT.exactSpaces, emptyLines: CHAT.exactEmptyLines };

/**
 * A click on a button that allows counts only once the card has been on screen for {@link CARD_ARM_MS}: a quick second
 * click meant for the card before must not answer the next one, which takes its place. A key press too.
 */
export function useArmed(): (event: { timeStamp: number }) => boolean {
  const shownAt = useRef<number | null>(null);
  useEffect(() => {
    shownAt.current = performance.now();
  }, []);
  return (event) => shownAt.current !== null && event.timeStamp - shownAt.current >= CARD_ARM_MS;
}

function exactNote(lines: number): string {
  return lines === 1 ? CHAT.exactLong : fill(CHAT.exactLines, { n: lines });
}

export function Exact(props: { text: string; label: string; id?: string }): ReactElement {
  return <ExactText {...props} overflowNote={exactNote} gapWords={GAP_WORDS} />;
}

export function permissionTitle(access: Access): string {
  switch (access) {
    case 'bash':
      return CHAT.permRun;
    case 'edit':
      return CHAT.permEdit;
    case 'mcp':
      return CHAT.permTool;
    default:
      return CHAT.permOther;
  }
}

/** The sentence for the page's status region when a card arrives. */
export function asksSentence(interaction: Interaction): string {
  if (interaction.kind === 'plan') return CHAT.status.asksPlan;
  if (interaction.kind === 'question') return CHAT.status.asksQuestion;
  if (interaction.kind === 'mcp-input') {
    return fill(interaction.mode === 'url' ? CHAT.status.asksMcpLink : CHAT.status.asksMcp, {
      server: interaction.server,
    });
  }
  switch (interaction.access) {
    case 'bash':
      return CHAT.status.asksRun;
    case 'edit':
      return CHAT.status.asksEdit;
    case 'mcp':
      return CHAT.status.asksTool;
    default:
      return CHAT.status.asksOther;
  }
}

export interface CardButton {
  option: Option;
  label: string;
  variant: 'primary' | 'secondary' | 'ghost';
  always: boolean;
}

/**
 * The buttons of a permission card, in order: Deny, Allow once, then the "always" option worded for what the agent
 * saves. Options are matched by id and kind; anything else the agent offers is not drawn. Outside Ask only Deny and
 * Allow once: an "always" answer would make the agent stop asking.
 */
export function cardButtons(
  interaction: Pick<PermissionInteraction, 'access' | 'options'>,
  permissionMode: PermissionMode,
): CardButton[] {
  const find = (id: string, kinds: readonly Option['kind'][]): Option | undefined =>
    interaction.options.find((option) => option.optionId === id && kinds.includes(option.kind));
  const out: CardButton[] = [];
  const deny = find('reject-once', ['reject_once']);
  if (deny !== undefined) out.push({ option: deny, label: CHAT.deny, variant: 'secondary', always: false });
  const once = find('allow-once', ['allow_once']);
  if (once !== undefined) out.push({ option: once, label: CHAT.allowOnce, variant: 'primary', always: false });
  if (permissionMode !== 'ask') return out;
  if (interaction.access === 'edit') {
    const edits = find('allow-edits-session', ['allow_always', 'allow_once']);
    if (edits !== undefined) out.push({ option: edits, label: CHAT.allowEditsSession, variant: 'ghost', always: true });
  } else {
    const always = find('always-allow', ['allow_always']);
    if (always !== undefined) {
      const label =
        interaction.access === 'bash'
          ? CHAT.allowAlwaysCommand
          : interaction.access === 'mcp'
            ? CHAT.allowAlwaysTool
            : CHAT.allowAlwaysOther;
      out.push({ option: always, label, variant: 'ghost', always: true });
    }
  }
  return out;
}

/**
 * The card's note lines: the agent's own words for its "always" option, what allowing all edits also allows, then what
 * Deny does and where grants live.
 */
export function cardNotes(access: Access, buttons: readonly CardButton[], ofRun = false): string[] {
  const always = buttons.find((button) => button.always);
  const notes: string[] = [];
  if (always !== undefined) notes.push(fill(CHAT.agentWording, { name: oneLine(always.option.name) }));
  if (always !== undefined && access === 'edit') notes.push(CHAT.editsSessionNote);
  const deny = ofRun ? CHAT.denyRunNote : CHAT.denyNote;
  notes.push(always !== undefined && (access === 'bash' || access === 'mcp') ? `${deny} ${CHAT.grantsNote}` : deny);
  return notes;
}

/** Focus goes to the card's safe button, unless the user is typing in the composer. */
function useArrivalFocus(
  target: RefObject<HTMLButtonElement | null>,
  composer: RefObject<HTMLTextAreaElement | null>,
  id: string,
): void {
  useEffect(() => {
    const area = composer.current;
    if (area !== null && document.activeElement === area && area.value.trim() !== '') return;
    target.current?.focus({ preventScroll: true });
  }, [id]);
}

export interface CardBase {
  /** The conversation's folder. */
  folder: string | null;
  /** "1 of 3" when more wait. */
  position: { index: number; count: number };
  composerRef: RefObject<HTMLTextAreaElement | null>;
  /** Resolves when the request was sent (an MCP form the host refused resolves with its field errors); rejects when it was refused. */
  onAnswer: (answer: InteractionAnswer) => Promise<unknown>;
}

type From = PermissionInteraction['from'];

/** The menu row that opens the never-allow step: it names the short words. */
export function neverMenuText(never: NonNullable<PermissionInteraction['never']>): string {
  return fill(CHAT.never.menu, { words: oneLine(never.short) });
}

/** The menu row that opens the note step: it names the agent that asked. */
export function denyWhyText(from: PermissionInteraction['from']): string {
  return from === undefined ? CHAT.denyWhy : fill(CHAT.denyWhyHelper, { label: oneLine(from.label) });
}

/** Who asks, when it is an agent and not the conversation's own turn. */
export function fromText(from: From): string | null {
  if (from === undefined) return null;
  return from.run === null
    ? fill(CHAT.permFromNoRun, { label: from.label })
    : fill(CHAT.permFrom, { label: from.label, run: from.run });
}

export function metaText(folder: string | null, from?: From): string | undefined {
  const who = fromText(from);
  const elsewhere = from?.cwd ?? null;
  const where =
    elsewhere !== null
      ? fill(CHAT.permFromFolder, { folder: elsewhere })
      : folder === null
        ? null
        : fill(CHAT.permIn, { folder });
  const parts = [who, where].filter((part): part is string => part !== null);
  return parts.length === 0 ? undefined : parts.join(', ');
}

export function Meta({ folder, position, from }: Pick<CardBase, 'folder' | 'position'> & { from?: From }): ReactNode {
  const who = fromText(from);
  const elsewhere = from?.cwd ?? null;
  if (folder === null && elsewhere === null && who === null && position.count <= 1) return undefined;
  return (
    <>
      {who !== null ? <span data-asked-by="">{who}</span> : null}
      {elsewhere !== null ? (
        <span className="inline-flex min-w-0" data-works-in="">
          {fill(CHAT.permFromFolder, { folder: '' })}
          <PathText path={elsewhere} mono={false} maxChars={META_PATH_CHARS} />
        </span>
      ) : folder !== null ? (
        <span className="inline-flex min-w-0">
          {'in '}
          <PathText path={folder} mono={false} maxChars={META_PATH_CHARS} />
        </span>
      ) : null}
      {position.count > 1 ? <span>{fill(CHAT.permCount, { i: position.index, n: position.count })}</span> : null}
    </>
  );
}

/**
 * The reason line of a card under the shown mode: with no "Allow once" offered, that (and the hint to Ask only when
 * Ask draws an answer the mode hides); else Auto's own reason.
 */
export function cardReason(
  interaction: Pick<PermissionInteraction, 'access' | 'options' | 'autoAsk' | 'autoAskDetail' | 'neverUnread'>,
  permissionMode: PermissionMode,
  folder: string | null,
  temp: string | null = null,
): string | null {
  if (permissionMode === 'ask') return null;
  const p = CHAT.permission;
  if (permissionMode === 'bypass' && interaction.neverUnread === true) {
    return fill(CHAT.never.stillAsks, { mode: p.modes.bypass, why: CHAT.never.unread });
  }
  const once = interaction.options.some((option) => option.optionId === 'allow-once' && option.kind === 'allow_once');
  if (!once) {
    const line = fill(p.noAllowOnce, { mode: p.modes[permissionMode] });
    const more = cardButtons(interaction, 'ask').length > cardButtons(interaction, permissionMode).length;
    return more ? `${line} ${p.noAllowOnceAsk}` : line;
  }
  if (permissionMode === 'auto' && interaction.autoAsk !== undefined) {
    return autoAskText(interaction.autoAsk, folder, temp, interaction.autoAskDetail);
  }
  return null;
}

/** A permission request of the agent: the exact command or file, its warnings, and the answers the agent offered. */
export function PermissionCard(
  props: CardBase & {
    interaction: PermissionInteraction;
    permissionMode: PermissionMode;
    /** The temp folder Auto covers, as people type it. */
    temp: string | null;
  },
): ReactElement {
  const { interaction } = props;
  const buttons = cardButtons(interaction, props.permissionMode);
  const reason = cardReason(interaction, props.permissionMode, props.folder, props.temp);
  const [pressed, setPressed] = useState<string | null>(null);
  const [writing, setWriting] = useState(false);
  const [note, setNote] = useState('');
  const [picking, setPicking] = useState(false);
  const [scope, setScope] = useState<'short' | 'exact'>('short');
  const first = useRef<HTMLButtonElement | null>(null);
  const field = useRef<HTMLTextAreaElement | null>(null);
  const pick = useRef<HTMLDivElement | null>(null);
  const neverButton = useRef<HTMLButtonElement | null>(null);
  useArrivalFocus(first, props.composerRef, interaction.id);
  const armed = useArmed();
  const exactId = useId();
  const landsId = useId();
  const wrote = useRef(false);
  useEffect(() => {
    if (writing) {
      wrote.current = true;
      field.current?.focus({ preventScroll: true });
    } else if (wrote.current) {
      wrote.current = false;
      first.current?.focus({ preventScroll: true });
    }
  }, [writing]);
  const picked = useRef(false);
  useEffect(() => {
    if (picking) {
      picked.current = true;
      const radio = pick.current?.querySelector<HTMLElement>('[role="radio"][aria-checked="true"]');
      (radio ?? neverButton.current)?.focus({ preventScroll: true });
    } else if (picked.current) {
      picked.current = false;
      first.current?.focus({ preventScroll: true });
    }
  }, [picking]);

  const answer = (event: MouseEvent, button: CardButton): void => {
    if (pressed !== null) return;
    if (button.option.kind !== 'reject_once' && !armed(event)) return;
    const optionId = button.option.optionId;
    setPressed(optionId);
    props.onAnswer({ optionId }).catch(() => setPressed(null));
  };

  const deny = buttons.find((button) => button.option.kind === 'reject_once');
  const sendNote = (): void => {
    const text = note.trim();
    if (pressed !== null || deny === undefined || text === '') return;
    setPressed('note');
    props.onAnswer({ optionId: deny.option.optionId, note: text }).catch(() => setPressed(null));
  };

  const never = interaction.access === 'bash' ? (interaction.never ?? null) : null;
  const sendNever = (): void => {
    if (pressed !== null || never === null) return;
    setPressed('never');
    props.onAnswer({ never: never.short === never.exact ? 'short' : scope }).catch(() => setPressed(null));
  };

  let body: ReactNode;
  const describedBy = [exactId];
  if (interaction.access === 'bash' && interaction.command !== null) {
    body = <Exact text={interaction.command} label={CHAT.permCommand} id={exactId} />;
  } else if (interaction.access === 'edit' && interaction.path !== null) {
    if (interaction.resolvedPath !== null) describedBy.push(landsId);
    body = (
      <>
        <Exact text={interaction.path} label={CHAT.permFile} id={exactId} />
        {interaction.resolvedPath !== null ? (
          <Exact text={interaction.resolvedPath} label={CHAT.permLands} id={landsId} />
        ) : null}
        {interaction.input !== '' ? <Exact text={interaction.input} label={CHAT.permInput} /> : null}
      </>
    );
  } else {
    body = <Exact text={interaction.input} label={CHAT.permInput} id={exactId} />;
  }

  return (
    <InteractionCard
      askId={interaction.id}
      icon={<Shield size={16} strokeWidth={1.75} />}
      title={permissionTitle(interaction.access)}
      meta={<Meta folder={props.folder} position={props.position} from={interaction.from} />}
      metaText={metaText(props.folder, interaction.from)}
      describedBy={describedBy}
      subtitle={cardDescription(interaction) ?? undefined}
      subtitleTitle={oneLine(interaction.title)}
      warnings={cardWarnings(interaction)}
      warningIcon={<TriangleAlert size={12} strokeWidth={1.75} />}
      reason={
        reason === null
          ? undefined
          : {
              icon:
                props.permissionMode === 'bypass' ? (
                  <ShieldOff size={12} strokeWidth={1.75} />
                ) : (
                  <ShieldCheck size={12} strokeWidth={1.75} />
                ),
              text: reason,
            }
      }
      busy={pressed !== null}
      onKeyDown={
        picking
          ? (event) => {
              if (event.key !== 'Escape' || pressed !== null) return;
              event.preventDefault();
              event.stopPropagation();
              setPicking(false);
            }
          : undefined
      }
      notes={
        writing
          ? [interaction.access === 'bash' ? CHAT.denyWritingNote : CHAT.denyWritingNoteOther]
          : picking
            ? [CHAT.never.note]
            : cardNotes(interaction.access, buttons, interaction.from !== undefined && interaction.from.run !== null)
      }
      actions={
        picking ? (
          <>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={pressed !== null}
              onClick={() => setPicking(false)}
            >
              {CHAT.planBack}
            </Button>
            <Button
              ref={neverButton}
              type="button"
              variant="primary"
              size="sm"
              loading={pressed === 'never'}
              disabled={pressed !== null && pressed !== 'never'}
              data-never-confirm=""
              onClick={sendNever}
            >
              {CHAT.never.confirm}
            </Button>
          </>
        ) : writing ? (
          <>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={pressed !== null}
              onClick={() => setWriting(false)}
            >
              {CHAT.planBack}
            </Button>
            <Button
              type="button"
              variant="primary"
              size="sm"
              loading={pressed === 'note'}
              disabled={note.trim() === '' || (pressed !== null && pressed !== 'note')}
              onClick={sendNote}
            >
              {CHAT.denySend}
            </Button>
          </>
        ) : (
          buttons.map((button, index) =>
            button === deny ? (
              <SplitButton
                key={button.option.optionId}
                ref={index === 0 ? first : undefined}
                label={button.label}
                variant={button.variant}
                loading={pressed === button.option.optionId}
                disabled={pressed !== null && pressed !== button.option.optionId}
                data-option={button.option.optionId}
                moreLabel={CHAT.denyMore}
                menuLabel={CHAT.denyMore}
                onClick={(event) => answer(event, button)}
              >
                <MenuItem label={denyWhyText(interaction.from)} onSelect={() => setWriting(true)} />
                {never === null ? null : <MenuItem label={neverMenuText(never)} onSelect={() => setPicking(true)} />}
              </SplitButton>
            ) : (
              <Button
                key={button.option.optionId}
                ref={index === 0 ? first : undefined}
                type="button"
                variant={button.variant}
                size="sm"
                loading={pressed === button.option.optionId}
                disabled={pressed !== null && pressed !== button.option.optionId}
                data-option={button.option.optionId}
                onClick={(event) => answer(event, button)}
              >
                {button.label}
              </Button>
            ),
          )
        )
      }
    >
      {body}
      {picking && never !== null ? (
        <div ref={pick} className="grid gap-1" data-never-pick="">
          {never.short === never.exact ? (
            <>
              <p className="m-0 text-body text-secondary">{CHAT.never.pick}</p>
              <p className="m-0 font-mono text-body" data-never-words="">
                {oneLine(never.short)}
              </p>
            </>
          ) : (
            <>
              <p className="m-0 text-body text-secondary" aria-hidden="true">
                {CHAT.never.pick}
              </p>
              <OptionList
                label={CHAT.never.pick}
                value={scope}
                onValueChange={(value) => setScope(value === 'exact' ? 'exact' : 'short')}
                options={[
                  { value: 'short', title: oneLine(never.short) },
                  { value: 'exact', title: oneLine(never.exact) },
                ]}
                wrap
              />
            </>
          )}
        </div>
      ) : null}
      {writing ? (
        <CardTextField
          label={CHAT.denyNoteLabel}
          value={note}
          onChange={setNote}
          fieldRef={field}
          maxLength={FEEDBACK_MAX_CHARS}
          countText={fill(CHAT.feedbackCount, { n: note.length, max: FEEDBACK_MAX_CHARS })}
          onEscape={() => {
            if (pressed === null) setWriting(false);
          }}
          onSubmit={sendNote}
        />
      ) : null}
    </InteractionCard>
  );
}

/** The agent asks to leave plan mode: approve the plan, or send what should change. */
export function PlanCard(props: CardBase & { interaction: PlanInteraction }): ReactElement {
  const { interaction } = props;
  const noText = interaction.plan.trim() === '';
  const [writing, setWriting] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [pressed, setPressed] = useState<'approved' | 'rejected' | null>(null);
  const first = useRef<HTMLButtonElement | null>(null);
  const field = useRef<HTMLTextAreaElement | null>(null);
  useArrivalFocus(first, props.composerRef, interaction.id);
  const armed = useArmed();
  const [wasWriting, setWasWriting] = useState(writing);
  if (wasWriting !== writing) setWasWriting(writing);
  useEffect(() => {
    if (writing) field.current?.focus({ preventScroll: true });
    else if (wasWriting) first.current?.focus({ preventScroll: true });
  }, [writing]);

  const send = (decision: 'approved' | 'rejected'): void => {
    if (pressed !== null) return;
    setPressed(decision);
    const body: InteractionAnswer =
      decision === 'approved' || noText ? { decision } : { decision, feedback: feedback.trim() };
    props.onAnswer(body).catch(() => setPressed(null));
  };

  const actions = writing ? (
    <>
      <Button type="button" variant="secondary" size="sm" disabled={pressed !== null} onClick={() => setWriting(false)}>
        {CHAT.planBack}
      </Button>
      <Button
        type="button"
        variant="primary"
        size="sm"
        loading={pressed === 'rejected'}
        disabled={feedback.trim() === '' || (pressed !== null && pressed !== 'rejected')}
        onClick={() => send('rejected')}
      >
        {CHAT.planSend}
      </Button>
    </>
  ) : (
    <>
      <Button
        ref={first}
        type="button"
        variant="secondary"
        size="sm"
        loading={noText && pressed === 'rejected'}
        disabled={pressed !== null && !(noText && pressed === 'rejected')}
        onClick={() => (noText ? send('rejected') : setWriting(true))}
      >
        {noText ? CHAT.planKeepPlanning : CHAT.planFeedback}
      </Button>
      <Button
        type="button"
        variant="primary"
        size="sm"
        loading={pressed === 'approved'}
        disabled={pressed !== null && pressed !== 'approved'}
        onClick={(event) => {
          if (armed(event)) send('approved');
        }}
      >
        {CHAT.planApprove}
      </Button>
    </>
  );

  return (
    <InteractionCard
      askId={interaction.id}
      icon={<ClipboardCheck size={16} strokeWidth={1.75} />}
      title={CHAT.planTitle}
      meta={<Meta folder={props.folder} position={props.position} />}
      metaText={metaText(props.folder)}
      busy={pressed !== null}
      actions={actions}
    >
      {noText ? (
        <p className="m-0 text-body text-secondary">{CHAT.planNoText}</p>
      ) : (
        <Well label={CHAT.planLabel}>
          <ChatMarkdown text={interaction.plan} links={{ mode: 'work' }} />
        </Well>
      )}
      {writing ? (
        <CardTextField
          label={CHAT.planFeedbackLabel}
          value={feedback}
          onChange={setFeedback}
          fieldRef={field}
          maxLength={FEEDBACK_MAX_CHARS}
          countText={fill(CHAT.feedbackCount, { n: feedback.length, max: FEEDBACK_MAX_CHARS })}
          onEscape={() => {
            if (pressed === null) setWriting(false);
          }}
        />
      ) : null}
    </InteractionCard>
  );
}
