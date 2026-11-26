import { ArrowUpRight, ChevronLeft, X } from 'lucide-react';
import { motion } from 'motion/react';
import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactElement } from 'react';
import { Link } from 'react-router-dom';

import { progressOf, transcriptPath } from '../lib/activity';
import { agentKeyOf } from '../lib/agent-keys';
import { canSendMessage, type MessageEnd, type MessageTarget, messageCount } from '../lib/agent-message';
import {
  type AgentField,
  type AgentFeed,
  agentField,
  type AgentViewPlace,
  contextWords,
  markCountWords,
  splitCopied,
} from '../lib/agent-view';
import {
  type AgentAsk,
  type AgentButtons,
  type AgentNow,
  type AgentRow as AgentRowData,
  agentButtons,
  agentLiveOf,
  agentState,
  stoppedByYou,
} from '../lib/agents';
import { CHAT, fill } from '../lib/chat-strings';
import type { AgentLive, HelperProgress, LiveActivity } from '../lib/chat-types';
import { type RowWords, rowWords } from '../lib/helper-words';
import { retryTitleOf } from '../lib/sessions';
import { useStickToBottom } from '../lib/stick-to-bottom';
import { toast } from '../lib/toast';
import type { TranscriptEntry } from '../lib/types';
import styles from './agent-view.module.css';
import { AgentControls, type AgentHandlers } from './arc-x/agent-controls/agent-controls';
import { AgentGlyph } from './arc-x/agent-row/agent-row';
import { JumpButton } from './arc-x/chat-frame/chat-frame';
import { Disclosure } from './arc-x/disclosure/disclosure';
import { IconButton } from './arc-x/icon-button/icon-button';
import { CardTextField } from './arc-x/interaction-card/interaction-card';
import { MiddleText } from './arc-x/middle-text/middle-text';
import { useStaticMotion } from './arc-x/motion/use-static-motion';
import { Button } from './arc/button/button';
import { ExpandableText } from './arc/expandable-text/expandable-text';
import { motionTokens } from './arc/motion-tokens';
import SegmentedControl from './arc/segmented-control/segmented-control';
import { TranscriptStream } from './transcript';
import { useNow } from './wait-clock';

/** What a message field sends with: the page's handler and the drafts it keeps per agent (by its newest attempt). */
export interface AgentMessaging {
  drafts: Map<string, string>;
  onSend: (target: MessageTarget, text: string) => Promise<MessageEnd>;
}

/** One agent now, as its view's head and field read it. */
export interface AgentFacts {
  now: AgentNow;
  progress: HelperProgress | null;
  held: AgentLive | null;
  ask: AgentAsk | null;
  words: RowWords;
  buttons: AgentButtons;
  field: AgentField;
}

/** The row's state, words, buttons and field from the host's live facts. */
export function agentFacts(
  row: AgentRowData,
  input: { agentLive: boolean; live: LiveActivity | undefined; asks: readonly AgentAsk[]; now: number },
): AgentFacts {
  const now = agentState(row, { live: input.agentLive, now: input.now });
  const running = now.state === 'running';
  const progress = running ? progressOf(input.live, row.childId) : null;
  const held = agentLiveOf(input.live, row);
  const ask = input.asks.find((item) => row.attempts.includes(item.child)) ?? null;
  const words = rowWords(now, {
    progress,
    now: input.now,
    ask: ask === null ? null : { title: ask.title, since: ask.openedAt },
    stopping: held?.stopping === true,
    stoppedByYou: stoppedByYou(input.live, row),
  });
  const workflow = row.run !== null;
  return {
    now,
    progress,
    held,
    ask,
    words,
    buttons: agentButtons({ workflow, state: now.state, held, askId: ask?.id ?? null }),
    field: agentField({ workflow, state: now.state, held, waits: ask !== null }),
  };
}

const MAC = typeof navigator !== 'undefined' && navigator.userAgent.includes('Mac');

/**
 * The agent's state with its word and facts, and its head buttons (Review, Stop). Message is the field under the
 * view, so it is not one of them.
 */
export function AgentStateLine(props: {
  row: AgentRowData;
  facts: AgentFacts;
  handlers: Pick<AgentHandlers, 'onStop' | 'onReview'> | null;
}): ReactElement {
  const { row, facts } = props;
  const { words } = facts;
  const pct = facts.progress === null ? null : fill(CHAT.agent.facts.contextPct, { pct: facts.progress.contextPct });
  const shown = words.facts.filter((fact) => fact !== pct);
  const handlers = props.handlers;
  const buttons: AgentButtons = { ...facts.buttons, message: null };
  return (
    <div className={styles.state} data-mark={words.mark} data-tone={words.tone}>
      <span className={styles.glyph} aria-hidden="true">
        <AgentGlyph mark={words.mark} tone={words.tone} pulse />
      </span>
      <span className={styles.word}>{words.word}</span>
      {shown.length > 0 ? <span className={styles.stateFacts}>{shown.join(' · ')}</span> : null}
      {handlers !== null ? (
        <AgentControls
          id={row.childId}
          label={row.label}
          workflow={row.run !== null}
          buttons={buttons}
          onStop={() => handlers.onStop({ id: row.childId, title: row.label })}
          onReview={handlers.onReview}
          className={styles.controls}
        />
      ) : null}
    </div>
  );
}

/**
 * Where a running Task subagent takes a message: ⌘↩ (Ctrl+Enter) sends, Enter is a new line, Escape calls
 * `onEscape`. One send at a time: the field is read-only while it goes. Its text is kept per agent in `drafts`. Any
 * other agent gets one line in its place.
 */
export function AgentMessageField(props: {
  target: MessageTarget;
  field: AgentField;
  messaging: AgentMessaging;
  /** Each new value past 0 puts focus in the field. */
  focusTick: number;
  onEscape?: () => void;
}): ReactElement {
  const { target, field, messaging } = props;
  if (field.kind === 'note') {
    return (
      <p className={styles.note} data-agent-note="">
        {field.text}
      </p>
    );
  }
  return (
    <MessageBox
      key={target.id}
      target={target}
      notice={field.notice}
      messaging={messaging}
      focusTick={props.focusTick}
      onEscape={props.onEscape}
    />
  );
}

function MessageBox(props: {
  target: MessageTarget;
  notice: string | null;
  messaging: AgentMessaging;
  focusTick: number;
  onEscape?: () => void;
}): ReactElement {
  const { target, messaging } = props;
  const { drafts } = messaging;
  const countId = useId();
  const hintId = useId();
  const reasonId = useId();
  const noticeId = useId();
  const [text, setText] = useState(() => drafts.get(target.id) ?? '');
  const [sending, setSending] = useState(false);
  const [reason, setReason] = useState<string | null>(null);
  const field = useRef<HTMLTextAreaElement | null>(null);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const tick = props.focusTick;
  useEffect(() => {
    if (tick > 0) field.current?.focus({ preventScroll: true });
  }, [tick]);
  const count = messageCount(text);
  const canSend = !sending && canSendMessage(text);
  const edit = (value: string): void => {
    setText(value);
    if (value === '') drafts.delete(target.id);
    else drafts.set(target.id, value);
  };
  const send = (): void => {
    if (!canSend) return;
    setSending(true);
    setReason(null);
    void messaging.onSend(target, text).then((end) => {
      if (!alive.current) {
        if (end.kind === 'stay') toast.error(end.reason);
        return;
      }
      setSending(false);
      if (end.kind === 'close') {
        if (end.clear) setText('');
        return;
      }
      setReason(end.reason);
      field.current?.focus();
    });
  };
  const described = [props.notice !== null ? noticeId : '', count !== null ? countId : '', hintId]
    .concat(reason !== null ? [reasonId] : [])
    .filter((id) => id !== '')
    .join(' ');
  return (
    <div className={styles.field} data-agent-field="">
      {props.notice !== null ? (
        <p id={noticeId} className={styles.notice}>
          {props.notice}
        </p>
      ) : null}
      <CardTextField
        label={fill(CHAT.agent.message, { label: target.title })}
        value={text}
        onChange={edit}
        fieldRef={field}
        readOnly={sending}
        describedBy={described}
        onSubmit={send}
        onEscape={props.onEscape}
      />
      {count !== null ? (
        <p id={countId} className={styles.count} data-tone={count.tooLong ? 'danger' : undefined}>
          {count.text}
        </p>
      ) : null}
      {reason !== null ? (
        <p id={reasonId} role="alert" className={styles.reason}>
          {reason}
        </p>
      ) : null}
      <div className={styles.fieldFoot}>
        <p id={hintId} className={styles.hint}>
          {MAC ? CHAT.agent.hint.subagent : CHAT.agent.hint.subagentCtrl}
        </p>
        <Button
          type="button"
          variant="primary"
          size="sm"
          aria-disabled={canSend ? undefined : true}
          aria-busy={sending ? true : undefined}
          onClick={send}
        >
          {sending ? CHAT.activity.messageSending : CHAT.activity.messageSend}
        </Button>
      </div>
    </div>
  );
}

/** The attempt's first message: its task, or a contract retry's fix-up, clamped to two lines. */
function PromptRow({ entry }: { entry: Extract<TranscriptEntry, { kind: 'user' }> }): ReactElement {
  return (
    <div className={styles.prompt} data-agent-prompt="">
      <span className={styles.promptLabel}>{retryTitleOf(entry.text) ?? CHAT.agent.view.prompt}</span>
      <ExpandableText lines={2} moreLabel="Show more" lessLabel="Show less" className={styles.promptText}>
        {entry.text}
      </ExpandableText>
    </div>
  );
}

export interface AgentViewProps {
  row: AgentRowData;
  /** The attempt shown, one of the row's attempts, and its list. */
  attempt: string;
  feed: AgentFeed | null;
  onRetry: () => void;
  onAttempt: (childId: string) => void;
  place: AgentViewPlace;
  /** The name of its run; null for a Task subagent. */
  runName: string | null;
  /** The conversation is loaded in a running agent process. */
  agentLive: boolean;
  live: LiveActivity | undefined;
  asks: readonly AgentAsk[];
  /** Requests and runs that wait for the person: `‹ Activity` carries the count while the rail is hidden. */
  needs: number;
  /** Calls a mode allowed for it and those the user denied with a note: a link to the lists under its row. */
  marks: { allowed: number; denied: number };
  /** Its session exists: Open in Sessions. */
  opens: boolean;
  handlers: Pick<AgentHandlers, 'onStop' | 'onReview'>;
  messaging: AgentMessaging;
  /** Each new value past 0 puts focus in the message field. */
  fieldTick: number;
  /** In the column, each new value past 0 puts focus on its title. */
  titleTick: number;
  onBack: () => void;
  onClose: () => void;
  /** Null: its row is not in the loaded thread. */
  onShowInThread: (() => void) | null;
  onShowMarks: (() => void) | null;
  /** `[` and `]`: the agent before or after it in its list. */
  onStep: (step: -1 | 1) => void;
}

/**
 * One agent beside the thread: its head (state, run and step, model, context, attempts, permission counts), its
 * lines as one stream that follows the end, and its message field. In the column it slides in and Escape closes it;
 * in a drawer the drawer's own title and close button stand for the label and ✕.
 */
export function AgentView(props: AgentViewProps): ReactElement {
  const { row, feed, place } = props;
  const staticMotion = useStaticMotion();
  const live = props.live;
  const [clockOn, setClockOn] = useState(true);
  const now = useNow(clockOn);
  const facts = agentFacts(row, { agentLive: props.agentLive, live, asks: props.asks, now });
  const ticking = facts.now.state === 'running' || facts.now.until !== null;
  if (ticking !== clockOn) setClockOn(ticking);
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null);
  const stick = useStickToBottom(scroller, `agent:${props.attempt}`);
  const title = useRef<HTMLHeadingElement | null>(null);
  const titleTick = props.titleTick;
  useEffect(() => {
    if (titleTick > 0 && place === 'column') title.current?.focus({ preventScroll: true });
  }, [titleTick, place]);
  const entries = feed?.entries ?? [];
  const { copied, own } = splitCopied(entries);
  const promptAt = own.findIndex((entry) => entry.kind === 'user');
  const prompt = promptAt === -1 ? null : (own[promptAt] as Extract<TranscriptEntry, { kind: 'user' }>);
  const rest = promptAt === -1 ? own : own.filter((_entry, n) => n !== promptAt);
  const attemptAt = row.attempts.indexOf(props.attempt);
  const model = row.entry?.model ?? null;
  const line3 = [
    props.runName !== null ? (row.phase !== null ? `${props.runName} › ${row.phase}` : props.runName) : null,
    model,
    facts.progress !== null ? contextWords(facts.progress) : null,
  ].filter((part): part is string => part !== null && part !== '');
  const marksWords = markCountWords(props.marks);
  const v = CHAT.agent.view;

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.defaultPrevented) return;
    const action = agentKeyOf(event, 'view', { pause: false });
    if (action === 'close' && place === 'column') {
      event.preventDefault();
      props.onClose();
    } else if (action === 'previous-agent' || action === 'next-agent') {
      event.preventDefault();
      props.onStep(action === 'previous-agent' ? -1 : 1);
    }
  };

  return (
    <motion.div
      className={styles.root}
      data-agent-view={row.childId}
      data-place={place}
      onKeyDown={onKeyDown}
      initial={staticMotion || place !== 'column' ? false : { opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={staticMotion ? { duration: 0 } : motionTokens.spring.smooth}
    >
      <header className={styles.head}>
        <div className={styles.bar}>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className={styles.back}
            aria-label={props.needs > 0 ? `${v.backName}, ${fill(v.needsYou, { n: props.needs })}` : v.backName}
            onClick={props.onBack}
          >
            <ChevronLeft size={14} strokeWidth={1.75} aria-hidden="true" />
            {v.back}
            {props.needs > 0 ? (
              <span className={styles.needs}>{` · ${fill(v.needsYou, { n: props.needs })}`}</span>
            ) : null}
          </Button>
          <span className={styles.barEnd}>
            {props.onShowInThread !== null ? (
              <Button type="button" variant="ghost" size="sm" onClick={props.onShowInThread}>
                {v.showInThread}
              </Button>
            ) : null}
            {props.opens ? (
              <Link className={`link ${styles.sessions}`} to={transcriptPath(row.childId)}>
                {v.openInSessions}
                <ArrowUpRight size={12} strokeWidth={2} aria-hidden="true" />
              </Link>
            ) : null}
            {place === 'column' ? (
              <IconButton
                label={fill(v.close, { label: row.label })}
                icon={<X size={14} strokeWidth={1.75} aria-hidden="true" />}
                onClick={props.onClose}
              />
            ) : null}
          </span>
        </div>
        <h2 ref={title} className={styles.title} tabIndex={-1} data-agent-title="" hidden={place !== 'column'}>
          <MiddleText text={row.label} />
        </h2>
        <AgentStateLine row={row} facts={facts} handlers={props.handlers} />
        {line3.length > 0 ? <p className={styles.line}>{line3.join(' · ')}</p> : null}
        {facts.words.sub.length > 0 || marksWords !== null || row.attempts.length > 1 ? (
          <div className={styles.subRow}>
            <p className={styles.line}>
              {facts.words.sub.join(' · ')}
              {marksWords !== null ? (
                <>
                  {facts.words.sub.length > 0 ? ' · ' : null}
                  {props.onShowMarks !== null ? (
                    <button
                      type="button"
                      className={`link ${styles.marks}`}
                      aria-label={fill(v.marksName, { label: row.label })}
                      onClick={props.onShowMarks}
                    >
                      {marksWords}
                    </button>
                  ) : (
                    marksWords
                  )}
                </>
              ) : null}
            </p>
            {row.attempts.length > 1 ? (
              <SegmentedControl
                label={v.attempts}
                value={props.attempt}
                onValueChange={props.onAttempt}
                options={row.attempts.map((id, n) => ({ value: id, label: fill(v.attempt, { n: n + 1 }) }))}
                className={styles.attempts}
              />
            ) : null}
          </div>
        ) : null}
      </header>
      <div className={styles.body}>
        <div
          ref={setScroller}
          className={styles.scroll}
          aria-busy={feed === null || feed.status === 'loading' ? true : undefined}
        >
          <div ref={stick.contentRef} className={styles.stream} data-agent-stream="">
            {copied.length > 0 ? (
              <Disclosure title={fill(v.copied, { n: Math.max(1, attemptAt) })} className={styles.copied}>
                <TranscriptStream sessionId={props.attempt} entries={copied} live={false} model={model} />
              </Disclosure>
            ) : null}
            {prompt !== null ? <PromptRow entry={prompt} /> : null}
            {rest.length > 0 ? (
              <TranscriptStream sessionId={props.attempt} entries={rest} live={feed?.live === true} model={model} />
            ) : null}
            {feed?.status === 'ready' && entries.length === 0 ? <p className={styles.empty}>{v.empty}</p> : null}
            {feed?.status === 'failed' ? (
              <div className={styles.failed} role="alert">
                <span>{v.failed}</span>
                <Button type="button" variant="secondary" size="sm" onClick={props.onRetry}>
                  {v.tryAgain}
                </Button>
              </div>
            ) : null}
          </div>
        </div>
        {stick.showJump ? (
          <div className={styles.jump}>
            <JumpButton onJump={() => stick.jump()} fresh={stick.fresh} />
          </div>
        ) : null}
      </div>
      <footer className={styles.foot}>
        <AgentMessageField
          target={{ id: row.childId, title: row.label }}
          field={facts.field}
          messaging={props.messaging}
          focusTick={props.fieldTick}
          onEscape={place === 'column' ? props.onClose : undefined}
        />
      </footer>
    </motion.div>
  );
}
