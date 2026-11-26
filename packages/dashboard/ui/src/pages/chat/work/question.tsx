import { CircleHelp } from 'lucide-react';
import { type KeyboardEvent, type ReactElement, useEffect, useId, useRef, useState } from 'react';

import {
  AnswerList,
  NO_ANSWER,
  pickRow,
  type AnswerOption,
  type AnswerState,
} from '../../../components/arc-x/answer-list/answer-list';
import { InteractionCard } from '../../../components/arc-x/interaction-card/interaction-card';
import { Steps } from '../../../components/arc-x/steps/steps';
import { Button } from '../../../components/arc/button/button';
import { CHAT, fill } from '../../../lib/chat-strings';
import type { QuestionInteraction } from '../../../lib/chat-types';
import {
  type CardBase,
  Exact,
  FEEDBACK_MAX_CHARS,
  type InteractionAnswer,
  Meta,
  metaText,
  useArmed,
} from './permission';

type Question = QuestionInteraction['questions'][number];
type Answer = { picks: number[]; other?: string } | null;
type Pressed = 'send' | 'skip' | 'discuss' | 'plan-now';

/** The step and the half-filled answers of open question cards, by request id: a card shown again keeps them. */
const drafts = new Map<string, { step: number; answers: AnswerState[] }>();
/** The conversation of each question card the page saw open. */
const draftOwners = new Map<string, string>();

/**
 * The open question cards of a conversation, as the page sees them now: the drafts of its cards that closed (answered,
 * skipped, stopped, given up) are dropped. Cards of other conversations keep theirs.
 */
export function keepQuestionDrafts(conversation: string, open: readonly string[]): void {
  for (const id of open) draftOwners.set(id, conversation);
  for (const [id, owner] of draftOwners) {
    if (owner !== conversation || open.includes(id)) continue;
    draftOwners.delete(id);
    drafts.delete(id);
  }
}

/**
 * The option whose preview shows under the list: the focused option, else the picked one; single select only (the
 * agent sends previews for single-select questions). Null when that option has none.
 */
export function previewOf(question: Question, state: AnswerState, focused: number | null): number | null {
  if (question.multi) return null;
  const count = question.options.length;
  const row = focused !== null && focused < count ? focused : state.otherOn ? null : (state.picks[0] ?? null);
  const preview = row === null ? null : (question.options[row]?.preview ?? null);
  return preview === null ? null : row;
}

const RECOMMENDED = ' (recommended)';

/** A label the agent marks "(Recommended)" shows the mark as a tag; the answer still names the option by index. */
export function optionView(option: Question['options'][number]): AnswerOption {
  const marked = option.label.toLowerCase().endsWith(RECOMMENDED) && option.label.length > RECOMMENDED.length;
  return {
    label: marked ? option.label.slice(0, option.label.length - RECOMMENDED.length) : option.label,
    description: option.description,
    ...(marked ? { tag: CHAT.ask.recommended } : {}),
  };
}

/** One question's answer as the route takes it: the picks, the user's own text when it is not blank; null for none. */
export function answerOf(state: AnswerState, question: Question): Answer {
  const own = state.otherOn ? state.otherText.trim() : '';
  if (!question.multi) {
    if (!state.otherOn && state.picks.length === 1) return { picks: [state.picks[0]!] };
    return own === '' ? null : { picks: [], other: own };
  }
  if (state.picks.length === 0 && own === '') return null;
  return own === '' ? { picks: [...state.picks] } : { picks: [...state.picks], other: own };
}

function inTextField(target: EventTarget | null): boolean {
  return target instanceof HTMLTextAreaElement || target instanceof HTMLInputElement;
}

/**
 * The agent asks its own questions: one question per step, options as a radio group or checkboxes, an "Other" row
 * with its own text, Skip; in plan mode also Talk it over and Plan now, which send the answers so far. The focused or
 * picked option's preview shows under the list as plain text. Nothing is picked when the card arrives, and Send (also
 * by Enter) counts only after the card has been on screen for the arm delay. The answer goes as option indexes; the
 * shown text is never the key. The step and the picks stay in page memory until the card closes, so a card shown
 * again (another conversation, another card in front) has them back.
 */
export function QuestionCard(
  props: CardBase & {
    interaction: QuestionInteraction;
    /** Words for the page's polite status region. */
    onStatus?: (text: string) => void;
  },
): ReactElement {
  const { interaction } = props;
  const questions = interaction.questions;
  const n = questions.length;
  const draft = drafts.get(interaction.id);
  const [step, setStep] = useState(() => (draft !== undefined && draft.step < n ? draft.step : 0));
  const [answers, setAnswers] = useState<AnswerState[]>(
    () => (draft?.answers.length === n ? draft.answers : null) ?? questions.map(() => NO_ANSWER),
  );
  const [pressed, setPressed] = useState<Pressed | null>(null);
  const [focused, setFocused] = useState<number | null>(null);
  const armed = useArmed();
  const group = useRef<HTMLDivElement | null>(null);
  const questionId = useId();
  const hintId = useId();
  const stepMoved = useRef(false);

  const question = questions[step]!;
  const count = question.options.length;
  const last = step === n - 1;
  const value = answers[step] ?? NO_ANSWER;
  const body = answers.map((state, index) => answerOf(state, questions[index]!));
  const done = body.map((answer) => answer !== null);
  const doneCount = done.filter(Boolean).length;
  const canSend = doneCount > 0 && pressed === null;
  const plan = interaction.planMode;
  const previewRow = previewOf(question, value, focused);
  const previewOption = previewRow === null ? null : question.options[previewRow]!;

  const focusGroup = (): void => {
    group.current?.querySelector<HTMLElement>('[tabindex="0"]')?.focus({ preventScroll: true });
  };
  useEffect(() => {
    const area = props.composerRef.current;
    if (area !== null && document.activeElement === area && area.value.trim() !== '') return;
    focusGroup();
  }, [interaction.id]);
  useEffect(() => {
    if (!stepMoved.current) return;
    stepMoved.current = false;
    focusGroup();
    props.onStatus?.(fill(CHAT.status.askStep, { i: step + 1, n }));
  }, [step]);

  const setValue = (next: AnswerState): void =>
    setAnswers((current) => {
      const all = current.map((state, index) => (index === step ? next : state));
      drafts.set(interaction.id, { step, answers: all });
      return all;
    });
  const reply = (kind: Pressed, answer: InteractionAnswer): void => {
    setPressed(kind);
    props
      .onAnswer(answer)
      .then(() => drafts.delete(interaction.id))
      .catch(() => setPressed(null));
  };
  const send = (): void => {
    if (canSend) reply('send', { answers: body });
  };
  const skip = (): void => {
    if (pressed === null) reply('skip', { reply: 'skip' });
  };
  const planReply = (kind: 'discuss' | 'plan-now'): void => {
    if (pressed === null && plan) reply(kind, { reply: kind, answers: body });
  };
  const move = (to: number): void => {
    if (pressed !== null || to < 0 || to >= n) return;
    stepMoved.current = true;
    setFocused(null);
    drafts.set(interaction.id, { step: to, answers });
    setStep(to);
  };
  const onEnter = (): void => {
    if (last) send();
    else move(step + 1);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (pressed !== null || inTextField(event.target) || event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key.length !== 1 || event.key < '1' || event.key > '9') return;
    const row = Number(event.key) - 1;
    if (row > count) return;
    event.preventDefault();
    const next = pickRow(value, row, count, question.multi, question.multi);
    setValue(next);
    const target =
      row === count && next.otherOn
        ? group.current?.parentElement?.querySelector<HTMLElement>('[data-other-field]')
        : group.current?.querySelector<HTMLElement>(`[data-row="${row}"]`);
    target?.focus({ preventScroll: true });
  };

  const stepWords = (i: number): string => fill(CHAT.ask.step, { i: i + 1, n });
  return (
    <InteractionCard
      askId={interaction.id}
      icon={<CircleHelp size={16} strokeWidth={1.75} />}
      title={n === 1 ? CHAT.ask.title : fill(CHAT.ask.titleMany, { n })}
      meta={<Meta folder={props.folder} position={props.position} />}
      metaText={metaText(props.folder)}
      describedBy={[questionId]}
      busy={pressed !== null}
      onKeyDown={onKeyDown}
      notes={
        plan
          ? [CHAT.ask.skipNote, CHAT.ask.discussNote, CHAT.ask.planNowNote, CHAT.ask.waitNote]
          : [CHAT.ask.skipNote, CHAT.ask.waitNote]
      }
      actions={
        <>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            loading={pressed === 'skip'}
            disabled={pressed !== null && pressed !== 'skip'}
            data-ask-skip=""
            onClick={skip}
          >
            {n > 1 ? CHAT.ask.skipMany : CHAT.ask.skip}
          </Button>
          {plan
            ? (['discuss', 'plan-now'] as const).map((kind) => (
                <Button
                  key={kind}
                  type="button"
                  variant="ghost"
                  size="sm"
                  loading={pressed === kind}
                  disabled={pressed !== null && pressed !== kind}
                  data-ask-plan={kind}
                  onClick={() => planReply(kind)}
                >
                  {kind === 'discuss' ? CHAT.ask.discuss : CHAT.ask.planNow}
                </Button>
              ))
            : null}
          <span className="flex-1" aria-hidden="true" />
          {step > 0 ? (
            <Button
              type="button"
              variant="secondary"
              size="sm"
              disabled={pressed !== null}
              onClick={() => move(step - 1)}
            >
              {CHAT.ask.back}
            </Button>
          ) : null}
          {last ? (
            <Button
              key="send"
              type="button"
              variant="primary"
              size="sm"
              loading={pressed === 'send'}
              disabled={!canSend && pressed !== 'send'}
              data-ask-send=""
              onClick={(event) => {
                if (armed(event)) send();
              }}
            >
              {n > 1 ? CHAT.ask.sendMany : CHAT.ask.send}
            </Button>
          ) : (
            <Button
              key="next"
              type="button"
              variant="primary"
              size="sm"
              disabled={pressed !== null}
              onClick={() => move(step + 1)}
            >
              {CHAT.ask.next}
            </Button>
          )}
        </>
      }
    >
      {n > 1 ? (
        <div className="flex min-w-0 items-center gap-3">
          <Steps
            label={CHAT.ask.stepsLabel}
            steps={questions.map((_, i) => stepWords(i))}
            current={step}
            done={done}
            bare
          />
          <span className="text-body text-secondary" aria-hidden="true">
            {stepWords(step)}
          </span>
        </div>
      ) : null}
      <div className="flex min-w-0 items-baseline justify-between gap-3">
        <p id={questionId} className="m-0 min-w-0 whitespace-pre-wrap text-body font-medium [overflow-wrap:anywhere]">
          {question.text}
        </p>
        {question.multi ? (
          <span id={hintId} className="shrink-0 text-caption text-secondary">
            {CHAT.ask.multiHint}
          </span>
        ) : null}
      </div>
      <AnswerList
        key={step}
        ref={group}
        label={question.text}
        describedBy={question.multi ? hintId : undefined}
        multi={question.multi}
        options={question.options.map(optionView)}
        value={value}
        onChange={setValue}
        onEnter={onEnter}
        enterReady={(event) => armed(event)}
        other={{
          label: CHAT.ask.other,
          fieldLabel: fill(CHAT.ask.otherLabel, { question: question.text }),
          placeholder: CHAT.ask.otherPlaceholder,
        }}
        otherMax={FEEDBACK_MAX_CHARS}
        disabled={pressed !== null}
        onFocusRow={setFocused}
      />
      {previewOption !== null ? (
        <div className="flex min-w-0 flex-col gap-1" data-ask-preview="">
          <p className="m-0 text-caption text-secondary" aria-hidden="true">
            {fill(CHAT.ask.preview, { label: optionView(previewOption).label })}
          </p>
          <Exact
            text={previewOption.preview!}
            label={fill(CHAT.ask.preview, { label: optionView(previewOption).label })}
          />
        </div>
      ) : null}
      {last && n > 1 && doneCount < n ? (
        <p className="m-0 text-caption text-secondary">{fill(CHAT.ask.answeredCount, { done: doneCount, n })}</p>
      ) : null}
    </InteractionCard>
  );
}
