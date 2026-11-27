import { ExternalLink, Plug, TriangleAlert } from 'lucide-react';
import { type KeyboardEvent, type MouseEvent, type ReactElement, useEffect, useId, useRef, useState } from 'react';

import { ButtonLink } from '../../../components/arc-x/button-link/button-link';
import { InteractionCard, Well } from '../../../components/arc-x/interaction-card/interaction-card';
import {
  initialValue,
  rfc3339,
  SchemaField,
  SchemaForm,
  type SchemaValue,
} from '../../../components/arc-x/schema-field/schema-field';
import { Button } from '../../../components/arc/button/button';
import { CHAT, fill } from '../../../lib/chat-strings';
import type { InteractionAnswerResult, McpField, McpFieldError, McpInteraction } from '../../../lib/chat-types';
import { type CardBase, type InteractionAnswer, Meta, metaText, useArmed } from './permission';

/** Typed values of open MCP cards, by request id: a card shown again keeps them until it closes. */
const drafts = new Map<string, SchemaValue[]>();
/** The conversation of each MCP card the page saw open. */
const draftOwners = new Map<string, string>();

/**
 * The open MCP cards of a conversation, as the page sees them now: the typed values of its cards that closed (sent,
 * declined, stopped, given up) are dropped. Cards of other conversations keep theirs.
 */
export function keepMcpDrafts(conversation: string, open: readonly string[]): void {
  for (const id of open) draftOwners.set(id, conversation);
  for (const [id, owner] of draftOwners) {
    if (owner !== conversation || open.includes(id)) continue;
    draftOwners.delete(id);
    drafts.delete(id);
  }
}

/** The words of one refused value. */
export function errorText(error: McpFieldError): string {
  const words = CHAT.mcp.errors;
  return 'n' in error ? fill(words[error.code], { n: error.n }) : words[error.code];
}

/** The values as the answer route takes them: one per field, a date and time with the local zone's offset. */
export function valuesOf(fields: readonly McpField[], values: readonly SchemaValue[]): SchemaValue[] {
  return fields.map((field, i) => {
    const value = values[i] ?? null;
    return field.type === 'text' && field.format === 'date-time' && typeof value === 'string' ? rfc3339(value) : value;
  });
}

type McpCardProps = Omit<CardBase, 'onAnswer'> & {
  interaction: McpInteraction;
  onAnswer: (answer: InteractionAnswer) => Promise<InteractionAnswerResult>;
};

/** An MCP server asks for input: a form, or a page to open. */
export function McpCard(props: McpCardProps): ReactElement {
  return props.interaction.mode === 'url' ? <McpLinkCard {...props} /> : <McpFormCard {...props} />;
}

/**
 * An MCP server asks for input: its message, a form built from its schema, Decline and Send. The host checks the
 * values; a refused Send leaves the card open with each wrong field marked, a summary said at once, and focus on the
 * first wrong field. Send (also Enter in a one-line field) counts only after the arm delay. A form the app cannot
 * draw offers Decline only. Typed values stay in page memory until the card is answered, never anywhere else.
 */
function McpFormCard(props: McpCardProps): ReactElement {
  const { interaction } = props;
  const fields = interaction.form?.fields ?? [];
  const usable = interaction.unusable === undefined && interaction.form !== undefined;
  const [values, setValues] = useState<SchemaValue[]>(
    () => drafts.get(interaction.id) ?? fields.map((field) => initialValue(field)),
  );
  const [errors, setErrors] = useState<Array<McpFieldError | null>>([]);
  const [pressed, setPressed] = useState<'send' | 'decline' | null>(null);
  const [refusals, setRefusals] = useState(0);
  const armed = useArmed();
  const form = useRef<HTMLDivElement | null>(null);
  const declineButton = useRef<HTMLButtonElement | null>(null);
  const messageId = useId();
  const summaryId = useId();
  const wrong = errors.filter((error) => error !== null).length;

  const focusFirst = (selector: string): void => {
    form.current?.querySelector<HTMLElement>(selector)?.focus({ preventScroll: true });
  };
  const firstControl = 'input, select, textarea, button[role="switch"]';
  useEffect(() => {
    const area = props.composerRef.current;
    if (area !== null && document.activeElement === area && area.value.trim() !== '') return;
    if (usable && fields.length > 0) focusFirst(firstControl);
    else declineButton.current?.focus({ preventScroll: true });
  }, [interaction.id]);
  useEffect(() => {
    if (refusals === 0) return;
    const at = errors.findIndex((error) => error !== null);
    if (at === -1) return;
    const box = form.current?.querySelector<HTMLElement>(`[data-field="${at}"]`);
    box?.querySelector<HTMLElement>(firstControl)?.focus({ preventScroll: true });
  }, [refusals]);

  const setValue = (index: number, value: SchemaValue): void => {
    setValues((current) => {
      const next = current.map((v, i) => (i === index ? value : v));
      drafts.set(interaction.id, next);
      return next;
    });
    setErrors((current) => current.map((error, i) => (i === index ? null : error)));
  };
  const send = (): void => {
    if (pressed !== null || !usable) return;
    setPressed('send');
    props
      .onAnswer({ mcp: 'accept', values: valuesOf(fields, values) })
      .then((result) => {
        if (result.fields === undefined) {
          drafts.delete(interaction.id);
          return;
        }
        setErrors(result.fields);
        setRefusals((n) => n + 1);
        setPressed(null);
      })
      .catch(() => setPressed(null));
  };
  const decline = (): void => {
    if (pressed !== null) return;
    setPressed('decline');
    props
      .onAnswer({ mcp: 'decline' })
      .then(() => drafts.delete(interaction.id))
      .catch(() => setPressed(null));
  };
  const onEnter = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (!event.repeat && armed(event)) send();
  };

  const notes = usable
    ? [fill(CHAT.mcp.formNote, { server: interaction.server }), CHAT.mcp.secretNote]
    : [fill(CHAT.mcp.formNote, { server: interaction.server })];
  const leftOut = interaction.form?.leftOut ?? 0;
  return (
    <InteractionCard
      askId={interaction.id}
      icon={<Plug size={16} strokeWidth={1.75} />}
      title={fill(CHAT.mcp.formTitle, { server: interaction.server })}
      subtitle={fill(CHAT.mcp.from, { server: interaction.server })}
      meta={<Meta folder={props.folder} position={props.position} />}
      metaText={metaText(props.folder)}
      describedBy={[messageId, ...(wrong > 0 ? [summaryId] : [])]}
      busy={pressed !== null}
      warnings={
        interaction.unusable === undefined
          ? undefined
          : [fill(CHAT.mcp.unusable, { why: CHAT.mcp.why[interaction.unusable] })]
      }
      warningIcon={<TriangleAlert size={12} strokeWidth={1.75} />}
      notes={notes}
      actions={
        <>
          <Button
            ref={declineButton}
            type="button"
            variant="secondary"
            size="sm"
            loading={pressed === 'decline'}
            disabled={pressed !== null && pressed !== 'decline'}
            data-mcp-decline=""
            onClick={decline}
          >
            {CHAT.mcp.decline}
          </Button>
          <span className="flex-1" aria-hidden="true" />
          {usable ? (
            <Button
              type="button"
              variant="primary"
              size="sm"
              loading={pressed === 'send'}
              disabled={pressed !== null && pressed !== 'send'}
              data-mcp-send=""
              onClick={(event) => {
                if (armed(event)) send();
              }}
            >
              {CHAT.mcp.send}
            </Button>
          ) : null}
        </>
      }
    >
      <Well label={CHAT.mcp.message}>
        <p id={messageId} className="m-0 whitespace-pre-wrap text-body [overflow-wrap:anywhere]">
          {interaction.message}
        </p>
      </Well>
      {usable ? (
        <div ref={form} className="grid min-w-0 gap-2">
          {wrong > 0 ? (
            <p id={summaryId} role="alert" className="m-0 text-body font-medium text-danger">
              {wrong === 1 ? CHAT.mcp.errorCountOne : fill(CHAT.mcp.errorCount, { n: wrong })}
            </p>
          ) : null}
          {fields.length > 0 ? (
            <SchemaForm>
              {fields.map((field, i) => (
                <SchemaField
                  key={i}
                  field={field}
                  index={i}
                  value={values[i] ?? null}
                  error={errors[i] == null ? null : errorText(errors[i]!)}
                  disabled={pressed !== null}
                  onChange={(value) => setValue(i, value)}
                  onEnter={onEnter}
                />
              ))}
            </SchemaForm>
          ) : null}
          {leftOut > 0 ? (
            <p className="m-0 text-caption text-secondary">
              {leftOut === 1 ? CHAT.mcp.leftOutOne : fill(CHAT.mcp.leftOut, { n: leftOut })}
            </p>
          ) : null}
        </div>
      ) : null}
    </InteractionCard>
  );
}

/**
 * An MCP server asks the user to open a page: its message, the whole address and its site, Decline and Open page.
 * Open page is a real link that the window's external-link policy opens in the browser; the click (after the arm
 * delay, once) also sends `accept`, which says only that the page was opened. Focus arrives on Decline. An address
 * that is not a safe web address offers Decline only.
 */
function McpLinkCard(props: McpCardProps): ReactElement {
  const { interaction } = props;
  const link = interaction.unusable === undefined ? (interaction.link ?? null) : null;
  const [pressed, setPressed] = useState<'open' | 'decline' | null>(null);
  const armed = useArmed();
  const declineButton = useRef<HTMLButtonElement | null>(null);
  const messageId = useId();
  const addressId = useId();
  const siteId = useId();

  useEffect(() => {
    const area = props.composerRef.current;
    if (area !== null && document.activeElement === area && area.value.trim() !== '') return;
    declineButton.current?.focus({ preventScroll: true });
  }, [interaction.id]);

  const open = (event: MouseEvent<HTMLAnchorElement>): void => {
    if (pressed !== null || !armed(event)) {
      event.preventDefault();
      return;
    }
    setPressed('open');
    props.onAnswer({ mcp: 'accept' }).catch(() => setPressed(null));
  };
  const decline = (): void => {
    if (pressed !== null) return;
    setPressed('decline');
    props.onAnswer({ mcp: 'decline' }).catch(() => setPressed(null));
  };

  const linkWarnings =
    link === null
      ? []
      : [...(link.secure ? [] : [CHAT.mcp.notSecure]), ...(link.lookalike === true ? [CHAT.mcp.lookalike] : [])];
  const warnings =
    interaction.unusable !== undefined
      ? [fill(CHAT.mcp.unusable, { why: CHAT.mcp.why[interaction.unusable] })]
      : linkWarnings.length > 0
        ? linkWarnings
        : undefined;
  return (
    <InteractionCard
      askId={interaction.id}
      icon={<Plug size={16} strokeWidth={1.75} />}
      title={fill(CHAT.mcp.linkTitle, { server: interaction.server })}
      subtitle={fill(CHAT.mcp.from, { server: interaction.server })}
      meta={<Meta folder={props.folder} position={props.position} />}
      metaText={metaText(props.folder)}
      describedBy={link === null ? [messageId] : [messageId, addressId, siteId]}
      busy={pressed !== null}
      warnings={warnings}
      warningIcon={<TriangleAlert size={12} strokeWidth={1.75} />}
      notes={link === null ? undefined : [CHAT.mcp.linkNote]}
      actions={
        <>
          <Button
            ref={declineButton}
            type="button"
            variant="secondary"
            size="sm"
            loading={pressed === 'decline'}
            disabled={pressed !== null && pressed !== 'decline'}
            data-mcp-decline=""
            onClick={decline}
          >
            {CHAT.mcp.decline}
          </Button>
          <span className="flex-1" aria-hidden="true" />
          {link !== null ? (
            <ButtonLink
              href={link.url}
              variant="primary"
              size="sm"
              label={fill(CHAT.mcp.openLabel, { host: link.host })}
              onClick={open}
              onAuxClick={(event) => event.preventDefault()}
            >
              {CHAT.mcp.open}
              <ExternalLink size={14} strokeWidth={1.75} aria-hidden="true" />
            </ButtonLink>
          ) : null}
        </>
      }
    >
      <Well label={CHAT.mcp.message}>
        <p id={messageId} className="m-0 whitespace-pre-wrap text-body [overflow-wrap:anywhere]">
          {interaction.message}
        </p>
      </Well>
      {link !== null ? (
        <>
          <Well label={CHAT.mcp.address}>
            <p id={addressId} className="m-0 font-mono text-body [overflow-wrap:anywhere]" data-mcp-address="">
              {link.url}
            </p>
          </Well>
          <p id={siteId} className="m-0 text-body font-medium">
            {fill(CHAT.mcp.site, { host: link.host })}
          </p>
        </>
      ) : null}
    </InteractionCard>
  );
}
