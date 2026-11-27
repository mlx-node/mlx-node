/** A message to a running subagent: its size as the agent counts it, and how a send ended for the field. */

import { CHAT, fill } from './chat-strings';
import { SUBAGENT_MESSAGE_MAX_BYTES } from './chat-types';

/**
 * How a send ended for the field: `close` (sent, gone, unsure, or messages off; `clear` drops the text), or `stay`
 * with the reason in danger ink and the text kept.
 */
export type MessageEnd = { kind: 'close'; clear: boolean } | { kind: 'stay'; reason: string };

/** The agent a message goes to: its newest attempt's child session and its label. */
export interface MessageTarget {
  id: string;
  title: string;
}

/** The count under the field shows from this many bytes. */
export const COUNT_FROM_BYTES = 28 * 1024;

const ENCODER = new TextEncoder();

/** `text` in UTF-8 bytes, as the agent counts it. */
export function messageBytes(text: string): number {
  return ENCODER.encode(text).length;
}

/** Bytes in KB with one decimal: cut down under the limit, rounded up past it, so a text past it never reads "32.0". */
export function kbWords(bytes: number): string {
  const tenths = bytes > SUBAGENT_MESSAGE_MAX_BYTES ? Math.ceil((bytes * 10) / 1024) : Math.floor((bytes * 10) / 1024);
  return (tenths / 10).toFixed(1);
}

/** "Too long: 33.0 of 32 KB." */
export function tooLongWords(text: string): string {
  return fill(CHAT.activity.messageTooLong, { n: kbWords(messageBytes(text)) });
}

/** The line under the field: null under {@link COUNT_FROM_BYTES}, the count, or the too-long words past the limit. */
export function messageCount(text: string): { text: string; tooLong: boolean } | null {
  const bytes = messageBytes(text);
  if (bytes < COUNT_FROM_BYTES) return null;
  const tooLong = bytes > SUBAGENT_MESSAGE_MAX_BYTES;
  return { text: tooLong ? tooLongWords(text) : fill(CHAT.activity.messageCount, { n: kbWords(bytes) }), tooLong };
}

/** Send is on: some text that is not only spaces, within the limit. */
export function canSendMessage(text: string): boolean {
  return text.trim() !== '' && messageBytes(text) <= SUBAGENT_MESSAGE_MAX_BYTES;
}
