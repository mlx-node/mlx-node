import type { AssistantMessage, AssistantMessageEvent, ToolCall } from '@earendil-works/pi-ai';

/** Pi events hold a mutable cumulative message; sending it per token is quadratic. */
type CompactEvent<E> = E extends { partial: AssistantMessage }
  ? Omit<E, 'partial'> & { partial?: AssistantMessage; toolCall?: ToolCall }
  : E;
export type SharedEvent = CompactEvent<AssistantMessageEvent>;

export function encodeSharedEvent(event: AssistantMessageEvent): SharedEvent {
  if (!('partial' in event)) return event;
  const { partial, ...data } = event;
  if (event.type === 'start') return { ...data, partial: { ...partial, content: [] } };
  if (event.type === 'toolcall_start') {
    const block = partial.content[event.contentIndex];
    if (block?.type !== 'toolCall') throw new Error('Missing streamed tool call.');
    return { ...data, toolCall: block };
  }
  return data;
}

/** Rebuild Pi's shared partial-message object from linear-size wire updates. */
export class SharedEventDecoder {
  partial?: AssistantMessage;
  private readonly openBlocks = new Set<number>();

  decode(event: SharedEvent): AssistantMessageEvent {
    if (event.type === 'done' || event.type === 'error') return event;
    if (event.type === 'start') {
      if (!event.partial) throw new Error('Missing shared inference stream start.');
      this.partial = event.partial;
      return { ...event, partial: this.partial };
    }
    const partial = this.partial;
    if (!partial) throw new Error('Shared inference event arrived before stream start.');
    const index = event.contentIndex;
    switch (event.type) {
      case 'text_start':
        partial.content[index] = { type: 'text', text: '' };
        this.openBlocks.add(index);
        break;
      case 'thinking_start':
        partial.content[index] = { type: 'thinking', thinking: '' };
        this.openBlocks.add(index);
        break;
      case 'text_delta': {
        const block = partial.content[index];
        if (block?.type !== 'text') throw new Error('Missing streamed text block.');
        block.text += event.delta;
        break;
      }
      case 'thinking_delta': {
        const block = partial.content[index];
        if (block?.type !== 'thinking') throw new Error('Missing streamed thinking block.');
        block.thinking += event.delta;
        break;
      }
      case 'text_end':
        partial.content[index] = { type: 'text', text: event.content };
        this.openBlocks.delete(index);
        break;
      case 'thinking_end':
        partial.content[index] = { type: 'thinking', thinking: event.content };
        this.openBlocks.delete(index);
        break;
      case 'toolcall_start':
        if (!event.toolCall) throw new Error('Missing streamed tool call.');
        partial.content[index] = event.toolCall;
        break;
      case 'toolcall_delta':
        // Native tool calls arrive parsed in one piece; start already carries arguments.
        break;
      case 'toolcall_end':
        partial.content[index] = event.toolCall;
        break;
    }
    return { ...event, partial } as AssistantMessageEvent;
  }

  /** Balance partially emitted blocks before a transport failure/cancellation. */
  *finishOpenBlocks(): Generator<AssistantMessageEvent> {
    if (!this.partial) return;
    for (const index of this.openBlocks) {
      const block = this.partial.content[index];
      if (block?.type === 'text')
        yield { type: 'text_end', contentIndex: index, content: block.text, partial: this.partial };
      if (block?.type === 'thinking')
        yield { type: 'thinking_end', contentIndex: index, content: block.thinking, partial: this.partial };
    }
    this.openBlocks.clear();
  }
}
