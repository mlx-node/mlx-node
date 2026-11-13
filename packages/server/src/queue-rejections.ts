/** A queue whose full state the endpoints answer with HTTP 429. */
export type QueueRejectionKind = 'chat' | 'load' | 'audio' | 'decision';

let total = 0;

export function noteQueueRejection(_kind: QueueRejectionKind): void {
  total += 1;
}

/** Process-wide total of queue-full rejections. */
export function queueRejections(): number {
  return total;
}
