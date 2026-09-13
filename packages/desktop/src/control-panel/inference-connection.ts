/** Private MAIN ↔ CONTROL PANEL exchange. Credentials never enter renderer RPC. */
export interface InferenceConnection {
  url: string;
  token: string;
  model: string;
}
export const CONNECTION_REQUEST = 'mlx:inference-connection';
export const CONNECTION_REPLY = 'mlx:inference-connection-result';

export function createInferenceConnector(parent: {
  postMessage(message: unknown): void;
  on(event: 'message', listener: (event: { data: unknown }) => void): void;
}): { connect(): Promise<InferenceConnection>; close(): void } {
  let nextId = 0;
  const pending = new Map<
    number,
    { resolve(value: InferenceConnection): void; reject(error: Error): void; timer: ReturnType<typeof setTimeout> }
  >();
  parent.on('message', ({ data }) => {
    if (typeof data !== 'object' || data === null) return;
    const reply = data as { type?: unknown; id?: unknown; connection?: InferenceConnection; error?: string };
    if (reply.type !== CONNECTION_REPLY || typeof reply.id !== 'number') return;
    const request = pending.get(reply.id);
    if (!request) return;
    pending.delete(reply.id);
    clearTimeout(request.timer);
    if (reply.connection) request.resolve(reply.connection);
    else request.reject(new Error(reply.error || 'The local model service is unavailable.'));
  });
  return {
    connect: () =>
      new Promise((resolve, reject) => {
        const id = ++nextId;
        const timer = setTimeout(() => {
          pending.delete(id);
          reject(new Error('The local model service did not start. Try again.'));
        }, 60_000);
        pending.set(id, { resolve, reject, timer });
        try {
          parent.postMessage({ type: CONNECTION_REQUEST, id });
        } catch (error) {
          clearTimeout(timer);
          pending.delete(id);
          reject(error);
        }
      }),
    close: () => {
      for (const request of pending.values()) {
        clearTimeout(request.timer);
        request.reject(new Error('The control panel is closing.'));
      }
      pending.clear();
    },
  };
}
