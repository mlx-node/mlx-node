/**
 * The port shape the RPC layer binds to, plus binders for the two real ports it
 * has to drive.
 *
 * Electron's `MessagePortMain` (main process) and the web / `node:worker_threads`
 * `MessagePort` (renderer, and every test in this package) do not share an API:
 * one is an EventEmitter (`.on('message', …)`), the other an EventTarget
 * (`addEventListener`), they disagree on whether a `message` listener receives the
 * raw value or a `{ data }` wrapper, and both need an explicit `start()` before a
 * listener registered either of those ways receives anything at all.
 *
 * Neither concrete type is importable here — `electron` cannot resolve outside an
 * Electron process, and the DOM `MessagePort` lib is not in this package's `lib` —
 * so this interface is the entire coupling and both bind to it structurally.
 */

/** What the RPC layer needs told about a port. */
export interface RpcPortEvents {
  /** One inbound payload, already unwrapped from whatever event object carried it. */
  onMessage(data: unknown): void;
  /**
   * The channel is gone: the peer closed its end, or the process holding it
   * died. Delivered at most once.
   */
  onClose(): void;
}

/**
 * A bidirectional message channel. Deliberately three members:
 *
 * - `postMessage` — send. Note what it does NOT do: posting to a port whose peer
 *   has already closed does not throw (verified on `node:worker_threads`), so a
 *   send can never be used to detect a dead peer. That is why {@link RpcPortEvents.onClose}
 *   exists, and why the client keeps a per-request deadline on top of it.
 * - `listen` — register both events AND start delivery in one call, returning one
 *   detach. Fused because starting is a property of the underlying port rather
 *   than of the caller, and because a half-detached port (messages off, close
 *   still armed) is a leak shape worth making unrepresentable.
 * - `close` — release the port. Must tolerate being called on a port the peer
 *   already closed, since that is the normal shutdown race.
 */
export interface RpcPort {
  postMessage(message: unknown): void;
  listen(events: RpcPortEvents): () => void;
  close(): void;
}

/** Read the payload out of a `MessageEvent`-shaped object. */
function messageData(event: unknown): unknown {
  if (typeof event === 'object' && event !== null && 'data' in event) {
    return (event as { data: unknown }).data;
  }
  return event;
}

/** The slice of a web / `node:worker_threads` `MessagePort` the binder uses. */
export interface EventTargetPort {
  postMessage(message: unknown): void;
  addEventListener(type: string, listener: (event: unknown) => void): void;
  removeEventListener(type: string, listener: (event: unknown) => void): void;
  start(): void;
  close(): void;
}

/** The slice of an Electron `MessagePortMain` the binder uses. */
export interface EventEmitterPort {
  postMessage(message: unknown): void;
  on(event: string, listener: (payload: unknown) => void): void;
  off(event: string, listener: (payload: unknown) => void): void;
  start(): void;
  close(): void;
}

/**
 * Bind a web or `node:worker_threads` `MessagePort`.
 *
 * `addEventListener` rather than `onmessage` so binding twice (or alongside
 * unrelated code) cannot silently clobber a listener, which is also why `start()`
 * has to be explicit here — assigning `onmessage` would have started delivery as
 * a side effect.
 */
export function bindEventTargetPort(port: EventTargetPort): RpcPort {
  return {
    postMessage(message: unknown): void {
      port.postMessage(message);
    },
    listen(events: RpcPortEvents): () => void {
      const onMessage = (event: unknown): void => events.onMessage(messageData(event));
      const onClose = (): void => events.onClose();
      port.addEventListener('message', onMessage);
      port.addEventListener('close', onClose);
      port.start();
      return () => {
        port.removeEventListener('message', onMessage);
        port.removeEventListener('close', onClose);
      };
    },
    close(): void {
      port.close();
    },
  };
}

/**
 * Bind an Electron `MessagePortMain`.
 *
 * Its `'message'` listener receives a `MessageEvent`-shaped `{ data }` (unlike
 * `node:worker_threads`' `.on('message')`, which hands over the raw value), so the
 * unwrap is the same one the EventTarget binder needs.
 */
export function bindEventEmitterPort(port: EventEmitterPort): RpcPort {
  return {
    postMessage(message: unknown): void {
      port.postMessage(message);
    },
    listen(events: RpcPortEvents): () => void {
      const onMessage = (payload: unknown): void => events.onMessage(messageData(payload));
      const onClose = (): void => events.onClose();
      port.on('message', onMessage);
      port.on('close', onClose);
      port.start();
      return () => {
        port.off('message', onMessage);
        port.off('close', onClose);
      };
    },
    close(): void {
      port.close();
    },
  };
}
