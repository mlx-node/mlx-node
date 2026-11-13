/**
 * @vitest-environment happy-dom
 */

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { FOLDER_PICKED_MESSAGE, PICK_FOLDER_MESSAGE, pickFolder } from '../ui/src/lib/pick-folder.js';

let off: (() => void) | null = null;

afterEach(() => {
  off?.();
  off = null;
});

/** Plays the preload: answers each folder request with `answer(requestId)`, and records every request. */
function desktop(answer: (requestId: string) => unknown[]): unknown[] {
  const requests: unknown[] = [];
  const onMessage = (event: MessageEvent): void => {
    const data = event.data as { type?: unknown; requestId?: unknown };
    if (data?.type !== PICK_FOLDER_MESSAGE) return;
    requests.push(data);
    for (const reply of answer(String(data.requestId))) {
      window.dispatchEvent(new MessageEvent('message', { data: reply, source: window }));
    }
  };
  window.addEventListener('message', onMessage);
  off = () => window.removeEventListener('message', onMessage);
  return requests;
}

describe('pickFolder', () => {
  it('sends only a request id and resolves with the folder answered for that id', async () => {
    const requests = desktop((requestId) => [
      { type: FOLDER_PICKED_MESSAGE, requestId: 'someone-else', path: '/wrong' },
      { type: FOLDER_PICKED_MESSAGE, requestId, path: '/Users/me/code/app' },
    ]);
    expect(await pickFolder()).toBe('/Users/me/code/app');
    expect(requests).toHaveLength(1);
    const request = requests[0] as Record<string, unknown>;
    expect(Object.keys(request).sort()).toEqual(['requestId', 'type']);
    expect(typeof request.requestId).toBe('string');
  });

  it('resolves null for a cancelled dialog and an empty path', async () => {
    desktop((requestId) => [{ type: FOLDER_PICKED_MESSAGE, requestId, path: null }]);
    expect(await pickFolder()).toBeNull();
    off?.();
    desktop((requestId) => [{ type: FOLDER_PICKED_MESSAGE, requestId, path: '' }]);
    expect(await pickFolder()).toBeNull();
  });

  it('ignores an answer from another window', async () => {
    const onMessage = (event: MessageEvent): void => {
      const data = event.data as { type?: unknown; requestId?: unknown };
      if (data?.type !== PICK_FOLDER_MESSAGE) return;
      window.dispatchEvent(
        new MessageEvent('message', {
          data: { type: FOLDER_PICKED_MESSAGE, requestId: data.requestId, path: '/frame' },
        }),
      );
    };
    window.addEventListener('message', onMessage);
    off = () => window.removeEventListener('message', onMessage);
    expect(await pickFolder(20)).toBeNull();
  });

  it('resolves null when nothing answers', async () => {
    expect(await pickFolder(20)).toBeNull();
  });
});
