import { describe, expect, it, vi } from 'vite-plus/test';

import {
  createInferenceConnector,
  CONNECTION_REPLY,
  CONNECTION_REQUEST,
} from '../src/control-panel/inference-connection.js';

describe('private inference connection exchange', () => {
  it('matches replies and rejects pending requests on close', async () => {
    let receive: (event: { data: unknown }) => void = () => {};
    const postMessage = vi.fn();
    const connector = createInferenceConnector({
      postMessage,
      on: (_, listener) => {
        receive = listener;
      },
    });
    const first = connector.connect();
    expect(postMessage).toHaveBeenCalledWith({ type: CONNECTION_REQUEST, id: 1 });
    receive({
      data: {
        type: CONNECTION_REPLY,
        id: 1,
        connection: { url: 'http://127.0.0.1:1', token: 'private', model: 'default' },
      },
    });
    expect((await first).model).toBe('default');
    const pending = connector.connect();
    connector.close();
    await expect(pending).rejects.toThrow('closing');
  });
});
