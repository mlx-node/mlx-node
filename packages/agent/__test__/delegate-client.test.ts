import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { localCompletion } from '../src/delegate.js';

afterEach(() => {
  vi.unstubAllGlobals();
});
const connection = { url: 'http://127.0.0.1:1234', token: 'private', model: 'chosen' };

describe('local delegation inference client', () => {
  it('verifies the exact default model and sends authenticated local inference', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ data: [{ id: 'chosen' }] }))
      .mockResolvedValueOnce(Response.json({ content: [{ type: 'text', text: '{"installed":false}' }] }));
    vi.stubGlobal('fetch', fetch);
    expect(await localCompletion(connection, 'Check', [{ role: 'user', content: 'Instructions' }])).toBe(
      '{"installed":false}',
    );
    const options = fetch.mock.calls[1][1];
    expect(options.headers['x-api-key']).toBe('private');
    expect(options.redirect).toBe('error');
    expect(JSON.parse(options.body)).toMatchObject({ model: 'chosen', temperature: 0, stream: false });
  });

  it('does not let an unknown model name fall back to a different loaded model', async () => {
    const fetch = vi.fn().mockResolvedValue(Response.json({ data: [{ id: 'different' }] }));
    vi.stubGlobal('fetch', fetch);
    await expect(localCompletion(connection, 'Check', [])).rejects.toThrow('default model is not available');
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('rejects cloud endpoints before sending instructions or credentials', async () => {
    const fetch = vi.fn();
    vi.stubGlobal('fetch', fetch);
    await expect(localCompletion({ ...connection, url: 'https://example.com' }, 'Check', [])).rejects.toThrow(
      'local inference',
    );
    expect(fetch).not.toHaveBeenCalled();
  });

  it('never treats truncated output as a valid detection', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValueOnce(Response.json({ data: [{ id: 'chosen' }] }))
        .mockResolvedValueOnce(Response.json({ content: [{ type: 'text', text: '{}' }], stop_reason: 'max_tokens' })),
    );
    await expect(localCompletion(connection, 'Check', [])).rejects.toThrow('output space');
  });

  it('propagates cancellation while waiting for the model service', async () => {
    const controller = new AbortController();
    const fetch = vi.fn(
      (_url, options) =>
        new Promise((_resolve, reject) => {
          options.signal.addEventListener('abort', () => reject(options.signal.reason), { once: true });
        }),
    );
    vi.stubGlobal('fetch', fetch);
    const completion = localCompletion(connection, 'Check', [], controller.signal);
    controller.abort(new Error('closed'));
    await expect(completion).rejects.toThrow('closed');
  });
});
