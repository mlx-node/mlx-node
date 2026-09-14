import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { localCompletion } from '../src/delegate.js';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});
const connection = { url: 'http://127.0.0.1:1234', token: 'private', model: 'chosen' };

describe('local delegation inference client', () => {
  it('allows a larger thinking budget while keeping a bounded request timeout', async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      'fetch',
      vi.fn(
        (_url, options) =>
          new Promise((_resolve, reject) => {
            options.signal.addEventListener('abort', () => reject(options.signal.reason), { once: true });
          }),
      ),
    );
    let completed = false;
    const completion = localCompletion(connection, 'Check', []);
    const result = expect(
      completion.finally(() => {
        completed = true;
      }),
    ).rejects.toThrow('timed out');
    await vi.advanceTimersByTimeAsync(120_000);
    expect(completed).toBe(false);
    await vi.advanceTimersByTimeAsync(480_000);
    await result;
  });

  it('verifies the exact default model and sends authenticated local inference', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ data: [{ id: 'chosen' }] }))
      .mockResolvedValueOnce(Response.json({ status: 'completed', output_text: '{"installed":false}' }));
    vi.stubGlobal('fetch', fetch);
    expect(await localCompletion(connection, 'Check', [{ role: 'user', content: 'Instructions' }])).toBe(
      '{"installed":false}',
    );
    const options = fetch.mock.calls[1][1];
    expect(options.headers['x-api-key']).toBe('private');
    expect(options.redirect).toBe('error');
    expect(fetch.mock.calls[1][0].pathname).toBe('/v1/responses');
    expect(JSON.parse(options.body)).toEqual({
      model: 'chosen',
      instructions: 'Check',
      input: [{ role: 'user', content: 'Instructions' }],
      max_output_tokens: 16384,
      temperature: 0,
      stream: false,
      reasoning: { effort: 'medium' },
      store: false,
    });
  });

  it('does not let an unknown model name fall back to a different loaded model', async () => {
    const fetch = vi.fn().mockResolvedValue(Response.json({ data: [{ id: 'different' }] }));
    vi.stubGlobal('fetch', fetch);
    await expect(localCompletion(connection, 'Check', [])).rejects.toThrow('default model is not available');
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('preserves an explicit caller output budget while enabling thinking', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(Response.json({ data: [{ id: 'chosen' }] }))
      .mockResolvedValueOnce(Response.json({ status: 'completed', output_text: '{}' }));
    vi.stubGlobal('fetch', fetch);
    await localCompletion(connection, 'Check', [], undefined, 2048);
    expect(JSON.parse(fetch.mock.calls[1][1].body)).toMatchObject({
      max_output_tokens: 2048,
      reasoning: { effort: 'medium' },
    });
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
        .mockResolvedValueOnce(
          Response.json({
            status: 'incomplete',
            output_text: '{}',
            incomplete_details: { reason: 'max_output_tokens' },
          }),
        ),
    );
    await expect(localCompletion(connection, 'Check', [])).rejects.toThrow('output space');
  });

  it.each(['incomplete', 'failed', undefined])(
    'rejects a non-completed response even with usable-looking output: %s',
    async (status) => {
      vi.stubGlobal(
        'fetch',
        vi
          .fn()
          .mockResolvedValueOnce(Response.json({ data: [{ id: 'chosen' }] }))
          .mockResolvedValueOnce(Response.json({ status, output_text: '{}' })),
      );
      await expect(localCompletion(connection, 'Check', [])).rejects.toThrow('did not complete');
    },
  );

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
