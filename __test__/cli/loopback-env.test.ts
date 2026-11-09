import { describe, expect, it } from 'vite-plus/test';

import { withLoopbackNoProxy } from '../../packages/cli/src/commands/loopback-env.js';

const LOOPBACK = '127.0.0.1,localhost,::1';
const PROXY = 'http://proxy.example:3128';

// Splash `test_every_client_reaches_the_server_without_a_proxy` case table.
// Mutation: skip it (return env unchanged) → red on every proxy row.
describe('withLoopbackNoProxy', () => {
  it.each([
    [{ HTTPS_PROXY: PROXY }, {}, [LOOPBACK, LOOPBACK]],
    [{ http_proxy: PROXY }, { NO_PROXY: 'corp.example' }, [`corp.example,${LOOPBACK}`, `corp.example,${LOOPBACK}`]],
    [
      { all_proxy: PROXY },
      { NO_PROXY: '', no_proxy: 'corp.example, localhost' },
      ['corp.example,localhost,127.0.0.1,::1', 'corp.example,localhost,127.0.0.1,::1'],
    ],
    [{ ALL_PROXY: PROXY }, { no_proxy: `${LOOPBACK},corp.example` }, [undefined, `${LOOPBACK},corp.example`]],
    [{ HTTP_PROXY: PROXY }, { NO_PROXY: '*' }, ['*', undefined]],
    [{ https_proxy: PROXY }, { NO_PROXY: '*', no_proxy: '*' }, ['*', '*']],
    [{}, {}, [undefined, undefined]],
    [{ http_proxy: '' }, { NO_PROXY: 'corp.example' }, ['corp.example', undefined]],
  ] as const)('proxies %j with %j', (proxies, noProxy, expected) => {
    const env = withLoopbackNoProxy({ PATH: '/usr/bin', ...proxies, ...noProxy });
    expect([env.NO_PROXY, env.no_proxy]).toEqual(expected);
    for (const [name, value] of Object.entries(proxies)) expect(env[name]).toBe(value);
    expect(env.PATH).toBe('/usr/bin');
  });

  it('does not change the environment it was given', () => {
    const parent = { HTTP_PROXY: PROXY };
    withLoopbackNoProxy(parent);
    expect(parent).toEqual({ HTTP_PROXY: PROXY });
  });
});
