import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';

import { describe, expect, it } from 'vite-plus/test';

import {
  bracketHost,
  hostUrl,
  isLoopbackBindHost,
  normalizeLoopbackBindHost,
  pickFreePort,
} from '../../../packages/server/src/host/net.js';

describe('bracketHost', () => {
  it('leaves IPv4 literals and hostnames untouched', () => {
    expect(bracketHost('127.0.0.1')).toBe('127.0.0.1');
    expect(bracketHost('localhost')).toBe('localhost');
    expect(bracketHost('my-mac.local')).toBe('my-mac.local');
  });

  it('brackets bare IPv6 literals', () => {
    expect(bracketHost('::1')).toBe('[::1]');
    expect(bracketHost('2001:db8::1')).toBe('[2001:db8::1]');
    expect(bracketHost('fe80::1%en0')).toBe('[fe80::1%en0]');
  });

  it('does not double-bracket an already-bracketed literal', () => {
    expect(bracketHost('[::1]')).toBe('[::1]');
  });
});

describe('hostUrl', () => {
  it('builds a parseable URL for a loopback bind', () => {
    expect(hostUrl('127.0.0.1', 8080)).toBe('http://127.0.0.1:8080');
    expect(new URL(hostUrl('127.0.0.1', 8080)).port).toBe('8080');
  });

  it('builds a parseable URL for an IPv6 bind', () => {
    // The whole reason `bracketHost` exists: `http://::1:8080` is ambiguous
    // and `new URL` rejects it, which would crash whoever printed it.
    const url = hostUrl('::1', 8080);
    expect(url).toBe('http://[::1]:8080');
    expect(new URL(url).port).toBe('8080');
  });

  it('advertises a connectable loopback for a wildcard bind', () => {
    // `http://0.0.0.0:port` is not a connectable address for a client and
    // `http://:8080` is malformed; each wildcard maps to the loopback of its
    // own address family.
    expect(hostUrl('0.0.0.0', 1234)).toBe('http://127.0.0.1:1234');
    expect(hostUrl('', 1234)).toBe('http://127.0.0.1:1234');
    expect(hostUrl('::', 1234)).toBe('http://[::1]:1234');
  });
});

describe('normalizeLoopbackBindHost', () => {
  it('removes URL brackets from the exact IPv6 loopback bind', () => {
    expect(normalizeLoopbackBindHost('[::1]')).toBe('::1');
    expect(normalizeLoopbackBindHost('::1')).toBe('::1');
  });

  it('does not normalize unsafe or non-loopback inputs', () => {
    for (const host of [
      '[::]',
      '[2001:db8::1]',
      '[fe80::1%en0]',
      '[::ffff:127.0.0.1]',
      '::',
      '2001:db8::1',
      '0.0.0.0',
      'my-mac.local',
    ]) {
      expect(normalizeLoopbackBindHost(host), host).toBe(host);
    }
  });
});

describe('isLoopbackBindHost', () => {
  it('accepts every loopback form a host can be asked to bind', () => {
    for (const host of ['127.0.0.1', '127.0.0.2', '127.255.255.254', 'localhost', '::1', '[::1]', '::ffff:127.0.0.1']) {
      expect(isLoopbackBindHost(host), host).toBe(true);
    }
  });

  it('rejects the wildcards, which `hostUrl` ADVERTISES as loopback', () => {
    // The trap this predicate exists for. `hostUrl('0.0.0.0', p)` returns
    // `http://127.0.0.1:p` because that is what a client should connect to —
    // so a reachability check written against the advertised URL would call a
    // bind on every interface safe. These bind every interface.
    for (const host of ['0.0.0.0', '::', '']) {
      expect(isLoopbackBindHost(host), host).toBe(false);
    }
  });

  it('rejects routable addresses and names that merely look loopback-ish', () => {
    for (const host of [
      '192.168.1.10',
      '10.0.0.1',
      '0.0.0.1',
      // Off by one octet from the loopback block, and a real routable address.
      '128.0.0.1',
      '2001:db8::1',
      'my-mac.local',
      // Resolves wherever DNS says; the name is not the loopback name.
      'localhost.evil.example',
      // A prefix match on '127.' would accept this.
      '127.0.0.1.evil.example',
    ]) {
      expect(isLoopbackBindHost(host), host).toBe(false);
    }
  });
});

describe('pickFreePort', () => {
  it('returns a port that is actually bindable', async () => {
    const port = await pickFreePort();
    expect(Number.isInteger(port)).toBe(true);
    expect(port).toBeGreaterThan(0);

    // The contract is "you can listen here next", so prove it rather than
    // just range-checking the number.
    const server = createServer();
    try {
      await new Promise<void>((resolve, reject) => {
        server.once('error', reject);
        server.listen(port, '127.0.0.1', resolve);
      });
      expect((server.address() as AddressInfo).port).toBe(port);
    } finally {
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });

  it('releases the probe socket, so the same port can be taken twice', async () => {
    // A leaked probe listener would leave the port occupied and make the very
    // next `listen()` fail with EADDRINUSE. Deliberately NOT asserting that
    // two calls return DIFFERENT ports — that is kernel ephemeral-port policy,
    // not our contract.
    const port = await pickFreePort();
    for (let attempt = 0; attempt < 2; attempt++) {
      const server = createServer();
      await new Promise<void>((resolve, reject) => {
        server.once('error', reject);
        server.listen(port, '127.0.0.1', resolve);
      });
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
});
