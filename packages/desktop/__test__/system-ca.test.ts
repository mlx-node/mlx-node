import { X509Certificate } from 'node:crypto';
import { mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { keychainCaRootsPem, parseTrustSettingsDump, prepareExtraCaBundle, type ExecText } from '../src/main/system-ca.js';

const FIXTURES = join(__dirname, 'fixtures');
const ROOT_CA = readFileSync(join(FIXTURES, 'test-root-ca.pem'), 'utf8').trim();
const LEAF = readFileSync(join(FIXTURES, 'test-leaf.pem'), 'utf8').trim();
const ROOT_SHA1 = new X509Certificate(ROOT_CA).fingerprint.replaceAll(':', '');

/** Must match the constant in system-ca.ts: the keychain whose roots are implicitly trusted. */
const SYSTEM_ROOTS = '/System/Library/Keychains/SystemRootCertificates.keychain';

const EMPTY_DUMP = `{
  "trustList" => {
  }
  "trustVersion" => 1
}
`;

/** plutil -p shaped trust dump, built with real cert SHA-1s. A missing `result` omits the key entirely. */
function trustDump(
  entries: Array<{ sha1: string; settings: Array<{ policy?: string; policyString?: string; result?: number }> }>,
): string {
  const body = entries
    .map(({ sha1, settings }) => {
      const items = settings
        .map((s, i) => {
          const policyLine = s.policy === undefined ? '' : `          "kSecTrustSettingsPolicyName" => "${s.policy}"\n`;
          const stringLine =
            s.policyString === undefined ? '' : `          "kSecTrustSettingsPolicyString" => "${s.policyString}"\n`;
          const resultLine = s.result === undefined ? '' : `          "kSecTrustSettingsResult" => ${s.result}\n`;
          return `        ${i} => {\n${policyLine}${stringLine}${resultLine}        }`;
        })
        .join(',\n');
      return `    "${sha1}" => {\n      "trustSettings" => [\n${items}\n      ]\n    }`;
    })
    .join(',\n');
  return `{\n  "trustList" => {\n${body}\n  }\n  "trustVersion" => 1\n}\n`;
}

/** A `security`/`plutil` stand-in: canned per-keychain certs and canned trust dumps. */
function fakeExec(opts: { certs: Readonly<Record<string, string>>; trust?: string }): ExecText {
  return async (cmd, args) => {
    if (cmd === 'security' && args[0] === 'find-certificate') {
      const output = opts.certs[args[args.length - 1]];
      if (output === undefined) throw new Error('keychain unreadable');
      return output;
    }
    if (cmd === 'security' && args[0] === 'trust-settings-export') return '';
    if (cmd === 'plutil') return opts.trust ?? EMPTY_DUMP;
    throw new Error(`unexpected call: ${cmd} ${args.join(' ')}`);
  };
}

function tmpDir(): string {
  return mkdtempSync(join(tmpdir(), 'mlx-system-ca-'));
}

describe('parseTrustSettingsDump', () => {
  it('reads a real plutil -p export: mkcert trusted, an entry with zero settings not', () => {
    // Captured verbatim from `plutil -p` of `security trust-settings-export -d`.
    const real = `{
  "trustList" => {
    "9C29EB274CB463788ACFC21705615CBE81EDB0AA" => {
      "issuerName" => {length = 148, bytes = 0x308191311e301c060355040a13156d6b ... 6f6f6f6b6c796e29}
      "modDate" => 2026-05-28 09:35:58 +0000
      "serialNumber" => {length = 16, bytes = 0x7bb75db9cea8232c4b35626c0240c670}
      "trustSettings" => [
        0 => {
          "kSecTrustSettingsPolicy" => {length = 9, bytes = 0x2a864886f763640103}
          "kSecTrustSettingsPolicyName" => "sslServer"
          "kSecTrustSettingsResult" => 1
        }
        1 => {
          "kSecTrustSettingsPolicy" => {length = 9, bytes = 0x2a864886f763640102}
          "kSecTrustSettingsPolicyName" => "basicX509"
          "kSecTrustSettingsResult" => 1
        }
      ]
    }
    "21261EB01273C866A943ACBD51E04D3CBFDC395B" => {
      "issuerName" => {length = 149, bytes = 0x308192310b3009060355040613025553 ... 63616c2043657274}
      "modDate" => 2026-07-17 03:41:09 +0000
      "serialNumber" => {length = 3, bytes = 0x04a8a6}
    }
  }
  "trustVersion" => 1
}
`;
    const decisions = parseTrustSettingsDump(real);
    expect(decisions.get('9C29EB274CB463788ACFC21705615CBE81EDB0AA')).toEqual({ allowForSsl: true, denyForSsl: false });
    // Present in the keychain with NO trust records: installed, but not trusted.
    expect(decisions.get('21261EB01273C866A943ACBD51E04D3CBFDC395B')).toEqual({ allowForSsl: false, denyForSsl: false });
  });

  it('records a deny, and ignores allows for non-SSL policies', () => {
    const dump = trustDump([
      { sha1: 'A'.repeat(40), settings: [{ policy: 'sslServer', result: 3 }] },
      { sha1: 'B'.repeat(40), settings: [{ policy: 'smime', result: 1 }] },
      { sha1: 'C'.repeat(40), settings: [{ result: 1 }] }, // no policy: applies to all
    ]);
    const decisions = parseTrustSettingsDump(dump);
    expect(decisions.get('A'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: true });
    expect(decisions.get('B'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false });
    expect(decisions.get('C'.repeat(40))).toEqual({ allowForSsl: true, denyForSsl: false });
  });

  it('treats a missing result key as TrustRoot (the add-trusted-cert export shape)', () => {
    // `security add-trusted-cert -r trustRoot -p ssl` exports an item with a
    // policy and NO kSecTrustSettingsResult; verify-cert confirms such a cert
    // is trusted, so the schema default is allow.
    const dump = trustDump([{ sha1: 'D'.repeat(40), settings: [{ policy: 'sslServer' }] }]);
    expect(parseTrustSettingsDump(dump).get('D'.repeat(40))).toEqual({ allowForSsl: true, denyForSsl: false });
  });

  it('reads an explicitly empty trustSettings array as always-trust', () => {
    // SecTrustSettings.h: an empty array means "always trust this cert" with
    // result TrustRoot — and is "definitely not the same as *no* Trust
    // Settings". Corporate roots added without a policy export this shape.
    const dump = trustDump([{ sha1: 'E'.repeat(40), settings: [] }]);
    expect(parseTrustSettingsDump(dump).get('E'.repeat(40))).toEqual({ allowForSsl: true, denyForSsl: false });
  });

  it('ignores hostname-constrained records in both directions', () => {
    // kSecTrustSettingsPolicyString scopes the record (sslServer trust valid
    // for one hostname). NODE_EXTRA_CA_CERTS is process-wide and cannot carry
    // the constraint: exporting the allow would broaden a narrow trust into
    // an any-host anchor, and exporting the deny would strip a good root.
    const dump = trustDump([
      { sha1: 'F'.repeat(40), settings: [{ policy: 'sslServer', policyString: 'internal.example.com', result: 1 }] },
      { sha1: '0'.repeat(40), settings: [{ policy: 'sslServer', policyString: 'evil.example.com', result: 3 }] },
    ]);
    const decisions = parseTrustSettingsDump(dump);
    expect(decisions.get('F'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false });
    expect(decisions.get('0'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false });
  });
});

describe('keychainCaRootsPem', () => {
  it('includes Apple system roots with no trust records (implicitly trusted)', async () => {
    const pem = await keychainCaRootsPem(fakeExec({ certs: { [SYSTEM_ROOTS]: ROOT_CA } }), [SYSTEM_ROOTS]);
    expect(pem).toContain(ROOT_CA);
  });

  it('excludes a system root the user or admin explicitly denied', async () => {
    const pem = await keychainCaRootsPem(
      fakeExec({
        certs: { [SYSTEM_ROOTS]: ROOT_CA },
        trust: trustDump([{ sha1: ROOT_SHA1, settings: [{ policy: 'sslServer', result: 3 }] }]),
      }),
      [SYSTEM_ROOTS],
    );
    expect(pem).toBe('');
  });

  it('includes a non-system CA only with an explicit SSL-trust record', async () => {
    const trusted = await keychainCaRootsPem(
      fakeExec({
        certs: { '/k/System': ROOT_CA },
        trust: trustDump([{ sha1: ROOT_SHA1, settings: [{ policy: 'sslServer', result: 1 }] }]),
      }),
      ['/k/System'],
    );
    expect(trusted).toContain(ROOT_CA);

    // The regression the review caught: installed in a keychain, zero trust
    // records — must NOT become an anchor.
    const installedOnly = await keychainCaRootsPem(fakeExec({ certs: { '/k/System': ROOT_CA } }), ['/k/System']);
    expect(installedOnly).toBe('');
  });

  it('lets a deny beat an allow on another SSL policy', async () => {
    const pem = await keychainCaRootsPem(
      fakeExec({
        certs: { '/k/System': ROOT_CA },
        trust: trustDump([
          { sha1: ROOT_SHA1, settings: [{ policy: 'basicX509', result: 1 }, { policy: 'sslServer', result: 3 }] },
        ]),
      }),
      ['/k/System'],
    );
    expect(pem).toBe('');
  });

  it('still drops non-CA certificates even when explicitly trusted', async () => {
    const leafSha1 = new X509Certificate(LEAF).fingerprint.replaceAll(':', '');
    const pem = await keychainCaRootsPem(
      fakeExec({
        certs: { '/k/login': `${ROOT_CA}\n${LEAF}` },
        trust: trustDump([
          { sha1: ROOT_SHA1, settings: [{ policy: 'sslServer', result: 1 }] },
          { sha1: leafSha1, settings: [{ policy: 'sslServer', result: 1 }] },
        ]),
      }),
      ['/k/login'],
    );
    expect(pem).toContain(ROOT_CA);
    expect(pem).not.toContain(LEAF);
  });

  it('dedupes a root present in several keychains and skips unreadable ones', async () => {
    const pem = await keychainCaRootsPem(fakeExec({ certs: { [SYSTEM_ROOTS]: ROOT_CA, '/k/a': `${ROOT_CA}\n` } }), [
      SYSTEM_ROOTS,
      '/k/missing',
      '/k/a',
    ]);
    expect(pem.match(/BEGIN CERTIFICATE/g)).toHaveLength(1);
  });

  it('skips unparseable blocks', async () => {
    const pem = await keychainCaRootsPem(
      fakeExec({ certs: { [SYSTEM_ROOTS]: '-----BEGIN CERTIFICATE-----\nnot-a-cert\n-----END CERTIFICATE-----\n' + ROOT_CA } }),
      [SYSTEM_ROOTS],
    );
    expect(pem.match(/BEGIN CERTIFICATE/g)).toHaveLength(1);
  });
});

describe('prepareExtraCaBundle', () => {
  it('is a no-op off macOS', async () => {
    const result = await prepareExtraCaBundle({
      platform: 'linux',
      dir: tmpDir(),
      exec: fakeExec({ certs: { [SYSTEM_ROOTS]: ROOT_CA } }),
      keychains: [SYSTEM_ROOTS],
    });
    expect(result).toBeNull();
  });

  it('writes a 0600 bundle of trusted roots and returns its path', async () => {
    const dir = tmpDir();
    const result = await prepareExtraCaBundle({
      platform: 'darwin',
      dir,
      exec: fakeExec({ certs: { [SYSTEM_ROOTS]: `${ROOT_CA}\n${LEAF}` } }),
      keychains: [SYSTEM_ROOTS],
    });
    expect(result).toBe(join(dir, 'system-ca-roots.pem'));
    const written = readFileSync(result!, 'utf8');
    expect(written).toContain(ROOT_CA);
    expect(written).not.toContain(LEAF);
    expect(statSync(result!).mode & 0o777).toBe(0o600);
  });

  it('merges an inherited NODE_EXTRA_CA_CERTS bundle instead of dropping it', async () => {
    const dir = tmpDir();
    const inherited = join(dir, 'inherited.pem');
    writeFileSync(inherited, `${LEAF}\n`);
    const result = await prepareExtraCaBundle({
      platform: 'darwin',
      dir,
      exec: fakeExec({ certs: { [SYSTEM_ROOTS]: ROOT_CA } }),
      inheritedPath: inherited,
      keychains: [SYSTEM_ROOTS],
    });
    const written = readFileSync(result!, 'utf8');
    expect(written).toContain(ROOT_CA);
    expect(written).toContain(LEAF); // the developer's own bundle survives
  });

  it('returns null without writing when there is nothing to add', async () => {
    const dir = tmpDir();
    const result = await prepareExtraCaBundle({
      platform: 'darwin',
      dir,
      exec: fakeExec({ certs: { '/k/System': ROOT_CA } }), // installed, but trusted by no one
      inheritedPath: join(dir, 'does-not-exist.pem'),
      keychains: ['/k/System'],
    });
    expect(result).toBeNull();
  });
});
