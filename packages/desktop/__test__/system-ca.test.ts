import { X509Certificate } from 'node:crypto';
import { mkdtempSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import {
  keychainCaRootsPem,
  macosKeychains,
  parseTrustSettingsDump,
  prepareExtraCaBundle,
  type ExecText,
} from '../src/main/system-ca.js';

const FIXTURES = join(__dirname, 'fixtures');
const ROOT_CA = readFileSync(join(FIXTURES, 'test-root-ca.pem'), 'utf8').trim();
const LEAF = readFileSync(join(FIXTURES, 'test-leaf.pem'), 'utf8').trim();
const ROOT_SHA1 = new X509Certificate(ROOT_CA).fingerprint.replaceAll(':', '');

/** An arbitrary keychain path stand-in; candidates from ANY keychain now require explicit trust. */
const SYSTEM_ROOTS = '/k/some-keychain';

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
/** How one trust-settings domain's export behaves: ok (dump follows), empty (errSecNoTrustSettings exit-1 shape), or a read failure. */
type DomainBehavior = 'ok' | 'empty' | 'fail';

function fakeExec(opts: {
  certs: Readonly<Record<string, string>>;
  /** Dump for both domains, unless overridden per domain. */
  trust?: string;
  trustUser?: string;
  trustAdmin?: string;
  userExport?: DomainBehavior;
  adminExport?: DomainBehavior;
}): ExecText {
  let domain: 'user' | 'admin' = 'user';
  return async (cmd, args) => {
    if (cmd === 'security' && args[0] === 'find-certificate') {
      const output = opts.certs[args[args.length - 1]];
      if (output === undefined) throw new Error('keychain unreadable');
      return output;
    }
    if (cmd === 'security' && args[0] === 'trust-settings-export') {
      domain = args.includes('-d') ? 'admin' : 'user';
      const behavior = (domain === 'admin' ? opts.adminExport : opts.userExport) ?? 'ok';
      if (behavior === 'fail') throw new Error('export timed out');
      if (behavior === 'empty') {
        throw new Error(
          'Command failed: security trust-settings-export\nSecTrustSettingsCreateExternalRepresentation: No Trust Settings were found.',
        );
      }
      return '';
    }
    if (cmd === 'plutil') return (domain === 'admin' ? opts.trustAdmin : opts.trustUser) ?? opts.trust ?? EMPTY_DUMP;
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
    expect(decisions.get('9C29EB274CB463788ACFC21705615CBE81EDB0AA')).toEqual({
      allowForSsl: true,
      denyForSsl: false,
      scopedDenyForSsl: false,
    });
    // Present in the keychain with NO trust records: installed, but not trusted.
    expect(decisions.get('21261EB01273C866A943ACBD51E04D3CBFDC395B')).toEqual({
      allowForSsl: false,
      denyForSsl: false,
      scopedDenyForSsl: false,
    });
  });

  it('records a deny, and ignores allows for non-SSL policies', () => {
    const dump = trustDump([
      { sha1: 'A'.repeat(40), settings: [{ policy: 'sslServer', result: 3 }] },
      { sha1: 'B'.repeat(40), settings: [{ policy: 'smime', result: 1 }] },
      // basicX509 is a distinct policy scope, not SSL authorization:
      // Chromium's trust_store_mac.cc evaluates SSL trust settings against
      // the sslServer policy only.
      { sha1: 'E'.repeat(40), settings: [{ policy: 'basicX509', result: 1 }] },
      { sha1: 'C'.repeat(40), settings: [{ result: 1 }] }, // no policy: applies to all
    ]);
    const decisions = parseTrustSettingsDump(dump);
    expect(decisions.get('A'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: true, scopedDenyForSsl: false });
    expect(decisions.get('B'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false });
    expect(decisions.get('E'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false });
    expect(decisions.get('C'.repeat(40))).toEqual({ allowForSsl: true, denyForSsl: false, scopedDenyForSsl: false });
  });

  it('treats a missing result key as TrustRoot (the add-trusted-cert export shape)', () => {
    // `security add-trusted-cert -r trustRoot -p ssl` exports an item with a
    // policy and NO kSecTrustSettingsResult; verify-cert confirms such a cert
    // is trusted, so the schema default is allow.
    const dump = trustDump([{ sha1: 'D'.repeat(40), settings: [{ policy: 'sslServer' }] }]);
    expect(parseTrustSettingsDump(dump).get('D'.repeat(40))).toEqual({
      allowForSsl: true,
      denyForSsl: false,
      scopedDenyForSsl: false,
    });
  });

  it('reads an explicitly empty trustSettings array as always-trust', () => {
    // SecTrustSettings.h: an empty array means "always trust this cert" with
    // result TrustRoot — and is "definitely not the same as *no* Trust
    // Settings". Corporate roots added without a policy export this shape.
    const dump = trustDump([{ sha1: 'E'.repeat(40), settings: [] }]);
    expect(parseTrustSettingsDump(dump).get('E'.repeat(40))).toEqual({
      allowForSsl: true,
      denyForSsl: false,
      scopedDenyForSsl: false,
    });
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
    expect(decisions.get('F'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false });
    // A scoped deny is not a global removal, but it flags the cert: the
    // bundle cannot express "trusted except for host X", so a root with ANY
    // scoped distrust is not exported (see the keychainCaRootsPem test).
    expect(decisions.get('0'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false, scopedDenyForSsl: true });
  });

  it('ignores application- and key-usage-constrained records', () => {
    // A record with ANY key beyond the policy OID, its name and the result
    // is scoped to something this process-wide bundle cannot express (an
    // application, a key usage, an allowed error, a future schema addition):
    // fail closed by ignoring it. This is the live-export shape of an
    // application-scoped trust.
    const dump = `{
  "trustList" => {
    "1111111111111111111111111111111111111111" => {
      "trustSettings" => [
        0 => {
          "kSecTrustSettingsPolicy" => {length = 9, bytes = 0x2a864886f763640103}
          "kSecTrustSettingsPolicyName" => "sslServer"
          "kSecTrustSettingsApplication" => {length = 42, bytes = 0xdeadbeef}
          "kSecTrustSettingsResult" => 1
        }
      ]
    }
    "2222222222222222222222222222222222222222" => {
      "trustSettings" => [
        0 => {
          "kSecTrustSettingsPolicy" => {length = 9, bytes = 0x2a864886f763640103}
          "kSecTrustSettingsPolicyName" => "sslServer"
          "kSecTrustSettingsKeyUsage" => 16
          "kSecTrustSettingsResult" => 1
        }
      ]
    }
  }
  "trustVersion" => 1
}
`;
    const decisions = parseTrustSettingsDump(dump);
    expect(decisions.get('1'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false });
    expect(decisions.get('2'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false });
  });

  it('fails closed on a policy OID the export did not name', () => {
    // Every built-in policy exports kSecTrustSettingsPolicyName, so a record
    // carrying an OID but no name is an unidentifiable scope — NOT "applies
    // to every policy". Its allow grants nothing; its deny still counts.
    const dump = `{
  "trustList" => {
    "3333333333333333333333333333333333333333" => {
      "trustSettings" => [
        0 => {
          "kSecTrustSettingsPolicy" => {length = 9, bytes = 0x010203040506070809}
          "kSecTrustSettingsResult" => 1
        }
      ]
    }
    "4444444444444444444444444444444444444444" => {
      "trustSettings" => [
        0 => {
          "kSecTrustSettingsPolicy" => {length = 9, bytes = 0x010203040506070809}
          "kSecTrustSettingsResult" => 3
        }
      ]
    }
  }
  "trustVersion" => 1
}
`;
    const decisions = parseTrustSettingsDump(dump);
    expect(decisions.get('3'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false });
    expect(decisions.get('4'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: true, scopedDenyForSsl: false });
  });

  it('judges a record only after the item closes, whatever the key order', () => {
    // `plutil -p` emits keys sorted today, but the order is not contractual:
    // a Result seen before the policy name or a constraint must not be
    // applied to a half-read item. Verdicts land at closeItem.
    const dump = `{
  "trustList" => {
    "5555555555555555555555555555555555555555" => {
      "trustSettings" => [
        0 => {
          "kSecTrustSettingsResult" => 1
          "kSecTrustSettingsPolicyName" => "smime"
        }
      ]
    }
    "6666666666666666666666666666666666666666" => {
      "trustSettings" => [
        0 => {
          "kSecTrustSettingsResult" => 1
          "kSecTrustSettingsPolicy" => {length = 9, bytes = 0x2a864886f763640103}
          "kSecTrustSettingsPolicyName" => "sslServer"
          "kSecTrustSettingsPolicyString" => "internal.example.com"
        }
      ]
    }
  }
  "trustVersion" => 1
}
`;
    const decisions = parseTrustSettingsDump(dump);
    // smime allow: not SSL regardless of field order.
    expect(decisions.get('5'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false });
    // sslServer allow scoped to a hostname: the late constraint still applies.
    expect(decisions.get('6'.repeat(40))).toEqual({ allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false });
  });

  it('ignores a stray Result line outside any item', () => {
    const dump = `{
  "trustList" => {
    "7777777777777777777777777777777777777777" => {
      "trustSettings" => [
        0 => {
          "kSecTrustSettingsPolicy" => {length = 9, bytes = 0x2a864886f763640103}
          "kSecTrustSettingsPolicyName" => "smime"
        }
        "kSecTrustSettingsResult" => 1
      ]
    }
  }
  "trustVersion" => 1
}
`;
    expect(parseTrustSettingsDump(dump).get('7'.repeat(40))).toEqual({
      allowForSsl: false,
      denyForSsl: false,
      scopedDenyForSsl: false,
    });
  });

  it('treats an unrecognised schema key with digits as a constraint', () => {
    // The constraint detector matches any kSecTrustSettings* key beyond the
    // three whitelist entries — including hypothetical future keys carrying
    // digits, which a [A-Za-z]+ pattern would have let slip through.
    const dump = `{
  "trustList" => {
    "8888888888888888888888888888888888888888" => {
      "trustSettings" => [
        0 => {
          "kSecTrustSettingsPolicy" => {length = 9, bytes = 0x2a864886f763640103}
          "kSecTrustSettingsPolicyName" => "sslServer"
          "kSecTrustSettingsPolicy2" => 1
          "kSecTrustSettingsResult" => 1
        }
      ]
    }
  }
  "trustVersion" => 1
}
`;
    expect(parseTrustSettingsDump(dump).get('8'.repeat(40))).toEqual({
      allowForSsl: false,
      denyForSsl: false,
      scopedDenyForSsl: false,
    });
  });
});

describe('keychainCaRootsPem', () => {
  it("does not read Apple's system-roots keychain at all", async () => {
    // The bundle is ADDITIVE to Node's Mozilla store, which already covers
    // the public web PKI. Apple's keychain additionally carries roots under
    // platform restrictions an unconditional export cannot honor (Entrust
    // Root CA G2 is distrusted for post-2024-11-15 certificates), so the
    // keychain is not even consulted.
    expect(macosKeychains().some((k) => k.includes('SystemRootCertificates'))).toBe(false);
    const seen: string[] = [];
    const exec: ExecText = async (cmd, args) => {
      seen.push(args.join(' '));
      if (cmd === 'security' && args[0] === 'find-certificate') throw new Error('keychain unreadable');
      if (cmd === 'security' && args[0] === 'trust-settings-export') throw new Error('empty');
      throw new Error('unexpected');
    };
    await keychainCaRootsPem(exec, macosKeychains()).catch(() => undefined);
    expect(seen.some((a) => a.includes('SystemRootCertificates'))).toBe(false);
  });

  it('excludes a CA the user or admin explicitly denied, even with another allow', async () => {
    const pem = await keychainCaRootsPem(
      fakeExec({
        certs: { [SYSTEM_ROOTS]: ROOT_CA },
        trust: trustDump([
          {
            sha1: ROOT_SHA1,
            settings: [
              { policy: 'sslServer', result: 1 },
              { policy: 'sslServer', result: 3 },
            ],
          },
        ]),
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

  it('excludes a root with a scoped deny even when another record allows it', async () => {
    // Cross-record composition: an unconditional sslServer allow in one
    // record plus a hostname-scoped deny ("never trust for huggingface.co")
    // in another. Exporting unconditionally would grant trust for exactly
    // the denied host — the download host. The bundle cannot carry the
    // scope, so the root is not exported at all.
    const pem = await keychainCaRootsPem(
      fakeExec({
        certs: { '/k/System': ROOT_CA },
        trust: trustDump([
          {
            sha1: ROOT_SHA1,
            settings: [
              { policy: 'sslServer', result: 1 },
              { policy: 'sslServer', policyString: 'huggingface.co', result: 3 },
            ],
          },
        ]),
      }),
      ['/k/System'],
    );
    expect(pem).toBe('');
  });

  it('lets a deny beat an allow on another SSL policy', async () => {
    const pem = await keychainCaRootsPem(
      fakeExec({
        certs: { '/k/System': ROOT_CA },
        trust: trustDump([
          {
            sha1: ROOT_SHA1,
            settings: [
              { policy: 'basicX509', result: 1 },
              { policy: 'sslServer', result: 3 },
            ],
          },
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
    const pem = await keychainCaRootsPem(
      fakeExec({
        certs: { [SYSTEM_ROOTS]: ROOT_CA, '/k/a': `${ROOT_CA}\n` },
        trust: trustDump([{ sha1: ROOT_SHA1, settings: [{ policy: 'sslServer', result: 1 }] }]),
      }),
      [SYSTEM_ROOTS, '/k/missing', '/k/a'],
    );
    expect(pem.match(/BEGIN CERTIFICATE/g)).toHaveLength(1);
  });

  it('skips unparseable blocks', async () => {
    const pem = await keychainCaRootsPem(
      fakeExec({
        certs: { [SYSTEM_ROOTS]: '-----BEGIN CERTIFICATE-----\nnot-a-cert\n-----END CERTIFICATE-----\n' + ROOT_CA },
        trust: trustDump([{ sha1: ROOT_SHA1, settings: [{ policy: 'sslServer', result: 1 }] }]),
      }),
      [SYSTEM_ROOTS],
    );
    expect(pem.match(/BEGIN CERTIFICATE/g)).toHaveLength(1);
  });
});

describe('prepareExtraCaBundle', () => {
  it('treats an empty trust domain as no records, not a read failure', async () => {
    // `security trust-settings-export` exits 1 with "No Trust Settings were
    // found." (errSecNoTrustSettings) for a domain with zero records — the
    // normal state on a machine whose user never edited trust settings. The
    // admin-installed corp root must still be exported, or the fix dies on
    // exactly the stock machines it targets.
    const dir = tmpDir();
    const result = await prepareExtraCaBundle({
      platform: 'darwin',
      dir,
      exec: fakeExec({
        certs: { [SYSTEM_ROOTS]: ROOT_CA },
        userExport: 'empty',
        trustAdmin: trustDump([{ sha1: ROOT_SHA1, settings: [{ policy: 'sslServer', result: 1 }] }]),
      }),
      keychains: [SYSTEM_ROOTS],
    });
    expect(result).toBe(join(dir, 'system-ca-roots.pem'));
    expect(readFileSync(result!, 'utf8')).toContain(ROOT_CA);
  });

  it('ships no keychain roots when a trust domain cannot be read', async () => {
    // Decisions merge allow-OR/deny-OR across the user and admin domains, so
    // exporting with only the domains that read successfully would keep their
    // allows while silently dropping the failed domain's denies — restoring a
    // trust the user explicitly revoked. The whole keychain bundle is
    // abandoned instead (startup itself is unaffected).
    const dir = tmpDir();
    const result = await prepareExtraCaBundle({
      platform: 'darwin',
      dir,
      exec: fakeExec({
        certs: { [SYSTEM_ROOTS]: ROOT_CA },
        userExport: 'fail',
        trustAdmin: trustDump([{ sha1: ROOT_SHA1, settings: [{ policy: 'sslServer', result: 1 }] }]),
      }),
      keychains: [SYSTEM_ROOTS],
    });
    expect(result).toBeNull();
  });

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
      exec: fakeExec({
        certs: { [SYSTEM_ROOTS]: `${ROOT_CA}\n${LEAF}` },
        trust: trustDump([{ sha1: ROOT_SHA1, settings: [{ policy: 'sslServer', result: 1 }] }]),
      }),
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
      exec: fakeExec({
        certs: { [SYSTEM_ROOTS]: ROOT_CA },
        trust: trustDump([{ sha1: ROOT_SHA1, settings: [{ policy: 'sslServer', result: 1 }] }]),
      }),
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
