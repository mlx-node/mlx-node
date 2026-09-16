/**
 * Trust the macOS keychain's SSL-trusted CA roots in the CONTROL PANEL child's
 * TLS — and ONLY the SSL-trusted ones.
 *
 * Why this exists: a TLS-inspecting network (corporate Zscaler/Netskope-style
 * proxy, antivirus "web shield") re-signs every HTTPS certificate with its own
 * root CA. The browser works because macOS trusts that root; Node's undici
 * verifies against its bundled Mozilla store, never looks at the keychain, and
 * fails instantly with SELF_SIGNED_CERT_IN_CHAIN. The download runner lives in
 * the CONTROL PANEL utilityProcess, so every outbound HTTPS it makes —
 * modelInfo, the file list, the blobs, Xet — dies the same way.
 *
 * Why not simply export every CA cert in the keychains: **keychain membership
 * is not trust.** macOS keeps per-certificate trust settings (Keychain Access
 * → a certificate → Trust): a stored CA can be installed-but-untrusted, set to
 * "Never Trust", or trusted for a non-SSL purpose only. Exporting on
 * `basicConstraints CA:TRUE` alone would promote exactly those into TLS
 * anchors — a regression an adversarial review caught before this shipped.
 * Effective anchor trust is therefore computed, per candidate, from BOTH
 * sources:
 *
 *   - candidates come from `security find-certificate` over the three
 *     keychains, filtered to `X509Certificate.ca` (CA:TRUE) — measured on a
 *     real System.keychain: its first entry is a self-signed identity with
 *     `ca: false`, so subject===issuer is not a usable proxy;
 *   - trust comes from `security trust-settings-export` (user + admin
 *     domains), which keys records by the cert's SHA-1 and carries
 *     `kSecTrustSettingsResult` per policy. Apple's System Roots are trusted
 *     implicitly (that's what the keychain means); anything else needs an
 *     explicit allow record for `sslServer` or `basicX509` (the pair
 *     mkcert-style tools set), and a deny record always wins. Verified against
 *     a live keychain: an installed cert with zero trust records (Blizzard
 *     Battle.net Local Cert) is NOT effectively trusted, an mkcert root with
 *     `sslServer → TrustRoot` IS.
 *
 * `verify-cert -p ssl` was tried and rejected as the trust oracle: it
 * evaluates the candidate AS A LEAF, which even Apple's own system roots fail
 * (CSSMERR_TP_CERT_SUSPENDED), so it cannot answer "is this a trusted anchor".
 *
 * The fix is process-wide rather than per-call: dump the effectively-trusted
 * roots into a PEM bundle and hand the child `NODE_EXTRA_CA_CERTS` at fork
 * time (the variable is latched on first TLS use — the same fork-parameter
 * rule as `supervisor/env.ts`). One bundle covers `fetch`, Xet, and anything
 * else that ever opens a TLS socket in that process.
 *
 * Async throughout — MAIN never blocks (`index.ts`'s header is the rule).
 * Failures degrade to `null` (no env var), never to a launch failure. A trust
 * domain that fails to READ voids the entire keychain bundle (see
 * loadTrustDecisions): exporting the surviving domains' allows without the
 * failed domain's denies could restore a trust the user explicitly revoked.
 */

import { execFile } from 'node:child_process';
import { X509Certificate } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';

/** Text-returning command runner, injected so tests never spawn `security`. */
export type ExecText = (cmd: string, args: string[]) => Promise<string>;

const PEM_BLOCK_RE = /-----BEGIN CERTIFICATE-----\r?\n[\s\S]+?-----END CERTIFICATE-----/g;

export const EXTRA_CA_BUNDLE_FILE = 'system-ca-roots.pem';

/** Apple's own roots: trusted by construction; only an explicit deny can remove one. */
const SYSTEM_ROOTS_KEYCHAIN = '/System/Library/Keychains/SystemRootCertificates.keychain';

// `kSecTrustSettingsResult` values (SecTrustSettings.h). Only the two allow
// results and the deny one matter here; Unspecified (4) is treated as no
// record, matching the system's own default-deny.
const TRUST_ROOT = 1;
const TRUST_AS_ROOT = 2;
const DENY = 3;
// The policies SSL evaluation consults: the SSL policy itself, and the basic
// X.509 policy it is layered on. An allow for e.g. S/MIME alone does NOT make
// a cert an SSL anchor.
const SSL_POLICIES = new Set(['sslServer', 'basicX509']);

/**
 * The three places a TLS-inspecting root can land. The System Roots keychain is
 * Apple-shipped; System.keychain is where an admin-installed (corp/MDM) root
 * lives; login.keychain-db is where a user-accepted one does.
 */
export function macosKeychains(): string[] {
  return [
    SYSTEM_ROOTS_KEYCHAIN,
    '/Library/Keychains/System.keychain',
    join(homedir(), 'Library', 'Keychains', 'login.keychain-db'),
  ];
}

const execFileAsync = promisify(execFile);

/** The production runner: bounded so a hung keychain cannot stall bootstrap. */
export async function securityText(cmd: string, args: string[]): Promise<string> {
  const { stdout } = await execFileAsync(cmd, args, {
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
    timeout: 10_000,
  });
  return stdout;
}

interface Candidate {
  pem: string;
  /** SHA-1, uppercase hex with no colons — the key `trust-settings-export` uses. */
  sha1: string;
  /** SHA-256, used only for dedupe. */
  sha256: string;
  systemRoots: boolean;
}

/** One cert's explicit SSL-trust verdict, OR'd across the user and admin domains. */
export interface TrustDecision {
  allowForSsl: boolean;
  denyForSsl: boolean;
}

/**
 * Parse `plutil -p` of a `trust-settings-export` plist into per-cert
 * decisions. The export keys each entry by the cert's SHA-1 (40 uppercase hex
 * chars, indent 4); each `trustSettings` item (indent 8) pairs a policy name
 * with a numeric result. Two schema subtleties, both verified against live
 * keychain state:
 *
 *   - an item with NO `kSecTrustSettingsResult` key means TrustRoot — the
 *     schema default. `security add-trusted-cert -r trustRoot` exports exactly
 *     such an item, and `security verify-cert -p ssl` confirms the cert is
 *     trusted while the key is absent;
 *   - a result with NO policy name applies to every policy (older records),
 *     so it counts for SSL too;
 *   - an entry whose `trustSettings` array is EMPTY means "always trust this
 *     cert" (SecTrustSettings.h: "An empty Trust Settings array is definitely
 *     not the same as *no* Trust Settings").
 *
 * Constrained records are IGNORED for this process-wide decision. A record
 * carries a constraint whenever it has ANY key beyond the policy OID, the
 * policy name and the result — `kSecTrustSettingsPolicyString` (hostname),
 * `kSecTrustSettingsApplication`, `kSecTrustSettingsKeyUsage`, and anything
 * schema adds later. Exporting such a CA unconditionally would broaden a
 * narrow trust (SSL for one host, one app, one usage) into an any-host
 * anchor. NODE_EXTRA_CA_CERTS cannot carry constraints, so the record is
 * skipped in both directions — a scoped "never trust" must not remove a
 * good root either.
 *
 * The `-p` format is undocumented but has been stable for years, and it is the
 * only export form that avoids shipping a plist parser for three calls per
 * launch.
 */
export function parseTrustSettingsDump(text: string): Map<string, TrustDecision> {
  const decisions = new Map<string, TrustDecision>();
  let current: (TrustDecision & { items: number; sawArray: boolean }) | null = null;
  let item: { policy: string | null; sawResult: boolean; constrained: boolean } | null = null;
  const apply = (result: number, policy: string | null, constrained: boolean): void => {
    if (current === null) return;
    if (policy !== null && !SSL_POLICIES.has(policy)) return;
    // A constrained record proves nothing about other hosts: no allow, and no
    // global deny either (see the docstring above).
    if (constrained) return;
    if (result === TRUST_ROOT || result === TRUST_AS_ROOT) current.allowForSsl = true;
    if (result === DENY) current.denyForSsl = true;
  };
  const closeItem = (): void => {
    if (item !== null && !item.sawResult) apply(TRUST_ROOT, item.policy, item.constrained);
    item = null;
  };
  let currentSha1: string | null = null;
  const closeEntry = (): void => {
    closeItem();
    // An explicitly EMPTY trust-settings array is Apple's "always trust this
    // cert" (TrustRoot) encoding — and is NOT the same as an entry with no
    // trustSettings key at all, which is installed-but-untrusted.
    if (current !== null && current.sawArray && current.items === 0) current.allowForSsl = true;
    if (current !== null && currentSha1 !== null) {
      decisions.set(currentSha1, { allowForSsl: current.allowForSsl, denyForSsl: current.denyForSsl });
    }
    current = null;
    currentSha1 = null;
  };
  for (const line of text.split('\n')) {
    const key = /^\s{4}"([0-9A-F]{40})" => \{$/.exec(line);
    if (key !== null) {
      closeEntry();
      current = { allowForSsl: false, denyForSsl: false, items: 0, sawArray: false };
      currentSha1 = key[1];
      continue;
    }
    if (current === null) continue;
    if (/"trustSettings" => \[/.test(line)) {
      current.sawArray = true;
      continue;
    }
    if (/^\s{8}\d+ => \{$/.test(line)) {
      closeItem();
      current.items += 1;
      item = { policy: null, sawResult: false, constrained: false };
      continue;
    }
    if (/^\s{8}\}/.test(line)) {
      closeItem();
      continue;
    }
    if (/^\s{4}\}/.test(line)) {
      closeEntry();
      continue;
    }
    const policy = /"kSecTrustSettingsPolicyName" => "([^"]+)"/.exec(line);
    if (policy !== null) {
      if (item !== null) item.policy = policy[1];
      continue;
    }
    // Any other kSecTrustSettings* key on the item is a constraint this
    // process-wide bundle cannot honor (hostname, application, key usage,
    // allowed errors, future schema additions) — fail closed by ignoring
    // the record entirely. The negative lookahead whitelists the three keys
    // an unconstrained record is made of: the policy OID blob, its name,
    // and the result.
    if (/"kSecTrustSettings(?!Policy"|PolicyName"|Result")[A-Za-z]+"/.test(line)) {
      if (item !== null) item.constrained = true;
      continue;
    }
    const result = /"kSecTrustSettingsResult" => (\d+)/.exec(line);
    if (result !== null) {
      if (item !== null) item.sawResult = true;
      apply(Number(result[1]), item?.policy ?? null, item?.constrained ?? false);
    }
  }
  return decisions;
}

/**
 * `security find-certificate` over the given keychains, filtered to CA certs.
 * Unreadable keychains and unparseable blocks are skipped — the worst case is
 * the pre-fix behavior, never a crash.
 */
async function collectCandidates(exec: ExecText, keychains: readonly string[]): Promise<Candidate[]> {
  const outputs = await Promise.all(
    keychains.map(async (keychain) => {
      try {
        return { keychain, output: await exec('security', ['find-certificate', '-a', '-p', keychain]) };
      } catch {
        return { keychain, output: '' };
      }
    }),
  );
  const candidates: Candidate[] = [];
  for (const { keychain, output } of outputs) {
    for (const block of output.match(PEM_BLOCK_RE) ?? []) {
      try {
        const cert = new X509Certificate(block);
        if (!cert.ca) continue;
        candidates.push({
          pem: block,
          sha1: cert.fingerprint.replaceAll(':', ''),
          sha256: cert.fingerprint256,
          systemRoots: keychain === SYSTEM_ROOTS_KEYCHAIN,
        });
      } catch {
        // Not a parseable certificate block (or a format Node rejects): skip it.
      }
    }
  }
  return candidates;
}

/**
 * Effective SSL-anchor decisions from the user and admin trust-settings
 * domains. The system domain is not exported: its entries are the Apple
 * system roots themselves, whose trust is the implicit default already
 * encoded by `systemRoots` on the candidate.
 *
 * A domain that fails to READ rejects the whole load — this is the one
 * failure in this module that must NOT degrade locally. Decisions are merged
 * as allow-OR / deny-OR across domains, so proceeding with only the domains
 * that read successfully would silently drop the failed domain's DENIES while
 * keeping the others' allows: a root the user explicitly revoked would be
 * exported on the admin domain's say-so. Rejecting propagates to
 * `prepareExtraCaBundle`'s catch, which ships no keychain roots at all (the
 * inherited NODE_EXTRA_CA_CERTS still applies, and startup is unaffected).
 */
async function loadTrustDecisions(exec: ExecText): Promise<Map<string, TrustDecision>> {
  const merged = new Map<string, TrustDecision>();
  const tmp = await mkdtemp(join(tmpdir(), 'mlx-trust-'));
  try {
    for (const args of [[], ['-d']]) {
      const plist = join(tmp, `trust${args[0] ?? '-user'}.plist`);
      await exec('security', ['trust-settings-export', ...args, plist]);
      const dump = await exec('plutil', ['-p', plist]);
      for (const [sha1, decision] of parseTrustSettingsDump(dump)) {
        const entry = merged.get(sha1) ?? { allowForSsl: false, denyForSsl: false };
        entry.allowForSsl ||= decision.allowForSsl;
        entry.denyForSsl ||= decision.denyForSsl;
        merged.set(sha1, entry);
      }
    }
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
  return merged;
}

/**
 * Concatenated PEM of every effectively SSL-trusted CA root found in the
 * given keychains, deduped by fingerprint. The selection rule:
 *
 *   - Apple system roots: included unless explicitly denied;
 *   - anything else: included only with an explicit SSL/basicX509 allow
 *     record, and never when denied — keychain membership alone is not trust.
 */
export async function keychainCaRootsPem(exec: ExecText, keychains: readonly string[]): Promise<string> {
  const [candidates, decisions] = await Promise.all([collectCandidates(exec, keychains), loadTrustDecisions(exec)]);
  const seen = new Set<string>();
  const roots: string[] = [];
  for (const candidate of candidates) {
    if (seen.has(candidate.sha256)) continue;
    const decision = decisions.get(candidate.sha1);
    const trusted = candidate.systemRoots ? decision?.denyForSsl !== true : decision?.allowForSsl === true && decision?.denyForSsl !== true;
    if (!trusted) continue;
    seen.add(candidate.sha256);
    roots.push(candidate.pem);
  }
  return roots.length === 0 ? '' : `${roots.join('\n')}\n`;
}

/**
 * Write the extra-CA bundle for the CONTROL PANEL child and return its path,
 * or `null` when there is nothing to add (non-macOS, no trusted roots, no
 * inherited bundle, or ANY failure). The caller awaits this inside
 * `bootstrap()`, whose rejection handler is `app.exit(1)` — an optional TLS
 * convenience must never be a fatal startup dependency, so every failure
 * mode collapses to `null` here rather than rejecting.
 *
 * An inherited `NODE_EXTRA_CA_CERTS` (a developer's shell can carry one into an
 * unpackaged run) is MERGED into the bundle rather than dropped: the variable
 * names exactly one file, and handing the child only the keychain dump would
 * silently un-trust whatever the developer had configured.
 */
export async function prepareExtraCaBundle(opts: {
  platform: NodeJS.Platform;
  /** userData — per-user writable, survives updates, same trust domain as the login keychain. */
  dir: string;
  exec: ExecText;
  inheritedPath?: string | undefined;
  keychains?: readonly string[];
}): Promise<string | null> {
  if (opts.platform !== 'darwin') return null;
  const parts: string[] = [];
  if (opts.inheritedPath !== undefined && opts.inheritedPath !== '') {
    try {
      parts.push(await readFile(opts.inheritedPath, 'utf8'));
    } catch {
      // A dangling inherited path is not ours to fix; the keychain roots still apply.
    }
  }
  try {
    // Whole call inside the catch: collectCandidates swallows per-keychain
    // errors, but loadTrustDecisions' mkdtemp/cleanup can still reject
    // (ENOSPC), and that rejection must not escape.
    parts.push(await keychainCaRootsPem(opts.exec, opts.keychains ?? macosKeychains()));
  } catch (error) {
    console.warn('[mlx] could not export keychain CA roots:', error);
  }
  const pem = parts.join('\n').trim();
  if (pem === '') return null;
  const bundlePath = join(opts.dir, EXTRA_CA_BUNDLE_FILE);
  try {
    await mkdir(opts.dir, { recursive: true });
    await writeFile(bundlePath, `${pem}\n`, { mode: 0o600 });
  } catch {
    return null;
  }
  return bundlePath;
}
