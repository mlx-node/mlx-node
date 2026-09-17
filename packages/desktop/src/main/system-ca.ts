/**
 * Trust the OS trust store's extra CA roots in the CONTROL PANEL child's
 * TLS — the ones Node's bundled Mozilla store does not already carry.
 *
 * Why this exists: a TLS-inspecting network (corporate Zscaler/Netskope-style
 * proxy, antivirus "web shield") re-signs every HTTPS certificate with its own
 * root CA. The browser works because the OS trusts that root; Node's undici
 * verifies against its bundled Mozilla store, never looks at the OS store, and
 * fails instantly with SELF_SIGNED_CERT_IN_CHAIN. The download runner lives in
 * the CONTROL PANEL utilityProcess, so every outbound HTTPS it makes —
 * modelInfo, the file list, the blobs, Xet — dies the same way. macOS and
 * Linux are supported; each needs its own collector because the two trust
 * models differ fundamentally.
 *
 * macOS: **keychain membership is not trust.** macOS keeps per-certificate
 * trust settings (Keychain Access → a certificate → Trust): a stored CA can
 * be installed-but-untrusted, set to "Never Trust", or trusted for a non-SSL
 * purpose only. Exporting on `basicConstraints CA:TRUE` alone would promote
 * exactly those into TLS anchors — a regression an adversarial review caught
 * before this shipped. Effective anchor trust is therefore computed, per
 * candidate, from BOTH sources:
 *
 *   - candidates come from `security find-certificate` over the user and
 *     admin keychains, filtered to `X509Certificate.ca` (CA:TRUE) — measured
 *     on a real System.keychain: its first entry is a self-signed identity
 *     with `ca: false`, so subject===issuer is not a usable proxy. Apple's
 *     System Roots keychain is excluded (see `macosKeychains`): Node's
 *     Mozilla store already covers the public web PKI, and Apple's keychain
 *     carries platform-restricted roots an additive bundle cannot honor;
 *   - trust comes from `security trust-settings-export` (user + admin
 *     domains), which keys records by the cert's SHA-1 and carries
 *     `kSecTrustSettingsResult` per policy — PLUS keychain provenance for
 *     the System keychain itself: a CA:TRUE root there is trusted unless
 *     denied, because profile-driven (MDM/corporate) trust does not appear
 *     in `trust-settings-export` and the explicit-record rule therefore
 *     omitted exactly the intercepting roots this module exists to add.
 *     Verified against a live keychain: an installed cert with zero trust
 *     records in the LOGIN keychain (Blizzard Battle.net Local Cert) is NOT
 *     effectively trusted and stays excluded, an mkcert root with
 *     `sslServer → TrustRoot` IS trusted, and a deny or scoped deny vetoes
 *     a root in either keychain.
 *
 * `verify-cert -p ssl` was tried and rejected as the trust oracle: it
 * evaluates the candidate AS A LEAF, which even Apple's own system roots fail
 * (CSSMERR_TP_CERT_SUSPENDED), so it cannot answer "is this a trusted anchor".
 *
 * Linux: **the store itself is the trust decision.** `update-ca-certificates`
 * (Debian/Ubuntu) and `update-ca-trust` (Fedora/RHEL p11-kit) emit merged
 * bundles that already exclude whatever the admin disabled (`!` lines in
 * ca-certificates.conf) or blocklisted, so a certificate's presence in an
 * effective store path IS its SSL trust verdict — there is no separate
 * ledger to consult, and no deny channel to miss: a distrusted root is
 * absent from the store, not marked inside it. Sources follow the
 * OpenSSL/Go convention (`$SSL_CERT_FILE`, the colon-separated
 * `$SSL_CERT_DIR`, then the well-known merged bundles and hashed
 * directories); every parseable certificate in them is exported with no
 * `cert.ca` filter, because OpenSSL anchors chains on leaf store entries
 * too. Store dirs are read by MEMBERSHIP — a symlink or `HASH.N`-named
 * entry — because a stray file in the dir (mod_ssl's `localhost.crt`,
 * `make-dummy-cert`) is not a store entry OpenSSL would ever reach. An
 * unreadable or missing path simply contributes nothing — the macOS
 * "unreadable domain hides a deny" failure mode has no analog.
 * Collected roots are then reduced to the delta over Node's bundled store:
 * a stock machine yields an empty bundle and no env var at all.
 *
 * The fix is process-wide rather than per-call: dump the effectively-trusted
 * roots into a PEM bundle and hand the child `NODE_EXTRA_CA_CERTS` at fork
 * time (the variable is latched on first TLS use — the same fork-parameter
 * rule as `supervisor/env.ts`). One bundle covers `fetch`, Xet, and anything
 * else that ever opens a TLS socket in that process.
 *
 * Async throughout — MAIN never blocks (`index.ts`'s header is the rule).
 * Failures degrade to `null` (no env var), never to a launch failure. On
 * macOS a trust domain that fails to READ voids the keychain bundle (see
 * loadTrustDecisions): exporting without the failed domain's denies could
 * restore a trust the user explicitly revoked — System-keychain candidates
 * included, since "Never Trust" is settable on a System item. The refusal is
 * logged by name; an inherited NODE_EXTRA_CA_CERTS still applies, so a
 * managed machine can be unblocked by exporting its proxy root manually.
 */

import { execFile } from 'node:child_process';
import { X509Certificate } from 'node:crypto';
import { mkdir, mkdtemp, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { rootCertificates } from 'node:tls';
import { promisify } from 'node:util';

/** Text-returning command runner, injected so tests never spawn `security`. */
export type ExecText = (cmd: string, args: string[]) => Promise<string>;

const PEM_BLOCK_RE = /-----BEGIN CERTIFICATE-----\r?\n[\s\S]+?-----END CERTIFICATE-----/g;

export const EXTRA_CA_BUNDLE_FILE = 'system-ca-roots.pem';

// `kSecTrustSettingsResult` values (SecTrustSettings.h). Only the two allow
// results and the deny one matter here; Unspecified (4) is treated as no
// record, matching the system's own default-deny.
const TRUST_ROOT = 1;
const TRUST_AS_ROOT = 2;
const DENY = 3;
// The only policy whose allow record authorizes TLS server verification.
// basicX509 is NOT on the list: it is a distinct policy scope (Chromium's
// trust_store_mac.cc evaluates SSL trust settings against the sslServer
// policy only), so a basic-only allow must not promote a CA into a TLS
// anchor. An allow for e.g. S/MIME alone does not count either. A record
// with NO policy OID at all (the oldest export shape) applies to every
// policy and is handled in the parser, not here.
const SSL_POLICIES = new Set(['sslServer']);

/**
 * Marker for a record carrying `kSecTrustSettingsPolicy` (a policy OID blob)
 * but NO `kSecTrustSettingsPolicyName`. Every built-in policy exports its
 * name, so an unnamed OID is a custom or unrecognised policy scope — not
 * "applies to every policy". The parser fails closed on it: an allow with an
 * unidentifiable scope grants nothing, while a deny still counts (denying is
 * the safe direction under ambiguity).
 */
const UNNAMED_POLICY = 'unnamed-policy-oid';

/**
 * The two places a TLS-inspecting root can land: System.keychain is where an
 * admin-installed (corp/MDM) root lives; login.keychain-db is where a
 * user-accepted one does.
 *
 * Apple's SystemRootCertificates.keychain is deliberately NOT exported: the
 * bundle is ADDITIVE to Node's Mozilla store, which already carries the
 * public web PKI, and Apple's keychain also holds roots under platform
 * restrictions an unconditional export cannot honor (e.g. Entrust Root CA
 * G2, which Apple — and Node — distrust for certificates issued after
 * 2024-11-15). Exporting it would broaden trust beyond both stores.
 */
export function macosKeychains(): string[] {
  return ['/Library/Keychains/System.keychain', join(homedir(), 'Library', 'Keychains', 'login.keychain-db')];
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
  /** The keychain this cert was read from — decides which trust rule applies. */
  keychain: string;
}

/** One cert's explicit SSL-trust verdict, OR'd across the user and admin domains. */
export interface TrustDecision {
  allowForSsl: boolean;
  denyForSsl: boolean;
  /**
   * A DENY record exists but is scoped (hostname, application, usage) in a
   * way a process-wide bundle cannot express. Such a CA is not
   * unambiguously trusted: the user explicitly distrusted it for some host,
   * and an unconditional export would grant trust for exactly that host.
   */
  scopedDenyForSsl: boolean;
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
 *   - a record with NO policy OID at all (the oldest export shape) applies to
 *     every policy, so it counts for SSL too. A record carrying a policy OID
 *     but no `kSecTrustSettingsPolicyName` is the opposite case: built-in
 *     policies always export their name, so an unnamed OID is a scope this
 *     bundle cannot identify — its allows do not count (see UNNAMED_POLICY);
 *   - an entry whose `trustSettings` array is EMPTY means "always trust this
 *     cert" (SecTrustSettings.h: "An empty Trust Settings array is definitely
 *     not the same as *no* Trust Settings").
 *
 * Constrained records are never applied verbatim. A record carries a
 * constraint whenever it has ANY key beyond the policy OID, the policy name
 * and the result — `kSecTrustSettingsPolicyString` (hostname),
 * `kSecTrustSettingsApplication`, `kSecTrustSettingsKeyUsage`, and anything
 * schema adds later. NODE_EXTRA_CA_CERTS cannot carry constraints, so:
 *
 *   - a constrained ALLOW is ignored: exporting it would broaden a narrow
 *     trust (SSL for one host, one app, one usage) into an any-host anchor;
 *   - a constrained DENY does not remove the root globally (it may be
 *     distrusted for one host only), but it DOES flag the cert via
 *     `scopedDenyForSsl`, and `keychainCaRootsPem` refuses to export a
 *     flagged non-system root at all — an unconditional export would grant
 *     trust for exactly the host the user distrusted.
 *
 * The verdict is applied only when an item CLOSES: `-p` output is
 * undocumented and dictionary order is not part of the contract, so a Result
 * seen before the policy name or a constraint key must not be judged on a
 * half-read item.
 *
 * The `-p` format is undocumented but has been stable for years, and it is the
 * only export form that avoids shipping a plist parser for three calls per
 * launch.
 */
export function parseTrustSettingsDump(text: string): Map<string, TrustDecision> {
  const decisions = new Map<string, TrustDecision>();
  let current: (TrustDecision & { items: number; sawArray: boolean }) | null = null;
  let item: {
    policy: string | null;
    sawPolicyOid: boolean;
    result: number | null;
    constrained: boolean;
  } | null = null;
  const apply = (result: number, policy: string | null, constrained: boolean): void => {
    if (current === null) return;
    const isDeny = result === DENY;
    if (policy !== null) {
      // Scope check. `null` (record carried no policy OID — the oldest shape)
      // and a named sslServer both count. UNNAMED_POLICY counts for DENIES
      // only: an allow needs a scope the bundle can positively identify,
      // while a deny is the safe direction under ambiguity.
      if (!SSL_POLICIES.has(policy) && !(policy === UNNAMED_POLICY && isDeny)) return;
    }
    if (constrained) {
      // Not applied verbatim (see the docstring): a scoped allow grants
      // nothing, but a scoped deny still flags the cert as not
      // unambiguously trusted.
      if (isDeny) current.scopedDenyForSsl = true;
      return;
    }
    if (result === TRUST_ROOT || result === TRUST_AS_ROOT) current.allowForSsl = true;
    if (isDeny) current.denyForSsl = true;
  };
  const closeItem = (): void => {
    if (item !== null) {
      const policy = item.policy ?? (item.sawPolicyOid ? UNNAMED_POLICY : null);
      apply(item.result ?? TRUST_ROOT, policy, item.constrained);
    }
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
      decisions.set(currentSha1, {
        allowForSsl: current.allowForSsl,
        denyForSsl: current.denyForSsl,
        scopedDenyForSsl: current.scopedDenyForSsl,
      });
    }
    current = null;
    currentSha1 = null;
  };
  for (const line of text.split('\n')) {
    const key = /^\s{4}"([0-9A-F]{40})" => \{$/.exec(line);
    if (key !== null) {
      closeEntry();
      current = { allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false, items: 0, sawArray: false };
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
      item = { policy: null, sawPolicyOid: false, result: null, constrained: false };
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
    if (/"kSecTrustSettingsPolicy" =>/.test(line)) {
      if (item !== null) item.sawPolicyOid = true;
      continue;
    }
    // Any other kSecTrustSettings* key on the item is a constraint this
    // process-wide bundle cannot honor (hostname, application, key usage,
    // allowed errors, future schema additions) — fail closed by ignoring
    // the record entirely. The negative lookahead whitelists the three keys
    // an unconstrained record is made of: the policy OID blob, its name,
    // and the result.
    if (/"kSecTrustSettings(?!Policy"|PolicyName"|Result")[^"]+"/.test(line)) {
      if (item !== null) item.constrained = true;
      continue;
    }
    const result = /"kSecTrustSettingsResult" => (\d+)/.exec(line);
    if (result !== null) {
      // Buffered, not applied: the verdict is judged at closeItem once every
      // field of the record has been seen.
      if (item !== null) item.result = Number(result[1]);
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
        return { output: await exec('security', ['find-certificate', '-a', '-p', keychain]) };
      } catch {
        return { output: '' };
      }
    }),
  );
  const candidates: Candidate[] = [];
  for (const [index, { output }] of outputs.entries()) {
    const keychain = keychains[index] ?? '';
    for (const block of output.match(PEM_BLOCK_RE) ?? []) {
      try {
        const cert = new X509Certificate(block);
        if (!cert.ca) continue;
        candidates.push({
          pem: block,
          sha1: cert.fingerprint.replaceAll(':', ''),
          sha256: cert.fingerprint256,
          keychain,
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
 * domains. The system domain is not read: its entries concern the Apple
 * system roots, which this module no longer exports (see macosKeychains).
 *
 * A domain that fails to READ is reported in `unreadable` rather than
 * rejecting the load outright. Decisions are merged as allow-OR / deny-OR
 * across domains, so continuing with only the domains that read successfully
 * would silently drop the failed domain's DENIES while keeping the others'
 * allows: a root the user explicitly revoked could be exported on the other
 * domain's say-so. Callers therefore get the flag and must WITHHOLD every
 * candidate whose export depends on a deny being absent — which is every
 * candidate, System-keychain roots included ("Never Trust" is settable on a
 * System item). The flag still exists rather than a plain throw so the
 * refusal is one decision with one log line, and so a future caller may
 * confine it more precisely than "ship nothing".
 * One exception: a domain that has NO records at all exits the export with
 * errSecNoTrustSettings, which means "empty", not "unreadable" — that domain
 * contributes nothing and the other domain still loads.
 */
interface TrustDecisions {
  decisions: Map<string, TrustDecision>;
  /** True when at least one domain's export failed for a reason other than "no records". */
  unreadable: boolean;
}

async function loadTrustDecisions(exec: ExecText): Promise<TrustDecisions> {
  const merged = new Map<string, TrustDecision>();
  let unreadable = false;
  const tmp = await mkdtemp(join(tmpdir(), 'mlx-trust-'));
  try {
    for (const args of [[], ['-d']]) {
      const plist = join(tmp, `trust${args[0] ?? '-user'}.plist`);
      try {
        await exec('security', ['trust-settings-export', ...args, plist]);
      } catch (error) {
        // An EMPTY domain is not a read failure: `security
        // trust-settings-export` exits 1 with "SecTrustSettingsCreateExternal-
        // Representation: No Trust Settings were found." (errSecNoTrustSettings)
        // when the domain has no records at all — the normal state on a
        // machine whose user never touched Keychain Access trust. Any other
        // failure marks the load UNREADABLE and keeps parsing the rest.
        if (!/No Trust Settings were found|errSecNoTrustSettings/.test(String(error))) {
          console.warn('[mlx] trust-settings export failed; treating that domain as unreadable:', error);
          unreadable = true;
        }
        continue;
      }
      const dump = await exec('plutil', ['-p', plist]);
      for (const [sha1, decision] of parseTrustSettingsDump(dump)) {
        const entry = merged.get(sha1) ?? { allowForSsl: false, denyForSsl: false, scopedDenyForSsl: false };
        entry.allowForSsl ||= decision.allowForSsl;
        entry.denyForSsl ||= decision.denyForSsl;
        entry.scopedDenyForSsl ||= decision.scopedDenyForSsl;
        merged.set(sha1, entry);
      }
    }
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
  return { decisions: merged, unreadable };
}

/** The admin-writable keychain: where an MDM-managed or installer-placed root lives. */
const SYSTEM_KEYCHAIN = '/Library/Keychains/System.keychain';

/**
 * Concatenated PEM of every effectively SSL-trusted CA root found in the
 * given keychains, deduped by fingerprint. The selection rule:
 *
 *   - Apple system roots: included unless explicitly denied (denial is
 *     advisory here anyway — the bundle is ADDITIVE to Node's Mozilla
 *     store, which already carries the same roots);
 *   - a CA:TRUE root in the SYSTEM keychain: included unless denied (global
 *     or scoped), and only when every trust domain READ — an unreadable
 *     domain's denies are unknown, and a System-item deny is exactly the
 *     kind that hides there. System-keychain membership IS trust for this
 *     purpose — writing there needs admin rights, and corporate/MDM roots
 *     routinely carry NO explicit trust record at all: profile-driven trust
 *     does not surface in `trust-settings-export`, so the old
 *     explicit-allow-record rule silently omitted exactly the middlebox
 *     roots this module exists to add (measured: a Zscaler-style fleet
 *     root in the System keychain with zero user/admin records);
 *   - anything else (the login keychain): included only with an
 *     unconstrained sslServer allow record, never when denied, and never
 *     when a SCOPED deny exists — the bundle cannot express "trusted except
 *     for host X", so a root the user distrusted for any host is not
 *     exported at all. Login-keychain membership alone is not trust — the
 *     Battle.net-style junk cert lives there, and it stays excluded.
 */
export async function keychainCaRootsPem(exec: ExecText, keychains: readonly string[]): Promise<string> {
  const [candidates, trust] = await Promise.all([collectCandidates(exec, keychains), loadTrustDecisions(exec)]);
  const { decisions } = trust;
  const seen = new Set<string>();
  const roots: string[] = [];
  for (const candidate of candidates) {
    if (seen.has(candidate.sha256)) continue;
    const decision = decisions.get(candidate.sha1);
    // Any deny vetoes everywhere: a global deny is explicit distrust, and a
    // SCOPED deny makes the root ineligible because the bundle cannot
    // express the scope (it would grant trust for exactly the denied host).
    const denied = decision?.denyForSsl === true || decision?.scopedDenyForSsl === true;
    const untrustworthy =
      candidate.keychain === SYSTEM_KEYCHAIN
        ? // System-keychain membership IS trust (admin-gated; profile-driven
          // trust lives here), so an explicit deny is what vetoes it — and a
          // deny is exactly what an UNREADABLE domain hides. Measured on a real
          // keychain: records for System.keychain certificates appear in the
          // ADMIN domain's export (both of this machine's admin records point
          // at System-keychain certs; the system domain carries only bare
          // default entries, zero per-policy items) — so a failed admin read is
          // precisely when a System item's "Never Trust" is unknowable.
          // Shipping such a root then would bypass a revocation the user
          // performed, so it is withheld until every domain reads.
          denied || trust.unreadable
        : // Every other candidate comes from the user's login keychain, where
          // membership is NOT trust: an explicit, unconstrained sslServer
          // allow is required, and an UNREADABLE trust domain cannot supply
          // one.
          denied || decision?.allowForSsl !== true || trust.unreadable;
    if (untrustworthy) continue;
    seen.add(candidate.sha256);
    roots.push(candidate.pem);
  }
  return roots.length === 0 ? '' : `${roots.join('\n')}\n`;
}

/**
 * The effective OpenSSL-style store files on Linux, in search order. These
 * are the generated, post-decision artifacts — never the admin's source
 * directories (`/usr/local/share/ca-certificates`, `pki/ca-trust/source`),
 * whose contents only count once the update tool merges them into a bundle
 * or hashed dir below.
 */
const LINUX_CA_FILES = [
  '/etc/ssl/certs/ca-certificates.crt', // Debian/Ubuntu (update-ca-certificates)
  '/etc/pki/tls/certs/ca-bundle.crt', // Fedora/RHEL (→ p11-kit extracted)
  '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem', // p11-kit, TLS-scoped
  '/etc/ssl/ca-bundle.pem', // openSUSE
  '/etc/ssl/cert.pem', // Alpine
  '/etc/pki/tls/cacert.pem', // OpenELEC
];

/**
 * Hashed-store directories. Membership is defined by hash-name
 * reachability, not by presence: Debian links every cert twice (named and
 * `HASH.N` symlinks), p11-kit emits `HASH.N` entries — while a stray file
 * like Fedora's `localhost.crt` or `make-dummy-cert` sits in the dir
 * without ever being a store entry.
 */
const LINUX_CA_DIRS = ['/etc/ssl/certs', '/etc/pki/tls/certs'];

/**
 * The OpenSSL hashed-lookup entry name: 8 hex digits of subject-name hash,
 * a dot, a collision counter (`HASH.0`, `HASH.1`, …). `HASH.rN` CRL links
 * are correctly excluded — a CRL is never a CA anchor.
 */
const HASHED_ENTRY_RE = /^[0-9a-f]{8}\.\d+$/i;

/**
 * Where a Linux TLS-intercepting root can be found: the env vars cover
 * hand-rolled setups, the well-known paths cover the generated stores.
 * Unlike under OpenSSL, where SSL_CERT_FILE/SSL_CERT_DIR REPLACE the
 * defaults, here they are additive — this module can only ever add anchors
 * to Node's store, so unioning every source is the correct bias toward
 * coverage. Missing/unreadable paths contribute nothing.
 */
export interface LinuxCaSources {
  files: readonly string[];
  dirs: readonly string[];
}

export function linuxCaSources(env: NodeJS.ProcessEnv = process.env): LinuxCaSources {
  return {
    files: [env.SSL_CERT_FILE ?? '', ...LINUX_CA_FILES].filter((p) => p !== ''),
    dirs: [...(env.SSL_CERT_DIR ?? '').split(':'), ...LINUX_CA_DIRS].filter((p) => p !== ''),
  };
}

/**
 * Read one store source, bounded like the `security` calls' 64 MB
 * maxBuffer. `stat` follows symlinks, so a fifo/device/other non-regular
 * entry — env-controlled, or sitting in a scanned dir — is skipped BEFORE
 * an open that could block forever, and an oversize file never reaches
 * memory.
 */
async function readStoreEntry(path: string): Promise<string> {
  try {
    const info = await stat(path);
    if (!info.isFile() || info.size > 64 * 1024 * 1024) return '';
    return await readFile(path, 'utf8');
  } catch {
    return '';
  }
}

/** Fingerprints of Node's bundled Mozilla roots — the baseline to delta against. */
function bundledCaFingerprints(): Set<string> {
  const fingerprints = new Set<string>();
  for (const pem of rootCertificates) {
    try {
      fingerprints.add(new X509Certificate(pem).fingerprint256);
    } catch {
      // A bundled root Node itself cannot parse is no anchor either way.
    }
  }
  return fingerprints;
}

/**
 * Concatenated PEM of every certificate in the given Linux stores that Node
 * does not already bundle, deduped by fingerprint. Membership in an
 * effective store IS the trust decision (see the module header): bundle
 * files contribute every parseable block, while dirs contribute only
 * entries OpenSSL's hashed lookup could reach (a symlink, or a
 * `HASH.N`-named file) — a stray cert file is not a store entry. There is
 * deliberately no `cert.ca` filter and no fail-closed path: an unreadable
 * source contributes nothing rather than voiding the bundle, because there
 * is no hidden deny ledger an absent read could be covering for.
 *
 * Note PEM_BLOCK_RE only matches `BEGIN CERTIFICATE` — p11-kit's
 * distrust-flagged blocks (`ca-bundle.trust.crt`, `BEGIN TRUSTED
 * CERTIFICATE`) are excluded by that label alone. The exclusion is
 * load-bearing, not a nicety: those blocks carry the deny semantics this
 * format has.
 */
export async function linuxCaRootsPem(sources: LinuxCaSources): Promise<string> {
  const texts = await Promise.all([
    ...sources.files.map(readStoreEntry),
    ...sources.dirs.map(async (dir) => {
      let entries;
      try {
        entries = await readdir(dir, { withFileTypes: true });
      } catch {
        return '';
      }
      const parts = await Promise.all(
        entries
          .filter((entry) => entry.isSymbolicLink() || HASHED_ENTRY_RE.test(entry.name))
          .map((entry) => readStoreEntry(join(dir, entry.name))),
      );
      return parts.join('\n');
    }),
  ]);
  const seen = bundledCaFingerprints();
  const roots: string[] = [];
  for (const block of texts.join('\n').match(PEM_BLOCK_RE) ?? []) {
    let fingerprint: string;
    try {
      fingerprint = new X509Certificate(block).fingerprint256;
    } catch {
      continue;
    }
    if (seen.has(fingerprint)) continue;
    seen.add(fingerprint);
    roots.push(block);
  }
  return roots.length === 0 ? '' : `${roots.join('\n')}\n`;
}

/**
 * Write the extra-CA bundle for the CONTROL PANEL child and return its path,
 * or `null` when there is nothing to add (unsupported platform, no extra
 * roots beyond Node's bundled store, no inherited bundle, or ANY failure).
 * The caller awaits this inside `bootstrap()`, whose rejection handler is
 * `app.exit(1)` — an optional TLS convenience must never be a fatal startup
 * dependency, so every failure mode collapses to `null` here rather than
 * rejecting.
 *
 * An inherited `NODE_EXTRA_CA_CERTS` (a developer's shell can carry one into an
 * unpackaged run) is MERGED into the bundle rather than dropped: the variable
 * names exactly one file, and handing the child only the collected roots would
 * silently un-trust whatever the developer had configured.
 */
export async function prepareExtraCaBundle(opts: {
  platform: NodeJS.Platform;
  /** userData — per-user writable, survives updates, same trust domain as the login keychain. */
  dir: string;
  exec: ExecText;
  inheritedPath?: string | undefined;
  /** macOS override — the keychain list to scan (tests). */
  keychains?: readonly string[];
  /** Linux override — the store paths to scan (tests). */
  linuxSources?: LinuxCaSources;
}): Promise<string | null> {
  if (opts.platform !== 'darwin' && opts.platform !== 'linux') return null;
  const parts: string[] = [];
  if (opts.inheritedPath !== undefined && opts.inheritedPath !== '') {
    try {
      parts.push(await readFile(opts.inheritedPath, 'utf8'));
    } catch {
      // A dangling inherited path is not ours to fix; the collected roots still apply.
    }
  }
  try {
    // Whole call inside the catch: collectCandidates swallows per-keychain
    // errors and linuxCaRootsPem per-path ones, but loadTrustDecisions'
    // mkdtemp/cleanup can still reject (ENOSPC), and that rejection must
    // not escape.
    parts.push(
      await (opts.platform === 'darwin'
        ? keychainCaRootsPem(opts.exec, opts.keychains ?? macosKeychains())
        : linuxCaRootsPem(opts.linuxSources ?? linuxCaSources())),
    );
  } catch (error) {
    console.warn('[mlx] could not export system CA roots:', error);
  }
  const pem = parts.join('\n').trim();
  const certCount = (pem.match(/BEGIN CERTIFICATE/g) ?? []).length;
  if (pem === '') {
    // Named, not silent: an emptied bundle is the difference between a
    // downloading app and a TLS error on an intercepting network, and this
    // line is the only place that fact is observable (see the module
    // docstring for why the child's failures carry no such detail).
    console.warn('[mlx] no extra CA roots to bundle (system store contributed none)');
    return null;
  }
  const bundlePath = join(opts.dir, EXTRA_CA_BUNDLE_FILE);
  try {
    await mkdir(opts.dir, { recursive: true });
    await writeFile(bundlePath, `${pem}\n`, { mode: 0o600 });
  } catch {
    return null;
  }
  console.log(`[mlx] extra CA bundle: ${certCount} certificate(s) → ${bundlePath}`);
  return bundlePath;
}
