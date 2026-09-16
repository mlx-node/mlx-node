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
 *   - candidates come from `security find-certificate` over the user and
 *     admin keychains, filtered to `X509Certificate.ca` (CA:TRUE) — measured
 *     on a real System.keychain: its first entry is a self-signed identity
 *     with `ca: false`, so subject===issuer is not a usable proxy. Apple's
 *     System Roots keychain is excluded (see `macosKeychains`): Node's
 *     Mozilla store already covers the public web PKI, and Apple's keychain
 *     carries platform-restricted roots an additive bundle cannot honor;
 *   - trust comes from `security trust-settings-export` (user + admin
 *     domains), which keys records by the cert's SHA-1 and carries
 *     `kSecTrustSettingsResult` per policy. A candidate needs an explicit,
 *     unconstrained allow record for the `sslServer` policy (mkcert-style
 *     tools set one), a deny record always wins, and any scoped deny makes
 *     the root ineligible. Verified against a live keychain: an installed
 *     cert with zero trust records (Blizzard Battle.net Local Cert) is NOT
 *     effectively trusted, an mkcert root with `sslServer → TrustRoot` IS.
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
  for (const { output } of outputs) {
    for (const block of output.match(PEM_BLOCK_RE) ?? []) {
      try {
        const cert = new X509Certificate(block);
        if (!cert.ca) continue;
        candidates.push({
          pem: block,
          sha1: cert.fingerprint.replaceAll(':', ''),
          sha256: cert.fingerprint256,
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
 * A domain that fails to READ rejects the whole load — this is the one
 * failure in this module that must NOT degrade locally. Decisions are merged
 * as allow-OR / deny-OR across domains, so proceeding with only the domains
 * that read successfully would silently drop the failed domain's DENIES while
 * keeping the others' allows: a root the user explicitly revoked would be
 * exported on the admin domain's say-so. Rejecting propagates to
 * `prepareExtraCaBundle`'s catch, which ships no keychain roots at all (the
 * inherited NODE_EXTRA_CA_CERTS still applies, and startup is unaffected).
 * One exception: a domain that has NO records at all exits the export with
 * errSecNoTrustSettings, which means "empty", not "unreadable" — that domain
 * contributes nothing and the other domain still loads.
 */
async function loadTrustDecisions(exec: ExecText): Promise<Map<string, TrustDecision>> {
  const merged = new Map<string, TrustDecision>();
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
        // machine whose user never touched Keychain Access trust. Only that
        // specific failure means "no records"; anything else rejects the
        // whole load per the docstring above.
        if (!/No Trust Settings were found|errSecNoTrustSettings/.test(String(error))) throw error;
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
  return merged;
}

/**
 * Concatenated PEM of every effectively SSL-trusted CA root found in the
 * given keychains, deduped by fingerprint. The selection rule:
 *
 *   - Apple system roots: included unless explicitly denied (denial is
 *     advisory here anyway — the bundle is ADDITIVE to Node's Mozilla
 *     store, which already carries the same roots);
 *   - anything else: included only with an unconstrained sslServer allow
 *     record, never when denied, and never when a SCOPED deny exists —
 *     the bundle cannot express "trusted except for host X", so a root the
 *     user distrusted for any host is not exported at all. Keychain
 *     membership alone is not trust.
 */
export async function keychainCaRootsPem(exec: ExecText, keychains: readonly string[]): Promise<string> {
  const [candidates, decisions] = await Promise.all([collectCandidates(exec, keychains), loadTrustDecisions(exec)]);
  const seen = new Set<string>();
  const roots: string[] = [];
  for (const candidate of candidates) {
    if (seen.has(candidate.sha256)) continue;
    const decision = decisions.get(candidate.sha1);
    // Every candidate here comes from a user/admin keychain, so keychain
    // membership alone is NOT trust: an explicit, unconstrained sslServer
    // allow is required, any global deny vetoes, and any scoped deny makes
    // the root ineligible (the bundle cannot express the scope).
    const trusted =
      decision?.allowForSsl === true && decision?.denyForSsl !== true && decision?.scopedDenyForSsl !== true;
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
