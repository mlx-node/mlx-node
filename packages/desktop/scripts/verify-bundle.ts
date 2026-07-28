/**
 * Release gate for the packaged .app.
 *
 * The reason this exists: `codesign --verify --deep --strict` PASSES on a bundle
 * containing an unsigned Mach-O under Contents/Resources, and the notary service
 * rejects that same bundle hours later. Local verification is not a predictor of
 * notarization, so the release path needs the check codesign does not do —
 * "every Mach-O in here carries OUR Team ID".
 *
 * Measured on the artifacts this app ships (2026-07-27):
 *
 *   codesign -dv packages/core/npm/darwin-arm64/mlx-core.darwin-arm64.node
 *     Identifier=libmlx_core.dylib          <- a build artifact name, not a bundle id
 *     Signature=adhoc                        <- linker-signed
 *     TeamIdentifier=not set                 <- the notary refuses this
 *
 *   otool -l ... | grep LC_ID_DYLIB
 *     name /Users/<user>/workspace/github/mlx-node/target/.../libmlx_core.dylib
 *                                            <- ships a home directory to every user
 *
 * Both are fixed before signing (install_name_tool -id, codesign --identifier);
 * this script is what proves the fix actually happened.
 *
 *   usage: yarn oxnode packages/desktop/scripts/verify-bundle.ts <path-to-.app> [--team-id ID] [--notarized]
 *
 * --notarized adds the post-submission checks. Omit it for a locally signed build.
 *
 * Everything above `main()` is pure: string in, verdict out. That split is not
 * tidiness — the parsers below are the part that has actually been wrong before,
 * and a parser that can only be exercised by producing a signed bundle is a
 * parser nobody exercises.
 */

import { spawnSync } from 'node:child_process';
import type { Dirent } from 'node:fs';
import { closeSync, mkdtempSync, openSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// Pure: parsing
// ---------------------------------------------------------------------------

/**
 * Mach-O paths out of `file` output, deduped and sorted.
 *
 * Parsing `file` output for NAMES is fiddly and two obvious forms are both wrong.
 * A regex like `/^(.*): .*Mach-O.*$/` is WRONG here and was: `.*` is greedy, and
 * `file` prints
 *
 *   path: Mach-O universal binary with 2 architectures: [x86_64:...] [arm64:...]
 *
 * so the capture ran past the SECOND colon. Splitting on `': '` fails too, because
 * a universal binary emits one line PER ARCHITECTURE and those use a different
 * separator entirely:
 *
 *   path (for architecture x86_64):<TAB>Mach-O 64-bit dynamically linked ...
 *
 * So: cut at the FIRST colon, strip a trailing " (for architecture …)", dedupe.
 * Thin binaries parse fine under both broken versions, which is exactly why this
 * survived the first pass.
 *
 * The "does this line describe a Mach-O" test is `includes('Mach-O')` over the
 * WHOLE line, matching the shell version this replaced. That is deliberately
 * faithful rather than correct — see the note in `__test__/verify-bundle.test.ts`.
 */
export function parseMachOPaths(fileOutput: string): string[] {
  const found = new Set<string>();
  for (const line of fileOutput.split('\n')) {
    const colon = line.indexOf(':');
    if (colon === -1) continue;
    // Only the DESCRIPTION decides, never the path. `file` prints
    // `<path>: <description>`, and testing the whole line meant a plain text
    // file merely NAMED `about-Mach-O.txt` was handed to `codesign -dv` and
    // reported as `FAIL no Team ID` — a release blocked by a filename.
    if (!line.slice(colon + 1).includes('Mach-O')) continue;
    found.add(line.slice(0, colon).replace(/ \(for architecture [^)]*\)$/, ''));
  }
  // Code-unit order, not `localeCompare`: this list drives the order of the FAIL
  // lines, and a locale-sensitive sort would reorder a release log depending on
  // the runner's LC_COLLATE.
  return [...found].sort();
}

/**
 * The `TeamIdentifier=` value from `codesign -dv`, or `null` when the tool printed
 * none — which includes the case the gate exists for, a file that is not signed at
 * all and whose only output is `<path>: code object is not signed at all`.
 *
 * Reads MERGED stdout+stderr, because codesign writes all of it to stderr.
 */
export function parseTeamId(codesignOutput: string): string | null {
  const prefix = 'TeamIdentifier=';
  const values = codesignOutput
    .split('\n')
    .filter((line) => line.startsWith(prefix))
    .map((line) => line.slice(prefix.length));
  return values.length === 0 ? null : values.join('\n');
}

/** Dylib/framework paths out of `otool -l`, in load-command order, duplicates kept. */
export function parseLoadCommandPaths(otoolOutput: string): string[] {
  const paths: string[] = [];
  for (const line of otoolOutput.split('\n')) {
    // Only `name`, not `path`: `path` is LC_RPATH, and an rpath of
    // `@executable_path/../Frameworks` is the healthy case, not a leak.
    const match = /^ *name (.*) \(offset.*$/.exec(line);
    if (match !== null) paths.push(match[1]);
  }
  return paths;
}

/** `minos` values out of `otool -l`. A universal binary contributes one per slice. */
export function parseMinosValues(otoolOutput: string): string[] {
  const versions: string[] = [];
  for (const line of otoolOutput.split('\n')) {
    if (!/^ *minos /.test(line)) continue;
    const value = line.trim().split(/[ \t]+/)[1];
    if (value !== undefined && value !== '') versions.push(value);
  }
  return versions;
}

// ---------------------------------------------------------------------------
// Pure: versions
//
// A private implementation rather than an import from `min-os.ts`, which is what
// `package.ts` uses to WRITE the plist. Step [5/5] compares the two derivations,
// so sharing the comparator would make them agree by construction on the one class
// of bug — a mis-parse — that the step is there to catch.
//
// `sort -V` was avoided in the shell version because version sort is a GNU
// extension BSD sort has only sometimes had; here the ordering is explicit.
// ---------------------------------------------------------------------------

/** Numeric prefix of a version component, `0` when there is none — awk's coercion. */
function componentValue(component: string | undefined): number {
  const value = Number.parseFloat(component ?? '');
  return Number.isNaN(value) ? 0 : value;
}

/** Component-wise numeric compare. Missing components are 0, so `12` === `12.0`. */
export function compareVersions(a: string, b: string): number {
  const left = a.split('.');
  const right = b.split('.');
  for (let i = 0; i < Math.max(left.length, right.length); i++) {
    const diff = componentValue(left[i]) - componentValue(right[i]);
    if (diff !== 0) return diff < 0 ? -1 : 1;
  }
  return 0;
}

/**
 * The highest version, or `null` for an empty list.
 *
 * Highest and not lowest: every binary in the bundle has to load, so the bundle
 * can only run where the most demanding one runs.
 */
export function maxVersion(versions: readonly string[]): string | null {
  if (versions.length === 0) return null;
  return versions.reduce((best, v) => (compareVersions(v, best) > 0 ? v : best));
}

// ---------------------------------------------------------------------------
// Pure: verdicts
// ---------------------------------------------------------------------------

/** `ok` prints nothing; `FAIL` prints and fails the gate; `skip`/`scanned` are notes. */
export type Note = 'ok' | 'FAIL' | 'skip' | 'scanned';

export interface Verdict {
  readonly note: Note;
  readonly message: string;
}

export type TeamIdVerdict =
  | { readonly kind: 'missing' }
  | { readonly kind: 'mismatch'; readonly found: string }
  | { readonly kind: 'ok'; readonly found: string };

/**
 * `expected` empty means "any Team ID will do" — the local case, where there is no
 * APPLE_TEAM_ID to compare against but "signed by SOMEBODY" is still worth proving.
 *
 * `TeamIdentifier=not set` is only damning INSIDE our bundle. Apple's own platform
 * binaries report exactly the same thing because they are signed under Apple's
 * authority rather than a Team ID — which is why nothing here ever scans outside
 * the .app.
 */
export function checkTeamId(found: string | null, expected: string): TeamIdVerdict {
  if (found === null || found === '' || found === 'not set') return { kind: 'missing' };
  if (expected !== '' && found !== expected) return { kind: 'mismatch', found };
  return { kind: 'ok', found };
}

/**
 * Load-command paths that ship a build machine's layout.
 *
 * Not a signing failure, but it hands a developer's home directory (and the CI
 * layout) to every user, and it is invisible in the UI.
 */
export function buildPathLeaks(paths: readonly string[]): string[] {
  return paths.filter((p) => p.startsWith('/Users/') || p.includes('/target/'));
}

/**
 * `LSMinimumSystemVersion` against the highest `minos` in the bundle.
 *
 * Both directions are bugs: too low and the app launches where it cannot run, too
 * high and it refuses to launch where it could.
 */
export function checkMinOs(declared: string, required: string | null): Verdict {
  if (declared === '') {
    return { note: 'FAIL', message: "Info.plist has no LSMinimumSystemVersion — packager's default would apply" };
  }
  if (required === null) {
    return { note: 'FAIL', message: 'no LC_BUILD_VERSION on any Mach-O — cannot tell what the floor should be' };
  }
  if (compareVersions(declared, required) !== 0) {
    return { note: 'FAIL', message: `declares ${declared} but its binaries demand ${required}` };
  }
  return { note: 'ok', message: `declares ${declared}, binaries demand ${required}` };
}

/**
 * The bundle-relative path used in FAIL lines: a literal prefix strip, not
 * `path.relative`. The .app argument is echoed back exactly as the operator typed
 * it, so a relative argument produces relative FAIL lines and an absolute one
 * produces absolute ones — the paths in the log then match what was on the command
 * line. (A trailing slash on the argument defeats the strip and the full path is
 * printed; the shell version behaved identically.)
 */
export function stripAppPrefix(app: string, path: string): string {
  const prefix = `${app}/`;
  return path.startsWith(prefix) ? path.slice(prefix.length) : path;
}

export interface Options {
  readonly app: string;
  readonly teamId: string;
  readonly notarized: boolean;
}

export type ParsedArgs =
  | { readonly kind: 'options'; readonly options: Options }
  | { readonly kind: 'usage'; readonly message: string };

/**
 * `verify-bundle <path-to-.app> [--team-id ID] [--notarized]`.
 *
 * Anything starting with `-` that is not one of the two flags is rejected rather
 * than taken as a path, so a typo'd flag cannot silently become the bundle
 * argument. A repeated positional keeps the LAST one, matching the shell version.
 */
export function parseArgs(argv: readonly string[], env: { readonly APPLE_TEAM_ID?: string }): ParsedArgs {
  let app = '';
  let teamId = env.APPLE_TEAM_ID ?? '';
  let notarized = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--team-id') {
      const value = argv[i + 1];
      if (value === undefined) return { kind: 'usage', message: '--team-id needs a value' };
      teamId = value;
      i++;
    } else if (arg === '--notarized') {
      notarized = true;
    } else if (arg.startsWith('-')) {
      return { kind: 'usage', message: `unknown flag: ${arg}` };
    } else {
      app = arg;
    }
  }

  if (app === '') return { kind: 'usage', message: '' };
  return { kind: 'options', options: { app, teamId, notarized } };
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------

/** Enough for `otool -l` on the 233 MB addon (~10 KB) with several orders to spare. */
const MAX_BUFFER = 256 * 1024 * 1024;

/**
 * `find <root> -type f`: regular files only, symlinks neither reported nor
 * followed. `readdirSync(withFileTypes)` uses lstat semantics, so a symlink is
 * `isSymbolicLink()` and neither `isFile()` nor `isDirectory()` — same exclusion,
 * and the same refusal to descend through a symlinked directory.
 *
 * EVERY regular file, with no permission filter. Selecting by `-perm +111` is the
 * obvious approach and it is WRONG: a `.node` is often not executable and a
 * `.dylib` usually is not (both prebuilts under `node_modules/@mariozechner` are
 * mode 644), so an exec-bit filter walks straight past the files most likely to be
 * unsigned.
 */
export function listRegularFiles(root: string): string[] {
  const files: string[] = [];
  const stack: string[] = [root];
  for (let dir = stack.pop(); dir !== undefined; dir = stack.pop()) {
    let entries: Dirent[];
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      // `find` warns and carries on; so does this. A directory we cannot read is
      // reported by the steps that then find nothing in it.
      continue;
    }
    for (const entry of entries) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) stack.push(path);
      else if (entry.isFile()) files.push(path);
    }
  }
  return files;
}

/**
 * Batch size for the `file` sweep, in bytes of argv.
 *
 * ONE `file` pass over the whole bundle, batched. The obvious implementation —
 * one `file` per path — spawns a process per file, and this bundle has thousands;
 * two such passes took over two minutes and were killed. Batching turns it into a
 * handful of execs. Well under macOS's 1 MB ARG_MAX, which also has to hold the
 * environment.
 */
const ARGV_BUDGET = 128 * 1024;

/** Every Mach-O under `root`, deduped and sorted. */
export function findMachOFiles(root: string): string[] {
  const files = listRegularFiles(root);
  let output = '';
  for (let i = 0; i < files.length; ) {
    const batch: string[] = [];
    let bytes = 0;
    while (i < files.length && (batch.length === 0 || bytes + files[i].length + 1 < ARGV_BUDGET)) {
      bytes += files[i].length + 1;
      batch.push(files[i]);
      i++;
    }
    const result = spawnSync('file', batch, { encoding: 'utf-8', maxBuffer: MAX_BUFFER });
    if (result.error !== undefined) throw result.error;
    // `file` reports unreadable paths on stderr; `xargs -0 file` let those through
    // to the caller's stderr rather than swallowing them, and so does this.
    if (result.stderr !== '') process.stderr.write(result.stderr);
    output += result.stdout;
  }
  return parseMachOPaths(output);
}

interface Merged {
  readonly status: number;
  readonly output: string;
}

/**
 * Run a tool with stdout and stderr on the SAME descriptor — a real `2>&1`, not
 * two pipes concatenated afterwards. codesign and spctl write everything to
 * stderr, `stapler` everything to stdout, and a future tool that interleaves the
 * two would be reordered by any implementation that captured them separately.
 *
 * A non-zero exit is a RESULT here, never a throw. `codesign -dv` exits non-zero
 * on a file that is not signed at all — the exact case this gate exists to catch —
 * and in the shell version that needed an explicit `|| true` or `set -e` aborted
 * the scan mid-way, producing no FAIL line, no count and no later steps.
 */
function runMerged(command: string, args: readonly string[]): Merged {
  const dir = mkdtempSync(join(tmpdir(), 'verify-bundle-'));
  const path = join(dir, 'merged');
  const fd = openSync(path, 'w');
  let result;
  try {
    result = spawnSync(command, args, { stdio: ['ignore', fd, fd], maxBuffer: MAX_BUFFER });
  } finally {
    closeSync(fd);
  }
  try {
    const output = readFileSync(path, 'utf-8');
    // A tool that is not on PATH is a failed check, not a crash — `command not
    // found` is 127 in a shell and it stays 127 here.
    if (result.error !== undefined) return { status: 127, output: `${output}${command}: ${result.error.message}\n` };
    // `null` means killed by a signal; that is a failure, not a success.
    return { status: result.status ?? 1, output };
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

/** stdout only, stderr discarded — the `2>/dev/null` the shell version used. */
function runQuiet(command: string, args: readonly string[]): string {
  const result = spawnSync(command, args, {
    encoding: 'utf-8',
    maxBuffer: MAX_BUFFER,
    stdio: ['ignore', 'pipe', 'ignore'],
  });
  return result.stdout ?? '';
}

/** Lines of a tool's output, the way `sed` sees them: no phantom line after a trailing newline. */
function toLines(text: string): string[] {
  if (text === '') return [];
  const lines = text.split('\n');
  if (lines[lines.length - 1] === '') lines.pop();
  return lines;
}

function note(tag: Note, message: string): void {
  console.log(`  ${tag.padEnd(8)} ${message}`);
}

function echoIndented(text: string, pad: string): void {
  for (const line of toLines(text)) console.log(pad + line);
}

const USAGE = 'usage: verify-bundle <path-to-.app> [--team-id ID] [--notarized]';

export function main(argv: readonly string[], env: NodeJS.ProcessEnv): number {
  const parsed = parseArgs(argv, env);
  if (parsed.kind === 'usage') {
    if (parsed.message !== '') console.error(parsed.message);
    console.error(USAGE);
    return 2;
  }
  const { app, teamId, notarized } = parsed.options;
  if (!isDirectory(app)) {
    console.error(USAGE);
    return 2;
  }

  let fail = false;
  console.log(`==> ${app}`);

  // -------------------------------------------------------------------------
  // 1. What codesign DOES check. Necessary, and nowhere near sufficient.
  // -------------------------------------------------------------------------
  console.log('[1/5] codesign --verify --deep --strict');
  const seal = runMerged('codesign', ['--verify', '--deep', '--strict', '--verbose=4', app]);
  echoIndented(seal.output, '  ');
  if (seal.status === 0) {
    note('ok', 'bundle seal intact');
  } else {
    note('FAIL', 'bundle seal broken');
    fail = true;
  }

  // -------------------------------------------------------------------------
  // 2. The check codesign does NOT do.
  // -------------------------------------------------------------------------
  console.log('[2/5] every Mach-O carries a Team ID');
  const machO = findMachOFiles(app);
  for (const file of machO) {
    const verdict = checkTeamId(parseTeamId(runMerged('codesign', ['-dv', file]).output), teamId);
    if (verdict.kind === 'missing') {
      note('FAIL', `no Team ID: ${stripAppPrefix(app, file)}`);
      fail = true;
    } else if (verdict.kind === 'mismatch') {
      note('FAIL', `Team ID ${verdict.found} != ${teamId}: ${stripAppPrefix(app, file)}`);
      fail = true;
    }
  }
  note('scanned', `${machO.length} Mach-O files`);
  if (machO.length === 0) {
    note('FAIL', 'found no Mach-O at all -- wrong path?');
    fail = true;
  }

  // `otool -l` once per binary and reused by steps 3 and 5. The shell version ran
  // it twice; the output is the same both times, and step 5 is independent of step
  // 3 in what it DERIVES, which is the independence that matters.
  const loadCommands = new Map(machO.map((file) => [file, runQuiet('otool', ['-l', file])]));

  // -------------------------------------------------------------------------
  // 3. Build-path leak.
  // -------------------------------------------------------------------------
  console.log('[3/5] no build paths baked into load commands');
  for (const file of machO) {
    const leaks = buildPathLeaks(parseLoadCommandPaths(loadCommands.get(file) ?? ''));
    if (leaks.length > 0) {
      note('FAIL', `leaks a build path: ${stripAppPrefix(app, file)}`);
      for (const leak of leaks) console.log(`           ${leak}`);
      fail = true;
    }
  }

  // -------------------------------------------------------------------------
  // 4. Post-notarization only. Gatekeeper assessment + a stapled ticket, so the
  //    app opens on a machine that has never seen it and is offline.
  // -------------------------------------------------------------------------
  console.log('[4/5] notarization');
  if (notarized) {
    const gatekeeper = runMerged('spctl', ['-a', '-vvv', '-t', 'install', app]);
    echoIndented(gatekeeper.output, '  ');
    if (gatekeeper.status === 0) {
      note('ok', 'gatekeeper accepts');
    } else {
      note('FAIL', 'gatekeeper rejects');
      fail = true;
    }
    const ticket = runMerged('xcrun', ['stapler', 'validate', app]);
    echoIndented(ticket.output, '  ');
    if (ticket.status === 0) {
      note('ok', 'ticket stapled');
    } else {
      note('FAIL', 'no stapled ticket — the app needs network on first launch');
      fail = true;
    }
  } else {
    note('skip', 'pass --notarized after the notary returns');
  }

  // -------------------------------------------------------------------------
  // 5. LSMinimumSystemVersion vs what the binaries actually demand.
  //
  // @electron/packager writes Electron's template floor (12.0) regardless of what
  // the payload was built against. Release builds set MACOSX_DEPLOYMENT_TARGET=26.0,
  // so the addon's LC_BUILD_VERSION says minos 26.0 and dyld refuses to map it on
  // 12-15 -- but LaunchServices reads the plist, so the app OPENS there and only
  // then fails to load the addon. The supervisor sees a dead sidecar and restarts
  // it, forever, with nothing naming the real cause.
  //
  // Computed over EVERY Mach-O here, while package.ts sets the plist from three
  // representative binaries. Deliberately two independent derivations: if they
  // agreed by construction this step would only be able to pass.
  // -------------------------------------------------------------------------
  console.log("[5/5] LSMinimumSystemVersion matches the binaries' build floor");
  const plist = join(app, 'Contents', 'Info.plist');
  // plutil reports a missing key on stderr and prints nothing to stdout, so an
  // absent LSMinimumSystemVersion arrives here as '' rather than as an error string.
  const extract = ['-extract', 'LSMinimumSystemVersion', 'raw', '-o', '-', plist];
  const declared = stripTrailingNewlines(runQuiet('plutil', extract));
  const required = maxVersion(machO.flatMap((file) => parseMinosValues(loadCommands.get(file) ?? '')));
  const verdict = checkMinOs(declared, required);
  note(verdict.note, verdict.message);
  if (verdict.note === 'FAIL') fail = true;

  console.log('');
  if (fail) {
    console.log('BUNDLE GATE: FAIL');
    return 1;
  }
  console.log('BUNDLE GATE: PASS');
  return 0;
}

/** What `$(…)` does to a command substitution: strip trailing newlines, nothing else. */
function stripTrailingNewlines(text: string): string {
  return text.replace(/\n+$/, '');
}

/** `[ -d "$APP" ]`: follows symlinks, and a missing path is false rather than a throw. */
function isDirectory(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

// Guarded so the parsers above can be imported by a test without the gate running
// and calling process.exit under the test runner.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  process.exit(main(process.argv.slice(2), process.env));
}
