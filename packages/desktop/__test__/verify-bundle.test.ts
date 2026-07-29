/**
 * The release gate is the only thing between an unsigned Mach-O and a
 * notarization rejection that arrives hours later, and every one of its checks is
 * a string parsed out of a developer tool. Nothing here is hypothetical: every
 * fixture below is verbatim output captured from this repo's own artifacts on
 * 2026-07-28, because the two parse bugs this file pins were both found by running
 * the gate on a real bundle and not by reading the code.
 *
 * The two that matter:
 *
 *  - `file` output. A greedy `^(.*): .*Mach-O.*$` runs past the SECOND colon on a
 *    universal binary's header line, and splitting on `': '` misses the
 *    per-architecture lines entirely because those use `):<TAB>`. A THIN binary
 *    parses correctly under both broken forms, which is exactly why the bug
 *    survived a first pass — so the thin case is pinned here too, labelled, as the
 *    false negative it is.
 *  - The scan must sniff EVERY regular file. Both `@mariozechner` prebuilts in
 *    this repo's own node_modules are mode 644, so an exec-bit filter walks past
 *    the files most likely to be unsigned.
 */

import { execFileSync } from 'node:child_process';
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, beforeAll, describe, expect, it } from 'vite-plus/test';

import {
  buildPathLeaks,
  checkMinOs,
  checkTeamId,
  compareVersions,
  findMachOFiles,
  listRegularFiles,
  main,
  maxVersion,
  parseArgs,
  parseLoadCommandPaths,
  parseMachOPaths,
  parseMinosValues,
  parseTeamId,
  stripAppPrefix,
} from '../scripts/verify-bundle.js';

// ---------------------------------------------------------------------------
// `file` fixtures — verbatim, tabs and alignment padding included.
// ---------------------------------------------------------------------------

const UNIVERSAL_NAME = 'node_modules/@mariozechner/clipboard-darwin-universal/clipboard.darwin-universal.node';
const THIN_NAME = 'node_modules/@mariozechner/clipboard-darwin-arm64/clipboard.darwin-arm64.node';

/**
 * `file` on both `@mariozechner` prebuilts at once, exactly as `xargs -0 file`
 * invokes it. Three properties of this output are load-bearing and none are
 * obvious:
 *
 *  1. The universal binary's header line carries a SECOND colon, inside
 *     `architectures: [x86_64:...]`.
 *  2. It then emits one line PER ARCHITECTURE, and those separate the name from
 *     the description with `):` + a TAB, not `: `.
 *  3. Given more than one path, `file` pads names to align the descriptions — so
 *     the run of spaces after the colon is real and must not end up in the name.
 */
const FILE_BOTH_PREBUILTS = `${UNIVERSAL_NAME}: Mach-O universal binary with 2 architectures: [x86_64:Mach-O 64-bit dynamically linked shared library x86_64] [arm64]
${UNIVERSAL_NAME} (for architecture x86_64):\tMach-O 64-bit dynamically linked shared library x86_64
${UNIVERSAL_NAME} (for architecture arm64):\tMach-O 64-bit dynamically linked shared library arm64
${THIN_NAME}:         Mach-O 64-bit dynamically linked shared library arm64
`;

/** A slice of the real sweep over the packaged .app: aligned, and mostly not Mach-O. */
const FILE_BUNDLE_SLICE = `out/mlx-node.app/Contents/Info.plist:                                        XML 1.0 document text, ASCII text
out/mlx-node.app/Contents/MacOS/mlx-node:                                    Mach-O 64-bit executable arm64
out/mlx-node.app/Contents/Resources/native/mlx-core.darwin-arm64.node:       Mach-O 64-bit dynamically linked shared library arm64
out/mlx-node.app/Contents/Resources/app/package.json:                        JSON text data
out/mlx-node.app/Contents/Frameworks/Mantle.framework/Versions/A/Mantle:     Mach-O 64-bit dynamically linked shared library arm64
`;

// ---------------------------------------------------------------------------
// `codesign -dv` fixtures — merged stdout+stderr, which is where codesign writes.
// ---------------------------------------------------------------------------

/**
 * A file that is not signed AT ALL. One line, on stderr, and the process exits
 * non-zero — the exact case this gate exists to catch, and the one that produces
 * the least output. Captured by `codesign --remove-signature` on a copy of the
 * shipped `darwin-modifiers.node`.
 */
const CODESIGN_UNSIGNED = '/tmp/unsigned.node: code object is not signed at all\n';

/** The napi addon as it comes out of the linker: adhoc, and no Team ID. */
const CODESIGN_ADHOC = `Executable=/…/mlx-node.app/Contents/Resources/native/mlx-core.darwin-arm64.node
Identifier=mlx-core.darwin-arm64.node
Format=Mach-O thin (arm64)
CodeDirectory v=20400 size=479891 flags=0x20002(adhoc,linker-signed) hashes=14993+0 location=embedded
Signature=adhoc
Info.plist=not bound
TeamIdentifier=not set
Sealed Resources=none
Internal requirements=none
`;

/**
 * `/bin/ls`. Apple's own platform binaries report `TeamIdentifier=not set` too,
 * because they are signed under Apple's authority rather than a Team ID. That is
 * why the gate never scans outside the .app — this same output is fine there and
 * damning inside.
 */
const CODESIGN_APPLE_PLATFORM = `Executable=/bin/ls
Identifier=com.apple.ls
Format=Mach-O universal (x86_64 arm64e)
CodeDirectory v=20400 size=741 flags=0x0(none) hashes=18+2 location=embedded
Platform identifier=26
Signature size=4442
Info.plist=not bound
TeamIdentifier=not set
Sealed Resources=none
Internal requirements count=1 size=60
`;

/** What a correctly signed binary looks like. Captured from `/Applications/Ghostty.app`. */
const CODESIGN_DEVELOPER_ID = `Executable=/Applications/Ghostty.app/Contents/MacOS/ghostty
Identifier=com.mitchellh.ghostty
Format=app bundle with Mach-O universal (x86_64 arm64)
CodeDirectory v=20500 size=39809 flags=0x10000(runtime) hashes=1233+7 location=embedded
Signature size=9056
Info.plist entries=50
TeamIdentifier=24VZTF6M5V
Runtime Version=26.2.0
Sealed Resources version=2 rules=13 files=536
Internal requirements count=1 size=184
`;

// ---------------------------------------------------------------------------
// `otool -l` fixtures.
// ---------------------------------------------------------------------------

/**
 * `otool -l` on `@mariozechner/clipboard-darwin-arm64`, trimmed to its load
 * commands. Upstream's CI home is baked into LC_ID_DYLIB; this is the real reason
 * `stage-app.ts` refuses to carry the package at all.
 *
 * The LC_RPATH block at the end is not decoration. It prints `path`, not `name`,
 * and `@executable_path/../Frameworks` is the HEALTHY case — a parser that read
 * `path` lines too would report a leak on every correctly built binary.
 */
const OTOOL_LEAKY = `          cmd LC_ID_DYLIB
      cmdsize 136
         name /Users/runner/work/clipboard/clipboard/target/aarch64-apple-darwin/release/deps/libcrosscopy_clipboard.dylib (offset 24)
   time stamp 1 Thu Jan  1 08:00:01 1970
      current version 0.0.0
Load command 12
          cmd LC_LOAD_DYLIB
      cmdsize 88
         name /System/Library/Frameworks/AppKit.framework/Versions/C/AppKit (offset 24)
   time stamp 2 Thu Jan  1 08:00:02 1970
      current version 2575.60.5
Load command 13
          cmd LC_LOAD_DYLIB
      cmdsize 56
         name /usr/lib/libSystem.B.dylib (offset 24)
   time stamp 2 Thu Jan  1 08:00:02 1970
      current version 1351.0.0
Load command 14
          cmd LC_RPATH
      cmdsize 48
         path @executable_path/../Frameworks (offset 12)
`;

/** The same binary after `install_name_tool -id`, which is what package.ts does. */
const OTOOL_CLEAN = OTOOL_LEAKY.replace(
  '/Users/runner/work/clipboard/clipboard/target/aarch64-apple-darwin/release/deps/libcrosscopy_clipboard.dylib',
  '@rpath/mlx-core.darwin-arm64.node',
);

/** LC_BUILD_VERSION out of the shipped addon: a local build reports `minos 11.0`. */
const OTOOL_MINOS = `Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 11.0
      sdk 26.5
   ntools 1
     tool 3
  version 1267.0
Load command 10
      cmd LC_MAIN
`;

/** A universal binary prints one LC_BUILD_VERSION per slice, and they can disagree. */
const OTOOL_MINOS_UNIVERSAL = `x (architecture x86_64):
Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 12.0
      sdk 26.5
x (architecture arm64):
Load command 9
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 26.0
      sdk 26.5
`;

// ---------------------------------------------------------------------------

describe('parseMachOPaths', () => {
  it('takes the name off a universal binary without running past the second colon', () => {
    // THE bug. The header line is
    //   <name>: Mach-O universal binary with 2 architectures: [x86_64:...] [arm64]
    // and a greedy `^(.*): .*Mach-O.*$` matches the LAST `: ` that still leaves a
    // "Mach-O" behind it, capturing
    //   <name>: Mach-O universal binary with 2 architectures
    // as the FILE NAME. codesign is then handed a path that does not exist, and
    // the binary it was supposed to check is never looked at.
    const paths = parseMachOPaths(FILE_BOTH_PREBUILTS);
    expect(paths).toContain(UNIVERSAL_NAME);
    for (const path of paths) {
      expect(path).not.toContain('Mach-O');
      expect(path).not.toContain('architectures');
    }
  });

  it('collapses the per-architecture lines into ONE entry', () => {
    // The universal binary contributes three lines — a header plus one per slice —
    // and all three name the same file. Three entries would mean codesign runs
    // three times on it and, under a `': '` split, twice on a path with
    // ` (for architecture …)` glued to the end.
    expect(parseMachOPaths(FILE_BOTH_PREBUILTS)).toStrictEqual([THIN_NAME, UNIVERSAL_NAME]);
  });

  it('handles a thin binary — the case both broken parsers got RIGHT', () => {
    // Pinned deliberately. This is the false negative: a bundle of thin binaries
    // parses perfectly under a greedy regex and under a `': '` split, so the gate
    // looked correct until something universal turned up in it.
    const thinOnly = `${THIN_NAME}: Mach-O 64-bit dynamically linked shared library arm64\n`;
    expect(parseMachOPaths(thinOnly)).toStrictEqual([THIN_NAME]);
  });

  it('does not take the alignment padding into the name', () => {
    // `file` pads names when given more than one path, and `xargs` always gives it
    // more than one.
    expect(parseMachOPaths(FILE_BUNDLE_SLICE)).toStrictEqual([
      'out/mlx-node.app/Contents/Frameworks/Mantle.framework/Versions/A/Mantle',
      'out/mlx-node.app/Contents/MacOS/mlx-node',
      'out/mlx-node.app/Contents/Resources/native/mlx-core.darwin-arm64.node',
    ]);
  });

  it('decides on the description, not the path — a file merely NAMED Mach-O is not one', () => {
    // The shell version tested the whole `file` line for /Mach-O/, so a plain
    // text file whose NAME contained the string was handed to `codesign -dv`
    // and reported as `FAIL no Team ID`. A release blocked by a filename.
    //
    // The failure is worse than a stray line: `[2/5]` is the step that exists
    // to catch a genuinely unsigned binary, so a false entry there trains
    // whoever reads the log to treat its output as noise.
    expect(parseMachOPaths('Contents/Resources/about-Mach-O.txt: ASCII text\n')).toStrictEqual([]);

    // The other half of the same rule: a path that says nothing about Mach-O
    // must still be picked up when the DESCRIPTION does. Without this, testing
    // only the path would pass the assertion above for the wrong reason.
    expect(
      parseMachOPaths('Contents/Resources/native/addon.node: Mach-O 64-bit dynamically linked shared library arm64\n'),
    ).toStrictEqual(['Contents/Resources/native/addon.node']);
  });

  it('ignores lines that are not Mach-O, and lines with no colon at all', () => {
    expect(parseMachOPaths('a.plist: XML 1.0 document text\nb.json: JSON text data\n')).toStrictEqual([]);
    expect(parseMachOPaths('Mach-O with no colon anywhere\n')).toStrictEqual([]);
    expect(parseMachOPaths('')).toStrictEqual([]);
  });

  it('sorts by code unit so a release log does not reorder with the runner locale', () => {
    // `sort -u` under LC_COLLATE=en_US.UTF-8 orders these libEGL, libffmpeg,
    // libGLESv2; under LC_ALL=C it orders them libEGL, libGLESv2, libffmpeg. The
    // shell version inherited whichever the runner happened to have. This one does
    // not have that degree of freedom.
    const out = `libffmpeg.dylib: Mach-O 64-bit x
libGLESv2.dylib: Mach-O 64-bit x
libEGL.dylib: Mach-O 64-bit x
`;
    expect(parseMachOPaths(out)).toStrictEqual(['libEGL.dylib', 'libGLESv2.dylib', 'libffmpeg.dylib']);
  });
});

describe('parseTeamId', () => {
  it('is null for a file that is not signed at all', () => {
    // Not an empty string and not a throw: `codesign -dv` also EXITS non-zero here,
    // and in the shell version that non-zero exit had to be swallowed explicitly or
    // `set -e` aborted the scan on the first unsigned file — no FAIL line, no
    // count, no later steps, and an operator reading the abort as something else.
    expect(parseTeamId(CODESIGN_UNSIGNED)).toBeNull();
  });

  it('reads "not set" out of a linker-signed addon rather than treating it as absent', () => {
    // The distinction is worth keeping: "not set" means codesign ran and answered,
    // null means it had nothing to answer with. Both fail, for different reasons.
    expect(parseTeamId(CODESIGN_ADHOC)).toBe('not set');
    expect(parseTeamId(CODESIGN_APPLE_PLATFORM)).toBe('not set');
  });

  it('reads a real Team ID', () => {
    expect(parseTeamId(CODESIGN_DEVELOPER_ID)).toBe('24VZTF6M5V');
  });

  it('anchors at the start of the line', () => {
    // `Identifier=` and `TeamIdentifier=` differ by a prefix, and an unanchored
    // match would read the Team ID off the Identifier line.
    expect(parseTeamId('Identifier=com.example.app\n')).toBeNull();
    expect(parseTeamId('XTeamIdentifier=NOPE\n')).toBeNull();
  });
});

describe('checkTeamId', () => {
  it('fails an unsigned file and an adhoc-signed one alike', () => {
    expect(checkTeamId(null, '')).toStrictEqual({ kind: 'missing' });
    expect(checkTeamId('not set', '')).toStrictEqual({ kind: 'missing' });
    expect(checkTeamId('', '')).toStrictEqual({ kind: 'missing' });
  });

  it('accepts any Team ID when none was demanded', () => {
    // The local case: no APPLE_TEAM_ID to compare against, but "signed by
    // somebody" is still worth proving.
    expect(checkTeamId('24VZTF6M5V', '')).toStrictEqual({ kind: 'ok', found: '24VZTF6M5V' });
  });

  it('rejects a Team ID that is not ours', () => {
    // A binary signed by a DIFFERENT team passes `codesign --verify` and still
    // fails notarization.
    expect(checkTeamId('24VZTF6M5V', 'ABCDE12345')).toStrictEqual({ kind: 'mismatch', found: '24VZTF6M5V' });
    expect(checkTeamId('ABCDE12345', 'ABCDE12345')).toStrictEqual({ kind: 'ok', found: 'ABCDE12345' });
  });
});

describe('parseLoadCommandPaths / buildPathLeaks', () => {
  it('finds the build path an upstream prebuilt bakes in', () => {
    expect(buildPathLeaks(parseLoadCommandPaths(OTOOL_LEAKY))).toStrictEqual([
      '/Users/runner/work/clipboard/clipboard/target/aarch64-apple-darwin/release/deps/libcrosscopy_clipboard.dylib',
    ]);
  });

  it('does not flag system frameworks or an @rpath install name', () => {
    expect(buildPathLeaks(parseLoadCommandPaths(OTOOL_CLEAN))).toStrictEqual([]);
  });

  it('reads `name` load commands and not `path` ones', () => {
    // LC_RPATH prints `path`. `@executable_path/../Frameworks` is how a correctly
    // built bundle finds its own Frameworks directory — reading those too would
    // not leak anything, but a `path /Users/...` rpath is a different check with a
    // different remedy, and the shell version only ever looked at `name`.
    const paths = parseLoadCommandPaths(OTOOL_LEAKY);
    expect(paths).toHaveLength(3);
    expect(paths).not.toContain('@executable_path/../Frameworks');
  });

  it('strips the trailing "(offset N)" and nothing else', () => {
    expect(parseLoadCommandPaths('         name @rpath/libfoo.dylib (offset 24)\n')).toStrictEqual([
      '@rpath/libfoo.dylib',
    ]);
  });

  it('catches a /target/ path that is not under /Users/', () => {
    // CI checkouts land in /home/runner or /private/var; the giveaway is the cargo
    // target directory, wherever it sits.
    expect(buildPathLeaks(['/build/mlx-node/target/release/libmlx_core.dylib'])).toHaveLength(1);
    expect(buildPathLeaks(['/usr/lib/libSystem.B.dylib', '@rpath/x.node'])).toStrictEqual([]);
  });
});

describe('parseMinosValues / maxVersion', () => {
  it('reads the floor out of real otool output', () => {
    expect(parseMinosValues(OTOOL_MINOS)).toStrictEqual(['11.0']);
    expect(maxVersion(parseMinosValues(OTOOL_MINOS))).toBe('11.0');
  });

  it('takes the HIGHEST slice of a universal binary, not the first', () => {
    // The x86_64 block prints first and says 12.0. Trusting it would advertise an
    // app that runs on 12 while its arm64 slice needs 26.
    expect(parseMinosValues(OTOOL_MINOS_UNIVERSAL)).toStrictEqual(['12.0', '26.0']);
    expect(maxVersion(parseMinosValues(OTOOL_MINOS_UNIVERSAL))).toBe('26.0');
  });

  it('is empty when nothing carries LC_BUILD_VERSION', () => {
    // Must not become "0.0", which would read as "runs everywhere".
    expect(parseMinosValues('Load command 0\n      cmd LC_SEGMENT_64\n')).toStrictEqual([]);
    expect(maxVersion([])).toBeNull();
  });

  it('will not take the number off the following line', () => {
    // otool does not emit this, and that is the point: the pattern must not be ABLE
    // to span a line, or which field it reads depends on output nobody pinned.
    expect(parseMinosValues('    minos\n26.0\n')).toStrictEqual([]);
  });

  it('does not match `sdk` or a word that merely starts with minos', () => {
    expect(parseMinosValues('      sdk 26.5\n    minosaur 1.0\n')).toStrictEqual([]);
  });
});

describe('compareVersions', () => {
  it('orders numerically, not lexically', () => {
    // The whole point: "26.0" < "9.0" as strings, and 26.0 is the release floor.
    // `sort -V` would have handled it and was avoided as a GNU extension BSD sort
    // has only sometimes had.
    expect(compareVersions('26.0', '9.0')).toBeGreaterThan(0);
    expect(compareVersions('9.0', '12.0')).toBeLessThan(0);
    expect(maxVersion(['11.0', '26.0', '9.0', '12.0'])).toBe('26.0');
  });

  it('treats a missing component as zero', () => {
    expect(compareVersions('12', '12.0')).toBe(0);
    expect(compareVersions('12.0.0', '12.0')).toBe(0);
    expect(compareVersions('12.0.1', '12.0')).toBeGreaterThan(0);
  });
});

describe('checkMinOs', () => {
  it('passes when the plist and the binaries agree', () => {
    expect(checkMinOs('12.0', '12.0')).toStrictEqual({ note: 'ok', message: 'declares 12.0, binaries demand 12.0' });
    expect(checkMinOs('12', '12.0').note).toBe('ok');
  });

  it('fails in BOTH directions', () => {
    // Too low and the app launches where it cannot run — packager's 12.0 template
    // against an addon built at 26.0, which is the bug this step was written for.
    // Too high and it refuses to launch where it could.
    expect(checkMinOs('12.0', '26.0')).toStrictEqual({
      note: 'FAIL',
      message: 'declares 12.0 but its binaries demand 26.0',
    });
    expect(checkMinOs('26.0', '12.0').note).toBe('FAIL');
  });

  it('fails rather than guessing when either side is missing', () => {
    expect(checkMinOs('', '26.0').note).toBe('FAIL');
    expect(checkMinOs('26.0', null).note).toBe('FAIL');
  });
});

describe('stripAppPrefix', () => {
  it('shows paths relative to the bundle', () => {
    expect(stripAppPrefix('out/mlx-node.app', 'out/mlx-node.app/Contents/MacOS/mlx-node')).toBe(
      'Contents/MacOS/mlx-node',
    );
  });

  it('leaves a path that is not under the bundle alone', () => {
    expect(stripAppPrefix('out/mlx-node.app', '/elsewhere/x')).toBe('/elsewhere/x');
  });
});

describe('parseArgs', () => {
  it('takes the bundle path, the team id and the notarized flag', () => {
    expect(parseArgs(['out/x.app'], {})).toStrictEqual({
      kind: 'options',
      options: { app: 'out/x.app', teamId: '', notarized: false },
    });
    expect(parseArgs(['out/x.app', '--team-id', 'ABCDE12345', '--notarized'], {})).toStrictEqual({
      kind: 'options',
      options: { app: 'out/x.app', teamId: 'ABCDE12345', notarized: true },
    });
  });

  it('defaults the team id from APPLE_TEAM_ID, and lets the flag override it', () => {
    // The release workflow passes the secret through the environment, not argv.
    expect(parseArgs(['out/x.app'], { APPLE_TEAM_ID: 'FROMENV12' })).toMatchObject({
      options: { teamId: 'FROMENV12' },
    });
    expect(parseArgs(['out/x.app', '--team-id', 'FLAG12345'], { APPLE_TEAM_ID: 'FROMENV12' })).toMatchObject({
      options: { teamId: 'FLAG12345' },
    });
  });

  it('rejects an unknown flag instead of treating it as the bundle path', () => {
    // A typo'd flag silently becoming the .app argument is how a release gate
    // ends up verifying a directory that does not exist and reporting usage.
    expect(parseArgs(['--notarised'], {})).toStrictEqual({ kind: 'usage', message: 'unknown flag: --notarised' });
  });

  it('rejects --team-id with no value', () => {
    expect(parseArgs(['out/x.app', '--team-id'], {}).kind).toBe('usage');
  });

  it('needs a bundle path', () => {
    expect(parseArgs([], {}).kind).toBe('usage');
    expect(parseArgs(['--notarized'], {}).kind).toBe('usage');
  });
});

describe('main', () => {
  it('exits 2 on a usage error rather than 1', () => {
    // 1 means the gate FAILED, which is a release-blocking verdict about the
    // bundle. A mistyped command line must never be reported as that.
    expect(main([], {})).toBe(2);
    expect(main(['--nope'], {})).toBe(2);
    expect(main(['/no/such/bundle.app'], {})).toBe(2);
  });
});

/**
 * The scan itself, against a real directory tree.
 *
 * `/bin/echo` is a genuine universal Mach-O, so one fixture covers both hazards at
 * once: it is copied in at mode 644 — the mode both `.node` prebuilts in this
 * repo's node_modules actually have — and its `file` output is the multi-line
 * per-architecture form.
 */
describe('findMachOFiles', () => {
  let root = '';

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'verify-bundle-test-'));
    // A universal Mach-O with NO exec bit, named like the addon. `find -perm +111`
    // does not see this file; the notary does.
    copyFileSync('/bin/echo', join(root, 'addon.node'));
    chmodSync(join(root, 'addon.node'), 0o644);
    // A dylib, also not executable — the usual mode for one.
    copyFileSync('/bin/echo', join(root, 'libthing.dylib'));
    chmodSync(join(root, 'libthing.dylib'), 0o644);
    // An executable Mach-O, nested, so the walk has to recurse to reach it.
    mkdirSync(join(root, 'nested'));
    copyFileSync('/bin/echo', join(root, 'nested', 'tool'));
    chmodSync(join(root, 'nested', 'tool'), 0o755);
    writeFileSync(join(root, 'notes.txt'), 'not a binary\n');
    // `find -type f` neither reports symlinks nor follows them, and neither does
    // the walk. Signing the target twice through two names is how an inside-out
    // signing order gets undone.
    symlinkSync(join(root, 'addon.node'), join(root, 'addon-link.node'));
  });

  afterAll(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('finds Mach-O files that carry no exec bit', () => {
    // Scanning by `-perm +111` is the obvious approach and it is WRONG. Both of
    // these are mode 644 and both would be invisible to it — and a `.node` is
    // precisely the kind of file that reaches a bundle unsigned.
    const found = findMachOFiles(root);
    expect(found).toContain(join(root, 'addon.node'));
    expect(found).toContain(join(root, 'libthing.dylib'));
  });

  it('recurses into subdirectories', () => {
    expect(findMachOFiles(root)).toContain(join(root, 'nested', 'tool'));
  });

  it('does not report files that are not Mach-O', () => {
    expect(findMachOFiles(root)).not.toContain(join(root, 'notes.txt'));
  });

  it('does not follow symlinks, so a binary is never checked twice', () => {
    expect(findMachOFiles(root)).not.toContain(join(root, 'addon-link.node'));
  });

  it('lists every regular file, whatever its mode', () => {
    const listed = listRegularFiles(root);
    expect(listed).toContain(join(root, 'addon.node'));
    expect(listed).toContain(join(root, 'notes.txt'));
  });

  it('agrees with `find -type f` on the same tree', () => {
    // The port replaced `find | xargs` with a readdir walk, so the walk is pinned
    // against the thing it replaced rather than against my belief about it.
    const fromFind = execFileSync('find', [root, '-type', 'f'], { encoding: 'utf-8' })
      .split('\n')
      .filter((line) => line !== '')
      .sort();
    expect(listRegularFiles(root).sort()).toStrictEqual(fromFind);
  });
});
