/** Where an edit really lands once links on disk are followed, and which folders the work-folder rule refuses. */

import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { agentPath } from '../src/chat/agent/command-flags.js';
import { checkWorkFolder, editLanding, landingPath } from '../src/chat/work-folder.js';

let root = '';

beforeEach(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), 'landing-')));
  mkdirSync(join(root, 'work', 'src'), { recursive: true });
  mkdirSync(join(root, 'outside'), { recursive: true });
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe('landingPath', () => {
  it('a plain path lands where it is written, a new file included', async () => {
    expect(await landingPath(join(root, 'work', 'src', 'a.ts'))).toBe(join(root, 'work', 'src', 'a.ts'));
    expect(await landingPath(join(root, 'work', 'new', 'deep', 'b.ts'))).toBe(
      join(root, 'work', 'new', 'deep', 'b.ts'),
    );
  });

  it('a linked folder inside the work folder leads outside it', async () => {
    symlinkSync(join(root, 'outside'), join(root, 'work', 'docs'));
    expect(await landingPath(join(root, 'work', 'docs', 'x.plist'))).toBe(join(root, 'outside', 'x.plist'));
  });

  it('a dangling link is followed to its target', async () => {
    symlinkSync(join(root, 'outside', 'later.txt'), join(root, 'work', 'later.txt'));
    expect(await landingPath(join(root, 'work', 'later.txt'))).toBe(join(root, 'outside', 'later.txt'));
  });

  describe('`..` after a linked folder is followed on disk, as the agent writes it', () => {
    const lands = (path: string) => landingPath(agentPath(path, { cwd: join(root, 'work'), home: join(root, 'home') }));

    beforeEach(() => {
      mkdirSync(join(root, 'outside', 'deep'));
      symlinkSync(join(root, 'outside', 'deep'), join(root, 'work', 'link'));
    });

    it('an existing file', async () => {
      writeFileSync(join(root, 'outside', 'victim.txt'), 'v');
      expect(await lands('link/../victim.txt')).toBe(join(root, 'outside', 'victim.txt'));
    });

    it('a file not there yet, and a new folder whose `..` comes back', async () => {
      expect(await lands('link/../new.txt')).toBe(join(root, 'outside', 'new.txt'));
      expect(await lands('fresh/../src/new.txt')).toBe(join(root, 'work', 'src', 'new.txt'));
    });

    it('a new folder whose `..` comes back to a link on disk', async () => {
      expect(await lands('fresh/../link/new.txt')).toBe(join(root, 'outside', 'deep', 'new.txt'));
      symlinkSync(join(root, 'outside', 'victim.txt'), join(root, 'work', 'notes.md'));
      expect(await lands('fresh/more/../../notes.md')).toBe(join(root, 'outside', 'victim.txt'));
    });

    it('a dangling link inside a linked folder, its relative target read from its own folder', async () => {
      symlinkSync('../z.txt', join(root, 'outside', 'deep', 'dl'));
      expect(await lands('link/dl')).toBe(join(root, 'outside', 'z.txt'));
    });

    it('a quoted or space-led path, and the work folder repeated without its slash', async () => {
      expect(await lands('"../esc.txt"')).toBe(join(root, 'esc.txt'));
      expect(await lands(' ../esc.txt')).toBe(join(root, 'esc.txt'));
      expect(await lands(`${join(root, 'work').slice(1)}/../x.txt`)).toBe(join(root, 'x.txt'));
    });
  });

  it('a loop of links cannot be known', async () => {
    symlinkSync(join(root, 'work', 'b'), join(root, 'work', 'a'));
    symlinkSync(join(root, 'work', 'a'), join(root, 'work', 'b'));
    expect(await landingPath(join(root, 'work', 'a', 'x'))).toBeNull();
  });
});

describe('editLanding', () => {
  it('a missing file whose folder holds a name that differs only by U+00A0 or U+202F: unknown', async () => {
    writeFileSync(join(root, 'outside', 'victim.txt'), 'v');
    symlinkSync(join(root, 'outside', 'victim.txt'), join(root, 'work', 'n b.txt'));
    writeFileSync(join(root, 'work', 'shot 1.png'), 'p');
    expect(await editLanding(join(root, 'work', 'n b.txt'))).toBeNull();
    expect(await editLanding(join(root, 'work', 'shot 1.png'))).toBeNull();
    expect(await editLanding(join(root, 'work', 'shot 1.png'))).toBeNull();
  });

  it('the exact name, a new name with no look-alike, and a missing folder land as written', async () => {
    writeFileSync(join(root, 'work', 'n b.txt'), 'x');
    writeFileSync(join(root, 'work', 'n b.txt'), 'y');
    expect(await editLanding(join(root, 'work', 'n b.txt'))).toBe(join(root, 'work', 'n b.txt'));
    expect(await editLanding(join(root, 'work', 'new.txt'))).toBe(join(root, 'work', 'new.txt'));
    expect(await editLanding(join(root, 'work', 'no', 'such b.txt'))).toBe(join(root, 'work', 'no', 'such b.txt'));
  });
});

describe('checkWorkFolder', () => {
  it('a folder that holds a refused folder is refused even when it does not hold the home folder', async () => {
    mkdirSync(join(root, 'home'), { recursive: true });
    mkdirSync(join(root, 'library', 'Keychains', 'inner'), { recursive: true });
    const rules = { home: join(root, 'home'), refused: [join(root, 'library', 'Keychains')] };
    expect(await checkWorkFolder(join(root, 'library'), rules)).toEqual({ ok: false, problem: 'app-data' });
    expect(await checkWorkFolder(join(root, 'library', 'Keychains', 'inner'), rules)).toEqual({
      ok: false,
      problem: 'app-data',
    });
    expect(await checkWorkFolder(join(root, 'work'), rules)).toEqual({ ok: true, path: join(root, 'work') });
  });

  it('a folder that holds where a linked app data folder really is, is refused', async () => {
    const home = join(root, 'home');
    mkdirSync(home);
    mkdirSync(join(root, 'ext', 'data'), { recursive: true });
    mkdirSync(join(root, 'ext', 'projects', 'foo'), { recursive: true });
    symlinkSync(join(root, 'ext', 'data'), join(home, '.mlx-node'));
    const rules = { home, refused: [join(home, '.mlx-node')] };
    expect(await checkWorkFolder(join(root, 'ext'), rules)).toEqual({ ok: false, problem: 'app-data' });
    expect(await checkWorkFolder(join(root, 'ext', 'projects', 'foo'), rules)).toEqual({
      ok: true,
      path: join(root, 'ext', 'projects', 'foo'),
    });
  });

  it('a folder picked through an app data folder is refused even when a link leads out of it', async () => {
    const home = join(root, 'home');
    mkdirSync(join(home, '.mlx-node'), { recursive: true });
    mkdirSync(join(root, 'disk', '.cache', 'models'), { recursive: true });
    mkdirSync(join(root, 'disk', 'projects', 'foo'), { recursive: true });
    symlinkSync(join(root, 'disk', '.cache', 'models'), join(home, '.mlx-node', 'models'));
    const rules = { home, refused: [join(home, '.mlx-node')] };
    expect(await checkWorkFolder(join(home, '.mlx-node', 'models'), rules)).toEqual({ ok: false, problem: 'app-data' });
    expect(await checkWorkFolder(join(root, 'disk', 'projects', 'foo'), rules)).toEqual({
      ok: true,
      path: join(root, 'disk', 'projects', 'foo'),
    });
  });

  it.runIf(process.platform === 'darwin')('the top of the data volume is the top of the disk', async () => {
    mkdirSync(join(root, 'home'));
    expect(await checkWorkFolder('/System/Volumes/Data', { home: join(root, 'home'), refused: [] })).toEqual({
      ok: false,
      problem: 'root',
    });
  });
});
