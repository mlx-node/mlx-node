/**
 * The bundled skills: the shipped manifest matches the shipped files, the Chat prompt ends with the skill body, and
 * the installer writes exactly the manifest's files, replaces a tampered copy, clears crash leftovers and removes
 * unwanted folders, while a corrupt bundle (bad hash, link, path that leaves the folder) installs nothing.
 */

import { createHash } from 'node:crypto';
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { CHAT_PROMPT_HEAD, chatPrompt, skillBody } from '../src/chat/agent/profiles.js';
import { INSTALLED_FILE, SkillsInstaller } from '../src/chat/skills/install.js';
import {
  CHAT_SKILL_BUNDLE,
  CHAT_SKILL_FILE,
  SkillBundleError,
  bundlePathSegments,
  defaultSkillsDir,
  loadManifest,
  parseManifest,
  readChatSkill,
} from '../src/chat/skills/manifest.js';

let base: string;
let skillsDir: string;
let pluginsDir: string;

function installer(): SkillsInstaller {
  return new SkillsInstaller({ skillsDir, pluginsDir, now: () => 1_000 });
}

function manifestText(): string {
  return readFileSync(join(skillsDir, 'manifest.json'), 'utf8');
}

function editManifest(edit: (m: { bundles: Record<string, { files: Array<Record<string, unknown>> }> }) => void): void {
  const manifest = JSON.parse(manifestText()) as { bundles: Record<string, { files: Array<Record<string, unknown>> }> };
  edit(manifest);
  writeFileSync(join(skillsDir, 'manifest.json'), JSON.stringify(manifest));
}

const SKILL_SOURCE = (): string => join(skillsDir, 'own', 'chat-web', CHAT_SKILL_FILE);

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'mlx-chat-skills-'));
  skillsDir = join(base, 'skills');
  pluginsDir = join(base, 'root', 'plugins');
  cpSync(defaultSkillsDir(), skillsDir, { recursive: true });
});

afterEach(() => {
  rmSync(base, { recursive: true, force: true });
});

describe('the shipped bundle', () => {
  it('matches its manifest file for file', () => {
    const dir = defaultSkillsDir();
    const manifest = loadManifest(dir);
    for (const [id, bundle] of Object.entries(manifest.bundles)) {
      const root = join(dir, ...bundle.root.split('/'));
      const listed = bundle.files.map((f) => f.path).sort();
      const onDisk: string[] = [];
      const walk = (d: string, prefix: string): void => {
        for (const e of readdirSync(d, { withFileTypes: true })) {
          const rel = prefix === '' ? e.name : `${prefix}/${e.name}`;
          if (e.isDirectory()) walk(join(d, e.name), rel);
          else onDisk.push(rel);
        }
      };
      walk(root, '');
      expect([id, onDisk.sort()]).toEqual([id, listed]);
      for (const file of bundle.files) {
        const data = readFileSync(join(root, file.path));
        expect([file.path, data.length, createHash('sha256').update(data).digest('hex')]).toEqual([
          file.path,
          file.bytes,
          file.sha256,
        ]);
      }
    }
    expect(manifest.bundles[CHAT_SKILL_BUNDLE]!.files.map((f) => f.path)).toContain(CHAT_SKILL_FILE);
  });

  it('gives the Chat prompt: the head, a blank line, then the skill without its frontmatter', () => {
    const skill = readChatSkill();
    expect(skill.startsWith('---\nname: web-tools\n')).toBe(true);
    const prompt = chatPrompt(skill);
    expect(prompt.startsWith(`${CHAT_PROMPT_HEAD}\n\n`)).toBe(true);
    expect(prompt.endsWith(skillBody(skill))).toBe(true);
    expect(skillBody(skill).startsWith('---')).toBe(false);
    expect(skillBody(skill)).toContain('web_search');
  });

  it('accepts only the allowed paths', () => {
    expect(bundlePathSegments('skills/web-tools/SKILL.md')).toEqual(['skills', 'web-tools', 'SKILL.md']);
    expect(bundlePathSegments('skills/a/references/b.md')).toEqual(['skills', 'a', 'references', 'b.md']);
    expect(bundlePathSegments('LICENSE')).toEqual(['LICENSE']);
    for (const bad of [
      '../x',
      '/etc/passwd',
      'skills/../SKILL.md',
      'skills/a/../../SKILL.md',
      'skills/A/SKILL.md',
      'skills/a/run.sh',
      'skills/a/references/x.sh',
      'skills/a/references/.md',
      'skills//SKILL.md',
      'skills/a/SKILL.md/x',
    ]) {
      expect([bad, bundlePathSegments(bad)]).toEqual([bad, null]);
    }
    expect(() =>
      parseManifest('{"version":1,"bundles":{"x":{"root":"../up","upstream":null,"skills":[],"files":[]}}}'),
    ).toThrow(SkillBundleError);
    expect(() => parseManifest('not json')).toThrow(SkillBundleError);
  });
});

describe('SkillsInstaller', () => {
  it('installs exactly the manifest files and a record, then verifies it', async () => {
    const skills = installer();
    expect(await skills.sync([CHAT_SKILL_BUNDLE])).toEqual({ [CHAT_SKILL_BUNDLE]: 'ok' });
    const target = join(pluginsDir, CHAT_SKILL_BUNDLE);
    expect(readFileSync(join(target, CHAT_SKILL_FILE))).toEqual(readFileSync(SKILL_SOURCE()));
    expect(JSON.parse(readFileSync(join(target, INSTALLED_FILE), 'utf8'))).toMatchObject({
      version: 1,
      id: CHAT_SKILL_BUNDLE,
      bundle: 'own/chat-web',
      installedAt: 1_000,
    });
    expect(statSync(pluginsDir).mode & 0o777).toBe(0o700);
    expect(statSync(join(target, CHAT_SKILL_FILE)).mode & 0o777).toBe(0o600);
    expect(await skills.verified(CHAT_SKILL_BUNDLE)).toBe(true);
    expect(readdirSync(pluginsDir)).toEqual([CHAT_SKILL_BUNDLE]);
  });

  it('replaces a copy with a changed file, an extra file or a link', async () => {
    const skills = installer();
    const target = join(pluginsDir, CHAT_SKILL_BUNDLE);
    const tamper: Array<() => void> = [
      () => writeFileSync(join(target, CHAT_SKILL_FILE), 'changed'),
      () => writeFileSync(join(target, 'skills', 'web-tools', 'extra.md'), 'x'),
      () => symlinkSync('/etc', join(target, 'skills', 'link')),
      () => writeFileSync(join(target, INSTALLED_FILE), '{}'),
    ];
    for (const change of tamper) {
      await skills.sync([CHAT_SKILL_BUNDLE]);
      change();
      expect(await skills.verified(CHAT_SKILL_BUNDLE)).toBe(false);
      expect(await skills.sync([CHAT_SKILL_BUNDLE])).toEqual({ [CHAT_SKILL_BUNDLE]: 'ok' });
      expect(await skills.verified(CHAT_SKILL_BUNDLE)).toBe(true);
      expect(readFileSync(join(target, CHAT_SKILL_FILE))).toEqual(readFileSync(SKILL_SOURCE()));
    }
  });

  it('clears crash leftovers and removes folders nobody wants', async () => {
    mkdirSync(join(pluginsDir, '.tmp-chat-web-abc'), { recursive: true });
    mkdirSync(join(pluginsDir, '.old-chat-web-abc'), { recursive: true });
    mkdirSync(join(pluginsDir, 'retired-provider'), { recursive: true });
    await installer().sync([CHAT_SKILL_BUNDLE]);
    expect(readdirSync(pluginsDir)).toEqual([CHAT_SKILL_BUNDLE]);
    await installer().sync([]);
    expect(readdirSync(pluginsDir)).toEqual([]);
  });

  it('installs nothing from a bundle with a wrong hash, a link or an unknown id', async () => {
    writeFileSync(SKILL_SOURCE(), 'not the shipped text');
    expect(await installer().sync([CHAT_SKILL_BUNDLE])).toEqual({ [CHAT_SKILL_BUNDLE]: 'skill-bundle-corrupt' });
    expect(existsSync(join(pluginsDir, CHAT_SKILL_BUNDLE))).toBe(false);
    expect(() => readChatSkill(skillsDir)).toThrow(SkillBundleError);

    cpSync(join(defaultSkillsDir(), 'own'), join(skillsDir, 'own'), { recursive: true });
    rmSync(SKILL_SOURCE());
    symlinkSync(join(defaultSkillsDir(), 'own', 'chat-web', CHAT_SKILL_FILE), SKILL_SOURCE());
    expect(await installer().sync([CHAT_SKILL_BUNDLE])).toEqual({ [CHAT_SKILL_BUNDLE]: 'skill-bundle-corrupt' });
    expect(readdirSync(pluginsDir)).toEqual([]);

    expect(await installer().sync(['../escape', 'missing'])).toEqual({
      '../escape': 'skill-bundle-corrupt',
      missing: 'skill-bundle-corrupt',
    });
    expect(existsSync(join(base, 'root', 'escape'))).toBe(false);
  });

  it('refuses a manifest whose path leaves the bundle folder', async () => {
    editManifest((m) => {
      m.bundles[CHAT_SKILL_BUNDLE]!.files[0]!.path = 'skills/../../../manifest.json';
    });
    expect(await installer().sync([CHAT_SKILL_BUNDLE])).toEqual({ [CHAT_SKILL_BUNDLE]: 'skill-bundle-corrupt' });
    expect(readdirSync(pluginsDir)).toEqual([]);
  });
});
