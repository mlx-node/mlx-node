/**
 * The skills bundle as shipped: every file on disk is in the manifest with its hash and size and nothing else is;
 * only allowed paths, no links, size and count caps; each SKILL.md has its folder's name and a description, no
 * `allowed-tools`, no pointer to an install file and no install or login step outside the one fixed sentence;
 * vendored bundles carry their license and a provenance whose commit is the manifest's.
 */

import { createHash } from 'node:crypto';
import { lstatSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { chatPrompt, skillBody } from '../src/chat/agent/profiles.js';
import {
  bundlePathSegments,
  defaultSkillsDir,
  loadManifest,
  MAX_BUNDLE_FILES,
  MAX_SKILL_FILE_BYTES,
  readChatSkill,
} from '../src/chat/skills/manifest.js';

const SKILLS = defaultSkillsDir();
const manifest = loadManifest(SKILLS);
const FIXED =
  'The firecrawl CLI is already installed and signed in through FIRECRAWL_API_KEY. Never install it. Never run firecrawl login, init or setup.';
const FORBIDDEN = ['firecrawl login', '--api-key', 'npm install -g', 'npm i -g', '@latest'];

function walk(dir: string, prefix = ''): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const rel = prefix === '' ? entry.name : `${prefix}/${entry.name}`;
    if (entry.isDirectory()) out.push(...walk(join(dir, entry.name), rel));
    else out.push(rel);
  }
  return out;
}

/** The frontmatter lines between the first two `---` lines. */
function frontmatter(text: string): string[] {
  const lines = text.split('\n');
  if (lines[0] !== '---') return [];
  const end = lines.indexOf('---', 1);
  return end === -1 ? [] : lines.slice(1, end);
}

function fieldOf(lines: string[], name: string): string | null {
  const at = lines.findIndex((l) => l.startsWith(`${name}:`));
  if (at === -1) return null;
  const value = lines[at]!.slice(name.length + 1).trim();
  if (value !== '|' && value !== '>') return value;
  const block: string[] = [];
  for (const line of lines.slice(at + 1)) {
    if (!line.startsWith('  ')) break;
    block.push(line.trim());
  }
  return block.join(' ');
}

describe('skills bundle', () => {
  it('ships the five bundles with the search and read subset of skills', () => {
    expect(Object.keys(manifest.bundles).sort()).toEqual(['brave', 'chat-web', 'deepseek', 'exa', 'firecrawl']);
    expect(manifest.bundles.exa!.skills).toEqual(['exa-search', 'exa-contents']);
    expect(manifest.bundles.deepseek!.skills).toEqual(['deepseek-search']);
    expect(manifest.bundles.brave!.skills).toEqual(['web-search', 'llm-context', 'news-search']);
    expect(manifest.bundles.firecrawl!.skills).toEqual([
      'firecrawl',
      'firecrawl-search',
      'firecrawl-scrape',
      'firecrawl-crawl',
      'firecrawl-map',
    ]);
    expect(manifest.bundles['chat-web']!.skills).toEqual(['web-tools']);
    expect(manifest.bundles.exa!.upstream).toMatchObject({ repo: 'exa-labs/agent-skills', license: 'MIT' });
    expect(manifest.bundles.brave!.upstream).toMatchObject({ repo: 'brave/brave-search-skills', license: 'MIT' });
    expect(manifest.bundles.firecrawl!.upstream).toMatchObject({ repo: 'firecrawl/skills', license: 'ISC' });
    expect(manifest.bundles.deepseek!.upstream).toBeNull();
  });

  for (const [id, bundle] of Object.entries(manifest.bundles)) {
    describe(id, () => {
      const root = join(SKILLS, bundle.root);

      it('holds exactly the manifest files, each with its hash and size, no link, within the caps', () => {
        const onDisk = walk(root).sort();
        expect(onDisk).toEqual(bundle.files.map((f) => f.path).sort());
        expect(bundle.files.length).toBeLessThanOrEqual(MAX_BUNDLE_FILES);
        for (const file of bundle.files) {
          expect([file.path, bundlePathSegments(file.path) !== null]).toEqual([file.path, true]);
          const path = join(root, file.path);
          expect(lstatSync(path).isFile()).toBe(true);
          const data = readFileSync(path);
          expect([file.path, data.length, createHash('sha256').update(data).digest('hex')]).toEqual([
            file.path,
            file.bytes,
            file.sha256,
          ]);
          expect(data.length).toBeLessThanOrEqual(MAX_SKILL_FILE_BYTES);
        }
      });

      it('lists one SKILL.md per skill, named as its folder, with a description and no allowed-tools', () => {
        const skillFiles = bundle.files.filter((f) => f.path.endsWith('/SKILL.md')).map((f) => f.path);
        expect(skillFiles.sort()).toEqual(bundle.skills.map((s) => `skills/${s}/SKILL.md`).sort());
        for (const path of skillFiles) {
          const text = readFileSync(join(root, path), 'utf8');
          const front = frontmatter(text);
          expect([path, fieldOf(front, 'name')]).toEqual([path, path.split('/')[1]]);
          expect((fieldOf(front, 'description') ?? '').length).toBeGreaterThan(20);
          expect([path, front.some((l) => l.startsWith('allowed-tools'))]).toEqual([path, false]);
        }
      });

      it('names no install file and no install or login step outside the fixed sentence', () => {
        for (const file of bundle.files.filter((f) => f.path.endsWith('.md') && f.path !== 'PROVENANCE.md')) {
          const text = readFileSync(join(root, file.path), 'utf8').split(FIXED).join('');
          expect([file.path, text.includes('rules/')]).toEqual([file.path, false]);
          for (const word of FORBIDDEN)
            expect([file.path, word, text.includes(word)]).toEqual([file.path, word, false]);
        }
      });

      if (bundle.upstream !== null) {
        it('carries its license and a provenance with the manifest commit', () => {
          expect(bundle.files.map((f) => f.path)).toContain('LICENSE');
          const provenance = readFileSync(join(root, 'PROVENANCE.md'), 'utf8');
          expect(provenance).toContain(`- Commit: ${bundle.upstream!.commit}\n`);
          expect(provenance).toContain(`https://github.com/${bundle.upstream!.repo}`);
          expect(provenance).toContain('- Fetched: ');
          for (const file of bundle.files) {
            if (file.path !== 'PROVENANCE.md') expect(provenance).toContain(`\`${file.path}\``);
          }
          const license = readFileSync(join(root, 'LICENSE'), 'utf8');
          expect(license.startsWith(`${bundle.upstream!.license} License`)).toBe(true);
        });
      }
    });
  }

  it('records each local change of the Firecrawl bundle, and replaces each install pointer by the fixed sentence', () => {
    const provenance = readFileSync(join(SKILLS, 'vendor/firecrawl/PROVENANCE.md'), 'utf8');
    expect(provenance).toContain(FIXED);
    const main = readFileSync(join(SKILLS, 'vendor/firecrawl/skills/firecrawl/SKILL.md'), 'utf8');
    expect(main.split(FIXED).length - 1).toBe(4);
    expect(readFileSync(join(SKILLS, 'vendor/exa/PROVENANCE.md'), 'utf8')).toContain('None.');
    expect(readFileSync(join(SKILLS, 'vendor/brave/PROVENANCE.md'), 'utf8')).toContain('None.');
  });

  it('the Chat prompt ends with the web-tools skill body, byte for byte', () => {
    const skill = readChatSkill(SKILLS);
    expect(chatPrompt(skill).endsWith(skillBody(skill))).toBe(true);
  });

  it('the DeepSeek skill sends the key only as $DEEPSEEK_API_KEY in a header', () => {
    const text = readFileSync(join(SKILLS, 'own/deepseek/skills/deepseek-search/SKILL.md'), 'utf8');
    const uses = text.split('$DEEPSEEK_API_KEY').length - 1;
    const headers = text.split('-H "Authorization: Bearer $DEEPSEEK_API_KEY"').length - 1;
    expect(uses).toBe(headers + 1);
    expect(text).toContain('https://api.deepseek.com/v1/tools/search');
    expect(text).toContain('https://api.deepseek.com/v1/tools/fetch');
    expect(text.includes('curl -v')).toBe(true);
    expect(text).not.toContain('curl -sS -v');
  });
});
