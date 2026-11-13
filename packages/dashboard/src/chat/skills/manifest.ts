/**
 * The skills bundle shipped in the package (`packages/dashboard/skills`): `manifest.json` names every file of every
 * bundle with its sha256 and size. A file is used only when it matches; any mismatch is `skill-bundle-corrupt`.
 * Paths are built from manifest segments that pass a strict character check: no `..`, no absolute path, no link.
 */

import { createHash } from 'node:crypto';
import { lstatSync, readFileSync, type Stats } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const MAX_SKILL_FILE_BYTES = 64 * 1024;
export const MAX_BUNDLE_FILES = 40;
export const CHAT_SKILL_BUNDLE = 'chat-web';
export const CHAT_SKILL_FILE = 'skills/web-tools/SKILL.md';

export interface BundleFile {
  path: string;
  sha256: string;
  bytes: number;
}

export interface SkillBundle {
  root: string;
  upstream: { repo: string; commit: string; license: string } | null;
  skills: string[];
  files: BundleFile[];
}

export interface SkillsManifest {
  version: 1;
  bundles: Record<string, SkillBundle>;
}

/** A bundle file or the manifest does not match what was shipped. */
export class SkillBundleError extends Error {
  readonly code = 'skill-bundle-corrupt' as const;

  constructor() {
    super("The app's skill files are damaged. Reinstall the app.");
    this.name = 'SkillBundleError';
  }
}

/** The package's `skills/` folder, from `src/chat/skills` and from `dist/chat/skills` alike. */
export function defaultSkillsDir(): string {
  return fileURLToPath(new URL('../../../skills/', import.meta.url));
}

function isSegmentChar(code: number): boolean {
  return (code >= 0x61 && code <= 0x7a) || (code >= 0x30 && code <= 0x39) || code === 0x2d || code === 0x2e;
}

/** `a-z 0-9 - .`, not empty, not `.` or `..`. */
export function isSafeSegment(segment: string): boolean {
  if (segment === '' || segment === '.' || segment === '..' || segment.length > 100) return false;
  for (let i = 0; i < segment.length; i++) if (!isSegmentChar(segment.charCodeAt(i))) return false;
  return true;
}

/**
 * The segments of an allowed bundle path, or null. Allowed: `LICENSE`, `PROVENANCE.md`, `skills/<name>/SKILL.md`,
 * `skills/<name>/references/<file>.md`.
 */
export function bundlePathSegments(path: string): string[] | null {
  if (path === 'LICENSE' || path === 'PROVENANCE.md') return [path];
  const parts = path.split('/');
  if (parts[0] !== 'skills' || parts.length < 3) return null;
  if (!isSafeSegment(parts[1]!)) return null;
  if (parts.length === 3 && parts[2] === 'SKILL.md') return parts;
  if (parts.length === 4 && parts[2] === 'references') {
    const file = parts[3]!;
    if (isSafeSegment(file) && file.endsWith('.md') && file.length > 3) return parts;
  }
  return null;
}

function isHex(text: string, length: number): boolean {
  if (text.length !== length) return false;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (!((c >= 0x30 && c <= 0x39) || (c >= 0x61 && c <= 0x66))) return false;
  }
  return true;
}

function parseBundle(value: unknown): SkillBundle {
  if (typeof value !== 'object' || value === null) throw new SkillBundleError();
  const b = value as Record<string, unknown>;
  if (typeof b.root !== 'string') throw new SkillBundleError();
  const rootParts = b.root.split('/');
  if (rootParts.length !== 2 || !rootParts.every(isSafeSegment)) throw new SkillBundleError();
  let upstream: SkillBundle['upstream'] = null;
  if (b.upstream !== null) {
    const u = b.upstream as Record<string, unknown> | undefined;
    if (
      typeof u !== 'object' ||
      typeof u.repo !== 'string' ||
      typeof u.license !== 'string' ||
      typeof u.commit !== 'string' ||
      !isHex(u.commit, 40)
    ) {
      throw new SkillBundleError();
    }
    upstream = { repo: u.repo, commit: u.commit, license: u.license };
  }
  if (!Array.isArray(b.skills) || !b.skills.every((s) => typeof s === 'string' && isSafeSegment(s))) {
    throw new SkillBundleError();
  }
  if (!Array.isArray(b.files) || b.files.length > MAX_BUNDLE_FILES) throw new SkillBundleError();
  const seen = new Set<string>();
  const files: BundleFile[] = [];
  for (const f of b.files as unknown[]) {
    const file = f as Record<string, unknown>;
    if (typeof file !== 'object' || file === null || typeof file.path !== 'string') throw new SkillBundleError();
    if (bundlePathSegments(file.path) === null || seen.has(file.path)) throw new SkillBundleError();
    if (typeof file.sha256 !== 'string' || !isHex(file.sha256, 64)) throw new SkillBundleError();
    if (typeof file.bytes !== 'number' || !Number.isSafeInteger(file.bytes) || file.bytes < 0) {
      throw new SkillBundleError();
    }
    if (file.bytes > MAX_SKILL_FILE_BYTES) throw new SkillBundleError();
    seen.add(file.path);
    files.push({ path: file.path, sha256: file.sha256, bytes: file.bytes });
  }
  return { root: b.root, upstream, skills: [...(b.skills as string[])], files };
}

export function parseManifest(text: string): SkillsManifest {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SkillBundleError();
  }
  if (typeof raw !== 'object' || raw === null) throw new SkillBundleError();
  const m = raw as { version?: unknown; bundles?: unknown };
  if (m.version !== 1 || typeof m.bundles !== 'object' || m.bundles === null) throw new SkillBundleError();
  const bundles: Record<string, SkillBundle> = {};
  for (const [id, value] of Object.entries(m.bundles)) {
    if (!isSafeSegment(id)) throw new SkillBundleError();
    bundles[id] = parseBundle(value);
  }
  return { version: 1, bundles };
}

export function loadManifest(skillsDir: string): SkillsManifest {
  try {
    return parseManifest(readFileSync(join(skillsDir, 'manifest.json'), 'utf8'));
  } catch (error) {
    if (error instanceof SkillBundleError) throw error;
    throw new SkillBundleError();
  }
}

/** sha256 of the bundle's manifest entry: what an installed copy records to prove which bundle it holds. */
export function bundleEntrySha(bundle: SkillBundle): string {
  return createHash('sha256').update(JSON.stringify(bundle)).digest('hex');
}

export function sha256Hex(data: Buffer | string): string {
  return createHash('sha256').update(data).digest('hex');
}

/** One bundled file, read only when it is a regular file (no link on the way) that matches the manifest. */
export function readBundleFile(skillsDir: string, bundle: SkillBundle, file: BundleFile): Buffer {
  const segments = bundlePathSegments(file.path);
  if (segments === null) throw new SkillBundleError();
  let dir = skillsDir;
  for (const part of [...bundle.root.split('/'), ...segments.slice(0, -1)]) {
    dir = join(dir, part);
    const st = lstatOrNull(dir);
    if (st === null || !st.isDirectory()) throw new SkillBundleError();
  }
  const path = join(dir, segments[segments.length - 1]!);
  const st = lstatOrNull(path);
  if (st === null || !st.isFile() || st.size !== file.bytes) throw new SkillBundleError();
  const data = readFileSync(path);
  if (data.length !== file.bytes || sha256Hex(data) !== file.sha256) throw new SkillBundleError();
  return data;
}

function lstatOrNull(path: string): Stats | null {
  try {
    return lstatSync(path);
  } catch {
    return null;
  }
}

/** The bundled Chat skill text, checked against the manifest. */
export function readChatSkill(skillsDir: string = defaultSkillsDir()): string {
  const manifest = loadManifest(skillsDir);
  const bundle = manifest.bundles[CHAT_SKILL_BUNDLE];
  const file = bundle?.files.find((f) => f.path === CHAT_SKILL_FILE);
  if (bundle === undefined || file === undefined) throw new SkillBundleError();
  return readBundleFile(skillsDir, bundle, file).toString('utf8');
}
