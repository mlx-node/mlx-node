import { mkdir, mkdtemp, open, rename, rm, truncate, writeFile, type FileHandle } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import type {
  ExtensionAPI,
  ExtensionContext,
  InlineExtension,
  InputEvent,
  InputEventResult,
} from '@earendil-works/pi-coding-agent';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { attachStandaloneLocalImage, createLocalImageInputExtension } from '../src/extensions/local-image-input.js';

const PNG_BYTES = Buffer.from('89504e470d0a1a0a0000000d49484452', 'hex');
const createdDirectories: string[] = [];

async function imageFile(name = 'CleanShot 2026-07-30 at 09.18.26@2x.png'): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'mlx-local-image-input-'));
  createdDirectories.push(directory);
  const path = join(directory, name);
  await writeFile(path, PNG_BYTES);
  return path;
}

function input(text: string, overrides: Partial<InputEvent> = {}): InputEvent {
  return { type: 'input', text, source: 'interactive', ...overrides };
}

function context(
  inputTypes: Array<'text' | 'image'> = ['text', 'image'],
  mode: ExtensionContext['mode'] = 'tui',
): ExtensionContext {
  return { mode, model: { input: inputTypes } } as unknown as ExtensionContext;
}

afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(createdDirectories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

describe('local image input extension', () => {
  it('attaches a standalone absolute PNG path with macOS shell-escaped spaces', async () => {
    const path = await imageFile();
    const text = path.replaceAll(' ', '\\ ');

    const result = await attachStandaloneLocalImage(input(text), context());

    expect(result).toEqual({
      action: 'transform',
      text,
      images: [{ type: 'image', mimeType: 'image/png', data: PNG_BYTES.toString('base64') }],
    });
  });

  it('also attaches an ordinary standalone absolute image path without changing its text', async () => {
    const path = await imageFile('screenshot.png');

    const result = await attachStandaloneLocalImage(input(path), context());

    expect(result.action).toBe('transform');
    expect((result as Extract<InputEventResult, { action: 'transform' }>).text).toBe(path);
  });

  it.each([
    'Please inspect /tmp/screenshot.png',
    './screenshot.png',
    '/tmp/screenshot.png\nand explain it',
    '/tmp/screenshot.txt',
  ])('leaves non-standalone or non-image input unchanged: %s', async (text) => {
    await expect(attachStandaloneLocalImage(input(text), context())).resolves.toEqual({ action: 'continue' });
  });

  it('attaches on the first turn while discovery metadata is still conservatively text-only', async () => {
    const path = await imageFile();

    const result = await attachStandaloneLocalImage(input(path), context(['text']));

    expect(result.action).toBe('transform');
    expect((result as Extract<InputEventResult, { action: 'transform' }>).text).toBe(path);
  });

  it.each([
    { source: 'rpc' as const, mode: 'rpc' as const },
    { source: 'extension' as const, mode: 'tui' as const },
    { source: 'interactive' as const, mode: 'print' as const },
  ])('does not auto-read a path from $source input in $mode mode', async ({ source, mode }) => {
    const path = await imageFile();

    await expect(
      attachStandaloneLocalImage(input(path, { source }), context(['text', 'image'], mode)),
    ).resolves.toEqual({
      action: 'continue',
    });
  });

  it('does not duplicate an image already attached by Pi or another extension', async () => {
    const path = await imageFile();

    await expect(
      attachStandaloneLocalImage(
        input(path, { images: [{ type: 'image', mimeType: 'image/png', data: 'already-attached' }] }),
        context(),
      ),
    ).resolves.toEqual({ action: 'continue' });
  });

  it('leaves missing, non-file, and invalid-image paths unchanged', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'mlx-local-image-input-invalid-'));
    createdDirectories.push(directory);
    const imageDirectory = join(directory, 'folder.png');
    await mkdir(imageDirectory);
    const invalid = join(directory, 'not-an-image.png');
    await writeFile(invalid, 'plain text');

    await expect(attachStandaloneLocalImage(input(join(directory, 'missing.png')), context())).resolves.toEqual({
      action: 'continue',
    });
    await expect(attachStandaloneLocalImage(input(imageDirectory), context())).resolves.toEqual({
      action: 'continue',
    });
    await expect(attachStandaloneLocalImage(input(invalid), context())).resolves.toEqual({
      action: 'continue',
    });
  });

  it('reads from the validated handle when the pathname is replaced after fstat', async () => {
    const path = await imageFile('replace-race.png');
    const replacement = join(dirname(path), 'replacement.png');
    const replacementBytes = Buffer.concat([PNG_BYTES, Buffer.from('replacement')]);
    await writeFile(replacement, replacementBytes);

    const probe = await open(path, 'r');
    const fileHandlePrototype = Object.getPrototypeOf(probe) as FileHandle;
    await probe.close();
    const originalStat = Object.getOwnPropertyDescriptor(fileHandlePrototype, 'stat')?.value as FileHandle['stat'];
    const statSpy = vi.spyOn(fileHandlePrototype, 'stat').mockImplementationOnce(async function (this: FileHandle) {
      const metadata = await originalStat.call(this);
      await rename(replacement, path);
      return metadata;
    });

    const result = await attachStandaloneLocalImage(input(path), context());

    expect(statSpy).toHaveBeenCalledTimes(1);
    expect(result).toEqual({
      action: 'transform',
      text: path,
      images: [{ type: 'image', mimeType: 'image/png', data: PNG_BYTES.toString('base64') }],
    });
  });

  it('rejects same-inode growth after fstat without reading the enlarged file', async () => {
    const path = await imageFile('growth-race.png');
    const probe = await open(path, 'r');
    const fileHandlePrototype = Object.getPrototypeOf(probe) as FileHandle;
    await probe.close();
    const originalStat = Object.getOwnPropertyDescriptor(fileHandlePrototype, 'stat')?.value as FileHandle['stat'];
    const statSpy = vi.spyOn(fileHandlePrototype, 'stat').mockImplementationOnce(async function (this: FileHandle) {
      const metadata = await originalStat.call(this);
      await truncate(path, 20 * 1024 * 1024 + 1);
      return metadata;
    });

    await expect(attachStandaloneLocalImage(input(path), context())).resolves.toEqual({
      action: 'continue',
    });
    expect(statSpy).toHaveBeenCalledTimes(1);
  });

  it('does not read an image larger than the local attachment cap', async () => {
    const directory = await mkdtemp(join(tmpdir(), 'mlx-local-image-input-large-'));
    createdDirectories.push(directory);
    const path = join(directory, 'large.png');
    const file = await open(path, 'w');
    try {
      await file.truncate(20 * 1024 * 1024 + 1);
    } finally {
      await file.close();
    }

    await expect(attachStandaloneLocalImage(input(path), context())).resolves.toEqual({
      action: 'continue',
    });
  });

  it('registers one named input handler', () => {
    let handler: unknown;
    const pi = {
      on(event: string, candidate: unknown): void {
        if (event === 'input') handler = candidate;
      },
    } as unknown as ExtensionAPI;

    const extension: InlineExtension = createLocalImageInputExtension();
    expect(typeof extension).toBe('object');
    if (typeof extension === 'function') throw new Error('expected a named extension');
    expect(extension.name).toBe('mlx-local-image-input');
    void extension.factory(pi);
    expect(handler).toBe(attachStandaloneLocalImage);
  });
});
