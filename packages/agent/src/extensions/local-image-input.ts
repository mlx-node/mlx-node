/**
 * Attach the local image path inserted by Pi's interactive clipboard handler.
 *
 * Pi 0.81.1 writes a pasted clipboard image to a temporary file and inserts
 * only that absolute path into the editor. AgentSession's `input` event is the
 * last seam where the path can be upgraded to `ImageContent` before the user
 * message is committed. Keep this deliberately narrow:
 *
 * - TUI + interactive input only;
 * - the entire input must be one absolute image path;
 * - only `\ ` shell escapes (the macOS drag/paste shape) are decoded;
 * - extension and magic bytes must both identify a supported image.
 *
 * Every failed check returns `continue`, preserving the original text exactly.
 * Capability is intentionally not checked here: discovery advertises every
 * local model as text-only until its first resident load. The stream adapter's
 * authoritative `session.supportsImages()` decides whether native bytes or the
 * existing text-model placeholder reaches inference.
 */

import { open } from 'node:fs/promises';
import { extname, isAbsolute } from 'node:path';

import type {
  ExtensionAPI,
  ExtensionContext,
  InlineExtension,
  InputEvent,
  InputEventResult,
} from '@earendil-works/pi-coding-agent';

const MAX_LOCAL_IMAGE_BYTES = 20 * 1024 * 1024;
const IMAGE_EXTENSIONS = new Set(['.gif', '.jpeg', '.jpg', '.png', '.webp']);
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a] as const;

function startsWith(bytes: Uint8Array, signature: readonly number[], offset = 0): boolean {
  return (
    bytes.byteLength >= offset + signature.length && signature.every((byte, index) => bytes[offset + index] === byte)
  );
}

function startsWithAscii(bytes: Uint8Array, text: string, offset = 0): boolean {
  return startsWith(
    bytes,
    Array.from(text, (character) => character.charCodeAt(0)),
    offset,
  );
}

function detectImageMimeType(bytes: Uint8Array): string | undefined {
  if (startsWith(bytes, [0xff, 0xd8, 0xff]) && bytes[3] !== 0xf7) return 'image/jpeg';
  if (startsWith(bytes, PNG_SIGNATURE) && startsWithAscii(bytes, 'IHDR', 12)) return 'image/png';
  if (startsWithAscii(bytes, 'GIF87a') || startsWithAscii(bytes, 'GIF89a')) return 'image/gif';
  if (startsWithAscii(bytes, 'RIFF') && startsWithAscii(bytes, 'WEBP', 8)) return 'image/webp';
  return undefined;
}

/**
 * Resolve only the standalone path shape produced by Pi/macOS paste and drag.
 * This is not a shell parser: no quoting, expansion, commands, or general
 * backslash processing is performed.
 */
function standaloneImagePath(text: string): string | undefined {
  const trimmed = text.trim();
  if (trimmed.length === 0 || trimmed.includes('\n') || trimmed.includes('\r')) return undefined;

  const path = trimmed.replaceAll('\\ ', ' ');
  if (!isAbsolute(path) || !IMAGE_EXTENSIONS.has(extname(path).toLowerCase())) return undefined;
  return path;
}

/** @internal Exported for focused tests of Pi's input-transform contract. */
export async function attachStandaloneLocalImage(event: InputEvent, ctx: ExtensionContext): Promise<InputEventResult> {
  if (ctx.mode !== 'tui' || event.source !== 'interactive' || event.images?.length) {
    return { action: 'continue' };
  }

  const path = standaloneImagePath(event.text);
  if (path === undefined) return { action: 'continue' };

  try {
    // Open once so validation and reading stay bound to the same inode if
    // the pathname is replaced. FileHandle.readFile() is intentionally not
    // used: it may allocate from a file that grew after stat(). The exact,
    // positional loop below caps allocation at the validated size.
    const file = await open(path, 'r');
    try {
      const metadata = await file.stat();
      if (!metadata.isFile() || metadata.size === 0 || metadata.size > MAX_LOCAL_IMAGE_BYTES) {
        return { action: 'continue' };
      }

      const bytes = Buffer.allocUnsafe(metadata.size);
      let offset = 0;
      while (offset < bytes.byteLength) {
        const { bytesRead } = await file.read(bytes, offset, bytes.byteLength - offset, offset);
        if (bytesRead === 0) return { action: 'continue' };
        offset += bytesRead;
      }

      // Reject same-inode growth after fstat without ever allocating or
      // reading more than one byte beyond the validated size.
      const overflowProbe = Buffer.allocUnsafe(1);
      if ((await file.read(overflowProbe, 0, 1, metadata.size)).bytesRead !== 0) {
        return { action: 'continue' };
      }

      const mimeType = detectImageMimeType(bytes);
      if (mimeType === undefined) return { action: 'continue' };

      return {
        action: 'transform',
        text: event.text,
        images: [{ type: 'image', mimeType, data: bytes.toString('base64') }],
      };
    } finally {
      await file.close();
    }
  } catch {
    // Missing, unreadable, or racing files remain ordinary user text.
    return { action: 'continue' };
  }
}

export function createLocalImageInputExtension(): InlineExtension {
  return {
    name: 'mlx-local-image-input',
    factory: (pi: ExtensionAPI) => {
      pi.on('input', attachStandaloneLocalImage);
    },
  };
}
