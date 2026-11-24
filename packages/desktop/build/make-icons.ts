/**
 * Generate the macOS, Windows, and Microsoft Store MSIX app icons.
 *
 * macOS assets use `images/logo.png` (canonical MLX wordmark plus green node)
 * on a white rounded tile, with the compact X + node optical variant at
 * 16px..32px, and `tray-icon-source.png` thresholded into a pure black-and-alpha
 * menu-bar template (`iconTemplate.png` / `iconTemplate@2x.png`).
 *
 * Windows and Microsoft Store MSIX assets use `portato-logo-source.png`
 * (canonical 1024x1024 Portato mascot + wordmark):
 * - Square icons (`icon.ico` and square MSIX tiles) place the extracted Portato
 *   mascot emblem on a warm `#F1F2EC` rounded-square tile with transparent outer
 *   corners and a subtle dark keyline (`rgba(22, 25, 24, 0.22)`) so the icon
 *   reads cleanly on both dark (`#202020`) and light (`#F3F3F3`) Windows taskbars.
 * - Wide tiles and splash screens (`Wide310x150Logo*`, `SplashScreen*`) render a
 *   rounded `#F1F2EC` card with a centered horizontal lockup of the mascot emblem
 *   and the `portato` wordmark.
 * - MSIX PNGs are kept strictly below Microsoft Store WACK's 204,800-byte limit
 *   (quantizing oversized truecolor assets to 8-bit indexed PNG with `PLTE` + `tRNS`).
 *
 *     icon.iconset/ -> icon.icns      macOS app icon, 10 images, 16..1024 with @2x (darwin only)
 *     icon.ico                        Windows multi-resolution PNG-in-ICO, 16..256
 *     msix/*.png                      Microsoft Store MSIX visual tile & splash assets
 *     iconTemplate.png / @2x          menu bar, black + alpha, macOS recolours it
 *
 * Run from the repository root:
 *
 *     yarn workspace @mlx-node/desktop icons
 */

import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  CompressionType,
  FilterType,
  losslessCompressPngSync,
  pngQuantizeSync,
  ResizeFilterType,
  ResizeFit,
  Transformer,
} from '@napi-rs/image';

import { MAX_MSIX_ASSET_BYTES } from '../scripts/windows-msix.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..', '..', '..');
const LOGO_PATH = join(REPO_ROOT, 'images', 'logo.png');
const TRAY_SOURCE_PATH = join(HERE, 'tray-icon-source.png');
const PORTATO_SOURCE_PATH = join(HERE, 'portato-logo-source.png');

const CANVAS = 1024;
const SQUIRCLE = 824;
const SQUIRCLE_RADIUS = Math.trunc(SQUIRCLE * 0.225);
const SQUIRCLE_MARGIN = Math.trunc((CANVAS - SQUIRCLE) / 2);
const APP_SS = 4;
const TRAY_SS = 8;

type Bounds = readonly [left: number, top: number, right: number, bottom: number];
type Rgb = readonly [red: number, green: number, blue: number];

interface RgbaImage {
  width: number;
  height: number;
  pixels: Uint8Array;
}

// Bounds measured from the canonical 1024px logos.
const WORDMARK_BOUNDS: Bounds = [98, 404, 926, 630];
const COMPACT_BOUNDS: Bounds = [552, 404, 926, 630];
const PORTATO_EMBLEM_REGION: Bounds = [205, 160, 843, 775];
const PORTATO_WORDMARK_REGION: Bounds = [0, 795, CANVAS, CANVAS];
const PORTATO_BG_TOLERANCE = 26;
const PORTATO_FALLBACK_INK: Rgb = [9, 12, 32];

const PNG_OPTIONS = {
  compressionType: CompressionType.Best,
  filterType: FilterType.Adaptive,
} as const;

function scaled(value: number, size: number, supersampling = 1): number {
  return Math.round((value * size * supersampling) / CANVAS);
}

function blankImage(width: number, height: number, fill = 0): RgbaImage {
  return {
    width,
    height,
    pixels: new Uint8Array(width * height * 4).fill(fill),
  };
}

function decodeOpaqueImage(path: string): RgbaImage {
  const encoded = readFileSync(path);
  const metadata = new Transformer(encoded).metadataSync();
  if (!metadata.width || !metadata.height) throw new Error(`Could not read dimensions from ${path}`);

  // Both source PNGs are RGB. Overlaying them on an opaque RGBA canvas asks the
  // native decoder for a predictable four-channel buffer.
  const rgba = new Uint8Array(metadata.width * metadata.height * 4).fill(255);
  const pixels = Transformer.fromRgbaPixels(rgba, metadata.width, metadata.height)
    .overlay(encoded, 0, 0)
    .rawPixelsSync();
  return { width: metadata.width, height: metadata.height, pixels };
}

function decodeLogo(): RgbaImage {
  const logo = decodeOpaqueImage(LOGO_PATH);
  if (logo.width !== CANVAS || logo.height !== CANVAS) {
    throw new Error(`${LOGO_PATH} must be ${CANVAS}x${CANVAS}, got ${logo.width}x${logo.height}`);
  }
  return logo;
}

function decodePortatoSource(): RgbaImage {
  const logo = decodeOpaqueImage(PORTATO_SOURCE_PATH);
  if (logo.width !== CANVAS || logo.height !== CANVAS) {
    throw new Error(`${PORTATO_SOURCE_PATH} must be ${CANVAS}x${CANVAS}, got ${logo.width}x${logo.height}`);
  }
  return logo;
}

const SOURCE_LOGO = decodeLogo();
const TRAY_SOURCE = decodeOpaqueImage(TRAY_SOURCE_PATH);
const PORTATO_SOURCE = decodePortatoSource();

function cropImage(image: RgbaImage, bounds: Bounds): RgbaImage {
  const [left, top, right, bottom] = bounds;
  const width = right - left;
  const height = bottom - top;
  if (left < 0 || top < 0 || right > image.width || bottom > image.height || width <= 0 || height <= 0) {
    throw new Error(`Invalid crop ${bounds.join(',')} for ${image.width}x${image.height}`);
  }

  const result = blankImage(width, height);
  for (let y = 0; y < height; y += 1) {
    const sourceStart = ((top + y) * image.width + left) * 4;
    const targetStart = y * width * 4;
    result.pixels.set(image.pixels.subarray(sourceStart, sourceStart + width * 4), targetStart);
  }
  return result;
}

function resizeImage(image: RgbaImage, width: number, height: number): RgbaImage {
  const pixels = Transformer.fromRgbaPixels(image.pixels, image.width, image.height)
    .resize({
      width,
      height,
      filter: ResizeFilterType.Lanczos3,
      fit: ResizeFit.Fill,
    })
    .rawPixelsSync();
  return { width, height, pixels };
}

function encodePng(image: RgbaImage): Buffer {
  return Transformer.fromRgbaPixels(image.pixels, image.width, image.height).pngSync(PNG_OPTIONS);
}

function sourceOverPixel(target: Uint8Array, targetOffset: number, source: Uint8Array, sourceOffset: number): void {
  const sourceAlpha = source[sourceOffset + 3] / 255;
  if (sourceAlpha === 0) return;

  const targetAlpha = target[targetOffset + 3] / 255;
  const outputAlpha = sourceAlpha + targetAlpha * (1 - sourceAlpha);
  if (outputAlpha === 0) return;

  for (let channel = 0; channel < 3; channel += 1) {
    target[targetOffset + channel] = Math.round(
      (source[sourceOffset + channel] * sourceAlpha +
        target[targetOffset + channel] * targetAlpha * (1 - sourceAlpha)) /
        outputAlpha,
    );
  }
  target[targetOffset + 3] = Math.round(outputAlpha * 255);
}

function compositeAt(target: RgbaImage, source: RgbaImage, targetX = 0, targetY = 0): void {
  for (let sourceY = 0; sourceY < source.height; sourceY += 1) {
    const y = targetY + sourceY;
    if (y < 0 || y >= target.height) continue;

    for (let sourceX = 0; sourceX < source.width; sourceX += 1) {
      const x = targetX + sourceX;
      if (x < 0 || x >= target.width) continue;

      sourceOverPixel(target.pixels, (y * target.width + x) * 4, source.pixels, (sourceY * source.width + sourceX) * 4);
    }
  }
}

function roundedRectContains(
  x: number,
  y: number,
  left: number,
  top: number,
  right: number,
  bottom: number,
  radius: number,
): boolean {
  if (x < left || y < top || x >= right || y >= bottom) return false;
  if (radius <= 0) return true;

  const nearestX = Math.max(left + radius, Math.min(x, right - radius));
  const nearestY = Math.max(top + radius, Math.min(y, bottom - radius));
  const dx = x - nearestX;
  const dy = y - nearestY;
  return dx * dx + dy * dy <= radius * radius;
}

function drawTile(size: number): RgbaImage {
  const renderSize = size * APP_SS;
  const tile = blankImage(renderSize, renderSize);
  const left = scaled(SQUIRCLE_MARGIN, size, APP_SS);
  const top = left;
  const right = scaled(SQUIRCLE_MARGIN + SQUIRCLE, size, APP_SS);
  const bottom = right;
  const radius = scaled(SQUIRCLE_RADIUS, size, APP_SS);
  const mask = new Uint8Array(renderSize * renderSize);

  for (let y = 0; y < renderSize; y += 1) {
    for (let x = 0; x < renderSize; x += 1) {
      if (roundedRectContains(x + 0.5, y + 0.5, left, top, right, bottom, radius)) {
        mask[y * renderSize + x] = 255;
      }
    }
  }

  // A small built-in shadow makes the white tile retain its shape in asset
  // previews; macOS adds the larger system shadow in the Dock.
  const shadow = blankImage(renderSize, renderSize);
  for (let pixel = 0; pixel < mask.length; pixel += 1) {
    shadow.pixels[pixel * 4 + 3] = Math.trunc((mask[pixel] * 54) / 255);
  }
  const blurredShadow: RgbaImage = {
    width: renderSize,
    height: renderSize,
    pixels: Transformer.fromRgbaPixels(shadow.pixels, renderSize, renderSize)
      .blur(Math.max(1, scaled(18, size, APP_SS)))
      .rawPixelsSync(),
  };
  compositeAt(tile, blurredShadow);

  for (let pixel = 0; pixel < mask.length; pixel += 1) {
    if (mask[pixel] === 0) continue;
    const offset = pixel * 4;
    tile.pixels[offset] = 255;
    tile.pixels[offset + 1] = 255;
    tile.pixels[offset + 2] = 255;
    tile.pixels[offset + 3] = 255;
  }

  const keylineWidth = Math.max(APP_SS, scaled(2.5, size, APP_SS));
  const keyline = new Uint8Array([22, 25, 24, 38]);
  for (let y = 0; y < renderSize; y += 1) {
    for (let x = 0; x < renderSize; x += 1) {
      const px = x + 0.5;
      const py = y + 0.5;
      const outer = roundedRectContains(px, py, left, top, right, bottom, radius);
      const inner = roundedRectContains(
        px,
        py,
        left + keylineWidth,
        top + keylineWidth,
        right - keylineWidth,
        bottom - keylineWidth,
        Math.max(0, radius - keylineWidth),
      );
      if (outer && !inner) sourceOverPixel(tile.pixels, (y * renderSize + x) * 4, keyline, 0);
    }
  }

  return resizeImage(tile, size, size);
}

function appIcon(size: number): RgbaImage {
  const image = drawTile(size);
  const compact = size <= 32;
  const mark = cropImage(SOURCE_LOGO, compact ? COMPACT_BOUNDS : WORDMARK_BOUNDS);
  const targetWidth = scaled(compact ? 690 : 716, size);
  const targetHeight = Math.max(1, Math.round((mark.height * targetWidth) / mark.width));
  const resizedMark = resizeImage(mark, targetWidth, targetHeight);
  compositeAt(image, resizedMark, Math.trunc((size - targetWidth) / 2), Math.trunc((size - targetHeight) / 2));
  return image;
}

// No default threshold on purpose: the only call site keys at 80, and a silent
// fallback an order of magnitude lower would quietly change what counts as ink.
function foregroundMask(image: RgbaImage, threshold: number): RgbaImage {
  const result = blankImage(image.width, image.height);
  for (let pixel = 0; pixel < image.width * image.height; pixel += 1) {
    const offset = pixel * 4;
    const difference = Math.max(
      255 - image.pixels[offset],
      255 - image.pixels[offset + 1],
      255 - image.pixels[offset + 2],
    );
    result.pixels[offset + 3] = difference > threshold ? 255 : 0;
  }
  return result;
}

function alphaBounds(image: RgbaImage, label = 'Tray source'): Bounds {
  let left = image.width;
  let top = image.height;
  let right = -1;
  let bottom = -1;

  for (let y = 0; y < image.height; y += 1) {
    for (let x = 0; x < image.width; x += 1) {
      if (image.pixels[(y * image.width + x) * 4 + 3] === 0) continue;
      left = Math.min(left, x);
      top = Math.min(top, y);
      right = Math.max(right, x);
      bottom = Math.max(bottom, y);
    }
  }

  if (right < left || bottom < top) throw new Error(`${label} does not contain a foreground mark`);
  return [left, top, right + 1, bottom + 1];
}

function trayTemplate(size: number): RgbaImage {
  const renderSize = size * TRAY_SS;
  // A 13px-tall mark at 1x gives the hexagon enough pixels to keep its facets
  // and the negative-space M enough room to remain open.
  const box = Math.round(size * 0.8125 * TRAY_SS);
  const sourceMask = foregroundMask(TRAY_SOURCE, 80);
  const mark = cropImage(sourceMask, alphaBounds(sourceMask));
  // Fit inside a square box, not to the height alone. A mark wider than
  // renderSize/box (1.23:1) would scale past the canvas, and `compositeAt`
  // drops out-of-bounds columns silently: exit 0, `validateTemplate` still
  // passes, and a mark with its sides sheared off is just as black-and-alpha as
  // an intact one. The current 777x945 mark is bound by height either way.
  const scale = Math.min(box / mark.width, box / mark.height);
  const targetWidth = Math.max(1, Math.round(mark.width * scale));
  const targetHeight = Math.max(1, Math.round(mark.height * scale));
  const resizedMark = resizeImage(mark, targetWidth, targetHeight);

  const result = blankImage(renderSize, renderSize);
  const left = Math.trunc((renderSize - targetWidth) / 2);
  const top = Math.trunc((renderSize - targetHeight) / 2);
  compositeAt(result, resizedMark, left, top);
  return resizeImage(result, size, size);
}

function validateTemplate(image: RgbaImage, expectedSize: number): void {
  if (image.width !== expectedSize || image.height !== expectedSize) {
    throw new Error(`Expected ${expectedSize}x${expectedSize} tray image, got ${image.width}x${image.height}`);
  }

  let maxAlpha = 0;
  for (let offset = 0; offset < image.pixels.length; offset += 4) {
    if (image.pixels[offset] !== 0 || image.pixels[offset + 1] !== 0 || image.pixels[offset + 2] !== 0) {
      throw new Error('Tray template must contain only black RGB pixels');
    }
    maxAlpha = Math.max(maxAlpha, image.pixels[offset + 3]);
  }
  if (image.pixels[3] !== 0) throw new Error('Tray template corner must be transparent');
  if (maxAlpha !== 255) throw new Error('Tray template must contain opaque pixels');
}

const ICO_SIZES = [16, 24, 32, 48, 64, 128, 256] as const;

interface MsixAssetSpec {
  name: string;
  width: number;
  height: number;
  kind: 'square' | 'light-unplated' | 'wide';
}

const MSIX_ASSETS: readonly MsixAssetSpec[] = [
  // StoreLogo
  { name: 'StoreLogo.png', width: 50, height: 50, kind: 'square' },
  { name: 'StoreLogo.scale-100.png', width: 50, height: 50, kind: 'square' },
  { name: 'StoreLogo.scale-125.png', width: 63, height: 63, kind: 'square' },
  { name: 'StoreLogo.scale-150.png', width: 75, height: 75, kind: 'square' },
  { name: 'StoreLogo.scale-200.png', width: 100, height: 100, kind: 'square' },
  { name: 'StoreLogo.scale-400.png', width: 200, height: 200, kind: 'square' },

  // Square44x44Logo (scale variants)
  { name: 'Square44x44Logo.png', width: 44, height: 44, kind: 'square' },
  { name: 'Square44x44Logo.scale-100.png', width: 44, height: 44, kind: 'square' },
  { name: 'Square44x44Logo.scale-125.png', width: 55, height: 55, kind: 'square' },
  { name: 'Square44x44Logo.scale-150.png', width: 66, height: 66, kind: 'square' },
  { name: 'Square44x44Logo.scale-200.png', width: 88, height: 88, kind: 'square' },
  { name: 'Square44x44Logo.scale-400.png', width: 176, height: 176, kind: 'square' },

  // Square44x44Logo (targetsize variants)
  { name: 'Square44x44Logo.targetsize-16.png', width: 16, height: 16, kind: 'square' },
  { name: 'Square44x44Logo.targetsize-24.png', width: 24, height: 24, kind: 'square' },
  { name: 'Square44x44Logo.targetsize-32.png', width: 32, height: 32, kind: 'square' },
  { name: 'Square44x44Logo.targetsize-48.png', width: 48, height: 48, kind: 'square' },
  { name: 'Square44x44Logo.targetsize-256.png', width: 256, height: 256, kind: 'square' },

  // Square44x44Logo (targetsize altform-unplated variants — dark/default taskbar)
  { name: 'Square44x44Logo.targetsize-16_altform-unplated.png', width: 16, height: 16, kind: 'square' },
  { name: 'Square44x44Logo.targetsize-24_altform-unplated.png', width: 24, height: 24, kind: 'square' },
  { name: 'Square44x44Logo.targetsize-32_altform-unplated.png', width: 32, height: 32, kind: 'square' },
  { name: 'Square44x44Logo.targetsize-48_altform-unplated.png', width: 48, height: 48, kind: 'square' },
  { name: 'Square44x44Logo.targetsize-256_altform-unplated.png', width: 256, height: 256, kind: 'square' },

  // Square44x44Logo (targetsize altform-lightunplated variants — light taskbar)
  { name: 'Square44x44Logo.targetsize-16_altform-lightunplated.png', width: 16, height: 16, kind: 'light-unplated' },
  { name: 'Square44x44Logo.targetsize-24_altform-lightunplated.png', width: 24, height: 24, kind: 'light-unplated' },
  { name: 'Square44x44Logo.targetsize-32_altform-lightunplated.png', width: 32, height: 32, kind: 'light-unplated' },
  { name: 'Square44x44Logo.targetsize-48_altform-lightunplated.png', width: 48, height: 48, kind: 'light-unplated' },
  { name: 'Square44x44Logo.targetsize-256_altform-lightunplated.png', width: 256, height: 256, kind: 'light-unplated' },

  // Square71x71Logo
  { name: 'Square71x71Logo.png', width: 71, height: 71, kind: 'square' },
  { name: 'Square71x71Logo.scale-100.png', width: 71, height: 71, kind: 'square' },
  { name: 'Square71x71Logo.scale-125.png', width: 89, height: 89, kind: 'square' },
  { name: 'Square71x71Logo.scale-150.png', width: 107, height: 107, kind: 'square' },
  { name: 'Square71x71Logo.scale-200.png', width: 142, height: 142, kind: 'square' },
  { name: 'Square71x71Logo.scale-400.png', width: 284, height: 284, kind: 'square' },

  // Square150x150Logo
  { name: 'Square150x150Logo.png', width: 150, height: 150, kind: 'square' },
  { name: 'Square150x150Logo.scale-100.png', width: 150, height: 150, kind: 'square' },
  { name: 'Square150x150Logo.scale-125.png', width: 188, height: 188, kind: 'square' },
  { name: 'Square150x150Logo.scale-150.png', width: 225, height: 225, kind: 'square' },
  { name: 'Square150x150Logo.scale-200.png', width: 300, height: 300, kind: 'square' },
  { name: 'Square150x150Logo.scale-400.png', width: 600, height: 600, kind: 'square' },

  // Square310x310Logo
  { name: 'Square310x310Logo.png', width: 310, height: 310, kind: 'square' },
  { name: 'Square310x310Logo.scale-100.png', width: 310, height: 310, kind: 'square' },
  { name: 'Square310x310Logo.scale-200.png', width: 620, height: 620, kind: 'square' },

  // Wide310x150Logo
  { name: 'Wide310x150Logo.png', width: 310, height: 150, kind: 'wide' },
  { name: 'Wide310x150Logo.scale-100.png', width: 310, height: 150, kind: 'wide' },
  { name: 'Wide310x150Logo.scale-125.png', width: 388, height: 188, kind: 'wide' },
  { name: 'Wide310x150Logo.scale-150.png', width: 465, height: 225, kind: 'wide' },
  { name: 'Wide310x150Logo.scale-200.png', width: 620, height: 300, kind: 'wide' },
  { name: 'Wide310x150Logo.scale-400.png', width: 1240, height: 600, kind: 'wide' },

  // SplashScreen
  { name: 'SplashScreen.png', width: 620, height: 300, kind: 'wide' },
  { name: 'SplashScreen.scale-100.png', width: 620, height: 300, kind: 'wide' },
  { name: 'SplashScreen.scale-200.png', width: 1240, height: 600, kind: 'wide' },
];

const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function sampleCornerBackground(image: RgbaImage): Rgb {
  const corners: readonly (readonly [number, number])[] = [
    [0, 0],
    [image.width - 1, 0],
    [0, image.height - 1],
    [image.width - 1, image.height - 1],
  ];
  let sumR = 0;
  let sumG = 0;
  let sumB = 0;
  for (const [x, y] of corners) {
    const offset = (y * image.width + x) * 4;
    sumR += image.pixels[offset];
    sumG += image.pixels[offset + 1];
    sumB += image.pixels[offset + 2];
  }
  return [Math.round(sumR / 4), Math.round(sumG / 4), Math.round(sumB / 4)];
}

function maxChannelDiff(pixels: Uint8Array, offset: number, bg: Rgb): number {
  return Math.max(
    Math.abs(pixels[offset] - bg[0]),
    Math.abs(pixels[offset + 1] - bg[1]),
    Math.abs(pixels[offset + 2] - bg[2]),
  );
}

/**
 * Extract the Portato mascot emblem onto a transparent canvas.
 *
 * Uses an exterior-only 4-connected flood-fill starting from the crop borders
 * so interior white/cream artwork (snout, chest stripe, eyes, eyebrows) stays
 * 100% opaque, then computes anti-aliased alpha and color unmixing against the
 * background color on the 1-2px boundary band adjacent to the exterior so no
 * light halo appears on dark Windows taskbars.
 */
function extractPortatoEmblem(source: RgbaImage, bg: Rgb): RgbaImage {
  const crop = cropImage(source, PORTATO_EMBLEM_REGION);
  const { width, height, pixels } = crop;
  const totalPixels = width * height;
  const bgMask = new Uint8Array(totalPixels);
  const queue = new Int32Array(totalPixels);
  let tail = 0;

  const enqueueBg = (x: number, y: number): void => {
    const index = y * width + x;
    if (bgMask[index] !== 0) return;
    if (maxChannelDiff(pixels, index * 4, bg) <= PORTATO_BG_TOLERANCE) {
      bgMask[index] = 1;
      queue[tail] = index;
      tail += 1;
    }
  };

  for (let x = 0; x < width; x += 1) {
    enqueueBg(x, 0);
    enqueueBg(x, height - 1);
  }
  for (let y = 0; y < height; y += 1) {
    enqueueBg(0, y);
    enqueueBg(width - 1, y);
  }

  for (let head = 0; head < tail; head += 1) {
    const index = queue[head];
    const x = index % width;
    const y = (index - x) / width;
    if (x > 0) enqueueBg(x - 1, y);
    if (x + 1 < width) enqueueBg(x + 1, y);
    if (y > 0) enqueueBg(x, y - 1);
    if (y + 1 < height) enqueueBg(x, y + 1);
  }

  // BFS Manhattan distance from the flood-filled exterior up to 3px inward.
  const distance = new Uint8Array(totalPixels).fill(255);
  const distQueue = new Int32Array(totalPixels);
  let distTail = 0;
  for (let index = 0; index < totalPixels; index += 1) {
    if (bgMask[index] !== 0) {
      distance[index] = 0;
      distQueue[distTail] = index;
      distTail += 1;
    }
  }

  const enqueueDist = (x: number, y: number, nextDist: number): void => {
    const index = y * width + x;
    if (distance[index] <= nextDist) return;
    distance[index] = nextDist;
    distQueue[distTail] = index;
    distTail += 1;
  };

  for (let head = 0; head < distTail; head += 1) {
    const index = distQueue[head];
    const currentDist = distance[index];
    if (currentDist >= 3) continue;
    const x = index % width;
    const y = (index - x) / width;
    const nextDist = currentDist + 1;
    if (x > 0) enqueueDist(x - 1, y, nextDist);
    if (x + 1 < width) enqueueDist(x + 1, y, nextDist);
    if (y > 0) enqueueDist(x, y - 1, nextDist);
    if (y + 1 < height) enqueueDist(x, y + 1, nextDist);
  }

  const original = new Uint8Array(pixels);
  for (let index = 0; index < totalPixels; index += 1) {
    if (bgMask[index] !== 0) {
      const offset = index * 4;
      pixels[offset] = 0;
      pixels[offset + 1] = 0;
      pixels[offset + 2] = 0;
      pixels[offset + 3] = 0;
    }
  }

  const sampleInteriorColor = (cx: number, cy: number, radius: number): Rgb | null => {
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    let count = 0;
    const minY = Math.max(0, cy - radius);
    const maxY = Math.min(height - 1, cy + radius);
    const minX = Math.max(0, cx - radius);
    const maxX = Math.min(width - 1, cx + radius);
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const index = y * width + x;
        if (distance[index] < 3 || bgMask[index] !== 0) continue;
        const offset = index * 4;
        sumR += original[offset];
        sumG += original[offset + 1];
        sumB += original[offset + 2];
        count += 1;
      }
    }
    if (count === 0) return null;
    return [sumR / count, sumG / count, sumB / count];
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (distance[index] !== 1 && distance[index] !== 2) continue;

      const interior = sampleInteriorColor(x, y, 3) ?? sampleInteriorColor(x, y, 5);
      if (interior === null) continue;

      const dr = interior[0] - bg[0];
      const dg = interior[1] - bg[1];
      const db = interior[2] - bg[2];
      const denom = dr * dr + dg * dg + db * db;
      if (denom < 400) continue;

      const offset = index * 4;
      const cr = original[offset];
      const cg = original[offset + 1];
      const cb = original[offset + 2];
      const alpha = Math.max(0, Math.min(1, ((cr - bg[0]) * dr + (cg - bg[1]) * dg + (cb - bg[2]) * db) / denom));

      if (alpha < 0.05) {
        pixels[offset] = 0;
        pixels[offset + 1] = 0;
        pixels[offset + 2] = 0;
        pixels[offset + 3] = 0;
      } else if (alpha < 0.98) {
        const unmixedR = clampByte((cr - bg[0] * (1 - alpha)) / alpha);
        const unmixedG = clampByte((cg - bg[1] * (1 - alpha)) / alpha);
        const unmixedB = clampByte((cb - bg[2] * (1 - alpha)) / alpha);
        // Stabilize very low-alpha pixels against JPEG noise amplification.
        const weight = Math.min(1, alpha / 0.4);
        pixels[offset] = clampByte(interior[0] * (1 - weight) + unmixedR * weight);
        pixels[offset + 1] = clampByte(interior[1] * (1 - weight) + unmixedG * weight);
        pixels[offset + 2] = clampByte(interior[2] * (1 - weight) + unmixedB * weight);
        pixels[offset + 3] = clampByte(alpha * 255);
      }
    }
  }

  return cropImage(crop, alphaBounds(crop, 'Portato emblem'));
}

/**
 * Extract the lower `portato` wordmark with crisp anti-aliased alpha and
 * dark-navy ink color unmixing against the sampled canvas background.
 */
function extractPortatoWordmark(source: RgbaImage, bg: Rgb): RgbaImage {
  const crop = cropImage(source, PORTATO_WORDMARK_REGION);
  const { width, height, pixels } = crop;

  let inkSumR = 0;
  let inkSumG = 0;
  let inkSumB = 0;
  let inkCount = 0;
  for (let offset = 0; offset < pixels.length; offset += 4) {
    const r = pixels[offset];
    const g = pixels[offset + 1];
    const b = pixels[offset + 2];
    if (r < 30 && g < 30 && b < 45) {
      inkSumR += r;
      inkSumG += g;
      inkSumB += b;
      inkCount += 1;
    }
  }
  const ink: Rgb = inkCount > 0 ? [inkSumR / inkCount, inkSumG / inkCount, inkSumB / inkCount] : PORTATO_FALLBACK_INK;

  const dr = ink[0] - bg[0];
  const dg = ink[1] - bg[1];
  const db = ink[2] - bg[2];
  const denom = dr * dr + dg * dg + db * db;

  const result = blankImage(width, height);
  for (let offset = 0; offset < pixels.length; offset += 4) {
    const cr = pixels[offset];
    const cg = pixels[offset + 1];
    const cb = pixels[offset + 2];
    const rawAlpha = ((cr - bg[0]) * dr + (cg - bg[1]) * dg + (cb - bg[2]) * db) / denom;
    const alpha = Math.max(0, Math.min(1, (rawAlpha - 0.08) / (0.94 - 0.08)));
    if (alpha <= 0) continue;

    const unmixedR = clampByte((cr - bg[0] * (1 - alpha)) / alpha);
    const unmixedG = clampByte((cg - bg[1] * (1 - alpha)) / alpha);
    const unmixedB = clampByte((cb - bg[2] * (1 - alpha)) / alpha);
    const weight = Math.min(1, alpha / 0.5);
    result.pixels[offset] = clampByte(ink[0] * (1 - weight) + unmixedR * weight);
    result.pixels[offset + 1] = clampByte(ink[1] * (1 - weight) + unmixedG * weight);
    result.pixels[offset + 2] = clampByte(ink[2] * (1 - weight) + unmixedB * weight);
    result.pixels[offset + 3] = clampByte(alpha * 255);
  }

  return cropImage(result, alphaBounds(result, 'Portato wordmark'));
}

const PORTATO_BG = sampleCornerBackground(PORTATO_SOURCE);
const PORTATO_EMBLEM = extractPortatoEmblem(PORTATO_SOURCE, PORTATO_BG);
const PORTATO_WORDMARK = extractPortatoWordmark(PORTATO_SOURCE, PORTATO_BG);

const WINDOWS_TILE_KEYLINE_RGB: Rgb = [22, 25, 24];
const WINDOWS_TILE_KEYLINE_ALPHA = 0.22;
const WINDOWS_TILE_RADIUS_RATIO = 0.225;
const WINDOWS_TILE_SAMPLES = 8;

function drawWindowsTile(size: number, fill: Rgb): RgbaImage {
  const image = blankImage(size, size);
  const radius = size * WINDOWS_TILE_RADIUS_RATIO;
  const keylineWidth = Math.max(1, (size * 4) / CANVAS);
  const innerRadius = Math.max(0, radius - keylineWidth);
  const borderR = Math.round(
    WINDOWS_TILE_KEYLINE_RGB[0] * WINDOWS_TILE_KEYLINE_ALPHA + fill[0] * (1 - WINDOWS_TILE_KEYLINE_ALPHA),
  );
  const borderG = Math.round(
    WINDOWS_TILE_KEYLINE_RGB[1] * WINDOWS_TILE_KEYLINE_ALPHA + fill[1] * (1 - WINDOWS_TILE_KEYLINE_ALPHA),
  );
  const borderB = Math.round(
    WINDOWS_TILE_KEYLINE_RGB[2] * WINDOWS_TILE_KEYLINE_ALPHA + fill[2] * (1 - WINDOWS_TILE_KEYLINE_ALPHA),
  );

  const samples = WINDOWS_TILE_SAMPLES;
  const totalSamples = samples * samples;
  const safeEdge = Math.ceil(keylineWidth);
  const safeCorner = Math.ceil(radius);

  for (let y = 0; y < size; y += 1) {
    const inSafeY = y >= safeEdge && y + 1 <= size - safeEdge;
    const inMidY = y >= safeCorner && y + 1 <= size - safeCorner;
    for (let x = 0; x < size; x += 1) {
      const offset = (y * size + x) * 4;
      const inSafeX = x >= safeEdge && x + 1 <= size - safeEdge;
      const inMidX = x >= safeCorner && x + 1 <= size - safeCorner;

      if ((inSafeX && inMidY) || (inMidX && inSafeY)) {
        image.pixels[offset] = fill[0];
        image.pixels[offset + 1] = fill[1];
        image.pixels[offset + 2] = fill[2];
        image.pixels[offset + 3] = 255;
        continue;
      }

      let sumR = 0;
      let sumG = 0;
      let sumB = 0;
      let covered = 0;
      for (let sy = 0; sy < samples; sy += 1) {
        const py = y + (sy + 0.5) / samples;
        for (let sx = 0; sx < samples; sx += 1) {
          const px = x + (sx + 0.5) / samples;
          if (!roundedRectContains(px, py, 0, 0, size, size, radius)) continue;
          covered += 1;
          if (
            roundedRectContains(
              px,
              py,
              keylineWidth,
              keylineWidth,
              size - keylineWidth,
              size - keylineWidth,
              innerRadius,
            )
          ) {
            sumR += fill[0];
            sumG += fill[1];
            sumB += fill[2];
          } else {
            sumR += borderR;
            sumG += borderG;
            sumB += borderB;
          }
        }
      }
      if (covered === 0) continue;
      image.pixels[offset] = Math.round(sumR / covered);
      image.pixels[offset + 1] = Math.round(sumG / covered);
      image.pixels[offset + 2] = Math.round(sumB / covered);
      image.pixels[offset + 3] = Math.round((covered / totalSamples) * 255);
    }
  }

  return image;
}

function windowsAppIcon(size: number): RgbaImage {
  const image = drawWindowsTile(size, PORTATO_BG);
  const fillRatio = size <= 24 ? 0.9 : 0.86;
  const targetBox = Math.max(1, Math.round(size * fillRatio));
  const scale = Math.min(targetBox / PORTATO_EMBLEM.width, targetBox / PORTATO_EMBLEM.height);
  const targetWidth = Math.max(1, Math.round(PORTATO_EMBLEM.width * scale));
  const targetHeight = Math.max(1, Math.round(PORTATO_EMBLEM.height * scale));
  const resizedMark = resizeImage(PORTATO_EMBLEM, targetWidth, targetHeight);
  compositeAt(image, resizedMark, Math.round((size - targetWidth) / 2), Math.round((size - targetHeight) / 2));

  const cornerOffsets = [3, (size - 1) * 4 + 3, (size - 1) * size * 4 + 3, (size * size - 1) * 4 + 3];
  for (const alphaOffset of cornerOffsets) {
    if (image.pixels[alphaOffset] !== 0) {
      throw new Error(`windowsAppIcon(${size}) outer corner must remain transparent`);
    }
  }
  return image;
}

/**
 * Light-theme taskbar unplated icon: renders the dark-navy Portato mascot
 * directly on a transparent canvas so it achieves strong silhouette contrast
 * against Windows 11's light taskbar (#F3F3F3) without a cream backplate.
 */
function windowsLightUnplatedIcon(size: number): RgbaImage {
  const image = blankImage(size, size);
  const fillRatio = size <= 24 ? 0.94 : 0.9;
  const targetBox = Math.max(1, Math.round(size * fillRatio));
  const scale = Math.min(targetBox / PORTATO_EMBLEM.width, targetBox / PORTATO_EMBLEM.height);
  const targetWidth = Math.max(1, Math.round(PORTATO_EMBLEM.width * scale));
  const targetHeight = Math.max(1, Math.round(PORTATO_EMBLEM.height * scale));
  const resizedMark = resizeImage(PORTATO_EMBLEM, targetWidth, targetHeight);
  compositeAt(image, resizedMark, Math.round((size - targetWidth) / 2), Math.round((size - targetHeight) / 2));

  const cornerOffsets = [3, (size - 1) * 4 + 3, (size - 1) * size * 4 + 3, (size * size - 1) * 4 + 3];
  for (const alphaOffset of cornerOffsets) {
    if (image.pixels[alphaOffset] !== 0) {
      throw new Error(`windowsLightUnplatedIcon(${size}) outer corner must remain transparent`);
    }
  }
  return image;
}

function drawWideTile(width: number, height: number, fill: Rgb): RgbaImage {
  const ss = width >= 1000 ? 2 : APP_SS;
  const renderWidth = width * ss;
  const renderHeight = height * ss;
  const tile = blankImage(renderWidth, renderHeight);

  const margin = scaled(SQUIRCLE_MARGIN, height, ss);
  const left = margin;
  const top = margin;
  const right = renderWidth - margin;
  const bottom = renderHeight - margin;
  const cardHeight = bottom - top;
  const radius = Math.round(cardHeight * 0.225);
  const mask = new Uint8Array(renderWidth * renderHeight);

  for (let y = 0; y < renderHeight; y += 1) {
    for (let x = 0; x < renderWidth; x += 1) {
      if (roundedRectContains(x + 0.5, y + 0.5, left, top, right, bottom, radius)) {
        mask[y * renderWidth + x] = 255;
      }
    }
  }

  const shadow = blankImage(renderWidth, renderHeight);
  for (let pixel = 0; pixel < mask.length; pixel += 1) {
    shadow.pixels[pixel * 4 + 3] = Math.trunc((mask[pixel] * 54) / 255);
  }
  const blurredShadow: RgbaImage = {
    width: renderWidth,
    height: renderHeight,
    pixels: Transformer.fromRgbaPixels(shadow.pixels, renderWidth, renderHeight)
      .blur(Math.max(1, scaled(18, height, ss)))
      .rawPixelsSync(),
  };
  compositeAt(tile, blurredShadow);

  for (let pixel = 0; pixel < mask.length; pixel += 1) {
    if (mask[pixel] === 0) continue;
    const offset = pixel * 4;
    tile.pixels[offset] = fill[0];
    tile.pixels[offset + 1] = fill[1];
    tile.pixels[offset + 2] = fill[2];
    tile.pixels[offset + 3] = 255;
  }

  const keylineWidth = Math.max(ss, scaled(2.5, height, ss));
  const keyline = new Uint8Array([22, 25, 24, 38]);
  for (let y = 0; y < renderHeight; y += 1) {
    for (let x = 0; x < renderWidth; x += 1) {
      const px = x + 0.5;
      const py = y + 0.5;
      const outer = roundedRectContains(px, py, left, top, right, bottom, radius);
      const inner = roundedRectContains(
        px,
        py,
        left + keylineWidth,
        top + keylineWidth,
        right - keylineWidth,
        bottom - keylineWidth,
        Math.max(0, radius - keylineWidth),
      );
      if (outer && !inner) sourceOverPixel(tile.pixels, (y * renderWidth + x) * 4, keyline, 0);
    }
  }

  return resizeImage(tile, width, height);
}

function widePortatoTile(width: number, height: number): RgbaImage {
  const image = drawWideTile(width, height, PORTATO_BG);
  const margin = scaled(SQUIRCLE_MARGIN, height);
  const cardWidth = width - margin * 2;
  const cardHeight = height - margin * 2;

  const baseEmblemHeight = cardHeight * 0.68;
  const baseEmblemWidth = (PORTATO_EMBLEM.width * baseEmblemHeight) / PORTATO_EMBLEM.height;
  const baseWordmarkHeight = baseEmblemHeight * 0.5;
  const baseWordmarkWidth = (PORTATO_WORDMARK.width * baseWordmarkHeight) / PORTATO_WORDMARK.height;
  const baseGap = baseEmblemHeight * 0.16;
  const fitScale = Math.min(1, (cardWidth * 0.86) / (baseEmblemWidth + baseGap + baseWordmarkWidth));

  const emblemWidth = Math.max(1, Math.round(baseEmblemWidth * fitScale));
  const emblemHeight = Math.max(1, Math.round(baseEmblemHeight * fitScale));
  const wordmarkWidth = Math.max(1, Math.round(baseWordmarkWidth * fitScale));
  const wordmarkHeight = Math.max(1, Math.round(baseWordmarkHeight * fitScale));
  const gap = Math.max(1, Math.round(baseGap * fitScale));

  const totalWidth = emblemWidth + gap + wordmarkWidth;
  const startX = Math.round((width - totalWidth) / 2);
  const emblemY = Math.round((height - emblemHeight) / 2);
  const wordmarkY = Math.round((height - wordmarkHeight) / 2);

  compositeAt(image, resizeImage(PORTATO_EMBLEM, emblemWidth, emblemHeight), startX, emblemY);
  compositeAt(
    image,
    resizeImage(PORTATO_WORDMARK, wordmarkWidth, wordmarkHeight),
    startX + emblemWidth + gap,
    wordmarkY,
  );
  return image;
}

function validateRgbaImage(image: RgbaImage, expectedWidth: number, expectedHeight: number, label: string): void {
  if (image.width !== expectedWidth || image.height !== expectedHeight) {
    throw new Error(`Expected ${expectedWidth}x${expectedHeight} for ${label}, got ${image.width}x${image.height}`);
  }
  if (image.pixels.length !== expectedWidth * expectedHeight * 4) {
    throw new Error(`Invalid RGBA buffer length for ${label}: ${image.pixels.length}`);
  }

  let maxAlpha = 0;
  let minAlpha = 255;
  let hasForegroundInk = false;
  for (let offset = 0; offset < image.pixels.length; offset += 4) {
    const alpha = image.pixels[offset + 3];
    if (alpha > maxAlpha) maxAlpha = alpha;
    if (alpha < minAlpha) minAlpha = alpha;
    if (alpha > 0 && (image.pixels[offset] < 240 || image.pixels[offset + 1] < 240 || image.pixels[offset + 2] < 240)) {
      hasForegroundInk = true;
    }
  }
  if (maxAlpha !== 255) throw new Error(`${label} must contain opaque pixels`);
  if (minAlpha !== 0) throw new Error(`${label} must contain transparent background pixels`);
  if (!hasForegroundInk) throw new Error(`${label} must contain non-white foreground pixels`);
}

interface ValidateEncodedPngOptions {
  allowIndexed?: boolean;
  maxBytes?: number;
}

function validateEncodedPng(
  png: Buffer,
  expectedWidth: number,
  expectedHeight: number,
  label: string,
  options: ValidateEncodedPngOptions = {},
): void {
  if (png.length < 33 || !png.subarray(0, 8).equals(PNG_SIGNATURE)) {
    throw new Error(`${label} is not a valid PNG`);
  }
  if (options.maxBytes !== undefined && png.length >= options.maxBytes) {
    throw new Error(`${label} PNG size (${png.length} bytes) must be strictly under ${options.maxBytes} bytes`);
  }
  if (png.readUInt32BE(8) !== 13 || png.subarray(12, 16).toString('ascii') !== 'IHDR') {
    throw new Error(`${label} is missing PNG IHDR chunk`);
  }
  const width = png.readUInt32BE(16);
  const height = png.readUInt32BE(20);
  const bitDepth = png[24];
  const colorType = png[25];
  if (width !== expectedWidth || height !== expectedHeight) {
    throw new Error(`${label} PNG IHDR dimensions ${width}x${height} do not match ${expectedWidth}x${expectedHeight}`);
  }

  let offset = 8;
  let hasPlte = false;
  let plteLength = 0;
  let hasTrns = false;
  let trnsLength = 0;
  let hasIdat = false;
  let hasIend = false;
  while (offset + 12 <= png.length) {
    const chunkLength = png.readUInt32BE(offset);
    const chunkType = png.subarray(offset + 4, offset + 8).toString('ascii');
    const nextOffset = offset + 12 + chunkLength;
    if (nextOffset > png.length) {
      throw new Error(`${label} PNG chunk ${chunkType} exceeds buffer bounds`);
    }
    if (chunkType === 'PLTE') {
      hasPlte = true;
      plteLength = chunkLength;
    } else if (chunkType === 'tRNS') {
      hasTrns = true;
      trnsLength = chunkLength;
    } else if (chunkType === 'IDAT') {
      hasIdat = true;
    } else if (chunkType === 'IEND') {
      hasIend = true;
      offset = nextOffset;
      break;
    }
    offset = nextOffset;
  }

  if (!hasIdat || !hasIend || offset !== png.length) {
    throw new Error(`${label} PNG has malformed chunk framing`);
  }

  if (colorType === 6) {
    if (bitDepth !== 8) {
      throw new Error(`${label} RGBA PNG bit depth must be 8, got ${bitDepth}`);
    }
  } else if (colorType === 3 && options.allowIndexed) {
    if (bitDepth !== 1 && bitDepth !== 2 && bitDepth !== 4 && bitDepth !== 8) {
      throw new Error(`${label} indexed PNG has invalid bit depth ${bitDepth}`);
    }
    if (!hasPlte || plteLength === 0 || plteLength > 768 || plteLength % 3 !== 0) {
      throw new Error(`${label} indexed PNG is missing a valid PLTE chunk`);
    }
    if (!hasTrns || trnsLength === 0 || trnsLength > plteLength / 3) {
      throw new Error(`${label} indexed PNG is missing a valid tRNS alpha chunk`);
    }
  } else {
    const expected = options.allowIndexed ? 'RGBA (6) or indexed palette (3)' : 'RGBA (6)';
    throw new Error(`${label} PNG color type must be ${expected}, got ${colorType}`);
  }
}

function optimizeMsixPng(png: Buffer, width: number, height: number, label: string): Buffer {
  let candidate = png;
  if (candidate.length >= MAX_MSIX_ASSET_BYTES) {
    const lossless = losslessCompressPngSync(png, {
      strip: true,
      colorTypeReduction: false,
      grayscaleReduction: false,
    });
    if (lossless.length < candidate.length) candidate = lossless;
  }
  if (candidate.length >= MAX_MSIX_ASSET_BYTES) {
    candidate = pngQuantizeSync(png, { colors: 256, minQuality: 70, speed: 1 });
  }
  if (candidate.length >= MAX_MSIX_ASSET_BYTES) {
    candidate = pngQuantizeSync(png, { colors: 128, minQuality: 60, speed: 1 });
  }
  validateEncodedPng(candidate, width, height, label, {
    allowIndexed: true,
    maxBytes: MAX_MSIX_ASSET_BYTES,
  });
  return candidate;
}

interface IcoEntry {
  size: number;
  png: Buffer;
}

/**
 * Encode multiple PNG-compressed RGBA images into a Windows `.ico` container.
 *
 * Layout per the ICO specification:
 * - 6-byte ICONDIR header: reserved=0, type=1 (ICO), count=N
 * - N * 16-byte ICONDIRENTRY records (width/height byte is 0 for 256px)
 * - Raw PNG payloads back-to-back
 */
function encodeIco(entries: readonly IcoEntry[]): Buffer {
  if (entries.length === 0 || entries.length > 0xffff) {
    throw new Error(`ICO must contain between 1 and 65535 images, got ${entries.length}`);
  }

  const headerSize = 6 + entries.length * 16;
  let totalPayload = 0;
  for (const entry of entries) {
    if (!Number.isInteger(entry.size) || entry.size < 1 || entry.size > 256) {
      throw new Error(`ICO image size must be an integer in 1..256, got ${entry.size}`);
    }
    validateEncodedPng(entry.png, entry.size, entry.size, `icon.ico (${entry.size}x${entry.size})`);
    totalPayload += entry.png.length;
  }

  const buffer = Buffer.alloc(headerSize + totalPayload);
  buffer.writeUInt16LE(0, 0); // idReserved
  buffer.writeUInt16LE(1, 2); // idType = 1 (ICO)
  buffer.writeUInt16LE(entries.length, 4); // idCount

  let dataOffset = headerSize;
  for (let index = 0; index < entries.length; index += 1) {
    const { size, png } = entries[index];
    const entryOffset = 6 + index * 16;
    const dimByte = size === 256 ? 0 : size;
    buffer[entryOffset] = dimByte; // bWidth (0 means 256)
    buffer[entryOffset + 1] = dimByte; // bHeight (0 means 256)
    buffer[entryOffset + 2] = 0; // bColorCount (0 for >= 8bpp / PNG)
    buffer[entryOffset + 3] = 0; // bReserved
    buffer.writeUInt16LE(1, entryOffset + 4); // wPlanes
    buffer.writeUInt16LE(32, entryOffset + 6); // wBitCount (32-bit RGBA)
    buffer.writeUInt32LE(png.length, entryOffset + 8); // dwBytesInRes
    buffer.writeUInt32LE(dataOffset, entryOffset + 12); // dwImageOffset

    png.copy(buffer, dataOffset);
    dataOffset += png.length;
  }

  return buffer;
}

function validateIco(ico: Buffer, expectedSizes: readonly number[]): void {
  const headerSize = 6 + expectedSizes.length * 16;
  if (ico.length < headerSize) {
    throw new Error(`ICO buffer too small (${ico.length} bytes)`);
  }
  if (ico.readUInt16LE(0) !== 0 || ico.readUInt16LE(2) !== 1 || ico.readUInt16LE(4) !== expectedSizes.length) {
    throw new Error('ICO header does not match expected ICONDIR');
  }
  for (let index = 0; index < expectedSizes.length; index += 1) {
    const expectedSize = expectedSizes[index];
    const entryOffset = 6 + index * 16;
    const widthByte = ico[entryOffset];
    const heightByte = ico[entryOffset + 1];
    const expectedByte = expectedSize === 256 ? 0 : expectedSize;
    if (widthByte !== expectedByte || heightByte !== expectedByte) {
      throw new Error(
        `ICO entry ${index} dimensions (${widthByte}x${heightByte}) do not match expected ${expectedSize}x${expectedSize}`,
      );
    }
    const byteLength = ico.readUInt32LE(entryOffset + 8);
    const imageOffset = ico.readUInt32LE(entryOffset + 12);
    if (byteLength === 0 || imageOffset < headerSize || imageOffset + byteLength > ico.length) {
      throw new Error(`ICO entry ${index} has invalid payload range (${imageOffset}+${byteLength})`);
    }
    validateEncodedPng(
      ico.subarray(imageOffset, imageOffset + byteLength),
      expectedSize,
      expectedSize,
      `icon.ico[${index}]`,
    );
  }
}

function main(): void {
  // Everything that can reject does so before the first byte is written, so a
  // bad source leaves the committed assets untouched rather than half-updated.
  const tray1x = trayTemplate(16);
  const tray2x = trayTemplate(32);
  validateTemplate(tray1x, 16);
  validateTemplate(tray2x, 32);
  const tray1xPng = encodePng(tray1x);
  const tray2xPng = encodePng(tray2x);

  const renderedPngCache = new Map<string, Buffer>();
  const getMacSquarePng = (size: number): Buffer => {
    const key = `mac-square:${size}`;
    const cached = renderedPngCache.get(key);
    if (cached !== undefined) return cached;
    const image = appIcon(size);
    validateRgbaImage(image, size, size, `appIcon(${size})`);
    const png = encodePng(image);
    validateEncodedPng(png, size, size, `appIcon(${size})`);
    renderedPngCache.set(key, png);
    return png;
  };
  const getWindowsSquarePng = (size: number): Buffer => {
    const key = `win-square:${size}`;
    const cached = renderedPngCache.get(key);
    if (cached !== undefined) return cached;
    const image = windowsAppIcon(size);
    validateRgbaImage(image, size, size, `windowsAppIcon(${size})`);
    const png = encodePng(image);
    validateEncodedPng(png, size, size, `windowsAppIcon(${size})`);
    renderedPngCache.set(key, png);
    return png;
  };
  const getWindowsLightUnplatedPng = (size: number): Buffer => {
    const key = `win-light-unplated:${size}`;
    const cached = renderedPngCache.get(key);
    if (cached !== undefined) return cached;
    const image = windowsLightUnplatedIcon(size);
    validateRgbaImage(image, size, size, `windowsLightUnplatedIcon(${size})`);
    const png = encodePng(image);
    validateEncodedPng(png, size, size, `windowsLightUnplatedIcon(${size})`);
    renderedPngCache.set(key, png);
    return png;
  };
  const getWidePng = (width: number, height: number): Buffer => {
    const key = `wide:${width}x${height}`;
    const cached = renderedPngCache.get(key);
    if (cached !== undefined) return cached;
    const image = widePortatoTile(width, height);
    validateRgbaImage(image, width, height, `widePortatoTile(${width}x${height})`);
    const png = encodePng(image);
    validateEncodedPng(png, width, height, `widePortatoTile(${width}x${height})`);
    renderedPngCache.set(key, png);
    return png;
  };

  const getMsixPng = (spec: MsixAssetSpec): Buffer => {
    const key = `msix:${spec.kind}:${spec.width}x${spec.height}`;
    const cached = renderedPngCache.get(key);
    if (cached !== undefined) return cached;
    const rawPng =
      spec.kind === 'square'
        ? getWindowsSquarePng(spec.width)
        : spec.kind === 'light-unplated'
          ? getWindowsLightUnplatedPng(spec.width)
          : getWidePng(spec.width, spec.height);
    const png = optimizeMsixPng(rawPng, spec.width, spec.height, spec.name);
    renderedPngCache.set(key, png);
    return png;
  };

  const icoEntries: IcoEntry[] = ICO_SIZES.map((size) => ({ size, png: getWindowsSquarePng(size) }));
  const icoBuffer = encodeIco(icoEntries);
  validateIco(icoBuffer, ICO_SIZES);

  const msixOutputs = MSIX_ASSETS.map((spec) => ({
    name: spec.name,
    png: getMsixPng(spec),
  }));

  const iconsetOutputs =
    process.platform === 'darwin'
      ? [16, 32, 128, 256, 512].flatMap((points) => [
          { name: `icon_${points}x${points}.png`, png: getMacSquarePng(points) },
          { name: `icon_${points}x${points}@2x.png`, png: getMacSquarePng(points * 2) },
        ])
      : [];

  const icns = join(HERE, 'icon.icns');
  const loggedOutputs: string[] = [];

  if (process.platform === 'darwin') {
    const iconset = join(HERE, 'icon.iconset');
    mkdirSync(iconset, { recursive: true });

    // These are the ten names `iconutil` requires.
    for (const output of iconsetOutputs) {
      writeFileSync(join(iconset, output.name), output.png);
    }

    execFileSync('iconutil', ['-c', 'icns', iconset, '-o', icns], {
      stdio: 'inherit',
    });
    loggedOutputs.push(icns);
  } else {
    console.log(`${relative(REPO_ROOT, icns)}  skipped (iconutil is macOS-only)`);
  }

  const icoPath = join(HERE, 'icon.ico');
  writeFileSync(icoPath, icoBuffer);
  loggedOutputs.push(icoPath);

  const msixDir = join(HERE, 'msix');
  mkdirSync(msixDir, { recursive: true });
  let msixBytes = 0;
  for (const output of msixOutputs) {
    const outPath = join(msixDir, output.name);
    writeFileSync(outPath, output.png);
    msixBytes += output.png.length;
  }

  const tray1xPath = join(HERE, 'iconTemplate.png');
  const tray2xPath = join(HERE, 'iconTemplate@2x.png');
  writeFileSync(tray1xPath, tray1xPng);
  writeFileSync(tray2xPath, tray2xPng);
  loggedOutputs.push(tray1xPath, tray2xPath);

  for (const path of loggedOutputs) {
    console.log(`${relative(REPO_ROOT, path)}  ${statSync(path).size.toLocaleString('en-US')} bytes`);
  }
  console.log(
    `${relative(REPO_ROOT, msixDir)}  ${msixOutputs.length} files (${msixBytes.toLocaleString('en-US')} bytes)`,
  );
}

main();
