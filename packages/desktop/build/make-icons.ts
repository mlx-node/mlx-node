/**
 * Generate the macOS app and menu-bar icons from the mlx-node logo.
 *
 * `images/logo.png` is the canonical MLX wordmark plus green node used by the
 * mlx-node GitHub organization. The app icon presents that mark directly on a
 * macOS-style white tile. At 16px and 32px it uses the logo's compact X + node
 * portion as an optical variant, because shrinking the 3.7:1 wordmark that far
 * makes every letter unreadable.
 *
 * The menu-bar asset uses a dedicated hex-M mark generated for this app:
 * `tray-icon-source.png` is thresholded and resized into a pure black-and-alpha
 * template. macOS recolours that silhouette for light, dark and highlighted
 * states.
 *
 *     icon.iconset/ -> icon.icns      app icon, 10 images, 16..1024 with @2x
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

import { CompressionType, FilterType, ResizeFilterType, ResizeFit, Transformer } from '@napi-rs/image';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..', '..', '..');
const LOGO_PATH = join(REPO_ROOT, 'images', 'logo.png');
const TRAY_SOURCE_PATH = join(HERE, 'tray-icon-source.png');

const CANVAS = 1024;
const SQUIRCLE = 824;
const SQUIRCLE_RADIUS = Math.trunc(SQUIRCLE * 0.225);
const SQUIRCLE_MARGIN = Math.trunc((CANVAS - SQUIRCLE) / 2);
const APP_SS = 4;
const TRAY_SS = 8;

type Bounds = readonly [left: number, top: number, right: number, bottom: number];

interface RgbaImage {
  width: number;
  height: number;
  pixels: Uint8Array;
}

// Bounds measured from the canonical 1024px logo.
const WORDMARK_BOUNDS: Bounds = [98, 404, 926, 630];
const COMPACT_BOUNDS: Bounds = [552, 404, 926, 630];

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

const SOURCE_LOGO = decodeLogo();
const TRAY_SOURCE = decodeOpaqueImage(TRAY_SOURCE_PATH);

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

function alphaBounds(image: RgbaImage): Bounds {
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

  if (right < left || bottom < top) throw new Error('Tray source does not contain a foreground mark');
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

function writePng(path: string, image: RgbaImage): void {
  writeFileSync(path, encodePng(image));
}

function main(): void {
  // Everything that can reject does so before the first byte is written, so a
  // bad source leaves the committed assets untouched rather than half-updated.
  const tray1x = trayTemplate(16);
  const tray2x = trayTemplate(32);
  validateTemplate(tray1x, 16);
  validateTemplate(tray2x, 32);

  const iconset = join(HERE, 'icon.iconset');
  mkdirSync(iconset, { recursive: true });

  // These are the ten names `iconutil` requires.
  for (const points of [16, 32, 128, 256, 512]) {
    writePng(join(iconset, `icon_${points}x${points}.png`), appIcon(points));
    writePng(join(iconset, `icon_${points}x${points}@2x.png`), appIcon(points * 2));
  }

  const icns = join(HERE, 'icon.icns');
  execFileSync('iconutil', ['-c', 'icns', iconset, '-o', icns], {
    stdio: 'inherit',
  });

  const tray1xPath = join(HERE, 'iconTemplate.png');
  const tray2xPath = join(HERE, 'iconTemplate@2x.png');
  writePng(tray1xPath, tray1x);
  writePng(tray2xPath, tray2x);

  for (const path of [icns, tray1xPath, tray2xPath]) {
    console.log(`${relative(REPO_ROOT, path)}  ${statSync(path).size.toLocaleString('en-US')} bytes`);
  }
}

main();
