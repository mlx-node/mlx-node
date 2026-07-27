#!/usr/bin/env python3
"""
Generate the macOS icon set from the mlx-node mark.

`images/logo.png` cannot be used for either output. It is a 1024x1024 RGB image
with NO alpha channel, and its content is a horizontal WORDMARK -- "MLX" plus a
green hexagon, inked across x 98..925 / y 404..629, a 3.7:1 box. Scaled into a
square app icon that wordmark is illegible by 32px, and a menubar template image
must be pure black plus alpha, which the gradient X and the green hexagon cannot
survive. So the hexagon is promoted to the standalone mark and everything else is
dropped.

The hexagon in the wordmark is a POINTY-TOP regular hexagon: its bounding box is
128x148, and 128/148 = 0.865, which is sqrt(3)/2 to three places. Both outputs
reproduce that geometry rather than approximating it.

    icon.iconset/ -> icon.icns      app icon, 10 images, 16..1024 with @2x
    iconTemplate.png / @2x          menubar, black + alpha, macOS recolors it

PLACEHOLDER. The geometry, alpha, colour space and sizes are correct, so the
signing and packaging pipeline can be exercised end to end -- but this is derived
art, not a designed mark. Swapping in a real master means re-running this script
(or replacing its two outputs); nothing downstream hard-codes the shape.

    python3 packages/desktop/build/make-icons.py
"""

import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent

# Sampled from images/logo.png: the most common pixel inside the hexagon.
GREEN = (99, 152, 98, 255)
# The wordmark's ink is near-black; the squircle picks it up so the icon reads as
# part of the same identity rather than a lone green tile.
INK = (23, 25, 24, 255)

# Big Sur+ app-icon grid: the rounded square occupies 824 of the 1024 canvas,
# leaving a 100px margin on every side for the shadow the system composites.
# Apple's own corner radius is ~22.5% of the square's edge.
CANVAS = 1024
SQUIRCLE = 824
RADIUS = int(SQUIRCLE * 0.225)

# Supersample, then downsample with LANCZOS. Drawing a hexagon at final size
# leaves visibly stepped diagonals -- there is no anti-aliased polygon fill in
# PIL, so the smoothing has to come from the resample.
SS = 4


def pointy_top_hexagon(cx: float, cy: float, height: float) -> list[tuple[float, float]]:
    """Vertices of a regular pointy-top hexagon, matching the wordmark's."""
    r = height / 2.0  # circumradius: vertex to vertex vertically
    # Start at -90deg so a vertex points up, then every 60deg.
    return [
        (cx + r * math.cos(math.radians(-90 + 60 * i)), cy + r * math.sin(math.radians(-90 + 60 * i)))
        for i in range(6)
    ]


def app_icon() -> Image.Image:
    img = Image.new('RGBA', (CANVAS * SS, CANVAS * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    margin = (CANVAS - SQUIRCLE) // 2
    d.rounded_rectangle(
        [margin * SS, margin * SS, (margin + SQUIRCLE) * SS, (margin + SQUIRCLE) * SS],
        radius=RADIUS * SS,
        fill=INK,
    )

    # The glyph reads best at roughly half the square's edge; smaller and it
    # floats, larger and it crowds the corner radius at 16px.
    d.polygon(pointy_top_hexagon(CANVAS / 2 * SS, CANVAS / 2 * SS, SQUIRCLE * 0.52 * SS), fill=GREEN)

    return img.resize((CANVAS, CANVAS), Image.LANCZOS)


def tray_template(size: int) -> Image.Image:
    """
    A menubar template image: pure black, shaped entirely by its alpha channel.

    macOS ignores the colour and recolours the silhouette itself for light mode,
    dark mode and the active/highlighted state. Emitting anything but black would
    still render, but would fight that recolouring on one of those states.
    """
    img = Image.new('RGBA', (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Leave ~1px of breathing room at 1x so the glyph does not touch the menubar
    # text baseline box.
    d.polygon(pointy_top_hexagon(size / 2 * SS, size / 2 * SS, (size - 2) * SS), fill=(0, 0, 0, 255))
    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    iconset = HERE / 'icon.iconset'
    iconset.mkdir(parents=True, exist_ok=True)
    master = app_icon()

    # The 10 names `iconutil` expects. Anything missing and it fails outright.
    for pt in (16, 32, 128, 256, 512):
        master.resize((pt, pt), Image.LANCZOS).save(iconset / f'icon_{pt}x{pt}.png')
        master.resize((pt * 2, pt * 2), Image.LANCZOS).save(iconset / f'icon_{pt}x{pt}@2x.png')

    icns = HERE / 'icon.icns'
    subprocess.run(['iconutil', '-c', 'icns', str(iconset), '-o', str(icns)], check=True)

    tray_template(16).save(HERE / 'iconTemplate.png')
    tray_template(32).save(HERE / 'iconTemplate@2x.png')

    for p in (icns, HERE / 'iconTemplate.png', HERE / 'iconTemplate@2x.png'):
        print(f'{p.relative_to(HERE.parent.parent.parent)}  {p.stat().st_size:,} bytes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
