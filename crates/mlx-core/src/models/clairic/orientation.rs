//! Conservative page orientation from detached dots beside Latin letter stems.
//!
//! Pure-Rust port of upstream `clairic_runtime/orientation.py` without OpenCV.
//! Examines glyph geometry only (BT.601 grayscale -> Lanczos3 thumbnail to 1600x1600 ->
//! Otsu binary-inverse threshold -> 8-connected components -> stem/dot voting + median
//! letter aspect ratio). Pages without enough consistent evidence retain 0° rotation.

use image::{
    GrayImage, RgbImage,
    imageops::{self, FilterType},
};

/// Maximum thumbnail edge length matching Pillow `page.thumbnail((1600, 1600))`.
const THUMBNAIL_MAX_EDGE: u32 = 1600;
/// Maximum total connected components considered before bailing out conservatively.
const MAX_COMPONENTS: usize = 12_000;
/// Maximum stem candidates before bailing out conservatively.
const MAX_STEMS: usize = 1_200;
/// Maximum dot candidates before bailing out conservatively.
const MAX_DOTS: usize = 2_400;
/// Minimum letter-like components required to establish line orientation.
const MIN_LETTERS: usize = 80;
/// Minimum winning votes required to apply a rotation.
const MIN_WINNER_VOTES: usize = 12;

/// Bounding box and pixel area of an 8-connected foreground component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentStat {
    pub left: u32,
    pub top: u32,
    pub width: u32,
    pub height: u32,
    pub area: u32,
}

/// Convert an RGB image to BT.601 grayscale matching Pillow's `image.convert("L")`.
///
/// Uses Pillow's exact fixed-point ITU-R 601-2 coefficients:
/// `L = (19595 * R + 38470 * G + 7471 * B + 32768) >> 16`
/// (equivalent to `0.299 * R + 0.587 * G + 0.114 * B` rounded to nearest `u8`).
pub fn rgb_to_luma_bt601(image: &RgbImage) -> Vec<u8> {
    let raw = image.as_raw();
    let pixel_count = (image.width() as usize).saturating_mul(image.height() as usize);
    let mut out = Vec::with_capacity(pixel_count);
    let (chunks, _) = raw.as_chunks::<3>();
    for chunk in chunks {
        let r = u32::from(chunk[0]);
        let g = u32::from(chunk[1]);
        let b = u32::from(chunk[2]);
        let luma = (r * 19_595 + g * 38_470 + b * 7_471 + 32_768) >> 16;
        out.push(luma as u8);
    }
    out
}

/// Downscale a grayscale image in-place-style to fit within `(max_w, max_h)` while
/// preserving aspect ratio using Lanczos3, matching Pillow's `Image.thumbnail`.
fn thumbnail_gray(image: &RgbImage, max_edge: u32) -> (Vec<u8>, u32, u32) {
    let w = image.width();
    let h = image.height();
    let luma = rgb_to_luma_bt601(image);
    if w == 0 || h == 0 || (w <= max_edge && h <= max_edge) {
        return (luma, w, h);
    }
    let Some(gray) = GrayImage::from_raw(w, h, luma) else {
        return (Vec::new(), 0, 0);
    };
    let (new_w, new_h) = if w >= h {
        let scaled_h =
            ((f64::from(h) * f64::from(max_edge) / f64::from(w)).round() as u32).clamp(1, max_edge);
        (max_edge, scaled_h)
    } else {
        let scaled_w =
            ((f64::from(w) * f64::from(max_edge) / f64::from(h)).round() as u32).clamp(1, max_edge);
        (scaled_w, max_edge)
    };
    let resized = imageops::resize(&gray, new_w, new_h, FilterType::Lanczos3);
    (resized.into_raw(), new_w, new_h)
}

/// Compute the Otsu global threshold matching OpenCV's `getThreshVal_Otsu_8u`.
///
/// With `THRESH_BINARY_INV | THRESH_OTSU`, foreground (ink) pixels satisfy `pixel <= thresh`.
pub fn otsu_threshold(pixels: &[u8]) -> u8 {
    if pixels.is_empty() {
        return 0;
    }
    let mut hist = [0u64; 256];
    for &p in pixels {
        hist[usize::from(p)] += 1;
    }

    let scale = 1.0 / (pixels.len() as f64);
    let mut mu = 0.0f64;
    for (i, &count) in hist.iter().enumerate() {
        mu += (i as f64) * (count as f64);
    }
    mu *= scale;

    let mut mu1 = 0.0f64;
    let mut q1 = 0.0f64;
    let mut max_sigma = 0.0f64;
    let mut max_val: u8 = 0;

    for (i, &count) in hist.iter().enumerate() {
        let p_i = (count as f64) * scale;
        mu1 *= q1;
        q1 += p_i;
        let q2 = 1.0 - q1;

        if q1.min(q2) < f64::EPSILON || q1.max(q2) > 1.0 - f64::EPSILON {
            continue;
        }

        mu1 = (mu1 + (i as f64) * p_i) / q1;
        let mu2 = (mu - q1 * mu1) / q2;
        let diff = mu1 - mu2;
        let sigma = q1 * q2 * diff * diff;
        if sigma > max_sigma {
            max_sigma = sigma;
            max_val = i as u8;
        }
    }

    max_val
}

/// Union-Find (Disjoint Set Union) with path compression for two-pass 8-CCL.
struct DisjointSet {
    parent: Vec<u32>,
}

impl DisjointSet {
    fn new() -> Self {
        Self { parent: vec![0] }
    }

    fn make_set(&mut self) -> u32 {
        let id = self.parent.len() as u32;
        self.parent.push(id);
        id
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = self.parent[x as usize];
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) -> u32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        if ra < rb {
            self.parent[rb as usize] = ra;
            ra
        } else {
            self.parent[ra as usize] = rb;
            rb
        }
    }
}

/// Pure-Rust two-pass 8-connected components labeling returning per-component stats
/// (excluding background component 0), matching `cv2.connectedComponentsWithStats(..., connectivity=8)`.
pub fn connected_components_8(mask: &[bool], width: u32, height: u32) -> Vec<ComponentStat> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || mask.len() != w.saturating_mul(h) {
        return Vec::new();
    }

    let mut labels = vec![0u32; mask.len()];
    let mut dsu = DisjointSet::new();

    // Pass 1: assign provisional labels and record equivalences across 8-neighbors
    // already visited: NW (x-1, y-1), N (x, y-1), NE (x+1, y-1), W (x-1, y).
    for y in 0..h {
        let row_off = y * w;
        let prev_row_off = y.saturating_sub(1) * w;
        for x in 0..w {
            let idx = row_off + x;
            if !mask[idx] {
                continue;
            }
            let mut neighbors = [0u32; 4];
            let mut n_count = 0usize;

            if y > 0 {
                if x > 0 {
                    let l = labels[prev_row_off + x - 1];
                    if l != 0 {
                        neighbors[n_count] = l;
                        n_count += 1;
                    }
                }
                let l = labels[prev_row_off + x];
                if l != 0 {
                    neighbors[n_count] = l;
                    n_count += 1;
                }
                if x + 1 < w {
                    let l = labels[prev_row_off + x + 1];
                    if l != 0 {
                        neighbors[n_count] = l;
                        n_count += 1;
                    }
                }
            }
            if x > 0 {
                let l = labels[row_off + x - 1];
                if l != 0 {
                    neighbors[n_count] = l;
                    n_count += 1;
                }
            }

            if n_count == 0 {
                labels[idx] = dsu.make_set();
            } else {
                let mut root = dsu.find(neighbors[0]);
                for &other in &neighbors[1..n_count] {
                    root = dsu.union(root, other);
                }
                labels[idx] = root;
            }
        }
    }

    // Flatten roots and assign dense 0-based component IDs.
    let num_provisional = dsu.parent.len();
    let mut root_to_comp = vec![u32::MAX; num_provisional];
    let mut comp_count = 0usize;
    for label in 1..num_provisional {
        let root = dsu.find(label as u32) as usize;
        if root_to_comp[root] == u32::MAX {
            root_to_comp[root] = comp_count as u32;
            comp_count += 1;
        }
        root_to_comp[label] = root_to_comp[root];
    }

    if comp_count == 0 {
        return Vec::new();
    }

    // Pass 2: accumulate bounding boxes and pixel areas.
    let mut min_x = vec![u32::MAX; comp_count];
    let mut max_x = vec![0u32; comp_count];
    let mut min_y = vec![u32::MAX; comp_count];
    let mut max_y = vec![0u32; comp_count];
    let mut area = vec![0u32; comp_count];

    for y in 0..h {
        let y32 = y as u32;
        let row_off = y * w;
        for x in 0..w {
            let label = labels[row_off + x];
            if label == 0 {
                continue;
            }
            let cid = root_to_comp[label as usize] as usize;
            let x32 = x as u32;
            if x32 < min_x[cid] {
                min_x[cid] = x32;
            }
            if x32 > max_x[cid] {
                max_x[cid] = x32;
            }
            if y32 < min_y[cid] {
                min_y[cid] = y32;
            }
            if y32 > max_y[cid] {
                max_y[cid] = y32;
            }
            area[cid] += 1;
        }
    }

    let mut stats = Vec::with_capacity(comp_count);
    for cid in 0..comp_count {
        if area[cid] == 0 {
            continue;
        }
        stats.push(ComponentStat {
            left: min_x[cid],
            top: min_y[cid],
            width: max_x[cid] - min_x[cid] + 1,
            height: max_y[cid] - min_y[cid] + 1,
            area: area[cid],
        });
    }
    stats
}

/// Compute the median of a non-empty `f64` slice matching `numpy.median`.
fn median_f64(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Estimate clockwise rotation in degrees (`0`, `90`, `180`, or `270`) needed to
/// restore an upright page orientation, together with the 4-way stem/dot vote counts.
///
/// Exact port of `clairic_runtime/orientation.py::document_rotation`.
pub fn document_rotation(image: &RgbImage) -> (u16, [usize; 4]) {
    let mut votes = [0usize; 4];
    let (pixels, w, h) = thumbnail_gray(image, THUMBNAIL_MAX_EDGE);
    if pixels.is_empty() || w == 0 || h == 0 {
        return (0, votes);
    }

    let thresh = otsu_threshold(&pixels);
    let ink: Vec<bool> = pixels.iter().map(|&p| p <= thresh).collect();
    let stats = connected_components_8(&ink, w, h);

    if stats.is_empty() || stats.len() > MAX_COMPONENTS {
        return (0, votes);
    }

    let mut stems: Vec<&ComponentStat> = Vec::new();
    let mut dots: Vec<&ComponentStat> = Vec::new();

    for stat in &stats {
        let short = stat.width.min(stat.height);
        let long = stat.width.max(stat.height);
        if (9..=60).contains(&long) && f64::from(long) >= 2.5 * f64::from(short) && stat.area >= 8 {
            stems.push(stat);
        }
        if stat.width <= 12 && stat.height <= 12 && (2..=100).contains(&stat.area) {
            dots.push(stat);
        }
    }

    if stems.is_empty() || dots.is_empty() || stems.len() > MAX_STEMS || dots.len() > MAX_DOTS {
        return (0, votes);
    }

    for stem in &stems {
        let sx = f64::from(stem.left);
        let sy = f64::from(stem.top);
        let sw = f64::from(stem.width);
        let sh = f64::from(stem.height);
        let sa = f64::from(stem.area);

        let vertical = sh > sw;
        let center = if vertical {
            sx + sw / 2.0
        } else {
            sy + sh / 2.0
        };
        let (start, length) = if vertical { (sy, sh) } else { (sx, sw) };
        let thin = if vertical { sw } else { sh };
        let align_tol = 1.5_f64.max(thin * 0.6);
        let max_dot_size = length * 0.4;
        let max_dot_cross = 3.0_f64.max(thin * 2.0);
        let max_dot_area = sa * 0.7;
        let max_gap = length * 0.45;

        let mut any_before = false;
        let mut any_after = false;

        for dot in &dots {
            let dx = f64::from(dot.left);
            let dy = f64::from(dot.top);
            let dw = f64::from(dot.width);
            let dh = f64::from(dot.height);
            let da = f64::from(dot.area);

            let across = if vertical {
                dx + dw / 2.0
            } else {
                dy + dh / 2.0
            };
            let (left, size) = if vertical { (dy, dh) } else { (dx, dw) };
            let cross = if vertical { dw } else { dh };

            if (across - center).abs() > align_tol
                || size > max_dot_size
                || cross > max_dot_cross
                || da >= max_dot_area
            {
                continue;
            }

            if left + size <= start && (start - left - size) <= max_gap {
                any_before = true;
            }
            if left >= start + length && (left - start - length) <= max_gap {
                any_after = true;
            }
            if any_before && any_after {
                break;
            }
        }

        if vertical {
            votes[0] += usize::from(any_before);
            votes[2] += usize::from(any_after);
        } else {
            votes[1] += usize::from(any_before);
            votes[3] += usize::from(any_after);
        }
    }

    let mut letter_aspects: Vec<f64> = Vec::new();
    for stat in &stats {
        let short = stat.width.min(stat.height);
        let long = stat.width.max(stat.height);
        if stat.area >= 10 && short >= 3 && long <= 60 && long <= 4 * short {
            letter_aspects.push(f64::from(stat.width) / f64::from(stat.height));
        }
    }
    if letter_aspects.len() < MIN_LETTERS {
        return (0, votes);
    }

    let aspect = median_f64(&mut letter_aspects);
    let directions = if aspect < 0.8 {
        (0usize, 2usize)
    } else if aspect > 1.25 {
        (1usize, 3usize)
    } else {
        return (0, votes);
    };

    // Text-line direction excludes accidental dot/stem pairs in the other axis.
    let winner = if votes[directions.1] > votes[directions.0] {
        directions.1
    } else {
        directions.0
    };
    let runner = if winner == directions.1 {
        votes[directions.0]
    } else {
        votes[directions.1]
    };

    if votes[winner] < MIN_WINNER_VOTES
        || votes[winner] < 3 * runner.max(1)
        || (votes[winner] as f64) < 0.7 * ((votes[winner] + runner) as f64)
    {
        return (0, votes);
    }

    ((winner * 90) as u16, votes)
}

/// Rotate an RGB image clockwise by `0`, `90`, `180`, or `270` degrees.
pub fn rotate_cw(image: &RgbImage, degrees: u16) -> RgbImage {
    match degrees % 360 {
        90 => imageops::rotate90(image),
        180 => imageops::rotate180(image),
        270 => imageops::rotate270(image),
        _ => image.clone(),
    }
}

/// Detect page orientation and rotate clockwise to upright if sufficient evidence exists.
pub fn orient_document(image: &RgbImage) -> RgbImage {
    let (rotation, _) = document_rotation(image);
    if rotation == 0 {
        image.clone()
    } else {
        rotate_cw(image, rotation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    fn fill_rect(img: &mut RgbImage, x0: u32, y0: u32, w: u32, h: u32, color: Rgb<u8>) {
        for y in y0..(y0 + h).min(img.height()) {
            for x in x0..(x0 + w).min(img.width()) {
                img.put_pixel(x, y, color);
            }
        }
    }

    /// Build a synthetic upright document page containing Latin `i`-like stems with
    /// detached dots above them plus regular tall letter glyphs (aspect < 0.8).
    fn synthetic_upright_text_page() -> RgbImage {
        let mut img = RgbImage::from_pixel(600, 500, Rgb([255, 255, 255]));
        let black = Rgb([10, 10, 10]);

        // Draw 6 rows of 18 glyphs each (108 letters total >= 80).
        // Every 3rd glyph is an `i` (4x20 stem + 4x4 dot 4px above);
        // other glyphs are tall narrow letters (10x24 -> aspect ~0.42 < 0.8).
        for row in 0..6u32 {
            let base_y = 40 + row * 68;
            for col in 0..18u32 {
                let x = 40 + col * 28;
                if col % 3 == 0 {
                    // Detached dot above stem
                    fill_rect(&mut img, x + 3, base_y, 4, 4, black);
                    // Vertical stem (4x20 -> long=20, short=4, long >= 2.5*short, area=80)
                    fill_rect(&mut img, x + 3, base_y + 8, 4, 20, black);
                } else {
                    // Tall letter glyph (10x24 -> aspect = 0.416 < 0.8)
                    fill_rect(&mut img, x, base_y + 4, 10, 24, black);
                }
            }
        }
        img
    }

    #[test]
    fn otsu_and_ccl_basic() {
        let mut img = RgbImage::from_pixel(40, 40, Rgb([255, 255, 255]));
        fill_rect(&mut img, 5, 5, 4, 12, Rgb([20, 20, 20]));
        fill_rect(&mut img, 20, 20, 3, 3, Rgb([20, 20, 20]));
        let luma = rgb_to_luma_bt601(&img);
        let thresh = otsu_threshold(&luma);
        assert!((20..255).contains(&thresh));
        let mask: Vec<bool> = luma.iter().map(|&p| p <= thresh).collect();
        let stats = connected_components_8(&mask, 40, 40);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].area, 48);
        assert_eq!(stats[1].area, 9);
    }

    #[test]
    fn blank_page_returns_zero_rotation() {
        let blank = RgbImage::from_pixel(200, 200, Rgb([255, 255, 255]));
        let (rot, votes) = document_rotation(&blank);
        assert_eq!(rot, 0);
        assert_eq!(votes, [0, 0, 0, 0]);
    }

    #[test]
    fn detects_and_restores_all_four_orientations() {
        let upright = synthetic_upright_text_page();

        // 0°: already upright
        let (rot0, votes0) = document_rotation(&upright);
        assert_eq!(rot0, 0, "upright votes={votes0:?}");
        assert!(votes0[0] >= 12);

        // Rotate upright page by 270° CW (= 90° CCW) -> needs 90° CW to restore
        let rot_ccw_90 = rotate_cw(&upright, 270);
        let (detected_90, votes_90) = document_rotation(&rot_ccw_90);
        assert_eq!(detected_90, 90, "ccw90 votes={votes_90:?}");
        let restored_90 = orient_document(&rot_ccw_90);
        assert_eq!(restored_90.dimensions(), upright.dimensions());
        assert_eq!(restored_90.as_raw(), upright.as_raw());

        // Rotate upright page by 180° -> needs 180° CW to restore
        let rot_180 = rotate_cw(&upright, 180);
        let (detected_180, votes_180) = document_rotation(&rot_180);
        assert_eq!(detected_180, 180, "rot180 votes={votes_180:?}");
        let restored_180 = orient_document(&rot_180);
        assert_eq!(restored_180.as_raw(), upright.as_raw());

        // Rotate upright page by 90° CW -> needs 270° CW to restore
        let rot_cw_90 = rotate_cw(&upright, 90);
        let (detected_270, votes_270) = document_rotation(&rot_cw_90);
        assert_eq!(detected_270, 270, "cw90 votes={votes_270:?}");
        let restored_270 = orient_document(&rot_cw_90);
        assert_eq!(restored_270.dimensions(), upright.dimensions());
        assert_eq!(restored_270.as_raw(), upright.as_raw());
    }
}
