use std::collections::VecDeque;
use std::io::Cursor;

use image::metadata::Orientation;
use image::{
    DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits, RgbImage, Rgba, RgbaImage,
};
use napi::bindgen_prelude::{Error, Result};

use crate::array::attention::scaled_dot_product_attention;
use crate::array::{DType, MxArray};
use crate::nn::{Activations, LayerNorm, Linear};
use crate::tools::inkling::{CONTENT_IMAGE_ID, IMAGE_PAD_ID};
use crate::vision::validate::{MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, image_processing_error};

use super::config::Inkling2VisionConfig;

/// Maximum merged vision tokens per image (matches llama.cpp `mtmd-image.cpp` cap of 4096).
const MAX_VISION_TOKENS: usize = 4096;

/// Preprocessed image tensor ready for [`Inkling2VisionEncoder::encode`].
pub struct PreprocessedImage {
    /// Normalized pixel values `[H_pad, W_pad, 3]` in FP32.
    pub pixels: MxArray,
    /// Number of merged token rows (`H_pad / (patch_size * spatial_merge_size)`).
    pub grid_rows: usize,
    /// Number of merged token cols (`W_pad / (patch_size * spatial_merge_size)`).
    pub grid_cols: usize,
}

impl PreprocessedImage {
    pub fn num_tokens(&self) -> usize {
        self.grid_rows * self.grid_cols
    }
}

/// Preprocess raw image bytes (PNG/JPEG/WebP/etc.) according to Inkling2's PE-Lang spec:
/// 1. Apply EXIF orientation transpose before dimension/grid calculations.
/// 2. Optionally downscale only if aligned area exceeds `MAX_VISION_TOKENS` (`4096` merged cells)
///    using bilinear resampling (`FilterType::Triangle`; premultiplied RGBA for transparent images).
/// 3. Composite alpha channel (if any) onto an 8×8 checkerboard (`204` / `153`) at final size.
/// 4. Pad bottom & right to multiples of `alignment = patch_size * spatial_merge_size` (`40`)
///    using pixel value **`-1.0`**.
/// 5. Normalize per channel: `(v / 255.0 - mean[c]) / std[c]` (for `mean=0.5, std=0.5`,
///    equals `2.0 * (v / 255.0) - 1.0`, mapping padding `-1` to `-257/255`).
pub fn preprocess_image_bytes(
    bytes: &[u8],
    config: &Inkling2VisionConfig,
) -> Result<PreprocessedImage> {
    if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
        return Err(image_processing_error(format!(
            "Inkling2 vision: image payload length {} out of range 1..={MAX_IMAGE_BYTES}",
            bytes.len()
        )));
    }
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| {
            image_processing_error(format!("Inkling2 vision: failed to read image: {e}"))
        })?;
    match reader.format() {
        Some(ImageFormat::Jpeg | ImageFormat::Png | ImageFormat::Gif | ImageFormat::WebP) => {}
        other => {
            return Err(image_processing_error(format!(
                "Inkling2 vision: unsupported image format {other:?}"
            )));
        }
    }
    let mut limits = Limits::default();
    limits.max_alloc = Some(MAX_IMAGE_PIXELS.saturating_mul(16).max(128 * 1024 * 1024));
    reader.limits(limits.clone());
    let mut decoder = reader.into_decoder().map_err(|e| {
        image_processing_error(format!("Inkling2 vision: failed to decode image: {e}"))
    })?;
    let (w, h) = decoder.dimensions();
    if w == 0 || h == 0 || u64::from(w).saturating_mul(u64::from(h)) > MAX_IMAGE_PIXELS {
        return Err(image_processing_error(format!(
            "Inkling2 vision: image dimensions {w}x{h} exceed {MAX_IMAGE_PIXELS} pixel limit"
        )));
    }
    // Enforce the same output-buffer reservation and remaining decoder allocation
    // budget that `ImageReader::decode` applies before `DynamicImage::from_decoder`.
    limits.reserve(decoder.total_bytes()).map_err(|e| {
        image_processing_error(format!(
            "Inkling2 vision: decoded image exceeds allocation limit: {e}"
        ))
    })?;
    decoder.set_limits(limits).map_err(|e| {
        image_processing_error(format!(
            "Inkling2 vision: failed to apply decoder allocation limits: {e}"
        ))
    })?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).map_err(|e| {
        image_processing_error(format!("Inkling2 vision: failed to decode image: {e}"))
    })?;
    img.apply_orientation(orientation);
    preprocess_dynamic_image(&img, config)
}

#[inline]
fn triangle_kernel(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 1.0 { 1.0 - ax } else { 0.0 }
}

/// Compute 1-D Triangle (bilinear) filter support `[left, right)`, pixel-center coordinate,
/// and reciprocal weight sum `inv_sum` matching `image::imageops::FilterType::Triangle`.
#[inline]
fn triangle_axis_span(out_idx: u32, src_len: u32, ratio: f32, sratio: f32) -> (u32, u32, f32, f32) {
    let input_c = (out_idx as f32 + 0.5) * ratio;
    let left = ((input_c - sratio).floor() as i64).clamp(0, i64::from(src_len) - 1) as u32;
    let right =
        ((input_c + sratio).ceil() as i64).clamp(i64::from(left) + 1, i64::from(src_len)) as u32;
    let center = input_c - 0.5;
    let mut sum = 0.0f32;
    for k in left..right {
        sum += triangle_kernel((k as f32 - center) / sratio);
    }
    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    (left, right, center, inv_sum)
}

#[inline]
fn triangle_axis_span_counted(
    out_idx: u32,
    src_len: u32,
    ratio: f32,
    sratio: f32,
    evals: Option<&mut u64>,
) -> (u32, u32, f32, f32) {
    let span = triangle_axis_span(out_idx, src_len, ratio, sratio);
    if let Some(count) = evals {
        *count += u64::from(span.1 - span.0);
    }
    span
}

#[inline]
fn unpremultiply_rgba_pixel(px: [f32; 4]) -> Rgba<u8> {
    let a_f = px[3].clamp(0.0, 1.0);
    let a_u8 = (a_f * 255.0).round().clamp(0.0, 255.0) as u8;
    if a_f <= 0.0 || a_u8 == 0 {
        Rgba([0, 0, 0, 0])
    } else {
        let r = ((px[0] / a_f) * 255.0).round().clamp(0.0, 255.0) as u8;
        let g = ((px[1] / a_f) * 255.0).round().clamp(0.0, 255.0) as u8;
        let b = ((px[2] / a_f) * 255.0).round().clamp(0.0, 255.0) as u8;
        Rgba([r, g, b, a_u8])
    }
}

/// Bilinear resize of an RGBA image in premultiplied-alpha space (matching PIL's
/// `Image.resize(..., Image.Resampling.BILINEAR)` so hidden RGB values under `alpha == 0`
/// cannot bleed into opaque borders).
///
/// Streams row-by-row directly from `&RgbaImage` (on-the-fly `u8 -> f32` premultiply) so we
/// never allocate a full-source `Rgba32FImage` (`16 B/src-pixel`); working memory is bounded
/// to at most a few `target_w`-wide `f32x4` rows.
fn resize_rgba_premultiplied_bilinear(rgba: &RgbaImage, target_w: u32, target_h: u32) -> RgbaImage {
    resize_rgba_premultiplied_bilinear_inner(rgba, target_w, target_h, None)
}

fn resize_rgba_premultiplied_bilinear_inner(
    rgba: &RgbaImage,
    target_w: u32,
    target_h: u32,
    mut v_weight_evals: Option<&mut u64>,
) -> RgbaImage {
    let (w, h) = rgba.dimensions();
    let mut out = RgbaImage::new(target_w, target_h);
    if w == 0 || h == 0 || target_w == 0 || target_h == 0 {
        return out;
    }

    let h_ratio = w as f32 / target_w as f32;
    let h_sratio = h_ratio.max(1.0);
    // Precompute horizontal filter taps when the flat weight table stays <= ~2 MiB;
    // fall back to on-the-fly tap evaluation for extreme ultra-wide strips.
    let precompute_h = u64::from(w) + u64::from(target_w) <= 262_144;
    let mut h_spans: Vec<(u32, usize, usize)> = Vec::new();
    let mut h_weights: Vec<f32> = Vec::new();
    if precompute_h {
        h_spans.reserve(target_w as usize);
        for outx in 0..target_w {
            let (left, right, center, inv_sum) = triangle_axis_span(outx, w, h_ratio, h_sratio);
            let start = h_weights.len();
            for x in left..right {
                h_weights.push(triangle_kernel((x as f32 - center) / h_sratio) * inv_sum);
            }
            h_spans.push((left, start, h_weights.len() - start));
        }
    }

    let resample_row_horizontal = |y: u32, dst_row: &mut [[f32; 4]]| {
        if precompute_h {
            for (outx, &(left, start, len)) in h_spans.iter().enumerate() {
                let mut acc = [0.0f32; 4];
                for (off, &wt) in h_weights[start..start + len].iter().enumerate() {
                    if wt == 0.0 {
                        continue;
                    }
                    let p = rgba.get_pixel(left + off as u32, y).0;
                    let a = f32::from(p[3]) / 255.0;
                    acc[0] += ((f32::from(p[0]) / 255.0) * a) * wt;
                    acc[1] += ((f32::from(p[1]) / 255.0) * a) * wt;
                    acc[2] += ((f32::from(p[2]) / 255.0) * a) * wt;
                    acc[3] += a * wt;
                }
                dst_row[outx] = acc;
            }
        } else {
            for outx in 0..target_w {
                let (left, right, center, inv_sum) = triangle_axis_span(outx, w, h_ratio, h_sratio);
                let mut acc = [0.0f32; 4];
                for x in left..right {
                    let wt = triangle_kernel((x as f32 - center) / h_sratio) * inv_sum;
                    if wt == 0.0 {
                        continue;
                    }
                    let p = rgba.get_pixel(x, y).0;
                    let a = f32::from(p[3]) / 255.0;
                    acc[0] += ((f32::from(p[0]) / 255.0) * a) * wt;
                    acc[1] += ((f32::from(p[1]) / 255.0) * a) * wt;
                    acc[2] += ((f32::from(p[2]) / 255.0) * a) * wt;
                    acc[3] += a * wt;
                }
                dst_row[outx as usize] = acc;
            }
        }
    };

    let v_ratio = h as f32 / target_h as f32;
    let v_sratio = v_ratio.max(1.0);
    let row_len = target_w as usize;

    if h >= target_h {
        // Downscale / 1:1 push-accumulate: with Triangle support == v_ratio, each source row
        // contributes to at most 2-3 active output rows regardless of vertical scale factor.
        // Cache `pending_span` so each output row's vertical weight sum is computed ONCE
        // rather than re-scanned on every intervening source row (`O(h + target_h)` total).
        struct ActiveOutRow {
            outy: u32,
            v_right: u32,
            center_y: f32,
            inv_sum_y: f32,
            accum: Vec<[f32; 4]>,
        }

        let mut next_outy = 0u32;
        let mut pending_span = Some(triangle_axis_span_counted(
            0,
            h,
            v_ratio,
            v_sratio,
            v_weight_evals.as_deref_mut(),
        ));
        let mut active: VecDeque<ActiveOutRow> = VecDeque::with_capacity(4);
        let mut free_bufs: Vec<Vec<[f32; 4]>> = Vec::with_capacity(4);
        let mut h_row = vec![[0.0f32; 4]; row_len];

        for y in 0..h {
            while let Some((v_left, v_right, center_y, inv_sum_y)) = pending_span {
                if v_left > y {
                    break;
                }
                let mut accum = free_bufs
                    .pop()
                    .unwrap_or_else(|| vec![[0.0f32; 4]; row_len]);
                accum.fill([0.0; 4]);
                active.push_back(ActiveOutRow {
                    outy: next_outy,
                    v_right,
                    center_y,
                    inv_sum_y,
                    accum,
                });
                next_outy += 1;
                pending_span = if next_outy < target_h {
                    Some(triangle_axis_span_counted(
                        next_outy,
                        h,
                        v_ratio,
                        v_sratio,
                        v_weight_evals.as_deref_mut(),
                    ))
                } else {
                    None
                };
            }

            if !active.is_empty() {
                resample_row_horizontal(y, &mut h_row);
                if let Some(count) = v_weight_evals.as_deref_mut() {
                    *count += active.len() as u64;
                }
                for row in &mut active {
                    let vy = triangle_kernel((y as f32 - row.center_y) / v_sratio) * row.inv_sum_y;
                    if vy != 0.0 {
                        for (acc, &hp) in row.accum.iter_mut().zip(&h_row) {
                            acc[0] += hp[0] * vy;
                            acc[1] += hp[1] * vy;
                            acc[2] += hp[2] * vy;
                            acc[3] += hp[3] * vy;
                        }
                    }
                }
            }

            while let Some(row) = active.pop_front() {
                if y + 1 < row.v_right {
                    active.push_front(row);
                    break;
                }
                for (outx, &px) in row.accum.iter().enumerate() {
                    out.put_pixel(outx as u32, row.outy, unpremultiply_rgba_pixel(px));
                }
                free_bufs.push(row.accum);
            }
        }
    } else {
        // Vertical upscale pull-window: with v_sratio == 1.0, each output row samples at most
        // 3 consecutive source rows (`O(target_h)` total vertical weight evaluations).
        let mut cached_start_y = 0u32;
        let mut cached_rows: VecDeque<Vec<[f32; 4]>> = VecDeque::with_capacity(4);
        let mut free_bufs: Vec<Vec<[f32; 4]>> = Vec::with_capacity(4);
        let mut out_row = vec![[0.0f32; 4]; row_len];

        for outy in 0..target_h {
            let (v_left, v_right, center_y, inv_sum_y) = triangle_axis_span_counted(
                outy,
                h,
                v_ratio,
                v_sratio,
                v_weight_evals.as_deref_mut(),
            );
            while cached_start_y < v_left {
                if let Some(buf) = cached_rows.pop_front() {
                    free_bufs.push(buf);
                }
                cached_start_y += 1;
            }
            if cached_rows.is_empty() {
                cached_start_y = v_left;
            }
            while cached_start_y + (cached_rows.len() as u32) < v_right {
                let src_y = cached_start_y + (cached_rows.len() as u32);
                let mut buf = free_bufs
                    .pop()
                    .unwrap_or_else(|| vec![[0.0f32; 4]; row_len]);
                resample_row_horizontal(src_y, &mut buf);
                cached_rows.push_back(buf);
            }

            out_row.fill([0.0; 4]);
            if let Some(count) = v_weight_evals.as_deref_mut() {
                *count += u64::from(v_right - v_left);
            }
            for y in v_left..v_right {
                let vy = triangle_kernel((y as f32 - center_y) / v_sratio) * inv_sum_y;
                if vy != 0.0 {
                    let src_row = &cached_rows[(y - cached_start_y) as usize];
                    for (acc, &hp) in out_row.iter_mut().zip(src_row) {
                        acc[0] += hp[0] * vy;
                        acc[1] += hp[1] * vy;
                        acc[2] += hp[2] * vy;
                        acc[3] += hp[3] * vy;
                    }
                }
            }
            for (outx, &px) in out_row.iter().enumerate() {
                out.put_pixel(outx as u32, outy, unpremultiply_rgba_pixel(px));
            }
        }
    }

    out
}

/// Composite an RGBA image onto the Inkling2 8×8 `(204, 153)` grey checkerboard using
/// exact integer rounding `(rgb * a + bg * (255 - a) + 127) / 255`.
fn composite_rgba_onto_checkerboard(rgba: &RgbaImage) -> RgbImage {
    let (w, h) = rgba.dimensions();
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y).0;
            let a = u16::from(p[3]);
            let bg: u16 = if ((x / 8) + (y / 8)) % 2 == 0 {
                204
            } else {
                153
            };
            let inv_a = 255 - a;
            let r = ((u16::from(p[0]) * a + bg * inv_a + 127) / 255) as u8;
            let g = ((u16::from(p[1]) * a + bg * inv_a + 127) / 255) as u8;
            let b = ((u16::from(p[2]) * a + bg * inv_a + 127) / 255) as u8;
            out.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    out
}

/// Compute largest aspect-preserving `(new_w, new_h)` with `1 <= new_w <= w`, `1 <= new_h <= h`,
/// and padded merged-token count `new_h.div_ceil(alignment) * new_w.div_ceil(alignment) <= max_tokens`.
pub(crate) fn fit_vision_dimensions(
    w: usize,
    h: usize,
    alignment: usize,
    max_tokens: usize,
) -> (usize, usize) {
    let grid_ok = |cw: usize, ch: usize| -> bool {
        ch.div_ceil(alignment)
            .saturating_mul(cw.div_ceil(alignment))
            <= max_tokens
    };
    if grid_ok(w, h) {
        return (w, h);
    }
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut best = (1usize, 1usize);
    for _ in 0..56 {
        let mid = 0.5 * (lo + hi);
        let cw = ((w as f64 * mid).floor() as usize).clamp(1, w);
        let ch = ((h as f64 * mid).floor() as usize).clamp(1, h);
        if grid_ok(cw, ch) {
            best = (cw, ch);
            lo = mid;
        } else {
            hi = mid;
        }
    }
    best
}

pub fn preprocess_dynamic_image(
    img: &DynamicImage,
    config: &Inkling2VisionConfig,
) -> Result<PreprocessedImage> {
    let alignment = config.alignment();
    if alignment == 0 {
        return Err(Error::from_reason(
            "Inkling2 vision: alignment (patch_size * spatial_merge_size) must be > 0",
        ));
    }

    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);
    if orig_w == 0 || orig_h == 0 {
        return Err(Error::from_reason(
            "image_processing_error: Could not process image (Inkling2 vision: empty image)",
        ));
    }

    // 1. Cap oversized images at MAX_VISION_TOKENS merged tokens (never upscaling, always
    //    checking the post-padding grid) using bilinear resampling (`FilterType::Triangle`).
    // 2. For transparent images, resize in premultiplied RGBA space FIRST and then composite
    //    onto the 8×8 `(204, 153)` checkerboard at final `(target_w, target_h)` so checkerboard
    //    squares stay crisp 8×8 pixels (matching Splash `server/images.py:249-298`).
    let (target_w, target_h) = fit_vision_dimensions(orig_w, orig_h, alignment, MAX_VISION_TOKENS);
    let rgb = if img.color().has_alpha() {
        let mut rgba = img.to_rgba8();
        if target_w != orig_w || target_h != orig_h {
            rgba = resize_rgba_premultiplied_bilinear(&rgba, target_w as u32, target_h as u32);
        }
        composite_rgba_onto_checkerboard(&rgba)
    } else {
        let mut rgb = img.to_rgb8();
        if target_w != orig_w || target_h != orig_h {
            rgb = image::imageops::resize(
                &rgb,
                target_w as u32,
                target_h as u32,
                image::imageops::FilterType::Triangle,
            );
        }
        rgb
    };
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);

    // 3 & 4. Pad bottom/right to multiples of `alignment` with `-1.0` and normalize.
    let grid_rows = h.div_ceil(alignment);
    let grid_cols = w.div_ceil(alignment);
    let total_tokens = grid_rows.saturating_mul(grid_cols);
    if total_tokens > MAX_VISION_TOKENS {
        return Err(Error::from_reason(format!(
            "Inkling2 vision: padded token grid {grid_rows}x{grid_cols} ({total_tokens}) exceeds limit {MAX_VISION_TOKENS}"
        )));
    }
    let h_pad = grid_rows * alignment;
    let w_pad = grid_cols * alignment;
    let inv_std = [
        1.0_f32 / config.image_std[0],
        1.0_f32 / config.image_std[1],
        1.0_f32 / config.image_std[2],
    ];
    let pad_norm = [
        (-1.0_f32 / 255.0 - config.image_mean[0]) * inv_std[0],
        (-1.0_f32 / 255.0 - config.image_mean[1]) * inv_std[1],
        (-1.0_f32 / 255.0 - config.image_mean[2]) * inv_std[2],
    ];

    let mut data = vec![0.0_f32; h_pad * w_pad * 3];
    for y in 0..h_pad {
        for x in 0..w_pad {
            let dst = (y * w_pad + x) * 3;
            if y < h && x < w {
                let px = rgb.get_pixel(x as u32, y as u32).0;
                data[dst] = (f32::from(px[0]) / 255.0 - config.image_mean[0]) * inv_std[0];
                data[dst + 1] = (f32::from(px[1]) / 255.0 - config.image_mean[1]) * inv_std[1];
                data[dst + 2] = (f32::from(px[2]) / 255.0 - config.image_mean[2]) * inv_std[2];
            } else {
                data[dst] = pad_norm[0];
                data[dst + 1] = pad_norm[1];
                data[dst + 2] = pad_norm[2];
            }
        }
    }

    let pixels = MxArray::from_float32(&data, &[h_pad as i64, w_pad as i64, 3])?;
    Ok(PreprocessedImage {
        pixels,
        grid_rows: h_pad / alignment,
        grid_cols: w_pad / alignment,
    })
}

/// Expand each `<|content_image|>` (`200005`) token in `prompt_ids` into
/// `[200005, 200054, ..., 200054]` (`count` copies of `IMAGE_PAD_ID = 200054`).
pub fn expand_image_tokens(prompt_ids: &[u32], image_token_counts: &[usize]) -> Result<Vec<u32>> {
    let total_pad: usize = image_token_counts.iter().sum();
    let mut expanded = Vec::with_capacity(prompt_ids.len() + total_pad);
    let mut img_idx = 0usize;
    for &tok in prompt_ids {
        expanded.push(tok);
        if tok == CONTENT_IMAGE_ID {
            let count = *image_token_counts.get(img_idx).ok_or_else(|| {
                Error::from_reason(format!(
                    "expand_image_tokens: prompt has more <|content_image|> markers than provided images ({})",
                    image_token_counts.len()
                ))
            })?;
            expanded.extend(std::iter::repeat_n(IMAGE_PAD_ID, count));
            img_idx += 1;
        }
    }
    if img_idx != image_token_counts.len() {
        return Err(Error::from_reason(format!(
            "expand_image_tokens: provided {} image(s) but prompt had {img_idx} <|content_image|> marker(s)",
            image_token_counts.len()
        )));
    }
    Ok(expanded)
}

/// Dense linear layer for the PE-Lang ViT that keeps matrix multiplications in `weight.dtype()`
/// (BF16 for BF16/Q8_0 checkpoints, FP32 for F32 checkpoints) while preserving an optional
/// higher-precision FP32 bias without triggering `addmm` weight promotion.
pub struct VisionLinear {
    pub linear: Linear,
    pub f32_bias: Option<MxArray>,
}

impl VisionLinear {
    pub fn from_weights(weight: &MxArray, bias: Option<&MxArray>) -> Result<Self> {
        let w_dtype = weight.dtype()?;
        if let Some(b) = bias
            && b.dtype()? != w_dtype
        {
            let linear = Linear::from_weights(weight, None)?;
            let b_shape = b.shape()?;
            let out_features = weight.shape_at(0)?;
            if b_shape.len() != 1 || b_shape[0] != out_features {
                return Err(Error::from_reason(format!(
                    "Inkling2 vision: bias shape {:?} does not match weight out_features {out_features}",
                    b_shape.as_ref()
                )));
            }
            let f32_bias = if b.dtype()? == DType::Float32 {
                b.clone()
            } else {
                b.astype(DType::Float32)?
            };
            return Ok(Self {
                linear,
                f32_bias: Some(f32_bias),
            });
        }
        Ok(Self {
            linear: Linear::from_weights(weight, bias)?,
            f32_bias: None,
        })
    }

    pub fn weight_dtype(&self) -> Result<DType> {
        self.linear.get_weight().dtype()
    }

    /// Run `x @ W^T + b` with GEMM in `weight.dtype()` and return in `out_dtype`.
    pub fn forward_cast(&self, x: &MxArray, out_dtype: DType) -> Result<MxArray> {
        let w_dtype = self.weight_dtype()?;
        let x_in = if x.dtype()? == w_dtype {
            x.clone()
        } else {
            x.astype(w_dtype)?
        };
        let y = self.linear.forward(&x_in)?;
        let y = if let Some(ref b) = self.f32_bias {
            let y_f32 = if y.dtype()? == DType::Float32 {
                y
            } else {
                y.astype(DType::Float32)?
            };
            y_f32.add(b)?
        } else {
            y
        };
        if y.dtype()? == out_dtype {
            Ok(y)
        } else {
            y.astype(out_dtype)
        }
    }

    /// Run `x @ W^T + b` with GEMM in `weight.dtype()` and return in FP32 for residual scaling.
    pub fn forward_f32(&self, x: &MxArray) -> Result<MxArray> {
        self.forward_cast(x, DType::Float32)
    }

    pub fn weight_arrays(&self) -> Vec<MxArray> {
        let mut arrays = Vec::with_capacity(2);
        arrays.push(self.linear.get_weight());
        if let Some(b) = self.linear.get_bias() {
            arrays.push(b);
        }
        if let Some(ref b) = self.f32_bias {
            arrays.push(b.clone());
        }
        arrays
    }

    pub(crate) fn ensure_cuda_device(&mut self) -> Result<usize> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(0);
        }
        let mut promoted = self.linear.ensure_cuda_device()?;
        if let Some(ref mut b) = self.f32_bias {
            promoted += b.promote_cuda_device()?;
        }
        Ok(promoted)
    }
}

/// One ViT block of the 23-layer PE-Lang vision encoder.
pub struct Inkling2VisionBlock {
    pub norm1: LayerNorm,
    pub q_proj: VisionLinear,
    pub k_proj: VisionLinear,
    pub v_proj: VisionLinear,
    pub o_proj: VisionLinear,
    /// LayerScale 1 (`[hidden_size]`, kept in FP32).
    pub ls1: MxArray,
    pub norm2: LayerNorm,
    pub fc1: VisionLinear,
    pub fc2: VisionLinear,
    /// LayerScale 2 (`[hidden_size]`, kept in FP32).
    pub ls2: MxArray,
}

impl Inkling2VisionBlock {
    /// Forward one ViT block keeping `hidden` and both LayerScale residual updates in FP32,
    /// casting to `matmul_dtype` (`q_proj` weight dtype) only after `norm1`/`norm2` for the
    /// linear projections and bidirectional SDPA (matching Splash `InklingVision.cpp` &
    /// `dense_linear.metal`).
    pub fn forward(
        &self,
        hidden: &MxArray,
        cos_table: &MxArray,
        sin_table: &MxArray,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<MxArray> {
        let n = hidden.shape()?[0];
        let d = hidden.shape()?[1];
        let matmul_dtype = self.q_proj.weight_dtype()?;

        let x_norm = self.norm1.forward(hidden)?;
        let x = if x_norm.dtype()? == matmul_dtype {
            x_norm
        } else {
            x_norm.astype(matmul_dtype)?
        };
        let q = self.q_proj.forward_cast(&x, matmul_dtype)?.reshape(&[
            n,
            num_heads as i64,
            head_dim as i64,
        ])?;
        let k = self.k_proj.forward_cast(&x, matmul_dtype)?.reshape(&[
            n,
            num_heads as i64,
            head_dim as i64,
        ])?;
        let v = self.v_proj.forward_cast(&x, matmul_dtype)?.reshape(&[
            n,
            num_heads as i64,
            head_dim as i64,
        ])?;

        // 2D interleaved axial RoPE on Q and K
        let q_rot = apply_2d_interleaved_rope(&q, cos_table, sin_table)?;
        let k_rot = apply_2d_interleaved_rope(&k, cos_table, sin_table)?;

        // Bidirectional SDPA: expects `[B=1, H, N, D]`
        let q_bhnd = q_rot.transpose(Some(&[1, 0, 2]))?.expand_dims(0)?;
        let k_bhnd = k_rot.transpose(Some(&[1, 0, 2]))?.expand_dims(0)?;
        let v_bhnd = v.transpose(Some(&[1, 0, 2]))?.expand_dims(0)?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let attn = scaled_dot_product_attention(&q_bhnd, &k_bhnd, &v_bhnd, scale, None)?
            .squeeze(Some(&[0]))?
            .transpose(Some(&[1, 0, 2]))?
            .reshape(&[n, d])?;

        let attn_out_f32 = self.o_proj.forward_f32(&attn)?;
        let ls1_f32 = if self.ls1.dtype()? == DType::Float32 {
            self.ls1.clone()
        } else {
            self.ls1.astype(DType::Float32)?
        };
        let h1 = hidden.add(&attn_out_f32.mul(&ls1_f32)?)?;

        let x2_norm = self.norm2.forward(&h1)?;
        let x2 = if x2_norm.dtype()? == matmul_dtype {
            x2_norm
        } else {
            x2_norm.astype(matmul_dtype)?
        };
        let fc1_out = self.fc1.forward_cast(&x2, matmul_dtype)?;
        let gelu_out = Activations::gelu_exact(&fc1_out)?;
        let mlp_out_f32 = self.fc2.forward_f32(&gelu_out)?;
        let ls2_f32 = if self.ls2.dtype()? == DType::Float32 {
            self.ls2.clone()
        } else {
            self.ls2.astype(DType::Float32)?
        };
        h1.add(&mlp_out_f32.mul(&ls2_f32)?)
    }

    pub fn weight_arrays(&self) -> Vec<MxArray> {
        let mut arrays = Vec::with_capacity(20);
        arrays.push(self.norm1.get_weight());
        arrays.push(self.norm1.get_bias());
        for l in [
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
            &self.fc1,
            &self.fc2,
        ] {
            arrays.extend(l.weight_arrays());
        }
        arrays.push(self.ls1.clone());
        arrays.push(self.norm2.get_weight());
        arrays.push(self.norm2.get_bias());
        arrays.push(self.ls2.clone());
        arrays
    }

    pub(crate) fn ensure_cuda_device(&mut self, layer_norm_eps: f64) -> Result<usize> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(0);
        }
        let mut promoted = 0usize;
        promoted += promote_layer_norm_cuda(&mut self.norm1, layer_norm_eps)?;
        promoted += self.q_proj.ensure_cuda_device()?;
        promoted += self.k_proj.ensure_cuda_device()?;
        promoted += self.v_proj.ensure_cuda_device()?;
        promoted += self.o_proj.ensure_cuda_device()?;
        promoted += self.ls1.promote_cuda_device()?;
        promoted += promote_layer_norm_cuda(&mut self.norm2, layer_norm_eps)?;
        promoted += self.fc1.ensure_cuda_device()?;
        promoted += self.fc2.ensure_cuda_device()?;
        promoted += self.ls2.promote_cuda_device()?;
        Ok(promoted)
    }
}

fn promote_layer_norm_cuda(ln: &mut LayerNorm, eps: f64) -> Result<usize> {
    let mut w = ln.get_weight();
    let mut b = ln.get_bias();
    let wb = w.promote_cuda_device()?;
    let bb = b.promote_cuda_device()?;
    if wb > 0 || bb > 0 {
        *ln = LayerNorm::from_weights(&w, Some(&b), Some(eps))?;
    }
    Ok(wb + bb)
}

/// Complete PE-Lang ViT + 2×2 spatial merge projector (`mmproj-BF16/F32/Q8_0.gguf` or SafeTensors).
pub struct Inkling2VisionEncoder {
    pub config: Inkling2VisionConfig,
    /// `[hidden_size]` (F32/BF16)
    pub class_embedding: MxArray,
    /// `[hidden_size]` (F32/BF16)
    pub class_position_embedding: MxArray,
    /// Patch embedding weight flattened to `[hidden_size, 3 * patch_size * patch_size]` in CHW `(c, ph, pw)` order.
    pub patch_weight_flat: MxArray,
    /// Base position embedding table `[grid_size * grid_size, hidden_size]` (F32, row-major `(y, x)`).
    pub position_embedding: MxArray,
    pub pre_ln: LayerNorm,
    pub blocks: Vec<Inkling2VisionBlock>,
    /// Projector `mm.1` (`4096 -> 4096`, optional bias) and `mm.2` (`4096 -> 2304`, optional bias).
    pub proj_fc1: VisionLinear,
    pub proj_fc2: VisionLinear,
}

impl Inkling2VisionEncoder {
    /// Encode raw image bytes (PNG/JPEG/etc.) into `[grid_rows * grid_cols, projection_dim]` BF16 embeddings,
    /// returning `(embeddings, grid_rows, grid_cols)`.
    pub fn encode_image_bytes(&self, bytes: &[u8]) -> Result<(MxArray, usize, usize)> {
        let prep = preprocess_image_bytes(bytes, &self.config)?;
        let emb = self.encode(&prep)?;
        Ok((emb, prep.grid_rows, prep.grid_cols))
    }

    /// Run ViT + 2×2 spatial merge + MLP projector on a preprocessed image.
    /// Keeps the residual stream in FP32 through `pre_ln` and all ViT blocks, then casts to
    /// projector dtype before the spatial merge + MLP and returns `[grid_rows * grid_cols, projection_dim]`
    /// in BF16 (ready to overwrite `IMAGE_PAD_ID` positions **after** `embed_norm`).
    pub fn encode(&self, prep: &PreprocessedImage) -> Result<MxArray> {
        let p = self.config.patch_size;
        let m = self.config.spatial_merge_size;
        let h_p = prep.grid_rows * m;
        let w_p = prep.grid_cols * m;
        let n_patches = h_p * w_p;
        let hidden_size = self.config.hidden_size;

        // 1. Patch embedding via exact stride=patch_size reshape + matmul in FP32
        // `[H_pad, W_pad, 3]` -> `[h_p, p, w_p, p, 3]` -> `[h_p, w_p, 3, p, p]` -> `[n_patches, 3*p*p]`
        let patches_chw = prep
            .pixels
            .reshape(&[h_p as i64, p as i64, w_p as i64, p as i64, 3])?
            .transpose(Some(&[0, 2, 4, 1, 3]))?
            .reshape(&[n_patches as i64, (3 * p * p) as i64])?;
        let w_t = self
            .patch_weight_flat
            .astype(DType::Float32)?
            .transpose(None)?;
        let patch_emb = patches_chw.matmul(&w_t)?; // `[n_patches, D]` FP32

        // 2. Bilinearly interpolate position embedding (`align_corners = false`) in FP32
        let pos_emb = bilinear_interp_posemb(
            &self.position_embedding,
            self.config.posemb_grid_size(),
            h_p,
            w_p,
        )?;
        let patch_with_pos = patch_emb.add(&pos_emb)?; // FP32

        // 3. Prepend CLS token (`class_embedding + class_position_embedding`) + `pre_ln` in FP32
        let cls = self
            .class_embedding
            .astype(DType::Float32)?
            .add(&self.class_position_embedding.astype(DType::Float32)?)?
            .reshape(&[1, hidden_size as i64])?;
        let seq = MxArray::concatenate(&cls, &patch_with_pos, 0)?; // `[n_patches + 1, D]` FP32
        let pre_ln_out = self.pre_ln.forward(&seq)?;
        let mut hidden = if pre_ln_out.dtype()? == DType::Float32 {
            pre_ln_out
        } else {
            pre_ln_out.astype(DType::Float32)?
        };

        // 4. Build 2D interleaved axial RoPE cos/sin tables (`CLS` at `(-1, -1)` -> pos `(0, 0)`, angle 0)
        let head_dim = self.config.head_dim();
        let (cos_table, sin_table) =
            build_2d_rope_tables(h_p, w_p, head_dim, self.config.rope_theta)?;

        // 5. ViT blocks (FP32 residual stream throughout)
        for blk in &self.blocks {
            hidden = blk.forward(
                &hidden,
                &cos_table,
                &sin_table,
                self.config.num_attention_heads,
                head_dim,
            )?;
        }

        // 6. Drop CLS, cast to projector weight dtype, and 2×2 spatial merge:
        // `(rows, m, cols, m, D)` -> `(rows, cols, m, m, D)` -> `(rows*cols, m*m*D)`
        let patches_out = hidden.slice(&[1, 0], &[(n_patches + 1) as i64, hidden_size as i64])?;
        let proj_dtype = self.proj_fc1.weight_dtype()?;
        let patches_cast = if patches_out.dtype()? == proj_dtype {
            patches_out
        } else {
            patches_out.astype(proj_dtype)?
        };
        let merged_dim = (m * m * hidden_size) as i64;
        let merged = patches_cast
            .reshape(&[
                prep.grid_rows as i64,
                m as i64,
                prep.grid_cols as i64,
                m as i64,
                hidden_size as i64,
            ])?
            .transpose(Some(&[0, 2, 1, 3, 4]))?
            .reshape(&[(prep.grid_rows * prep.grid_cols) as i64, merged_dim])?;

        // 7. Projector: `mm.2(gelu_exact(mm.1(merged)))` -> BF16
        let h_proj = self.proj_fc1.forward_cast(&merged, proj_dtype)?;
        let act = Activations::gelu_exact(&h_proj)?;
        let out = self.proj_fc2.forward_cast(&act, proj_dtype)?;
        if out.dtype()? == DType::BFloat16 {
            Ok(out)
        } else {
            out.astype(DType::BFloat16)
        }
    }

    pub fn weight_arrays(&self) -> Vec<MxArray> {
        let mut arrays = vec![
            self.class_embedding.clone(),
            self.class_position_embedding.clone(),
            self.patch_weight_flat.clone(),
            self.position_embedding.clone(),
            self.pre_ln.get_weight(),
            self.pre_ln.get_bias(),
        ];
        for b in &self.blocks {
            arrays.extend(b.weight_arrays());
        }
        arrays.extend(self.proj_fc1.weight_arrays());
        arrays.extend(self.proj_fc2.weight_arrays());
        arrays
    }

    pub(crate) fn ensure_cuda_device(&mut self) -> Result<usize> {
        if !crate::accelerator::cuda_ops_available() {
            return Ok(0);
        }
        let eps = self.config.layer_norm_eps;
        let mut promoted = 0usize;
        promoted += self.class_embedding.promote_cuda_device()?;
        promoted += self.class_position_embedding.promote_cuda_device()?;
        promoted += self.patch_weight_flat.promote_cuda_device()?;
        promoted += self.position_embedding.promote_cuda_device()?;
        promoted += promote_layer_norm_cuda(&mut self.pre_ln, eps)?;
        for blk in &mut self.blocks {
            promoted += blk.ensure_cuda_device(eps)?;
        }
        promoted += self.proj_fc1.ensure_cuda_device()?;
        promoted += self.proj_fc2.ensure_cuda_device()?;
        Ok(promoted)
    }
}

/// Bilinear interpolation of `[grid * grid, D]` position embeddings to `[h_p * w_p, D]`
/// with PyTorch `align_corners = false` in FP32.
fn bilinear_interp_posemb(
    pos_table: &MxArray,
    src_grid: usize,
    dst_h: usize,
    dst_w: usize,
) -> Result<MxArray> {
    let table_f32 = if pos_table.dtype()? == DType::Float32 {
        pos_table.clone()
    } else {
        pos_table.astype(DType::Float32)?
    };
    if dst_h == src_grid && dst_w == src_grid {
        return Ok(table_f32);
    }

    let hidden_dim = table_f32.shape_at(1)?;
    let table_2d = table_f32.reshape(&[src_grid as i64, src_grid as i64, hidden_dim])?;
    let interp =
        crate::vision::interpolate::bilinear_interpolate(&table_2d, dst_h as i64, dst_w as i64)?;
    interp.reshape(&[(dst_h * dst_w) as i64, hidden_dim])
}

/// Build 2D interleaved axial RoPE `(cos, sin)` tables `[h_p * w_p + 1, 1, head_dim / 2]` in FP32.
///
/// Matches vLLM `ops/vision.py` & llama.cpp `mtmd-graph.cpp`:
/// - `head_dim = 64` -> `half = 32` interleaved pairs `(2p, 2p+1)`:
///   - `p in 0..16` (first half dims `0..32`): X-axis `pos_w = x + 1`, `omega_p = theta^(-p / 16)`.
///   - `p in 16..32` (second half dims `32..64`): Y-axis `pos_h = y + 1`, `omega_{p-16} = theta^(-(p-16) / 16)`.
/// - Row `0` (CLS): `pos_w = 0, pos_h = 0` -> angle `0` (cos=1, sin=0).
fn build_2d_rope_tables(
    h_p: usize,
    w_p: usize,
    head_dim: usize,
    theta: f64,
) -> Result<(MxArray, MxArray)> {
    let half = head_dim / 2;
    let quarter = head_dim / 4;
    let n_tokens = h_p * w_p + 1;

    let mut freqs = Vec::with_capacity(quarter);
    for m in 0..quarter {
        let exponent = -4.0 * (m as f64) / (head_dim as f64);
        freqs.push(theta.powf(exponent));
    }

    let mut cos_vals = Vec::with_capacity(n_tokens * half);
    let mut sin_vals = Vec::with_capacity(n_tokens * half);

    // Row 0: CLS token -> angle 0 -> cos = 1, sin = 0
    cos_vals.extend(std::iter::repeat_n(1.0_f32, half));
    sin_vals.extend(std::iter::repeat_n(0.0_f32, half));

    for y in 0..h_p {
        let pos_h = (y + 1) as f64;
        for x in 0..w_p {
            let pos_w = (x + 1) as f64;
            // First half (dims 0..32): X-axis
            for &f in &freqs {
                let angle = pos_w * f;
                cos_vals.push(angle.cos() as f32);
                sin_vals.push(angle.sin() as f32);
            }
            // Second half (dims 32..64): Y-axis
            for &f in &freqs {
                let angle = pos_h * f;
                cos_vals.push(angle.cos() as f32);
                sin_vals.push(angle.sin() as f32);
            }
        }
    }

    let cos = MxArray::from_float32(&cos_vals, &[n_tokens as i64, 1, half as i64])?;
    let sin = MxArray::from_float32(&sin_vals, &[n_tokens as i64, 1, half as i64])?;
    Ok((cos, sin))
}

/// Apply interleaved 2D RoPE to `x` `[N, H, head_dim]` using FP32 `cos`, `sin` `[N, 1, head_dim/2]`.
fn apply_2d_interleaved_rope(
    x: &MxArray,
    cos_table: &MxArray,
    sin_table: &MxArray,
) -> Result<MxArray> {
    let shape = x.shape()?;
    let n = shape[0];
    let h = shape[1];
    let d = shape[2];
    let half = d / 2;
    let out_dtype = x.dtype()?;

    let x_f32 = x.astype(DType::Float32)?.reshape(&[n, h, half, 2])?;
    let even = x_f32
        .slice(&[0, 0, 0, 0], &[n, h, half, 1])?
        .squeeze(Some(&[-1]))?;
    let odd = x_f32
        .slice(&[0, 0, 0, 1], &[n, h, half, 2])?
        .squeeze(Some(&[-1]))?;

    let rot_even = even.mul(cos_table)?.sub(&odd.mul(sin_table)?)?;
    let rot_odd = even.mul(sin_table)?.add(&odd.mul(cos_table)?)?;

    let stacked = MxArray::stack(vec![&rot_even, &rot_odd], Some(-1))?.reshape(&[n, h, d])?;
    if out_dtype != DType::Float32 {
        stacked.astype(out_dtype)
    } else {
        Ok(stacked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_pads_to_40_with_minus_one_and_expands_tokens() {
        // 30×50 solid red image -> pads to 40×80 -> grid_rows=1, grid_cols=2 -> 2 image tokens
        let img = image::RgbImage::from_pixel(50, 30, image::Rgb([255, 0, 0]));
        let dyn_img = DynamicImage::ImageRgb8(img);
        let cfg = Inkling2VisionConfig::default();
        let prep = preprocess_dynamic_image(&dyn_img, &cfg).unwrap();
        assert_eq!(prep.grid_rows, 1);
        assert_eq!(prep.grid_cols, 2);
        assert_eq!(prep.num_tokens(), 2);
        assert_eq!(prep.pixels.shape().unwrap().as_ref(), &[40, 80, 3]);

        let vals = prep.pixels.to_float32().unwrap();
        // Top-left (0, 0) is red [255, 0, 0] -> [+1.0, -1.0, -1.0]
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - (-1.0)).abs() < 1e-5);
        assert!((vals[2] - (-1.0)).abs() < 1e-5);
        // Bottom-right padded pixel (39, 79) has -1 -> 2*(-1/255) - 1 = -257/255
        let pad_idx = (39 * 80 + 79) * 3;
        let expected_pad = -257.0_f32 / 255.0;
        assert!((vals[pad_idx] - expected_pad).abs() < 1e-5);

        let expanded = expand_image_tokens(&[200000, CONTENT_IMAGE_ID, 200010], &[2]).unwrap();
        assert_eq!(
            expanded,
            vec![200000, CONTENT_IMAGE_ID, IMAGE_PAD_ID, IMAGE_PAD_ID, 200010]
        );
    }

    #[test]
    fn rope_2d_keeps_cls_identity() {
        let (cos, sin) = build_2d_rope_tables(2, 2, 64, 10_000.0).unwrap();
        let x_vals: Vec<f32> = (0..(5 * 2 * 64)).map(|i| (i as f32) * 0.01).collect();
        let x = MxArray::from_float32(&x_vals, &[5, 2, 64]).unwrap();
        let rot = apply_2d_interleaved_rope(&x, &cos, &sin)
            .unwrap()
            .to_float32()
            .unwrap();
        // Row 0 (CLS) must be unchanged
        for i in 0..(2 * 64) {
            assert!((rot[i] - x_vals[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn fit_vision_dimensions_respects_padded_token_cap_without_upscaling() {
        let align = 40;
        let cases = [
            (200_000, 1),
            (1, 200_000),
            (2600, 2561),
            (2561, 2521),
            (2560, 2560),
            (5000, 3000),
        ];
        for (w, h) in cases {
            let (nw, nh) = fit_vision_dimensions(w, h, align, MAX_VISION_TOKENS);
            assert!(nw >= 1 && nw <= w, "upscaled width for {w}x{h}: {nw}");
            assert!(nh >= 1 && nh <= h, "upscaled height for {w}x{h}: {nh}");
            let tokens = nw.div_ceil(align) * nh.div_ceil(align);
            assert!(
                tokens <= MAX_VISION_TOKENS,
                "padded tokens {tokens} > {MAX_VISION_TOKENS} for {w}x{h} -> {nw}x{nh}"
            );
        }
        // End-to-end check on an extreme 200,000x1 strip: must not upscale or exceed 4096 tokens.
        let strip = image::RgbImage::from_pixel(200_000, 1, image::Rgb([128, 64, 32]));
        let prep = preprocess_dynamic_image(
            &DynamicImage::ImageRgb8(strip),
            &Inkling2VisionConfig::default(),
        )
        .unwrap();
        assert_eq!(prep.grid_rows, 1);
        assert!(prep.grid_cols <= MAX_VISION_TOKENS);
        assert!(prep.num_tokens() <= MAX_VISION_TOKENS);
    }

    #[test]
    fn preprocess_applies_exif_orientation_6_before_grid_and_pixels() {
        // Stored landscape 80x40 JPEG: left half [0..40) bright red, right half [40..80) bright blue.
        let mut stored = image::RgbImage::new(80, 40);
        for y in 0..40 {
            for x in 0..80 {
                let px = if x < 40 {
                    image::Rgb([255, 0, 0])
                } else {
                    image::Rgb([0, 0, 255])
                };
                stored.put_pixel(x, y, px);
            }
        }
        let mut raw_jpeg = Vec::new();
        DynamicImage::ImageRgb8(stored)
            .write_to(&mut Cursor::new(&mut raw_jpeg), image::ImageFormat::Jpeg)
            .unwrap();
        assert_eq!(&raw_jpeg[..2], &[0xFF, 0xD8]);

        // Splice minimal EXIF APP1 segment with Orientation = 6 (90° CW).
        let exif_app1: [u8; 36] = [
            0xFF, 0xE1, 0x00, 0x22, // APP1 marker + length (34 bytes)
            b'E', b'x', b'i', b'f', 0x00, 0x00, // Exif header
            b'I', b'I', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // Little-endian TIFF header
            0x01, 0x00, // 1 IFD0 entry
            0x12, 0x01, // Tag 0x0112 (Orientation)
            0x03, 0x00, // Type SHORT (3)
            0x01, 0x00, 0x00, 0x00, // Count = 1
            0x06, 0x00, 0x00, 0x00, // Value = 6 (Rotate90 CW)
            0x00, 0x00, 0x00, 0x00, // Next IFD = 0
        ];
        let mut oriented_jpeg = Vec::with_capacity(raw_jpeg.len() + exif_app1.len());
        oriented_jpeg.extend_from_slice(&raw_jpeg[..2]);
        oriented_jpeg.extend_from_slice(&exif_app1);
        oriented_jpeg.extend_from_slice(&raw_jpeg[2..]);

        let cfg = Inkling2VisionConfig::default();
        let prep = preprocess_image_bytes(&oriented_jpeg, &cfg).unwrap();
        // After 90° CW rotation: 40 wide x 80 tall -> grid_rows = 2, grid_cols = 1.
        assert_eq!(prep.grid_rows, 2);
        assert_eq!(prep.grid_cols, 1);
        assert_eq!(prep.pixels.shape().unwrap().as_ref(), &[80, 40, 3]);

        // Top half (y=10, x=20) should be red (~[+1, -1, -1]), bottom half (y=60, x=20) blue (~[-1, -1, +1]).
        let vals = prep.pixels.to_float32().unwrap();
        let top_idx = (10 * 40 + 20) * 3;
        assert!(
            vals[top_idx] > 0.7 && vals[top_idx + 2] < -0.7,
            "expected red top half after 90° CW EXIF rotate, got {:?}",
            &vals[top_idx..top_idx + 3]
        );
        let bot_idx = (60 * 40 + 20) * 3;
        assert!(
            vals[bot_idx] < -0.7 && vals[bot_idx + 2] > 0.7,
            "expected blue bottom half after 90° CW EXIF rotate, got {:?}",
            &vals[bot_idx..bot_idx + 3]
        );
    }

    #[test]
    fn preprocess_oversized_transparent_rgba_preserves_8x8_checkerboard() {
        // 5000x3000 fully transparent image -> downscales first, then composites onto 8x8 checkerboard.
        let rgba = RgbaImage::from_pixel(5000, 3000, Rgba([0, 0, 0, 0]));
        let cfg = Inkling2VisionConfig::default();
        let prep = preprocess_dynamic_image(&DynamicImage::ImageRgba8(rgba), &cfg).unwrap();
        assert!(prep.num_tokens() <= MAX_VISION_TOKENS);

        let shape = prep.pixels.shape().unwrap();
        let w_pad = shape[1] as usize;
        let vals = prep.pixels.to_float32().unwrap();
        let norm = |u: u8| 2.0_f32 * (f32::from(u) / 255.0) - 1.0;
        let check_px = |x: usize, y: usize, expected_u8: u8| {
            let idx = (y * w_pad + x) * 3;
            let want = norm(expected_u8);
            for c in 0..3 {
                assert!(
                    (vals[idx + c] - want).abs() < 1e-5,
                    "pixel ({x},{y}) ch {c}: got {}, expected {want} (u8={expected_u8})",
                    vals[idx + c]
                );
            }
        };
        check_px(0, 0, 204);
        check_px(7, 7, 204);
        check_px(8, 0, 153);
        check_px(0, 8, 153);
        check_px(8, 8, 204);
    }

    #[test]
    fn preprocess_premultiplied_rgba_resize_avoids_halo_and_blends_semitransparent_exact() {
        // 1) Oversized RGBA (3200x3200 -> resizes down): left half hidden red under alpha=0,
        // right half opaque white. Premultiplied resize must produce zero red tint on the seam.
        let mut rgba = RgbaImage::new(3200, 3200);
        for y in 0..3200 {
            for x in 0..3200 {
                let px = if x < 1600 {
                    Rgba([255, 0, 0, 0])
                } else {
                    Rgba([255, 255, 255, 255])
                };
                rgba.put_pixel(x, y, px);
            }
        }
        let cfg = Inkling2VisionConfig::default();
        let prep = preprocess_dynamic_image(&DynamicImage::ImageRgba8(rgba), &cfg).unwrap();
        let shape = prep.pixels.shape().unwrap();
        let (h_pad, w_pad) = (shape[0] as usize, shape[1] as usize);
        let vals = prep.pixels.to_float32().unwrap();
        // Every non-padded pixel must be neutral grey/white (`r == g == b`); red bleed would make r > g.
        let (target_w, target_h) =
            fit_vision_dimensions(3200, 3200, cfg.alignment(), MAX_VISION_TOKENS);
        assert!(target_w < 3200 && target_h < 3200);
        assert!(h_pad >= target_h && w_pad >= target_w);
        for y in 0..target_h.min(32) {
            for x in 0..target_w {
                let idx = (y * w_pad + x) * 3;
                assert!(
                    (vals[idx] - vals[idx + 1]).abs() < 1e-5
                        && (vals[idx + 1] - vals[idx + 2]).abs() < 1e-5,
                    "red halo leaked at ({x},{y}): {:?}",
                    &vals[idx..idx + 3]
                );
            }
        }

        // 2) Exact integer blend formula on semi-transparent pixel (a = 128) without resize.
        let semi = RgbaImage::from_pixel(16, 16, Rgba([100, 200, 50, 128]));
        let prep_semi = preprocess_dynamic_image(&DynamicImage::ImageRgba8(semi), &cfg).unwrap();
        let s_vals = prep_semi.pixels.to_float32().unwrap();
        let s_w = prep_semi.pixels.shape().unwrap()[1] as usize;
        let expected_u8 = |c: u16, bg: u16| ((c * 128 + bg * 127 + 127) / 255) as u8;
        let norm = |u: u8| 2.0_f32 * (f32::from(u) / 255.0) - 1.0;
        // (0, 0) sits on 204 square; (8, 0) sits on 153 square.
        for (x, bg) in [(0usize, 204u16), (8usize, 153u16)] {
            let idx = x * 3;
            let want = [
                norm(expected_u8(100, bg)),
                norm(expected_u8(200, bg)),
                norm(expected_u8(50, bg)),
            ];
            for c in 0..3 {
                assert!(
                    (s_vals[idx + c] - want[c]).abs() < 1e-5,
                    "semi-transparent blend mismatch at x={x} ch={c}: got {}, want {}",
                    s_vals[idx + c],
                    want[c]
                );
            }
        }
        // Verify padded region at y=20 is still -257/255.
        let pad_idx = (20 * s_w) * 3;
        assert!((s_vals[pad_idx] - (-257.0_f32 / 255.0)).abs() < 1e-5);
    }

    #[test]
    fn preprocess_oversized_rgb_uses_bilinear_without_catmull_rom_ringing() {
        // 3200x3200 high-contrast 2-stripe image (20 vs 235) where CatmullRom overshoots below 20 / above 235.
        let mut rgb = RgbImage::new(3200, 3200);
        for y in 0..3200 {
            for x in 0..3200 {
                let v = if x < 1600 { 20 } else { 235 };
                rgb.put_pixel(x, y, image::Rgb([v, v, v]));
            }
        }
        let cfg = Inkling2VisionConfig::default();
        let (target_w, target_h) =
            fit_vision_dimensions(3200, 3200, cfg.alignment(), MAX_VISION_TOKENS);
        let expected_rgb = image::imageops::resize(
            &rgb,
            target_w as u32,
            target_h as u32,
            image::imageops::FilterType::Triangle,
        );
        let prep = preprocess_dynamic_image(&DynamicImage::ImageRgb8(rgb), &cfg).unwrap();
        let w_pad = prep.pixels.shape().unwrap()[1] as usize;
        let vals = prep.pixels.to_float32().unwrap();
        let lo = 2.0_f32 * (20.0 / 255.0) - 1.0 - 1e-5;
        let hi = 2.0_f32 * (235.0 / 255.0) - 1.0 + 1e-5;
        for y in 0..target_h.min(8) {
            for x in 0..target_w {
                let idx = (y * w_pad + x) * 3;
                let exp_v = 2.0_f32
                    * (f32::from(expected_rgb.get_pixel(x as u32, y as u32).0[0]) / 255.0)
                    - 1.0;
                assert!(
                    (vals[idx] - exp_v).abs() < 1e-5,
                    "bilinear mismatch at ({x},{y}): got {}, want {exp_v}",
                    vals[idx]
                );
                assert!(
                    vals[idx] >= lo && vals[idx] <= hi,
                    "ringing/overshoot detected at ({x},{y}): {}",
                    vals[idx]
                );
            }
        }
    }

    #[test]
    fn vit_fp32_residual_stream_preserves_small_layerscale_updates_and_returns_bf16() {
        // Synthetic 23-block ViT with hidden_size=16, num_heads=4 (head_dim=4), patch_size=4,
        // spatial_merge_size=2, image_size=8 (posemb_grid_size=2), projection_dim=16.
        // With `hidden ~ 2.0`, BF16 ULP is `2^-6 = 0.015625`, so `ls1 = ls2 = 1e-4` updates of
        // `~1e-4` per sublayer (46 updates total -> delta ~ 0.0046) would be rounded to ZERO in
        // a BF16 residual stream, whereas an FP32 residual stream accumulates all 46 updates!
        let cfg = Inkling2VisionConfig {
            hidden_size: 16,
            intermediate_size: 16,
            num_hidden_layers: 23,
            num_attention_heads: 4,
            image_size: 8,
            patch_size: 4,
            spatial_merge_size: 2,
            projection_dim: 16,
            layer_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        };
        let d = 16i64;
        let ones_d = MxArray::from_float32(&[1.0f32; 16], &[d]).unwrap();
        let zeros_d = MxArray::from_float32(&[0.0f32; 16], &[d]).unwrap();
        // Pre-LN bias = 2.0 pushes every hidden channel to ~2.0 (BF16 ULP = 0.015625).
        let pre_ln_bias = MxArray::from_float32(&[2.0f32; 16], &[d]).unwrap();
        let zeros_mat_bf16 = MxArray::zeros(&[d, d], Some(DType::BFloat16)).unwrap();
        // Constant +1.0 bias on o_proj and fc2 ensures each sublayer contributes `+1.0 * 1e-4` in FP32.
        let one_bias_f32 = MxArray::from_float32(&[1.0f32; 16], &[d]).unwrap();
        let ls_small = MxArray::from_float32(&[1e-4_f32; 16], &[d]).unwrap();

        let make_block = |ls: &MxArray| -> Inkling2VisionBlock {
            Inkling2VisionBlock {
                norm1: LayerNorm::from_weights(&ones_d, Some(&zeros_d), Some(1e-5)).unwrap(),
                q_proj: VisionLinear::from_weights(&zeros_mat_bf16, Some(&zeros_d)).unwrap(),
                k_proj: VisionLinear::from_weights(&zeros_mat_bf16, Some(&zeros_d)).unwrap(),
                v_proj: VisionLinear::from_weights(&zeros_mat_bf16, Some(&zeros_d)).unwrap(),
                o_proj: VisionLinear::from_weights(&zeros_mat_bf16, Some(&one_bias_f32)).unwrap(),
                ls1: ls.clone(),
                norm2: LayerNorm::from_weights(&ones_d, Some(&zeros_d), Some(1e-5)).unwrap(),
                fc1: VisionLinear::from_weights(&zeros_mat_bf16, Some(&zeros_d)).unwrap(),
                fc2: VisionLinear::from_weights(&zeros_mat_bf16, Some(&one_bias_f32)).unwrap(),
                ls2: ls.clone(),
            }
        };

        // Identity-like projector that reads the first 16 channels directly (`fc1` = [64, 64] I,
        // `fc2` = [16, 64] selecting first 16 channels) so the final hidden shift is observable.
        let mut fc1_w = vec![0.0f32; 64 * 64];
        for i in 0..64 {
            fc1_w[i * 64 + i] = 1.0;
        }
        let mut fc2_w = vec![0.0f32; 16 * 64];
        for i in 0..16 {
            fc2_w[i * 64 + i] = 1.0;
        }
        let build_enc = |ls: &MxArray| Inkling2VisionEncoder {
            config: cfg.clone(),
            class_embedding: zeros_d.clone(),
            class_position_embedding: zeros_d.clone(),
            patch_weight_flat: MxArray::zeros(&[d, 3 * 4 * 4], Some(DType::Float32)).unwrap(),
            position_embedding: MxArray::zeros(&[4, d], Some(DType::Float32)).unwrap(),
            pre_ln: LayerNorm::from_weights(&ones_d, Some(&pre_ln_bias), Some(1e-5)).unwrap(),
            blocks: (0..23).map(|_| make_block(ls)).collect(),
            proj_fc1: VisionLinear::from_weights(
                &MxArray::from_float32(&fc1_w, &[64, 64]).unwrap(),
                None,
            )
            .unwrap(),
            proj_fc2: VisionLinear::from_weights(
                &MxArray::from_float32(&fc2_w, &[16, 64]).unwrap(),
                None,
            )
            .unwrap(),
        };

        let img = RgbImage::from_pixel(8, 8, image::Rgb([128, 128, 128]));
        let prep = preprocess_dynamic_image(&DynamicImage::ImageRgb8(img), &cfg).unwrap();

        let enc_small = build_enc(&ls_small);
        let enc_zero = build_enc(&zeros_d);
        let out_small = enc_small.encode(&prep).unwrap();
        let out_zero = enc_zero.encode(&prep).unwrap();
        assert_eq!(out_small.dtype().unwrap(), DType::BFloat16);
        assert_eq!(out_small.shape().unwrap().as_ref(), &[1, 16]);

        // Directly run the 23 blocks on FP32 `hidden` to verify the exact FP32 residual delta = 46 * 1e-4 = 0.0046.
        let (cos_t, sin_t) = build_2d_rope_tables(2, 2, 4, 10_000.0).unwrap();
        let mut h_small = MxArray::from_float32(&vec![2.0f32; 5 * 16], &[5, 16]).unwrap();
        let mut h_zero = h_small.clone();
        for (bs, bz) in enc_small.blocks.iter().zip(enc_zero.blocks.iter()) {
            h_small = bs.forward(&h_small, &cos_t, &sin_t, 4, 4).unwrap();
            h_zero = bz.forward(&h_zero, &cos_t, &sin_t, 4, 4).unwrap();
        }
        assert_eq!(h_small.dtype().unwrap(), DType::Float32);
        let v_small = h_small.to_float32().unwrap();
        let v_zero = h_zero.to_float32().unwrap();
        let delta = v_small[16] - v_zero[16];
        assert!(
            (delta - 0.0046).abs() < 1e-5,
            "FP32 residual stream must accumulate 46 * 1e-4 = 0.0046 (BF16 would round to 0), got {delta}"
        );
        // Also verify `encode` output matches BF16 projection cleanly with finite values.
        assert!(!out_small.has_nan_or_inf().unwrap());
        assert!(!out_zero.has_nan_or_inf().unwrap());
    }

    #[test]
    fn preprocess_image_bytes_enforces_byte_pixel_format_and_alloc_limits_before_decode() {
        let cfg = Inkling2VisionConfig::default();

        // 1. Empty payload rejected with structured prefix.
        let err = preprocess_image_bytes(&[], &cfg)
            .err()
            .expect("empty bytes must fail");
        assert!(
            err.reason
                .starts_with(crate::vision::validate::IMAGE_PROCESSING_ERROR_PREFIX),
            "missing image_processing_error prefix: {}",
            err.reason
        );

        // 2. > 32 MiB encoded payload rejected before decoding.
        let oversized = vec![0u8; MAX_IMAGE_BYTES + 1];
        let err = preprocess_image_bytes(&oversized, &cfg)
            .err()
            .expect("oversized bytes must fail");
        assert!(
            err.reason
                .starts_with(crate::vision::validate::IMAGE_PROCESSING_ERROR_PREFIX),
            "missing image_processing_error prefix: {}",
            err.reason
        );

        // 3. > 32 MP PNG IHDR (8000x5000 = 40 MP) rejected before decoding body.
        let mut png = Vec::new();
        DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, image::Rgb([10, 20, 30])))
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .unwrap();
        let mut over_32mp = png.clone();
        over_32mp[16..20].copy_from_slice(&8000u32.to_be_bytes());
        over_32mp[20..24].copy_from_slice(&5000u32.to_be_bytes());
        let err = preprocess_image_bytes(&over_32mp, &cfg)
            .err()
            .expect("40 MP PNG header must be rejected before decode");
        assert!(
            err.reason
                .starts_with(crate::vision::validate::IMAGE_PROCESSING_ERROR_PREFIX),
            "missing image_processing_error prefix: {}",
            err.reason
        );

        // 4. Zero-dimension PNG IHDR rejected before decoding.
        let mut zero_dim = png;
        zero_dim[16..20].copy_from_slice(&0u32.to_be_bytes());
        let err = preprocess_image_bytes(&zero_dim, &cfg)
            .err()
            .expect("0-width PNG header must be rejected before decode");
        assert!(
            err.reason
                .starts_with(crate::vision::validate::IMAGE_PROCESSING_ERROR_PREFIX),
            "missing image_processing_error prefix: {}",
            err.reason
        );

        // 5. Unsupported BMP format rejected with structured prefix.
        let bmp: &[u8] = &[
            0x42, 0x4d, 0x3a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
            0x28, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00,
            0x00, 0x00,
        ];
        let err = preprocess_image_bytes(bmp, &cfg)
            .err()
            .expect("BMP payload must be rejected");
        assert!(
            err.reason
                .starts_with(crate::vision::validate::IMAGE_PROCESSING_ERROR_PREFIX),
            "missing image_processing_error prefix: {}",
            err.reason
        );
    }

    #[test]
    fn resize_rgba_premultiplied_bilinear_matches_rgba32f_reference() {
        use image::Rgba32FImage;

        let reference_resize = |rgba: &RgbaImage, tw: u32, th: u32| -> RgbaImage {
            let (w, h) = rgba.dimensions();
            let mut premul = Rgba32FImage::new(w, h);
            for (src, dst) in rgba.pixels().zip(premul.pixels_mut()) {
                let a = f32::from(src[3]) / 255.0;
                *dst = Rgba([
                    (f32::from(src[0]) / 255.0) * a,
                    (f32::from(src[1]) / 255.0) * a,
                    (f32::from(src[2]) / 255.0) * a,
                    a,
                ]);
            }
            let resized =
                image::imageops::resize(&premul, tw, th, image::imageops::FilterType::Triangle);
            let mut out = RgbaImage::new(tw, th);
            for (src, dst) in resized.pixels().zip(out.pixels_mut()) {
                *dst = unpremultiply_rgba_pixel(src.0);
            }
            out
        };

        // Test downscale (square + non-square), 1:1, extreme aspect downscale, and vertical upscale.
        let cases = [
            (97u32, 83u32, 37u32, 29u32),
            (120, 45, 51, 19),
            (64, 64, 64, 64),
            (40, 128, 40, 1),
            (31, 17, 19, 37),
        ];
        for (w, h, tw, th) in cases {
            let mut img = RgbaImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let px = if x < w / 3 {
                        // Hidden bright red under alpha=0 to catch any premultiplication bug.
                        Rgba([255, 0, 0, 0])
                    } else if x < (2 * w) / 3 {
                        // Semi-transparent gradient.
                        Rgba([
                            ((x * 19 + y * 7) % 256) as u8,
                            ((x * 11 + y * 23) % 256) as u8,
                            ((x * 5 + y * 31) % 256) as u8,
                            (((x + y) * 13) % 255 + 1) as u8,
                        ])
                    } else {
                        // Opaque pattern.
                        Rgba([
                            ((x * 37 + y * 3) % 256) as u8,
                            ((x * 13 + y * 41) % 256) as u8,
                            ((x * 29 + y * 17) % 256) as u8,
                            255,
                        ])
                    };
                    img.put_pixel(x, y, px);
                }
            }
            let got = resize_rgba_premultiplied_bilinear(&img, tw, th);
            let want = reference_resize(&img, tw, th);
            assert_eq!(got.dimensions(), want.dimensions());
            for y in 0..th {
                for x in 0..tw {
                    let g = got.get_pixel(x, y).0;
                    let e = want.get_pixel(x, y).0;
                    for c in 0..4 {
                        let diff = g[c].abs_diff(e[c]);
                        assert!(
                            diff <= 1,
                            "({w}x{h}->{tw}x{th}) mismatch at ({x},{y}) ch {c}: got {g:?}, ref {e:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn resize_rgba_premultiplied_bilinear_tall_strip_is_linear_and_exact() {
        use image::Rgba32FImage;

        // 1. Tall narrow strip `2 x 200_000 -> 2 x 40` (v_ratio = 5000):
        // Recomputing `triangle_axis_span` on every intervening source row would perform
        // ~2 billion vertical weight evaluations; caching `pending_span` must stay strictly
        // `O(h + target_h)` (`<= 4*h + 6*target_h`, ~800k evaluations).
        let (w, h, tw, th) = (2u32, 200_000u32, 2u32, 40u32);
        let mut strip = RgbaImage::new(w, h);
        for y in 0..h {
            let band = y / 50_000;
            let px0 = match band {
                0 => Rgba([255, 0, 0, 0]),     // hidden red under alpha=0
                1 => Rgba([40, 180, 90, 128]), // semi-transparent green
                2 => Rgba([20, 60, 220, 255]), // opaque blue
                _ => Rgba([240, 210, 30, 192]),
            };
            let px1 = Rgba([
                ((y * 7) % 256) as u8,
                ((y * 13) % 256) as u8,
                ((y * 29) % 256) as u8,
                if y % 3 == 0 { 0 } else { 200 },
            ]);
            strip.put_pixel(0, y, px0);
            strip.put_pixel(1, y, px1);
        }

        let mut v_evals = 0u64;
        let got = resize_rgba_premultiplied_bilinear_inner(&strip, tw, th, Some(&mut v_evals));
        let max_linear_evals = 4 * u64::from(h) + 6 * u64::from(th);
        assert!(
            v_evals <= max_linear_evals,
            "tall strip downscale exceeded O(h + target_h) vertical weight evals: got {v_evals}, cap {max_linear_evals}"
        );

        let mut premul = Rgba32FImage::new(w, h);
        for (src, dst) in strip.pixels().zip(premul.pixels_mut()) {
            let a = f32::from(src[3]) / 255.0;
            *dst = Rgba([
                (f32::from(src[0]) / 255.0) * a,
                (f32::from(src[1]) / 255.0) * a,
                (f32::from(src[2]) / 255.0) * a,
                a,
            ]);
        }
        let resized_ref =
            image::imageops::resize(&premul, tw, th, image::imageops::FilterType::Triangle);
        for y in 0..th {
            for x in 0..tw {
                let g = got.get_pixel(x, y).0;
                let e = unpremultiply_rgba_pixel(resized_ref.get_pixel(x, y).0).0;
                for c in 0..4 {
                    assert!(
                        g[c].abs_diff(e[c]) <= 1,
                        "tall strip mismatch at ({x},{y}) ch {c}: got {g:?}, ref {e:?}"
                    );
                }
            }
        }

        // 2. Extreme 100,000:1 downscale `1 x 200_000 -> 1 x 2` must also remain strictly linear.
        let extreme = RgbaImage::from_pixel(1, 200_000, Rgba([100, 150, 200, 128]));
        let mut extreme_evals = 0u64;
        let extreme_out =
            resize_rgba_premultiplied_bilinear_inner(&extreme, 1, 2, Some(&mut extreme_evals));
        assert!(
            extreme_evals <= 4 * 200_000 + 6 * 2,
            "extreme 1x200_000 -> 1x2 exceeded linear vertical weight evals: got {extreme_evals}"
        );
        for y in 0..2 {
            assert_eq!(extreme_out.get_pixel(0, y).0, [100, 150, 200, 128]);
        }

        // 3. Vertical upscale branch `2 x 40 -> 2 x 400` must remain O(target_h).
        let small = RgbaImage::from_pixel(2, 40, Rgba([64, 128, 192, 255]));
        let mut up_evals = 0u64;
        let up_out = resize_rgba_premultiplied_bilinear_inner(&small, 2, 400, Some(&mut up_evals));
        assert!(
            up_evals <= 6 * 400,
            "upscale exceeded O(target_h) vertical weight evals: got {up_evals}"
        );
        assert_eq!(up_out.get_pixel(0, 200).0, [64, 128, 192, 255]);
    }
}
