use image::DynamicImage;
use napi::bindgen_prelude::{Error, Result};

use crate::array::attention::scaled_dot_product_attention;
use crate::array::{DType, MxArray};
use crate::nn::{Activations, LayerNorm, Linear};
use crate::tools::inkling::{CONTENT_IMAGE_ID, IMAGE_PAD_ID};

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
/// 1. Composite alpha channel (if any) onto an 8×8 checkerboard (`204` / `153`).
/// 2. Optionally downscale only if aligned area exceeds `MAX_VISION_TOKENS` (`4096` merged cells).
/// 3. Pad bottom & right to multiples of `alignment = patch_size * spatial_merge_size` (`40`)
///    using pixel value **`-1.0`**.
/// 4. Normalize per channel: `(v / 255.0 - mean[c]) / std[c]` (for `mean=0.5, std=0.5`,
///    equals `2.0 * (v / 255.0) - 1.0`, mapping padding `-1` to `-257/255`).
pub fn preprocess_image_bytes(
    bytes: &[u8],
    config: &Inkling2VisionConfig,
) -> Result<PreprocessedImage> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| Error::from_reason(format!("Inkling2 vision: failed to decode image: {e}")))?;
    preprocess_dynamic_image(&img, config)
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

    // 1. Alpha-composite onto 8×8 checkerboard (204 / 153) when transparent, else RGB8.
    let mut rgb = if img.color().has_alpha() {
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let mut out = image::RgbImage::new(w, h);
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
    } else {
        img.to_rgb8()
    };

    let (mut w, mut h) = (rgb.width() as usize, rgb.height() as usize);
    if w == 0 || h == 0 {
        return Err(Error::from_reason("Inkling2 vision: empty image"));
    }

    // 2. Cap oversized images at MAX_VISION_TOKENS merged tokens (never upscaling, always
    //    checking the post-padding grid).
    let (target_w, target_h) = fit_vision_dimensions(w, h, alignment, MAX_VISION_TOKENS);
    if target_w != w || target_h != h {
        rgb = image::imageops::resize(
            &rgb,
            target_w as u32,
            target_h as u32,
            image::imageops::FilterType::CatmullRom,
        );
        w = target_w;
        h = target_h;
    }

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

/// One ViT block of the 23-layer PE-Lang vision encoder.
pub struct Inkling2VisionBlock {
    pub norm1: LayerNorm,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub ls1: MxArray,
    pub norm2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
    pub ls2: MxArray,
}

impl Inkling2VisionBlock {
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
        let act_dtype = hidden.dtype()?;

        let x = self.norm1.forward(hidden)?;
        let q = self
            .q_proj
            .forward(&x)?
            .reshape(&[n, num_heads as i64, head_dim as i64])?;
        let k = self
            .k_proj
            .forward(&x)?
            .reshape(&[n, num_heads as i64, head_dim as i64])?;
        let v = self
            .v_proj
            .forward(&x)?
            .reshape(&[n, num_heads as i64, head_dim as i64])?;

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

        let attn_out = self.o_proj.forward(&attn)?;
        let ls1_cast = self.ls1.astype(act_dtype)?;
        let h1 = hidden.add(&attn_out.mul(&ls1_cast)?)?;

        let x2 = self.norm2.forward(&h1)?;
        let fc1_out = self.fc1.forward(&x2)?;
        let gelu_out = Activations::gelu_exact(&fc1_out)?;
        let mlp_out = self.fc2.forward(&gelu_out)?;
        let ls2_cast = self.ls2.astype(act_dtype)?;
        h1.add(&mlp_out.mul(&ls2_cast)?)
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
            arrays.push(l.get_weight());
            if let Some(b) = l.get_bias() {
                arrays.push(b);
            }
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

/// Complete PE-Lang ViT + 2×2 spatial merge projector (`mmproj-BF16.gguf`).
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
    /// Projector `mm.1` (`4096 -> 4096`, no bias) and `mm.2` (`4096 -> 2304`, no bias).
    pub proj_fc1: Linear,
    pub proj_fc2: Linear,
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
    /// Returns `[grid_rows * grid_cols, projection_dim]` in BF16 (ready to overwrite `IMAGE_PAD_ID`
    /// positions **after** `embed_norm`).
    pub fn encode(&self, prep: &PreprocessedImage) -> Result<MxArray> {
        let p = self.config.patch_size;
        let m = self.config.spatial_merge_size;
        let h_p = prep.grid_rows * m;
        let w_p = prep.grid_cols * m;
        let n_patches = h_p * w_p;
        let hidden_size = self.config.hidden_size;

        // 1. Patch embedding via exact stride=patch_size reshape + matmul
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
        let patch_emb = patches_chw.matmul(&w_t)?.astype(DType::BFloat16)?; // `[n_patches, 1024]`

        // 2. Bilinearly interpolate 32×32 position embedding (`align_corners = false`)
        let pos_emb = bilinear_interp_posemb(
            &self.position_embedding,
            self.config.posemb_grid_size(),
            h_p,
            w_p,
        )?
        .astype(DType::BFloat16)?;
        let patch_with_pos = patch_emb.add(&pos_emb)?;

        // 3. Prepend CLS token (`class_embedding + class_position_embedding`) + `pre_ln`
        let cls = self
            .class_embedding
            .astype(DType::Float32)?
            .add(&self.class_position_embedding.astype(DType::Float32)?)?
            .astype(DType::BFloat16)?
            .reshape(&[1, hidden_size as i64])?;
        let seq = MxArray::concatenate(&cls, &patch_with_pos, 0)?; // `[n_patches + 1, 1024]`
        let mut hidden = self.pre_ln.forward(&seq)?;

        // 4. Build 2D interleaved axial RoPE cos/sin tables (`CLS` at `(-1, -1)` -> pos `(0, 0)`, angle 0)
        let head_dim = self.config.head_dim();
        let (cos_table, sin_table) =
            build_2d_rope_tables(h_p, w_p, head_dim, self.config.rope_theta)?;

        // 5. 23 ViT blocks
        for blk in &self.blocks {
            hidden = blk.forward(
                &hidden,
                &cos_table,
                &sin_table,
                self.config.num_attention_heads,
                head_dim,
            )?;
        }

        // 6. Drop CLS, 2×2 spatial merge: `(rows, 2, cols, 2, D)` -> `(rows, cols, 2, 2, D)` -> `(rows*cols, 4*D)`
        let patches_out = hidden.slice(&[1, 0], &[(n_patches + 1) as i64, hidden_size as i64])?;
        let merged_dim = (m * m * hidden_size) as i64;
        let merged = patches_out
            .reshape(&[
                prep.grid_rows as i64,
                m as i64,
                prep.grid_cols as i64,
                m as i64,
                hidden_size as i64,
            ])?
            .transpose(Some(&[0, 2, 1, 3, 4]))?
            .reshape(&[(prep.grid_rows * prep.grid_cols) as i64, merged_dim])?;

        // 7. Projector: `mm.2(gelu_exact(mm.1(merged)))`
        let h_proj = self.proj_fc1.forward(&merged)?;
        let act = Activations::gelu_exact(&h_proj)?;
        self.proj_fc2.forward(&act)
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
        arrays.push(self.proj_fc1.get_weight());
        if let Some(b) = self.proj_fc1.get_bias() {
            arrays.push(b);
        }
        arrays.push(self.proj_fc2.get_weight());
        if let Some(b) = self.proj_fc2.get_bias() {
            arrays.push(b);
        }
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
}
