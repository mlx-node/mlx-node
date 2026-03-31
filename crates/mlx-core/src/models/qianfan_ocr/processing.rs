//! Image Processing for Qianfan-OCR (InternVL Dynamic Tiling)
//!
//! InternVL uses a "dynamic tiling" approach: instead of resizing images to a
//! single fixed size, it splits images into 1-12 tiles of 448x448 plus an
//! optional thumbnail, allowing the model to handle various resolutions and
//! aspect ratios.
//!
//! Pipeline:
//! 1. Find the best (rows, cols) grid for the image's aspect ratio
//! 2. Resize image to (cols * 448, rows * 448)
//! 3. Split into rows * cols tiles of 448x448
//! 4. Optionally append a thumbnail (original resized to 448x448)
//! 5. Normalize each tile with ImageNet mean/std
//! 6. Stack into a single MxArray [num_tiles, 448, 448, 3] (NHWC for MLX)

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use napi::bindgen_prelude::*;

use crate::array::{DType, MxArray};
use crate::models::qianfan_ocr::config::QianfanOCRConfig;

// ============================================================================
// ProcessedImage
// ============================================================================

/// Result of processing a single image through dynamic tiling.
pub(crate) struct ProcessedImage {
    /// Pixel values as MxArray [num_tiles, 448, 448, 3] in bf16 (NHWC)
    pub pixel_values: MxArray,
    /// Total number of tiles (including thumbnail if added)
    pub num_tiles: u32,
}

// ============================================================================
// find_closest_aspect_ratio
// ============================================================================

/// Find the best (rows, cols) grid that matches the image's aspect ratio.
///
/// Enumerates all valid (rows, cols) pairs where `min_num <= rows*cols <= max_num`,
/// then picks the pair whose target ratio (cols/rows) is closest to the image's
/// aspect ratio. Ties are broken by preferring grids whose total pixel area is
/// closer to (but not less than half of) the original image area.
fn find_closest_aspect_ratio(
    aspect_ratio: f64,
    min_num: u32,
    max_num: u32,
    image_size: u32,
    orig_width: u32,
    orig_height: u32,
) -> (u32, u32) {
    // Collect all valid (rows, cols) pairs
    let mut target_ratios: Vec<(u32, u32)> = Vec::new();
    for n in min_num..=max_num {
        for i in 1..=n {
            for j in 1..=n {
                let product = i * j;
                if product >= min_num && product <= max_num {
                    // Deduplicate: only add if not already present
                    let pair = (i, j);
                    if !target_ratios.contains(&pair) {
                        target_ratios.push(pair);
                    }
                }
            }
        }
    }

    // Sort by total tiles (rows * cols) ascending for stable tie-breaking
    target_ratios.sort_by_key(|&(r, c)| r * c);

    let area = orig_width as f64 * orig_height as f64;
    let mut best_ratio = (1u32, 1u32);
    let mut best_ratio_diff = f64::INFINITY;

    for &(rows, cols) in &target_ratios {
        // Target aspect ratio: cols / rows (width_tiles / height_tiles)
        let target_aspect_ratio = cols as f64 / rows as f64;
        let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();

        if ratio_diff < best_ratio_diff {
            best_ratio_diff = ratio_diff;
            best_ratio = (rows, cols);
        } else if (ratio_diff - best_ratio_diff).abs() < f64::EPSILON {
            // Tied — prefer grid whose area covers the original image
            let grid_area = rows as f64 * cols as f64 * (image_size as f64).powi(2);
            if area > 0.5 * grid_area {
                best_ratio = (rows, cols);
            }
        }
    }

    best_ratio
}

// ============================================================================
// dynamic_preprocess
// ============================================================================

/// Split an image into tiles using dynamic resolution tiling.
///
/// 1. Find best (rows, cols) grid for the image's aspect ratio
/// 2. Resize to (cols * image_size, rows * image_size)
/// 3. Split into rows * cols tiles of image_size x image_size
/// 4. If `use_thumbnail` and more than 1 tile, append a thumbnail
fn dynamic_preprocess(
    image: &DynamicImage,
    min_num: u32,
    max_num: u32,
    image_size: u32,
    use_thumbnail: bool,
) -> Vec<DynamicImage> {
    let (orig_width, orig_height) = image.dimensions();
    let aspect_ratio = orig_width as f64 / orig_height as f64;

    let (rows, cols) = find_closest_aspect_ratio(
        aspect_ratio,
        min_num,
        max_num,
        image_size,
        orig_width,
        orig_height,
    );

    let target_width = cols * image_size;
    let target_height = rows * image_size;

    // Resize to target grid dimensions (bicubic / CatmullRom)
    let resized = image.resize_exact(target_width, target_height, FilterType::CatmullRom);

    // Split into tiles
    let mut tiles: Vec<DynamicImage> = Vec::with_capacity((rows * cols + 1) as usize);
    for row in 0..rows {
        for col in 0..cols {
            let x = col * image_size;
            let y = row * image_size;
            let tile = resized.crop_imm(x, y, image_size, image_size);
            tiles.push(tile);
        }
    }

    // Optionally append thumbnail
    if use_thumbnail && tiles.len() > 1 {
        let thumbnail = image.resize_exact(image_size, image_size, FilterType::CatmullRom);
        tiles.push(thumbnail);
    }

    tiles
}

// ============================================================================
// QianfanImageProcessor
// ============================================================================

/// Image processor for Qianfan-OCR using InternVL-style dynamic tiling.
pub(crate) struct QianfanImageProcessor {
    image_size: u32,
    min_dynamic_patch: u32,
    max_dynamic_patch: u32,
    dynamic_image_size: bool,
    use_thumbnail: bool,
    mean: [f32; 3],
    std: [f32; 3],
}

impl QianfanImageProcessor {
    /// Create a new processor from model configuration.
    pub fn new(config: &QianfanOCRConfig) -> Self {
        Self {
            image_size: config.vision_config.image_size as u32,
            min_dynamic_patch: config.min_dynamic_patch as u32,
            max_dynamic_patch: config.max_dynamic_patch as u32,
            dynamic_image_size: config.dynamic_image_size,
            use_thumbnail: config.use_thumbnail,
            // ImageNet normalization constants
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    /// Process a single image from raw encoded bytes (PNG/JPEG/etc).
    ///
    /// Returns pixel values as [num_tiles, 448, 448, 3] in bf16 (NHWC format
    /// for MLX's vision encoder).
    pub fn process(&self, image_bytes: &[u8]) -> Result<ProcessedImage> {
        let img = image::load_from_memory(image_bytes).map_err(|e| {
            Error::new(
                Status::GenericFailure,
                format!("Failed to decode image: {e}"),
            )
        })?;

        let tiles = if self.dynamic_image_size {
            dynamic_preprocess(
                &img,
                self.min_dynamic_patch,
                self.max_dynamic_patch,
                self.image_size,
                self.use_thumbnail,
            )
        } else {
            // Static: single tile resized to image_size x image_size
            vec![img.resize_exact(
                self.image_size,
                self.image_size,
                image::imageops::FilterType::CatmullRom,
            )]
        };

        let num_tiles = tiles.len() as u32;
        let h = self.image_size as usize;
        let w = self.image_size as usize;
        let channels = 3usize;

        // Pre-allocate for all tiles in NHWC layout: [num_tiles, H, W, C]
        let mut pixel_data: Vec<f32> = Vec::with_capacity(num_tiles as usize * h * w * channels);

        for tile in &tiles {
            let rgb = tile.to_rgb8();
            for chunk in rgb.as_raw().chunks_exact(channels) {
                for (c, &byte) in chunk.iter().enumerate() {
                    // Rescale to [0, 1] then normalize with ImageNet stats
                    let value = byte as f32 / 255.0;
                    let normalized = (value - self.mean[c]) / self.std[c];
                    pixel_data.push(normalized);
                }
            }
        }

        // Create MxArray [num_tiles, H, W, C] — NHWC for MLX vision encoder
        let pixel_values = MxArray::from_float32(
            &pixel_data,
            &[num_tiles as i64, h as i64, w as i64, channels as i64],
        )?;

        // Convert to bf16 for inference
        let pixel_values = pixel_values.astype(DType::BFloat16)?;

        Ok(ProcessedImage {
            pixel_values,
            num_tiles,
        })
    }

    /// Process multiple images from raw encoded bytes.
    pub fn process_many(&self, images: &[&[u8]]) -> Result<Vec<ProcessedImage>> {
        images.iter().map(|data| self.process(data)).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Convert BigInt64Array from shape() to Vec<i64> for easy assertion.
    fn shape_to_vec(shape: napi::bindgen_prelude::BigInt64Array) -> Vec<i64> {
        shape.as_ref().to_vec()
    }

    // -- find_closest_aspect_ratio tests --

    #[test]
    fn test_find_closest_aspect_ratio_square() {
        // 1:1 image -> should pick (1, 1) since target ratio 1/1 = 1.0 matches exactly
        let (rows, cols) = find_closest_aspect_ratio(1.0, 1, 12, 448, 448, 448);
        assert_eq!((rows, cols), (1, 1));
    }

    #[test]
    fn test_find_closest_aspect_ratio_landscape_2_1() {
        // 2:1 landscape -> target ratio 2.0 -> best is (1, 2) since 2/1 = 2.0
        let (rows, cols) = find_closest_aspect_ratio(2.0, 1, 12, 448, 896, 448);
        assert_eq!(cols as f64 / rows as f64, 2.0);
        assert_eq!((rows, cols), (1, 2));
    }

    #[test]
    fn test_find_closest_aspect_ratio_portrait_1_3() {
        // 1:3 portrait -> aspect_ratio = 1/3 ≈ 0.333
        // Best should be (3, 1) since 1/3 = 0.333
        let (rows, cols) = find_closest_aspect_ratio(1.0 / 3.0, 1, 12, 448, 448, 1344);
        assert_eq!((rows, cols), (3, 1));
    }

    #[test]
    fn test_find_closest_aspect_ratio_extreme_wide() {
        // Very wide: 10:1 aspect ratio
        // Max tiles = 12, so best could be (1, 10), (1, 11), (1, 12), (2, 5), (2, 6), etc.
        // (1, 10) gives ratio 10.0 — exact match
        let (rows, cols) = find_closest_aspect_ratio(10.0, 1, 12, 448, 4480, 448);
        assert_eq!((rows, cols), (1, 10));
    }

    #[test]
    fn test_find_closest_aspect_ratio_extreme_tall() {
        // Very tall: 1:10 aspect ratio = 0.1
        // (10, 1) gives ratio 1/10 = 0.1 — exact match
        let (rows, cols) = find_closest_aspect_ratio(0.1, 1, 12, 448, 448, 4480);
        assert_eq!((rows, cols), (10, 1));
    }

    #[test]
    fn test_find_closest_aspect_ratio_4_3() {
        // 4:3 aspect ratio ≈ 1.333
        // Candidates: (3, 4) -> 4/3 = 1.333 — exact match
        let (rows, cols) = find_closest_aspect_ratio(4.0 / 3.0, 1, 12, 448, 1200, 900);
        assert_eq!(cols as f64 / rows as f64, 4.0 / 3.0);
    }

    #[test]
    fn test_find_closest_aspect_ratio_min_num_respected() {
        // With min_num = 1, a tiny image should still get (1,1)
        let (rows, cols) = find_closest_aspect_ratio(1.0, 1, 12, 448, 100, 100);
        assert!(rows * cols >= 1);
    }

    #[test]
    fn test_find_closest_aspect_ratio_tie_break_area() {
        // When two ratios tie, the one whose area covers more of the original wins.
        // 1.5 aspect ratio: (2, 3) -> 3/2 = 1.5, (4, 6) -> 6/4 = 1.5
        // Both match exactly. For a large image, the larger grid should win.
        let (rows, cols) = find_closest_aspect_ratio(1.5, 1, 12, 448, 2000, 1333);
        // Both (2,3) and (4,6) are 1.5 — area decides
        // Original area = 2000*1333 = 2,666,000
        // (2,3) grid area = 6 * 448^2 = 1,203,264 -> 0.5 * 1,203,264 = 601,632 < 2,666,000 -> upgrade
        // (4,6) would have product 24 which exceeds max_num=12, so (2,3) is the largest valid match
        assert_eq!(cols as f64 / rows as f64, 1.5);
    }

    // -- dynamic_preprocess tests --

    #[test]
    fn test_dynamic_preprocess_exact_single_tile() {
        // 448x448 image -> (1,1) grid -> 1 tile, no thumbnail (only 1 tile)
        let img = DynamicImage::new_rgb8(448, 448);
        let tiles = dynamic_preprocess(&img, 1, 12, 448, true);
        // 1 tile, thumbnail not added because tiles.len() == 1
        assert_eq!(tiles.len(), 1);
        let (w, h) = tiles[0].dimensions();
        assert_eq!((w, h), (448, 448));
    }

    #[test]
    fn test_dynamic_preprocess_landscape_with_thumbnail() {
        // 896x448 image -> aspect 2.0 -> (1,2) grid -> 2 tiles + 1 thumbnail = 3
        let img = DynamicImage::new_rgb8(896, 448);
        let tiles = dynamic_preprocess(&img, 1, 12, 448, true);
        assert_eq!(tiles.len(), 3); // 2 tiles + 1 thumbnail
        for tile in &tiles {
            let (w, h) = tile.dimensions();
            assert_eq!((w, h), (448, 448));
        }
    }

    #[test]
    fn test_dynamic_preprocess_no_thumbnail() {
        // Same image but use_thumbnail = false -> 2 tiles only
        let img = DynamicImage::new_rgb8(896, 448);
        let tiles = dynamic_preprocess(&img, 1, 12, 448, false);
        assert_eq!(tiles.len(), 2);
    }

    #[test]
    fn test_dynamic_preprocess_portrait() {
        // 448x1344 image -> aspect 1/3 -> (3,1) grid -> 3 tiles + thumbnail = 4
        let img = DynamicImage::new_rgb8(448, 1344);
        let tiles = dynamic_preprocess(&img, 1, 12, 448, true);
        assert_eq!(tiles.len(), 4); // 3 tiles + 1 thumbnail
    }

    #[test]
    fn test_dynamic_preprocess_small_image_upscaled() {
        // Small 100x100 image -> aspect 1.0 -> (1,1) -> 1 tile, no thumbnail
        let img = DynamicImage::new_rgb8(100, 100);
        let tiles = dynamic_preprocess(&img, 1, 12, 448, true);
        assert_eq!(tiles.len(), 1);
        let (w, h) = tiles[0].dimensions();
        assert_eq!((w, h), (448, 448));
    }

    #[test]
    fn test_dynamic_preprocess_tile_dimensions() {
        // 2x2 grid -> 4 tiles, all must be exactly image_size x image_size
        let img = DynamicImage::new_rgb8(896, 896);
        let tiles = dynamic_preprocess(&img, 1, 12, 448, false);
        // aspect 1.0 -> (1,1) or could be (2,2) depending on area tie-break
        // (1,1) ratio=1.0, (2,2) ratio=1.0 — tie, area decides
        // original 896*896 = 802,816. (2,2) grid_area = 4*448^2 = 802,816.
        // 0.5 * 802,816 = 401,408 < 802,816 -> pick (2,2)
        assert_eq!(tiles.len(), 4);
        for tile in &tiles {
            let (w, h) = tile.dimensions();
            assert_eq!((w, h), (448, 448));
        }
    }

    // -- Normalization tests --

    #[test]
    fn test_normalization_range() {
        // Create a small white image (all 255) and verify normalized values
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        // White pixel: (255/255 - mean) / std
        for c in 0..3 {
            let normalized = (1.0 - mean[c]) / std[c];
            // Should be positive and finite
            assert!(normalized.is_finite());
            assert!(normalized > 0.0);
        }

        // Black pixel: (0/255 - mean) / std
        for c in 0..3 {
            let normalized = (0.0 - mean[c]) / std[c];
            // Should be negative and finite
            assert!(normalized.is_finite());
            assert!(normalized < 0.0);
        }
    }

    #[test]
    fn test_normalization_values() {
        // Specific check: for channel 0 (R), mean=0.485, std=0.229
        // pixel=128 -> val=128/255 ≈ 0.502 -> (0.502 - 0.485) / 0.229 ≈ 0.074
        let val: f64 = 128.0 / 255.0;
        let normalized = (val - 0.485) / 0.229;
        assert!((normalized - 0.074).abs() < 0.01);
    }

    // -- QianfanImageProcessor tests --

    #[test]
    fn test_processor_from_config() {
        let config = QianfanOCRConfig::default();
        let processor = QianfanImageProcessor::new(&config);
        assert_eq!(processor.image_size, 448);
        assert_eq!(processor.min_dynamic_patch, 1);
        assert_eq!(processor.max_dynamic_patch, 12);
        assert!(processor.use_thumbnail);
        assert_eq!(processor.mean, [0.485, 0.456, 0.406]);
        assert_eq!(processor.std, [0.229, 0.224, 0.225]);
    }

    #[test]
    fn test_process_synthetic_image() {
        // Create a minimal valid PNG in memory
        let img = DynamicImage::new_rgb8(448, 448);
        let mut png_bytes: Vec<u8> = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut png_bytes),
            image::ImageFormat::Png,
        )
        .unwrap();

        let config = QianfanOCRConfig::default();
        let processor = QianfanImageProcessor::new(&config);
        let result = processor.process(&png_bytes).unwrap();

        // 448x448 -> 1 tile (no thumbnail for single tile)
        assert_eq!(result.num_tiles, 1);
        assert_eq!(
            shape_to_vec(result.pixel_values.shape().unwrap()),
            vec![1, 448, 448, 3]
        ); // NHWC
    }

    #[test]
    fn test_process_landscape_image() {
        // 896x448 -> should produce 3 tiles (2 grid + 1 thumbnail)
        let img = DynamicImage::new_rgb8(896, 448);
        let mut png_bytes: Vec<u8> = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut png_bytes),
            image::ImageFormat::Png,
        )
        .unwrap();

        let config = QianfanOCRConfig::default();
        let processor = QianfanImageProcessor::new(&config);
        let result = processor.process(&png_bytes).unwrap();

        assert_eq!(result.num_tiles, 3);
        assert_eq!(
            shape_to_vec(result.pixel_values.shape().unwrap()),
            vec![3, 448, 448, 3]
        );
    }

    #[test]
    fn test_process_invalid_bytes() {
        let config = QianfanOCRConfig::default();
        let processor = QianfanImageProcessor::new(&config);
        let result = processor.process(b"not an image");
        assert!(result.is_err());
    }

    #[test]
    fn test_process_many() {
        let img1 = DynamicImage::new_rgb8(448, 448);
        let img2 = DynamicImage::new_rgb8(896, 448);

        let mut bytes1: Vec<u8> = Vec::new();
        img1.write_to(
            &mut std::io::Cursor::new(&mut bytes1),
            image::ImageFormat::Png,
        )
        .unwrap();

        let mut bytes2: Vec<u8> = Vec::new();
        img2.write_to(
            &mut std::io::Cursor::new(&mut bytes2),
            image::ImageFormat::Png,
        )
        .unwrap();

        let config = QianfanOCRConfig::default();
        let processor = QianfanImageProcessor::new(&config);
        let results = processor
            .process_many(&[bytes1.as_slice(), bytes2.as_slice()])
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].num_tiles, 1); // 448x448 -> 1 tile
        assert_eq!(results[1].num_tiles, 3); // 896x448 -> 2+1 tiles
    }

    #[test]
    fn test_process_output_dtype() {
        let img = DynamicImage::new_rgb8(448, 448);
        let mut png_bytes: Vec<u8> = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut png_bytes),
            image::ImageFormat::Png,
        )
        .unwrap();

        let config = QianfanOCRConfig::default();
        let processor = QianfanImageProcessor::new(&config);
        let result = processor.process(&png_bytes).unwrap();

        // Should be bf16
        let dtype = result.pixel_values.dtype().unwrap();
        assert_eq!(dtype, DType::BFloat16);
    }
}
