//! Strict preflight image validation exported to NAPI (`validateImageBytes`)
//! and shared across Rust vision loaders.
//!
//! Accepts ONLY `JPEG | PNG | GIF | WebP`, enforces non-empty payloads up to
//! 32 MiB and valid dimensions up to 32 MP, and runs a full decode under
//! bounded [`image::Limits`] so truncated/corrupt bodies and unsupported
//! formats (BMP, TIFF, HEIC, garbage) fail deterministically with the
//! structured `image_processing_error:` prefix.

use std::io::Cursor;

use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};
use napi::bindgen_prelude::{AsyncTask, Env, Error, Result, Status, Task, Uint8Array};
use napi_derive::napi;

/// Structured prefix recognized by `@mlx-node/lm`'s `isImageProcessingError`
/// to map native image failures to HTTP 400 instead of HTTP 500.
pub const IMAGE_PROCESSING_ERROR_PREFIX: &str = "image_processing_error:";

/// Canonical client-safe error message returned for corrupt or unsupported images.
pub const IMAGE_PROCESSING_ERROR_MESSAGE: &str = "image_processing_error: Could not process image";

/// Maximum encoded image byte length accepted by preflight validation (32 MiB).
pub const MAX_IMAGE_BYTES: usize = 32 * 1024 * 1024;

/// Maximum decoded pixel area accepted by preflight validation (32 MP).
pub const MAX_IMAGE_PIXELS: u64 = 32_000_000;

/// Build a structured `image_processing_error:` NAPI error with diagnostic detail.
pub(crate) fn image_processing_error(detail: impl std::fmt::Display) -> Error {
    Error::new(
        Status::InvalidArg,
        format!("{IMAGE_PROCESSING_ERROR_MESSAGE} ({detail})"),
    )
}

/// Validate encoded image bytes in Rust:
/// - Non-empty and `<= 32 MiB`
/// - Format strictly in `{JPEG, PNG, GIF, WebP}` (rejects BMP, TIFF, HEIC, etc.)
/// - Dimensions `w > 0`, `h > 0`, and `w * h <= 32 MP`
/// - Full decode under bounded `image::Limits` so truncated/corrupt bodies fail
pub fn validate_image_bytes_slice(bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() || bytes.len() > MAX_IMAGE_BYTES {
        return Err(Error::new(
            Status::InvalidArg,
            IMAGE_PROCESSING_ERROR_MESSAGE,
        ));
    }

    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| Error::new(Status::InvalidArg, IMAGE_PROCESSING_ERROR_MESSAGE))?;

    match reader.format() {
        Some(ImageFormat::Jpeg | ImageFormat::Png | ImageFormat::Gif | ImageFormat::WebP) => {}
        _ => {
            return Err(Error::new(
                Status::InvalidArg,
                IMAGE_PROCESSING_ERROR_MESSAGE,
            ));
        }
    }

    let mut limits = Limits::default();
    limits.max_alloc = Some(MAX_IMAGE_PIXELS.saturating_mul(16).max(128 * 1024 * 1024));
    reader.limits(limits);

    let decoder = reader
        .into_decoder()
        .map_err(|_| Error::new(Status::InvalidArg, IMAGE_PROCESSING_ERROR_MESSAGE))?;

    let (w, h) = decoder.dimensions();
    if w == 0 || h == 0 || u64::from(w).saturating_mul(u64::from(h)) > MAX_IMAGE_PIXELS {
        return Err(Error::new(
            Status::InvalidArg,
            IMAGE_PROCESSING_ERROR_MESSAGE,
        ));
    }

    DynamicImage::from_decoder(decoder)
        .map_err(|_| Error::new(Status::InvalidArg, IMAGE_PROCESSING_ERROR_MESSAGE))?;

    Ok(())
}

/// Validate raw encoded image bytes before starting inference or committing SSE
/// headers (exported to JS as `validateImageBytes`).
#[napi]
pub fn validate_image_bytes(data: Uint8Array) -> Result<()> {
    validate_image_bytes_slice(data.as_ref())
}

/// Async libuv worker-pool task backing [`validate_image_bytes_async`].
pub struct ValidateImageBytesTask {
    bytes: Option<Vec<u8>>,
}

impl ValidateImageBytesTask {
    fn from_slice(slice: &[u8]) -> Self {
        let bytes = if slice.is_empty() || slice.len() > MAX_IMAGE_BYTES {
            None
        } else {
            Some(slice.to_vec())
        };
        Self { bytes }
    }
}

impl Task for ValidateImageBytesTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<Self::Output> {
        let Some(ref bytes) = self.bytes else {
            return Err(Error::new(
                Status::InvalidArg,
                IMAGE_PROCESSING_ERROR_MESSAGE,
            ));
        };
        validate_image_bytes_slice(bytes)
    }

    fn resolve(&mut self, _env: Env, _output: Self::Output) -> Result<Self::JsValue> {
        Ok(())
    }
}

/// Validate raw encoded image bytes on libuv's worker thread pool without
/// blocking the JS event loop (exported to JS as `validateImageBytesAsync`).
#[napi(ts_return_type = "Promise<void>")]
pub fn validate_image_bytes_async(data: Uint8Array) -> AsyncTask<ValidateImageBytesTask> {
    AsyncTask::new(ValidateImageBytesTask::from_slice(data.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    const ONE_PIXEL_GIF: &[u8] = &[
        0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00,
        0x00, 0xff, 0xff, 0xff, 0x21, 0xf9, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44, 0x01, 0x00, 0x3b,
    ];

    fn encode_test_image(w: u32, h: u32, fmt: ImageFormat) -> Vec<u8> {
        let img = RgbImage::from_pixel(w, h, Rgb([12, 34, 56]));
        let mut cursor = Cursor::new(Vec::new());
        img.write_to(&mut cursor, fmt).unwrap();
        cursor.into_inner()
    }

    #[test]
    fn accepts_valid_png_jpeg_gif_webp() {
        let png = encode_test_image(4, 4, ImageFormat::Png);
        let jpeg = encode_test_image(4, 4, ImageFormat::Jpeg);
        let webp = encode_test_image(4, 4, ImageFormat::WebP);

        validate_image_bytes_slice(&png).expect("valid PNG");
        validate_image_bytes_slice(&jpeg).expect("valid JPEG");
        validate_image_bytes_slice(ONE_PIXEL_GIF).expect("valid GIF");
        validate_image_bytes_slice(&webp).expect("valid WebP");
        validate_image_bytes(Uint8Array::from(png)).expect("NAPI wrapper accepts valid PNG");
    }

    #[test]
    fn rejects_empty_garbage_truncated_and_oversized_payloads() {
        // Empty
        let err = validate_image_bytes_slice(&[]).unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);

        // Random garbage
        let err = validate_image_bytes_slice(b"not an image payload").unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);

        // Truncated PNG (valid IHDR header, truncated body)
        let png = encode_test_image(16, 16, ImageFormat::Png);
        let truncated = &png[..png.len() / 2];
        let err = validate_image_bytes_slice(truncated).unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);

        // Truncated JPEG (SOI + header only, no SOS/EOI)
        let jpeg = encode_test_image(16, 16, ImageFormat::Jpeg);
        let err = validate_image_bytes_slice(&jpeg[..jpeg.len() / 2]).unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);

        // > 32 MiB payload
        let too_big = vec![0u8; MAX_IMAGE_BYTES + 1];
        let err = validate_image_bytes_slice(&too_big).unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);
    }

    #[test]
    fn rejects_bmp_tiff_and_heic() {
        // Valid 1x1 24-bit BMP
        let bmp: &[u8] = &[
            0x42, 0x4d, 0x3a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
            0x28, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
            0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00,
            0x00, 0x00,
        ];
        let err = validate_image_bytes_slice(bmp).unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);

        // Little-endian TIFF header
        let tiff: &[u8] = &[
            0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let err = validate_image_bytes_slice(tiff).unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);

        // HEIC ISO-BMFF `ftypheic` box
        let heic: &[u8] = &[
            0x00, 0x00, 0x00, 0x18, b'f', b't', b'y', b'p', b'h', b'e', b'i', b'c', 0x00, 0x00,
            0x00, 0x00, b'm', b'i', b'f', b'1', b'h', b'e', b'i', b'c',
        ];
        let err = validate_image_bytes_slice(heic).unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);
    }

    #[test]
    fn rejects_zero_and_over_32mp_dimensions() {
        // Craft a minimal PNG IHDR with 8000x5000 = 40 MP (> 32 MP cap)
        let mut png = encode_test_image(1, 1, ImageFormat::Png);
        // IHDR width at bytes 16..20, height at bytes 20..24 (big-endian)
        png[16..20].copy_from_slice(&8000u32.to_be_bytes());
        png[20..24].copy_from_slice(&5000u32.to_be_bytes());
        let err = validate_image_bytes_slice(&png).unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);

        // Zero width
        let mut zero_png = encode_test_image(1, 1, ImageFormat::Png);
        zero_png[16..20].copy_from_slice(&0u32.to_be_bytes());
        let err = validate_image_bytes_slice(&zero_png).unwrap_err();
        assert_eq!(err.reason, IMAGE_PROCESSING_ERROR_MESSAGE);
    }

    #[test]
    fn async_task_matches_sync_validation() {
        let png = encode_test_image(4, 4, ImageFormat::Png);
        let jpeg = encode_test_image(4, 4, ImageFormat::Jpeg);
        let webp = encode_test_image(4, 4, ImageFormat::WebP);

        ValidateImageBytesTask::from_slice(&png)
            .compute()
            .expect("async task accepts valid PNG");
        ValidateImageBytesTask::from_slice(&jpeg)
            .compute()
            .expect("async task accepts valid JPEG");
        ValidateImageBytesTask::from_slice(ONE_PIXEL_GIF)
            .compute()
            .expect("async task accepts valid GIF");
        ValidateImageBytesTask::from_slice(&webp)
            .compute()
            .expect("async task accepts valid WebP");

        // Empty and > 32 MiB short-circuit without allocating and reject in compute()
        let mut empty_task = ValidateImageBytesTask::from_slice(&[]);
        assert!(empty_task.bytes.is_none());
        assert_eq!(
            empty_task.compute().unwrap_err().reason,
            IMAGE_PROCESSING_ERROR_MESSAGE
        );

        let too_big = vec![0u8; MAX_IMAGE_BYTES + 1];
        let mut oversized_task = ValidateImageBytesTask::from_slice(&too_big);
        assert!(oversized_task.bytes.is_none());
        assert_eq!(
            oversized_task.compute().unwrap_err().reason,
            IMAGE_PROCESSING_ERROR_MESSAGE
        );

        // Corrupt / truncated payloads reject with structured error
        let truncated = &png[..png.len() / 2];
        assert_eq!(
            ValidateImageBytesTask::from_slice(truncated)
                .compute()
                .unwrap_err()
                .reason,
            IMAGE_PROCESSING_ERROR_MESSAGE
        );
    }
}
