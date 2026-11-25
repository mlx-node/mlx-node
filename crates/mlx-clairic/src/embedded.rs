//! Embedded image validation, EXIF/alpha normalization, and rendered PDF page handling.
//!
//! Mirrors upstream `clairic_runtime/{embedded.py, api.py (ClairImage), documents.py}`.

use std::{fmt, io::Cursor, sync::LazyLock};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use image::{
    DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits, RgbImage, metadata::Orientation,
};
use regex::Regex;
use serde_json::Value;

use super::{
    errors::{ClairicError, Result, invalid_request},
    orientation::{orient_document, rotate_cw},
    settings::{MAX_IMAGE_BYTES, MAX_IMAGES, MAX_IMAGES_BYTES, MAX_PIXELS},
};

static STRICT_DATA_URI: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?is)^data:(image/(?:png|jpe?g|webp));base64,(.*)$").ok());

static PERMISSIVE_DATA_URL: LazyLock<Option<Regex>> =
    LazyLock::new(|| Regex::new(r"(?s)^data:([^;,]+);base64,(.*)$").ok());

/// Stable validation error matching upstream `embedded.EmbeddedImageError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedImageError {
    pub code: &'static str,
    pub message: String,
}

impl EmbeddedImageError {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn into_clairic_error(self) -> ClairicError {
        invalid_request(format!("{} ({})", self.message, self.code))
    }
}

impl fmt::Display for EmbeddedImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for EmbeddedImageError {}

impl From<EmbeddedImageError> for ClairicError {
    fn from(err: EmbeddedImageError) -> Self {
        err.into_clairic_error()
    }
}

/// Structured input for an embedded image before validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddedImageInput {
    /// A `data:image/{png,jpeg,webp};base64,...` URI string.
    DataUri(String),
    /// An explicit `{ content_type, base64, rotation? }` specification.
    Object {
        content_type: String,
        base64: String,
        rotation: Option<u16>,
    },
}

/// Validated embedded PNG, JPEG, or WebP image (port of upstream `api.ClairImage`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedImage {
    /// Canonical MIME type: `"image/png"`, `"image/jpeg"`, or `"image/webp"`.
    pub content_type: String,
    /// Canonical base64 encoding of the validated image bytes.
    pub base64: String,
    /// Explicit clockwise rotation (`0`, `90`, `180`, `270`) when provided by caller;
    /// `None` means auto-orient via [`orient_document`].
    pub rotation: Option<u16>,
}

impl ValidatedImage {
    /// Validate a JSON value (either a data URI string or `{content_type, base64, rotation?}` object)
    /// in strict mode, rejecting remote URLs, extra keys, MIME mismatches, or oversize rasters.
    pub fn from_json(value: &Value) -> std::result::Result<Self, EmbeddedImageError> {
        parse_embedded_image_json(value, true, MAX_IMAGE_BYTES, MAX_PIXELS)
    }

    /// Validate a structured [`EmbeddedImageInput`] in strict mode.
    pub fn from_input(input: &EmbeddedImageInput) -> std::result::Result<Self, EmbeddedImageError> {
        match input {
            EmbeddedImageInput::DataUri(uri) => {
                normalize_data_uri(uri, true, MAX_IMAGE_BYTES, MAX_PIXELS)
            }
            EmbeddedImageInput::Object {
                content_type,
                base64,
                rotation,
            } => {
                if let Some(deg) = rotation {
                    validate_rotation(*deg)?;
                }
                normalize_parts(
                    content_type,
                    base64,
                    *rotation,
                    true,
                    MAX_IMAGE_BYTES,
                    MAX_PIXELS,
                )
            }
        }
    }

    /// Decoded payload size in bytes computed from canonical base64 without re-decoding.
    pub fn decoded_size(&self) -> usize {
        decoded_base64_size(&self.base64)
    }

    /// Whether an explicit `rotation` field was supplied by the caller.
    pub fn has_explicit_rotation(&self) -> bool {
        self.rotation.is_some()
    }

    /// Effective explicit rotation degrees (`0` when unset).
    pub fn rotation_degrees(&self) -> u16 {
        self.rotation.unwrap_or(0)
    }

    /// Decode the validated image to an `RgbImage`:
    /// 1. Applies EXIF orientation (`ImageDecoder::orientation()` -> `DynamicImage::apply_orientation`).
    /// 2. Composites any alpha channel onto a solid white `(255, 255, 255)` RGB background.
    /// 3. Applies explicit clockwise rotation if provided, otherwise runs [`orient_document`].
    pub fn to_rgb_image(&self) -> Result<RgbImage> {
        let bytes = decode_flexible_base64(&self.base64)?;
        let rgb = decode_bytes_to_rgb(&bytes, MAX_PIXELS)?;
        if let Some(deg) = self.rotation {
            Ok(if deg == 0 { rgb } else { rotate_cw(&rgb, deg) })
        } else {
            Ok(orient_document(&rgb))
        }
    }
}

/// Pre-rendered PDF page raster supplied from the host/JS PDF renderer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPdfPageData {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub document_index: usize,
    pub page_index: usize,
}

impl RenderedPdfPageData {
    /// Validate dimensions & buffer size, build an `RgbImage`, and run [`orient_document`].
    pub fn to_rgb_image(&self) -> Result<RgbImage> {
        if self.width == 0 || self.height == 0 {
            return Err(EmbeddedImageError::new(
                "invalid_image",
                "PDF page has invalid dimensions",
            )
            .into());
        }
        let pixels = u64::from(self.width).saturating_mul(u64::from(self.height));
        if pixels > MAX_PIXELS as u64 {
            return Err(EmbeddedImageError::new(
                "invalid_image",
                "PDF page dimensions are too large",
            )
            .into());
        }
        let expected_len = (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(3);
        if self.rgb.len() != expected_len {
            return Err(EmbeddedImageError::new(
                "invalid_image",
                format!(
                    "PDF page RGB buffer length {} does not match {}x{}x3 ({})",
                    self.rgb.len(),
                    self.width,
                    self.height,
                    expected_len
                ),
            )
            .into());
        }
        let image =
            RgbImage::from_raw(self.width, self.height, self.rgb.clone()).ok_or_else(|| {
                EmbeddedImageError::new("invalid_image", "PDF page RGB buffer is invalid")
            })?;
        Ok(orient_document(&image))
    }

    /// Raw RGB byte size of this rendered page.
    pub fn byte_len(&self) -> usize {
        self.rgb.len()
    }
}

/// Encode an `RgbImage` as PNG and wrap it as a [`ValidatedImage`] (used by PO supplements).
pub fn encode_rgb_png_to_validated(image: &RgbImage) -> Result<ValidatedImage> {
    let mut buffer = Cursor::new(Vec::new());
    image
        .write_to(&mut buffer, ImageFormat::Png)
        .map_err(|err| {
            EmbeddedImageError::new("invalid_image", format!("failed to encode PNG: {err}"))
        })?;
    let bytes = buffer.into_inner();
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(EmbeddedImageError::new(
            "image_too_large",
            "image exceeds the decoded byte limit",
        )
        .into());
    }
    Ok(ValidatedImage {
        content_type: "image/png".to_string(),
        base64: BASE64_STANDARD.encode(&bytes),
        rotation: None,
    })
}

/// Compute decoded byte length of base64 (canonical, wrapped, unpadded, or URL-safe)
/// without decoding again.
pub fn decoded_base64_size(encoded: &str) -> usize {
    let mut non_ws_unpadded = 0usize;
    for &b in encoded.as_bytes() {
        if !matches!(b, b' ' | b'\t' | b'\n' | b'\r' | b'=') {
            non_ws_unpadded += 1;
        }
    }
    (non_ws_unpadded * 3) / 4
}

/// Decode standard, whitespace-wrapped (` `, `\t`, `\r`, `\n`), unpadded, or
/// URL-safe (`-`, `_`) base64 while still rejecting invalid characters,
/// `len % 4 == 1`, interior/excess padding, or non-canonical trailing bits.
fn decode_flexible_base64(encoded: &str) -> std::result::Result<Vec<u8>, EmbeddedImageError> {
    let malformed = || EmbeddedImageError::new("invalid_image", "image base64 is malformed");
    let mut normalized = Vec::with_capacity(encoded.len() + 2);
    for &b in encoded.as_bytes() {
        match b {
            b' ' | b'\t' | b'\n' | b'\r' => continue,
            b'-' => normalized.push(b'+'),
            b'_' => normalized.push(b'/'),
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'+' | b'/' | b'=' => normalized.push(b),
            _ => return Err(malformed()),
        }
    }
    if normalized.is_empty() {
        return Ok(Vec::new());
    }
    if let Some(eq_pos) = normalized.iter().position(|&b| b == b'=') {
        let pad_len = normalized.len() - eq_pos;
        if pad_len > 2
            || normalized.len() % 4 != 0
            || !normalized[eq_pos..].iter().all(|&b| b == b'=')
        {
            return Err(malformed());
        }
    } else {
        match normalized.len() % 4 {
            0 => {}
            1 => return Err(malformed()),
            rem => {
                normalized.resize(normalized.len() + (4 - rem), b'=');
            }
        }
    }
    BASE64_STANDARD.decode(&normalized).map_err(|_| malformed())
}

/// Enforce the maximum count of images (or images + PDF pages combined).
pub fn validate_image_count(
    count: usize,
    max_images: usize,
) -> std::result::Result<(), EmbeddedImageError> {
    if count > max_images {
        return Err(EmbeddedImageError::new(
            "too_many_images",
            "request exceeds the image count limit",
        ));
    }
    Ok(())
}

/// Enforce the aggregate decoded byte limit across all request media items.
pub fn validate_total_image_bytes(
    sizes: impl IntoIterator<Item = usize>,
    max_bytes: usize,
) -> std::result::Result<(), EmbeddedImageError> {
    let mut total: usize = 0;
    for size in sizes {
        total = total.saturating_add(size);
        if total > max_bytes {
            return Err(EmbeddedImageError::new(
                "images_too_large",
                "images exceed the aggregate decoded byte limit",
            ));
        }
    }
    Ok(())
}

/// Validate combined count (`images.len() + pdf_pages.len() <= MAX_IMAGES`) and
/// aggregate image byte limit (`<= MAX_IMAGES_BYTES`).
pub fn validate_request_media(
    images: &[ValidatedImage],
    pdf_pages: &[RenderedPdfPageData],
    document_encoded_bytes: &[usize],
) -> std::result::Result<(), EmbeddedImageError> {
    let total_count = images.len().saturating_add(pdf_pages.len());
    validate_image_count(total_count, MAX_IMAGES)?;
    validate_total_image_bytes(
        images
            .iter()
            .map(ValidatedImage::decoded_size)
            .chain(document_encoded_bytes.iter().copied()),
        MAX_IMAGES_BYTES,
    )
}

/// Normalize and validate an embedded image JSON value matching upstream
/// `embedded.normalize_embedded_image` + `api.ClairImage.parse_image`.
pub fn parse_embedded_image_json(
    value: &Value,
    strict: bool,
    max_image_bytes: usize,
    max_pixels: usize,
) -> std::result::Result<ValidatedImage, EmbeddedImageError> {
    match value {
        Value::String(s) => normalize_data_uri(s, strict, max_image_bytes, max_pixels),
        Value::Object(map) => {
            let mut rotation: Option<u16> = None;
            let mut content_type: Option<&str> = None;
            let mut encoded: Option<&str> = None;

            for (key, val) in map {
                match key.as_str() {
                    "content_type" => {
                        content_type = Some(val.as_str().ok_or_else(|| {
                            EmbeddedImageError::new("invalid_image", "image fields must be strings")
                        })?);
                    }
                    "base64" => {
                        encoded = Some(val.as_str().ok_or_else(|| {
                            EmbeddedImageError::new("invalid_image", "image fields must be strings")
                        })?);
                    }
                    "rotation" => {
                        let raw_rot = val.as_i64().filter(|_| !val.is_f64()).ok_or_else(|| {
                            EmbeddedImageError::new(
                                "invalid_image",
                                "rotation must be 0, 90, 180 or 270 clockwise degrees",
                            )
                        })?;
                        let deg = u16::try_from(raw_rot).map_err(|_| {
                            EmbeddedImageError::new(
                                "invalid_image",
                                "rotation must be 0, 90, 180 or 270 clockwise degrees",
                            )
                        })?;
                        validate_rotation(deg)?;
                        rotation = Some(deg);
                    }
                    _ => {
                        return Err(EmbeddedImageError::new(
                            "invalid_image",
                            "image object has invalid fields",
                        ));
                    }
                }
            }

            let (Some(ct), Some(b64)) = (content_type, encoded) else {
                return Err(EmbeddedImageError::new(
                    "invalid_image",
                    "image object has invalid fields",
                ));
            };

            normalize_parts(ct, b64, rotation, strict, max_image_bytes, max_pixels)
        }
        _ => Err(EmbeddedImageError::new(
            "invalid_image",
            "image object has invalid fields",
        )),
    }
}

fn validate_rotation(deg: u16) -> std::result::Result<(), EmbeddedImageError> {
    match deg {
        0 | 90 | 180 | 270 => Ok(()),
        _ => Err(EmbeddedImageError::new(
            "invalid_image",
            "rotation must be 0, 90, 180 or 270 clockwise degrees",
        )),
    }
}

fn normalize_data_uri(
    uri: &str,
    strict: bool,
    max_image_bytes: usize,
    max_pixels: usize,
) -> std::result::Result<ValidatedImage, EmbeddedImageError> {
    let regex = if strict {
        STRICT_DATA_URI.as_ref()
    } else {
        PERMISSIVE_DATA_URL.as_ref()
    };
    let label = if strict { "URI" } else { "URL" };
    let Some(caps) = regex.and_then(|re| re.captures(uri)) else {
        return Err(EmbeddedImageError::new(
            "invalid_image",
            format!("image must be an embedded base64 data {label}"),
        ));
    };
    let (Some(ct), Some(b64)) = (caps.get(1), caps.get(2)) else {
        return Err(EmbeddedImageError::new(
            "invalid_image",
            format!("image must be an embedded base64 data {label}"),
        ));
    };
    normalize_parts(
        ct.as_str(),
        b64.as_str(),
        None,
        strict,
        max_image_bytes,
        max_pixels,
    )
}

fn canonical_content_type(
    raw: &str,
) -> std::result::Result<(&'static str, ImageFormat), EmbeddedImageError> {
    let lower = raw.to_ascii_lowercase();
    let normalized = if lower == "image/jpg" {
        "image/jpeg"
    } else {
        lower.as_str()
    };
    match normalized {
        "image/png" => Ok(("image/png", ImageFormat::Png)),
        "image/jpeg" => Ok(("image/jpeg", ImageFormat::Jpeg)),
        "image/webp" => Ok(("image/webp", ImageFormat::WebP)),
        _ => Err(EmbeddedImageError::new(
            "invalid_image",
            "image content type is not supported",
        )),
    }
}

fn normalize_parts(
    content_type: &str,
    encoded: &str,
    rotation: Option<u16>,
    strict: bool,
    max_image_bytes: usize,
    max_pixels: usize,
) -> std::result::Result<ValidatedImage, EmbeddedImageError> {
    let (canonical_type, expected_format) = canonical_content_type(content_type)?;

    let decoded = decode_flexible_base64(encoded)?;
    if decoded.is_empty() {
        return Err(EmbeddedImageError::new(
            "invalid_image",
            "image payload is empty",
        ));
    }
    if decoded.len() > max_image_bytes {
        return Err(EmbeddedImageError::new(
            "image_too_large",
            "image exceeds the decoded byte limit",
        ));
    }

    if strict {
        verify_raster(&decoded, expected_format, max_pixels)?;
    } else if !has_image_signature(canonical_type, &decoded) {
        return Err(EmbeddedImageError::new(
            "invalid_image",
            "image payload does not match its content type",
        ));
    }

    Ok(ValidatedImage {
        content_type: canonical_type.to_string(),
        base64: BASE64_STANDARD.encode(&decoded),
        rotation,
    })
}

/// Magic-byte check matching `embedded._has_image_signature`.
pub fn has_image_signature(content_type: &str, data: &[u8]) -> bool {
    match content_type {
        "image/png" => data.starts_with(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg" => data.starts_with(b"\xff\xd8\xff"),
        "image/webp" => data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP",
        _ => false,
    }
}

fn verify_raster(
    data: &[u8],
    expected_format: ImageFormat,
    max_pixels: usize,
) -> std::result::Result<(), EmbeddedImageError> {
    let reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|_| EmbeddedImageError::new("invalid_image", "image payload is invalid"))?;
    let detected = reader
        .format()
        .ok_or_else(|| EmbeddedImageError::new("invalid_image", "image payload is invalid"))?;
    if detected != expected_format {
        return Err(EmbeddedImageError::new(
            "invalid_image",
            "image payload does not match its content type",
        ));
    }
    let mut reader = reader;
    let mut limits = Limits::default();
    limits.max_alloc = Some((max_pixels as u64).saturating_mul(16).max(64 * 1024 * 1024));
    reader.limits(limits);

    let decoder = reader
        .into_decoder()
        .map_err(|_| EmbeddedImageError::new("invalid_image", "image payload is invalid"))?;
    let (w, h) = decoder.dimensions();
    if w == 0 || h == 0 || u64::from(w).saturating_mul(u64::from(h)) > max_pixels as u64 {
        return Err(EmbeddedImageError::new(
            "invalid_image",
            "image dimensions are too large",
        ));
    }
    DynamicImage::from_decoder(decoder)
        .map_err(|_| EmbeddedImageError::new("invalid_image", "image payload is invalid"))?;
    Ok(())
}

/// Decode verified image bytes to RGB with EXIF transposition and white-background alpha compositing.
fn decode_bytes_to_rgb(
    data: &[u8],
    max_pixels: usize,
) -> std::result::Result<RgbImage, EmbeddedImageError> {
    let mut reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|_| EmbeddedImageError::new("invalid_image", "image payload is invalid"))?;
    let mut limits = Limits::default();
    limits.max_alloc = Some((max_pixels as u64).saturating_mul(16).max(64 * 1024 * 1024));
    reader.limits(limits);

    let mut decoder = reader
        .into_decoder()
        .map_err(|_| EmbeddedImageError::new("invalid_image", "image payload is invalid"))?;
    let (w, h) = decoder.dimensions();
    if w == 0 || h == 0 || u64::from(w).saturating_mul(u64::from(h)) > max_pixels as u64 {
        return Err(EmbeddedImageError::new(
            "invalid_image",
            "image dimensions are too large",
        ));
    }
    let exif_orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut dyn_img = DynamicImage::from_decoder(decoder)
        .map_err(|_| EmbeddedImageError::new("invalid_image", "image payload is invalid"))?;
    dyn_img.apply_orientation(exif_orientation);

    if dyn_img.color().has_alpha() {
        Ok(composite_rgba_over_white(&dyn_img.to_rgba8()))
    } else {
        Ok(dyn_img.to_rgb8())
    }
}

/// Composite an RGBA image over solid white `(255, 255, 255)` matching `api.py:82-88`.
pub fn composite_rgba_over_white(rgba: &image::RgbaImage) -> RgbImage {
    let (w, h) = rgba.dimensions();
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
    let (pixels, _) = rgba.as_raw().as_chunks::<4>();
    for px in pixels {
        let a = u32::from(px[3]);
        let inv_a = 255 - a;
        let r = ((u32::from(px[0]) * a + 255 * inv_a + 127) / 255) as u8;
        let g = ((u32::from(px[1]) * a + 255 * inv_a + 127) / 255) as u8;
        let b = ((u32::from(px[2]) * a + 255 * inv_a + 127) / 255) as u8;
        out.push(r);
        out.push(g);
        out.push(b);
    }
    RgbImage::from_raw(w, h, out).unwrap_or_else(|| RgbImage::new(w, h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, Rgba, RgbaImage};
    use serde_json::json;

    fn encode_rgb(img: &RgbImage, format: ImageFormat) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        img.write_to(&mut cursor, format).unwrap();
        cursor.into_inner()
    }

    fn encode_rgba_png(img: &RgbaImage) -> Vec<u8> {
        let mut cursor = Cursor::new(Vec::new());
        img.write_to(&mut cursor, ImageFormat::Png).unwrap();
        cursor.into_inner()
    }

    #[test]
    fn decodes_png_jpeg_webp_data_uri_and_object() {
        let img = RgbImage::from_pixel(16, 12, Rgb([40, 80, 120]));

        // PNG data URI
        let png_b64 = BASE64_STANDARD.encode(encode_rgb(&img, ImageFormat::Png));
        let uri = format!("data:image/png;base64,{png_b64}");
        let parsed = ValidatedImage::from_json(&json!(uri)).unwrap();
        assert_eq!(parsed.content_type, "image/png");
        assert_eq!(parsed.rotation, None);
        let decoded = parsed.to_rgb_image().unwrap();
        assert_eq!(decoded.dimensions(), (16, 12));
        assert_eq!(decoded.get_pixel(0, 0).0, [40, 80, 120]);

        // JPEG object with image/jpg alias and explicit rotation
        let jpg_b64 = BASE64_STANDARD.encode(encode_rgb(&img, ImageFormat::Jpeg));
        let parsed_jpg = ValidatedImage::from_json(&json!({
            "content_type": "image/jpg",
            "base64": jpg_b64,
            "rotation": 90
        }))
        .unwrap();
        assert_eq!(parsed_jpg.content_type, "image/jpeg");
        assert_eq!(parsed_jpg.rotation, Some(90));
        let rot_decoded = parsed_jpg.to_rgb_image().unwrap();
        assert_eq!(rot_decoded.dimensions(), (12, 16));

        // WebP data URI
        let webp_b64 = BASE64_STANDARD.encode(encode_rgb(&img, ImageFormat::WebP));
        let webp_uri = format!("data:image/webp;base64,{webp_b64}");
        let parsed_webp = ValidatedImage::from_json(&json!(webp_uri)).unwrap();
        assert_eq!(parsed_webp.content_type, "image/webp");
        let decoded_webp = parsed_webp.to_rgb_image().unwrap();
        assert_eq!(decoded_webp.dimensions(), (16, 12));
    }

    #[test]
    fn composites_rgba_onto_white_background() {
        let mut rgba = RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0]));
        rgba.put_pixel(1, 1, Rgba([0, 0, 0, 255]));
        rgba.put_pixel(2, 2, Rgba([0, 0, 0, 128]));

        let b64 = BASE64_STANDARD.encode(encode_rgba_png(&rgba));
        let validated = ValidatedImage::from_json(&json!({
            "content_type": "image/png",
            "base64": b64,
            "rotation": 0
        }))
        .unwrap();
        let rgb = validated.to_rgb_image().unwrap();
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(rgb.get_pixel(1, 1).0, [0, 0, 0]);
        // Half-alpha black over white should be ~127
        let mid = rgb.get_pixel(2, 2).0[0];
        assert!((126..=128).contains(&mid), "mid={mid}");
    }

    #[test]
    fn rejects_mime_mismatch_remote_url_bad_rotation_and_limits() {
        let img = RgbImage::from_pixel(8, 8, Rgb([10, 20, 30]));
        let png_b64 = BASE64_STANDARD.encode(encode_rgb(&img, ImageFormat::Png));

        // MIME mismatch: PNG bytes declared as JPEG
        let err = ValidatedImage::from_json(&json!({
            "content_type": "image/jpeg",
            "base64": png_b64
        }))
        .unwrap_err();
        assert_eq!(err.code, "invalid_image");

        // Remote URL rejected
        let err = ValidatedImage::from_json(&json!("https://example.com/invoice.png")).unwrap_err();
        assert_eq!(err.code, "invalid_image");

        // Extra field rejected
        let err = ValidatedImage::from_json(&json!({
            "content_type": "image/png",
            "base64": png_b64,
            "url": "https://example.com/a.png"
        }))
        .unwrap_err();
        assert_eq!(err.code, "invalid_image");

        // Invalid rotation rejected
        let err = ValidatedImage::from_json(&json!({
            "content_type": "image/png",
            "base64": png_b64,
            "rotation": 45
        }))
        .unwrap_err();
        assert_eq!(err.code, "invalid_image");

        // > 4 MiB single image rejected
        let big = vec![0u8; MAX_IMAGE_BYTES + 1];
        let big_b64 = BASE64_STANDARD.encode(&big);
        let err = ValidatedImage::from_json(&json!({
            "content_type": "image/png",
            "base64": big_b64
        }))
        .unwrap_err();
        assert_eq!(err.code, "image_too_large");

        // > 8 MiB combined rejected
        let err =
            validate_total_image_bytes([MAX_IMAGE_BYTES, MAX_IMAGE_BYTES, 1], MAX_IMAGES_BYTES)
                .unwrap_err();
        assert_eq!(err.code, "images_too_large");

        // > 4 images rejected
        let err = validate_image_count(MAX_IMAGES + 1, MAX_IMAGES).unwrap_err();
        assert_eq!(err.code, "too_many_images");
    }

    #[test]
    fn rendered_pdf_page_validation_and_orientation() {
        let page = RenderedPdfPageData {
            rgb: vec![255u8; 20 * 10 * 3],
            width: 20,
            height: 10,
            document_index: 0,
            page_index: 0,
        };
        let rgb = page.to_rgb_image().unwrap();
        assert_eq!(rgb.dimensions(), (20, 10));

        let bad = RenderedPdfPageData {
            rgb: vec![255u8; 10],
            width: 20,
            height: 10,
            document_index: 0,
            page_index: 0,
        };
        assert!(bad.to_rgb_image().is_err());
    }

    #[test]
    fn accepts_wrapped_unpadded_and_urlsafe_base64_and_rejects_malformed() {
        // Choose dimensions/pixels whose PNG encoding includes '+', '/', and '=' padding.
        let mut img = RgbImage::new(11, 9);
        for (x, y, px) in img.enumerate_pixels_mut() {
            *px = Rgb([
                ((x * 23 + y * 11) % 256) as u8,
                ((x * 37 + y * 53) % 256) as u8,
                ((x * 71 + y * 17) % 256) as u8,
            ]);
        }
        let png_bytes = encode_rgb(&img, ImageFormat::Png);
        let canonical_b64 = BASE64_STANDARD.encode(&png_bytes);

        // 1. Whitespace-wrapped (CRLF, LF, spaces, tabs)
        let wrapped_b64: String = canonical_b64
            .as_bytes()
            .chunks(16)
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect::<Vec<_>>()
            .join("\r\n \t");
        assert_eq!(decoded_base64_size(&wrapped_b64), png_bytes.len());

        // 2. Unpadded
        let unpadded_b64 = canonical_b64.trim_end_matches('=').to_string();
        assert_eq!(decoded_base64_size(&unpadded_b64), png_bytes.len());

        // 3. URL-safe (- and _) + unpadded + wrapped
        let urlsafe_b64 = wrapped_b64
            .trim_end_matches('=')
            .replace('+', "-")
            .replace('/', "_");
        assert_eq!(decoded_base64_size(&urlsafe_b64), png_bytes.len());

        for variant in [&wrapped_b64, &unpadded_b64, &urlsafe_b64] {
            // Data URI form
            let uri = format!("data:image/png;base64,{variant}");
            let from_uri = ValidatedImage::from_json(&json!(uri)).unwrap();
            assert_eq!(from_uri.content_type, "image/png");
            assert_eq!(from_uri.base64, canonical_b64);
            assert_eq!(from_uri.to_rgb_image().unwrap().dimensions(), (11, 9));

            // Object form
            let from_obj = ValidatedImage::from_json(&json!({
                "content_type": "image/png",
                "base64": variant,
                "rotation": 0
            }))
            .unwrap();
            assert_eq!(from_obj.base64, canonical_b64);
            assert_eq!(from_obj.to_rgb_image().unwrap().dimensions(), (11, 9));
        }

        // Reject genuinely malformed base64 variants
        for bad_b64 in ["!!!", "A", "A=BC", "AA===", "AA="] {
            let uri_err =
                ValidatedImage::from_json(&json!(format!("data:image/png;base64,{bad_b64}")))
                    .unwrap_err();
            assert_eq!(uri_err.code, "invalid_image", "bad uri {bad_b64}");

            let obj_err = ValidatedImage::from_json(&json!({
                "content_type": "image/png",
                "base64": bad_b64
            }))
            .unwrap_err();
            assert_eq!(obj_err.code, "invalid_image", "bad obj {bad_b64}");
        }
    }
}
