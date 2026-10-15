# Changelog

## Unreleased

### Breaking changes

- `@mlx-node/core` / `@mlx-node/vlm`: removed the unused raw-RGB PP-OCR per-crop methods `TextDetModel.detectCrop()` and `TextRecModel.recognizeCrop()`. Encode the crop (PNG/JPEG) and call `TextDetModel.detect()` or `TextRecModel.recognize()` / `recognizeBatch()` instead.
