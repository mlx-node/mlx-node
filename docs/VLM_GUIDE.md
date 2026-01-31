# Vision Language Model (VLM) Guide

This guide covers the PaddleOCR-VL implementation in MLX-Node, a vision-language model optimized for OCR and document understanding tasks.

## Quick Start

```typescript
import { VLModel } from '@mlx-node/vlm';

// Load the model (tokenizer is loaded automatically)
const model = await VLModel.load('./models/paddleocr-vl-1.5');

// Simple OCR - just pass the image path
const text = model.ocr('./receipt.jpg');
console.log(text);

// Chat with image - also just pass the path
const result = model.chat([{ role: 'user', content: 'What text is in this image?' }], {
  imagePaths: ['./document.jpg'],
});
console.log(result.text);
```

## Installation

The VLM package is part of the MLX-Node workspace:

```bash
yarn add @mlx-node/vlm
```

## API Reference

### VLModel

The main model class combining vision encoder and language model.

```typescript
import { VLModel } from '@mlx-node/vlm';

// Load a model (recommended)
const model = await VLModel.load('./model-path');

// Load just the config
const config = await VLModel.loadConfig('./model-path');
console.log(config.visionConfig.hiddenSize);
```

#### Methods

**`chat(messages, options?)`**

High-level conversational interface for image understanding. Just pass image paths directly.

```typescript
const result = model.chat(
  [
    { role: 'system', content: 'You are a helpful OCR assistant.' },
    { role: 'user', content: 'Extract all text from this image.' },
  ],
  {
    imagePaths: ['./document.jpg'],
    maxNewTokens: 2048,
    temperature: 0.7,
  },
);

console.log(result.text); // Generated text
console.log(result.numTokens); // Number of tokens generated
```

**`ocr(imagePath, prompt?)`**

Convenience method for simple text extraction. Just pass an image path - no preprocessing needed.

```typescript
// Basic OCR
const text = model.ocr('./receipt.jpg');

// With custom prompt
const text = model.ocr('./document.png', 'Extract all dates from this image.');
```

**`generate(inputIds, pixelValues?, gridThw?, options?)`**

Low-level generation for custom workflows.

```typescript
const result = await model.generate(inputIds, pixelValues, gridThw, {
  maxNewTokens: 512,
  temperature: 0.8,
  topP: 0.95,
});
```

### Image Processing

Process images before feeding to the model.

```typescript
import { ImageProcessor, preprocessImage, preprocessImageBytes, smartResize } from '@mlx-node/vlm';

// Simple path-based processing
const processed = preprocessImage('./image.jpg');

// Buffer-based processing
const buffer = fs.readFileSync('./image.png');
const processed = preprocessImageBytes(buffer);

// Custom processor with config
const processor = new ImageProcessor({
  minPixels: 147384, // Minimum total pixels
  maxPixels: 2822400, // Maximum total pixels
  patchSize: 14, // Vision transformer patch size
  mergeSize: 2, // Spatial merge factor
});
const processed = processor.processFile('./image.jpg');

// Get target dimensions
const [height, width] = processor.getTargetSize(1024, 768);
```

### Configuration

```typescript
import { VLModel } from '@mlx-node/vlm';

// Load config from model directory
const config = await VLModel.loadConfig('./model-path');

// Vision config
console.log(config.visionConfig.hiddenSize); // 1152
console.log(config.visionConfig.numHiddenLayers); // 27
console.log(config.visionConfig.patchSize); // 14

// Text config (ERNIE)
console.log(config.textConfig.hiddenSize); // 1024
console.log(config.textConfig.numHiddenLayers); // 18
console.log(config.textConfig.headDim); // 128

// Special tokens
console.log(config.imageTokenId); // 100295
console.log(config.eosTokenId); // 2
```

## Architecture

### Model Components

PaddleOCR-VL consists of three main components:

1. **Vision Encoder** (`PaddleOCRVisionModel`)
   - Processes images into visual features
   - Uses 14x14 patch embeddings
   - 27 transformer layers with 1152 hidden dimensions
   - Vision RoPE for position encoding

2. **Spatial Projector** (`SpatialProjector`)
   - Merges visual features (2x2 spatial merge)
   - Projects to language model dimension

3. **Language Model** (`ERNIELanguageModel`)
   - ERNIE-based decoder with Multimodal RoPE (mRoPE)
   - 18 layers, 1024 hidden dimensions
   - 3D positional encoding: [temporal, height, width]

### Multimodal RoPE (Internal Architecture)

The language model uses specialized Multimodal Rotary Position Embeddings (mRoPE) to encode 3D positions for vision tokens. Unlike standard RoPE which uses 1D sequential positions, mRoPE encodes temporal, height, and width dimensions separately.

**How it works:**

The position encoding is configured via `mrope_section` in the model config:

- `[16, 24, 24]` means 16 dims for temporal, 24 for height, 24 for width
- These are doubled and summed: `[16, 24, 24] * 2 = [32, 48, 48]` -> cumulative `[32, 80, 128]`
- This matches `head_dim` (128 in the default config)

**Key parameters:**

- `head_dim`: 128 (dimension per attention head)
- `max_position_embeddings`: 131072 (maximum sequence length)
- `rope_theta`: 500000 (frequency base)

This is handled automatically by `VLModel` - users interact with the high-level API and do not need to configure mRoPE directly.

## Performance Tips

### 1. Use KV Cache

KV caching is automatically enabled during generation, providing 10-100x speedup:

```typescript
// KV caches are managed internally
model.chat(messages, { imagePaths: ['./image.jpg'] });

// Reset for new conversation
model.resetKvCaches();
```

### 2. Batch Processing

Process multiple images sequentially:

```typescript
// Each call manages its own KV cache state
const files = ['./doc1.jpg', './doc2.jpg', './doc3.jpg'];
const results = files.map((f) => model.ocr(f));
```

### 3. Image Size Optimization

The `smartResize` function optimizes image dimensions:

```typescript
import { smartResize } from '@mlx-node/vlm';

// Resize to fit within pixel bounds while maintaining aspect ratio
const [h, w] = smartResize(
  originalHeight,
  originalWidth,
  28, // factor (patchSize * mergeSize)
  147384, // minPixels
  2822400, // maxPixels
);
```

### 4. Memory Management

- Vision features are computed once and cached during generation
- KV cache grows with sequence length
- Call `model.resetKvCaches()` between unrelated conversations

## Example: Document OCR Pipeline

```typescript
import { VLModel } from '@mlx-node/vlm';
import * as fs from 'fs';
import * as path from 'path';

async function processDocuments(inputDir: string, outputDir: string) {
  // Load model (tokenizer is loaded automatically)
  const model = await VLModel.load('./models/paddleocr-vl');

  // Process each document
  const files = fs.readdirSync(inputDir).filter((f) => f.endsWith('.jpg') || f.endsWith('.png'));

  for (const file of files) {
    const imagePath = path.join(inputDir, file);

    // Extract text - just pass the path
    const text = model.ocr(imagePath);

    // Save result
    const outputPath = path.join(outputDir, `${path.basename(file, path.extname(file))}.txt`);
    fs.writeFileSync(outputPath, text);

    console.log(`Processed: ${file}`);

    // Reset cache between documents
    model.resetKvCaches();
  }
}

// Run
processDocuments('./documents', './output');
```

## Example: Interactive Image Chat

```typescript
import { VLModel } from '@mlx-node/vlm';
import * as readline from 'readline';

async function interactiveChat(imagePath: string) {
  // Load model (tokenizer is loaded automatically)
  const model = await VLModel.load('./models/paddleocr-vl');

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const messages: Array<{ role: string; content: string }> = [];

  console.log(`Loaded image: ${imagePath}`);
  console.log('Ask questions about the image (type "quit" to exit):\n');

  const askQuestion = () => {
    rl.question('You: ', (input) => {
      if (input.toLowerCase() === 'quit') {
        rl.close();
        return;
      }

      messages.push({ role: 'user', content: input });

      const result = model.chat(messages, { imagePaths: [imagePath] });

      messages.push({ role: 'assistant', content: result.text });

      console.log(`\nAssistant: ${result.text}\n`);

      askQuestion();
    });
  };

  askQuestion();
}

// Run
interactiveChat('./document.jpg');
```

## Troubleshooting

### Common Issues

**"Model not initialized"**

- Ensure weights are fully loaded before calling generate/chat
- Use `model.isInitialized` to check status

**"Aspect ratio too extreme"**

- Images with aspect ratio > 200 are rejected
- Crop or resize the image before processing

**Memory issues with large images**

- Use smaller `maxPixels` in ImageProcessor config
- Process images sequentially rather than in parallel

**Slow generation**

- Ensure KV cache is being used (don't create new model instances per call)
- Check that Metal acceleration is active

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Project overview
- [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md) - Development timeline
- [AUTOGRAD_INTEGRATION.md](AUTOGRAD_INTEGRATION.md) - Autograd system

## Model Weights

PaddleOCR-VL weights can be downloaded from Hugging Face:

- [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/paddleocr-vl-1.5)

Convert to SafeTensors format if needed:

```bash
python -m transformers.convert_checkpoint_to_safetensors ./model-dir
```
