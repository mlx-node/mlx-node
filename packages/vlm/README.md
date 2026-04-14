# @mlx-node/vlm

Vision-language models and document processing pipelines for Node.js on Apple Silicon. Extract text, tables, and structure from documents and images using PaddleOCR-VL, Qianfan-OCR (InternVL), and the PP-StructureV3 pipeline — all running locally on Metal GPU.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Node.js 18+

## Installation

```bash
npm install @mlx-node/vlm
```

## Multi-turn chat with VLMs

`QianfanOCRModel` conforms to the same `ChatSession<M>` surface as the language models in `@mlx-node/lm`, so a single session handle drives both single-shot and multi-turn VLM conversations:

```typescript
import { ChatSession } from '@mlx-node/lm';
import { QianfanOCRModel } from '@mlx-node/vlm';
import { readFileSync } from 'node:fs';

const model = await QianfanOCRModel.load('./models/Qianfan-VL');
const session = new ChatSession(model, { system: 'You read documents precisely.' });

// First turn with an image.
const r1 = await session.send('Extract the text from this receipt.', {
  images: [readFileSync('./receipt.jpg')],
});
console.log(r1.text);

// Text-only follow-up reuses the same KV cache against the same image.
const r2 = await session.send('What is the total price?');
console.log(r2.text);

// Swapping the image mid-session forcibly restarts the cache with the new image.
const r3 = await session.send('And this one?', {
  images: [readFileSync('./other-receipt.jpg')],
});
console.log(r3.text);
```

`loadSession()` from `@mlx-node/lm` cannot load Qianfan-OCR (that would introduce a circular package dependency), so construct the wrapper here with `QianfanOCRModel.load()` and pass it to `ChatSession` directly.

The one-shot `VLModel.chat()` API shown below for PaddleOCR-VL is a single-turn OCR entry point and is intentionally kept out of the session surface.

## Quick Start

### Document Structure Analysis

Use `StructureV3Pipeline` for layout detection + OCR without a VLM:

```typescript
import { StructureV3Pipeline } from '@mlx-node/vlm';

const pipeline = await StructureV3Pipeline.load({
  layoutModelPath: './models/PP-DocLayoutV3',
  textDetModelPath: './models/PP-OCRv5-det',
  textRecModelPath: './models/PP-OCRv5-rec',
  dictPath: './models/PP-OCRv5-rec/en_dict.txt',
});

const result = await pipeline.analyze('./document.png');
console.log(result.markdown);
```

### VLM-based OCR

Use `VLModel` for higher-quality OCR with document understanding:

```typescript
import { VLModel, parsePaddleResponse } from '@mlx-node/vlm';

const model = await VLModel.load('./models/PaddleOCR-VL-1.5-mlx');

const result = model.chat([{ role: 'user', content: 'Read the text in this image.' }], {
  images: [imageBuffer],
  maxNewTokens: 2048,
});

const formatted = parsePaddleResponse(result.text, { format: 'markdown' });
console.log(formatted);
```

## StructureV3Pipeline

The PP-StructureV3 pipeline combines multiple specialized models for fast, accurate document processing:

```
Input Image
    │
    ▼ (optional)
Orientation Correction → Dewarping
    │
    ▼
Layout Detection (25 categories)
    │
    ▼
Per-element Cropping
    │
    ▼
Text Detection → Text Recognition
    │
    ▼
Markdown Assembly
```

### Pipeline Options

```typescript
const result = await pipeline.analyze(imageBuffer, {
  layoutThreshold: 0.5, // Layout detection confidence threshold
  textDetThreshold: 0.3, // Text detection confidence threshold
  includeDetails: true, // Include per-element bounding boxes
  useDocOrientationClassify: true, // Auto-correct rotation (requires DocOrientationModel)
  useDocUnwarping: true, // Auto-correct perspective (requires DocUnwarpModel)
});

// result.elements — array of StructuredElement with label, bbox, text lines
// result.markdown — assembled markdown document
```

### OCR Without Layout

Run OCR directly on a pre-cropped image:

```typescript
const lines = await pipeline.ocrImage(croppedImageBuffer);
for (const line of lines) {
  console.log(`${line.text} (confidence: ${line.score.toFixed(2)})`);
}
```

## Individual Models

Each pipeline model can be used independently:

### DocLayoutModel — Document Layout Detection

PP-DocLayoutV3 with RT-DETR architecture, 25 layout categories (title, text, table, figure, formula, header, footer, etc.):

```typescript
import { DocLayoutModel } from '@mlx-node/vlm';

const layout = await DocLayoutModel.load('./models/PP-DocLayoutV3');
const elements = layout.detect(imageBuffer, 0.5); // threshold

for (const el of elements) {
  console.log(`${el.labelName} (${el.score.toFixed(2)}) at [${el.bbox}]`);
}
```

### TextDetModel — Text Line Detection

PP-OCRv5 DBNet with PPHGNetV2 backbone:

```typescript
import { TextDetModel } from '@mlx-node/vlm';

const detector = await TextDetModel.load('./models/PP-OCRv5-det');
const boxes = detector.detect(imageBuffer);
```

### TextRecModel — Text Recognition

PP-OCRv5 SVTR neck + CTC head:

```typescript
import { TextRecModel } from '@mlx-node/vlm';

const recognizer = await TextRecModel.load('./models/PP-OCRv5-rec');
const results = recognizer.recognizeBatch(croppedImages);
// [{ text: "Hello world", score: 0.98 }, ...]
```

### DocOrientationModel — Orientation Classification

Classifies document rotation (0/90/180/270 degrees):

```typescript
import { DocOrientationModel } from '@mlx-node/vlm';

const classifier = await DocOrientationModel.load('./models/PP-LCNet_x1_0_doc_ori-mlx');
const { angle, score } = classifier.classify(imageBuffer);
const { image } = classifier.classifyAndRotate(imageBuffer); // auto-correct
```

### DocUnwarpModel — Document Dewarping

UVDocNet-based perspective correction:

```typescript
import { DocUnwarpModel } from '@mlx-node/vlm';

const unwarper = await DocUnwarpModel.load('./models/UVDoc-mlx');
const { image } = unwarper.unwarp(imageBuffer);
```

## VLModel — Vision-Language Model

PaddleOCR-VL architecture (ERNIE language model + vision encoder):

```typescript
import { VLModel } from '@mlx-node/vlm';

const model = await VLModel.load('./models/PaddleOCR-VL-1.5-mlx');

// Single image chat
const result = model.chat(messages, {
  images: [imageBuffer],
  maxNewTokens: 2048,
  temperature: 0.1,
});

// Simple OCR
const text = model.ocr(imageBuffer);

// Batch OCR (multiple images)
const results = model.ocrBatch([image1, image2, image3]);

// Batch chat (different prompts per image)
const batchResults = model.batch([
  { messages: tableMessages, images: [tableImage] },
  { messages: formulaMessages, images: [formulaImage] },
]);
```

## Output Parsing

Parse VLM output into structured documents:

```typescript
import { parseVlmOutput, formatDocument, parsePaddleResponse, OutputFormat } from '@mlx-node/vlm';

// Two-step: parse then format
const doc = parseVlmOutput(result.text);
const markdown = formatDocument(doc, { format: OutputFormat.Markdown });

// One-step: parse and format
const html = parsePaddleResponse(result.text, { format: 'html' });
```

### XLSX Export

```typescript
import { saveToXlsx, documentToXlsx } from '@mlx-node/vlm';

// Direct save
saveToXlsx(result.text, './output.xlsx');

// Or get buffer
const doc = parseVlmOutput(result.text);
const buffer = documentToXlsx(doc);
```

## API Reference

### Pipeline

| Class/Function                         | Description                                                       |
| -------------------------------------- | ----------------------------------------------------------------- |
| `StructureV3Pipeline.load(config)`     | Load pipeline with layout, text detection, and recognition models |
| `pipeline.analyze(image, options?)`    | Full document analysis returning `StructuredDocument`             |
| `pipeline.ocrImage(image, threshold?)` | OCR on a single cropped image                                     |

### Models

| Class                 | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| `VLModel`             | PaddleOCR-VL vision-language model — `chat()`, `ocr()`, `ocrBatch()`, `batch()` |
| `DocLayoutModel`      | PP-DocLayoutV3 layout detection — `detect()`                                    |
| `TextDetModel`        | PP-OCRv5 text detection — `detect()`, `detectCrop()`                            |
| `TextRecModel`        | PP-OCRv5 text recognition — `recognize()`, `recognizeBatch()`                   |
| `DocOrientationModel` | Orientation classifier — `classify()`, `classifyAndRotate()`                    |
| `DocUnwarpModel`      | Document dewarping — `unwarp()`                                                 |

### Output

| Function                             | Description                              |
| ------------------------------------ | ---------------------------------------- |
| `parseVlmOutput(text)`               | Parse raw VLM text to `ParsedDocument`   |
| `formatDocument(doc, config?)`       | Format to markdown, HTML, plain, or JSON |
| `parsePaddleResponse(text, config?)` | Parse and format in one step             |
| `documentToXlsx(doc)`                | Convert parsed document to XLSX buffer   |
| `saveToXlsx(text, path)`             | Parse and save to XLSX file              |

### Configs

| Export                      | Description                                      |
| --------------------------- | ------------------------------------------------ |
| `PADDLEOCR_VL_CONFIGS`      | Pre-defined PaddleOCR-VL 1.5 config              |
| `createPaddleocrVlConfig()` | Default config factory                           |
| `OutputFormat`              | Enum: `Raw`, `Plain`, `Markdown`, `Html`, `Json` |

## License

[MIT](https://github.com/mlx-node/mlx-node/blob/main/LICENSE)
