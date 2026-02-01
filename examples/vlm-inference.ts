/**
 * VLM Inference Example (PaddleOCR-VL-1.5)
 *
 * Usage:
 *   oxnode examples/vlm-inference.ts [image_path] [format]
 *
 * Examples:
 *   oxnode examples/vlm-inference.ts                              # Text-only chat
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png           # OCR with markdown output
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png plain     # OCR with plain text
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png html      # OCR with HTML output
 *   oxnode examples/vlm-inference.ts ./examples/ocr.png raw       # OCR with raw tokens
 */
import { ChatRole } from '@mlx-node/core';
import { VLModel, parsePaddleResponse, type OutputFormat } from '@mlx-node/vlm';

const modelPath = '.cache/models/PaddleOCR-VL-1.5-mlx';
const imagePath = process.argv[2];
const outputFormat = (process.argv[3] || 'Markdown') as OutputFormat;

// === Load Model (includes tokenizer) ===
console.log('Loading VLM...');
console.time('Model load');
const model = await VLModel.load(modelPath);
console.timeEnd('Model load');

console.log('\n✅ Model loaded successfully!');
console.log(`   Vision: ${model.config.visionConfig.numHiddenLayers} layers`);
console.log(`   Language: ${model.config.textConfig.numHiddenLayers} layers`);

if (imagePath) {
  // === Simple OCR ===
  console.log(`\n--- OCR: ${imagePath} ---`);
  console.log(`   Output format: ${outputFormat}`);
  console.time('OCR');
  const vlmResult = model.chat([{ role: ChatRole.User, content: 'Extract the text in this image' }], {
    imagePaths: [imagePath],
  });
  console.timeEnd('OCR');

  // Parse and format the output
  const formatted = parsePaddleResponse(vlmResult.text, { format: outputFormat });

  console.log(`\n📝 Extracted Text:\n${formatted}`);
} else {
  // === Text-only Chat ===
  console.log('\n--- Text-only Chat ---');
  console.log('(Pass an image path to test with vision)');

  console.time('Chat');
  const result = model.chat([{ role: ChatRole.User, content: 'Hello! What can you help me with?' }], {
    maxNewTokens: 100,
    temperature: 0.0, // Greedy decoding
    repetitionPenalty: 1.5,
  });
  console.timeEnd('Chat');

  console.log(`\n📝 Response:\n${result.text}`);
  console.log(`\nTokens generated: ${result.numTokens}`);
}
