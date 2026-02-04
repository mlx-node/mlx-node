import { describe, expect, it } from 'vite-plus/test';
import { parseVlmOutput, documentToXlsx, saveToXlsx } from '@mlx-node/vlm';
import { existsSync, unlinkSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

describe('PaddleOCR-VL XLSX Export (Rust)', () => {
  it('should generate xlsx buffer from parsed document with a table', () => {
    const doc = parseVlmOutput('<fcel>Name<lcel>Age<ecel><nl><fcel>Alice<lcel>30<ecel>');
    const buffer = documentToXlsx(doc);

    expect(buffer).toBeInstanceOf(Buffer);
    expect(buffer.length).toBeGreaterThan(0);

    // XLSX files are ZIP archives starting with PK magic bytes
    expect(buffer[0]).toBe(0x50); // 'P'
    expect(buffer[1]).toBe(0x4b); // 'K'
  });

  it('should generate xlsx with multiple tables', () => {
    // Two tables separated by text
    const doc = parseVlmOutput('<fcel>A<ecel><nl><fcel>1<ecel><nl>Some text<nl><fcel>B<ecel><nl><fcel>2<ecel>');
    const buffer = documentToXlsx(doc);

    expect(buffer).toBeInstanceOf(Buffer);
    expect(buffer.length).toBeGreaterThan(0);
    // Valid ZIP/XLSX
    expect(buffer[0]).toBe(0x50);
    expect(buffer[1]).toBe(0x4b);
  });

  it('should handle empty cells in xlsx', () => {
    const doc = parseVlmOutput('<fcel>Name<ecel><ecel><nl><fcel>Alice<ecel><ecel>');
    const buffer = documentToXlsx(doc);

    expect(buffer).toBeInstanceOf(Buffer);
    expect(buffer.length).toBeGreaterThan(0);
    expect(buffer[0]).toBe(0x50);
    expect(buffer[1]).toBe(0x4b);
  });

  it('should save xlsx to file', () => {
    const filePath = join(tmpdir(), `paddleocr-test-${Date.now()}.xlsx`);
    try {
      saveToXlsx('<fcel>Name<lcel>Age<ecel><nl><fcel>Alice<lcel>30<ecel>', filePath);

      expect(existsSync(filePath)).toBe(true);
      const fileBuffer = readFileSync(filePath);
      expect(fileBuffer.length).toBeGreaterThan(0);
      expect(fileBuffer[0]).toBe(0x50);
      expect(fileBuffer[1]).toBe(0x4b);
    } finally {
      if (existsSync(filePath)) unlinkSync(filePath);
    }
  });
});
