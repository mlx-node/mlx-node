import { describe, it, expect } from 'vite-plus/test';
import { DocOrientationModel, DocUnwarpModel } from '@mlx-node/core';
import type { StructureV3Config, AnalyzeOptions } from '@mlx-node/vlm';

describe('DocOrientationModel', () => {
  it('should have a load factory method', () => {
    expect(typeof DocOrientationModel.load).toBe('function');
  });

  it('should throw on missing model path', () => {
    expect(() => DocOrientationModel.load('/nonexistent/path')).toThrow();
  });

  it('should throw on missing weights file', () => {
    // Create a temp dir without model.safetensors
    const os = require('node:os');
    const fs = require('node:fs');
    const path = require('node:path');
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'doc-ori-test-'));
    try {
      expect(() => DocOrientationModel.load(tmpDir)).toThrow('Weights file not found');
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});

describe('DocUnwarpModel', () => {
  it('should have a load factory method', () => {
    expect(typeof DocUnwarpModel.load).toBe('function');
  });

  it('should throw on missing model path', () => {
    expect(() => DocUnwarpModel.load('/nonexistent/path')).toThrow();
  });

  it('should throw on missing weights file', () => {
    const os = require('node:os');
    const fs = require('node:fs');
    const path = require('node:path');
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'doc-unwarp-test-'));
    try {
      expect(() => DocUnwarpModel.load(tmpDir)).toThrow('Weights file not found');
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });
});

describe('StructureV3Pipeline preprocessing integration', () => {
  it('should export DocOrientationModel and DocUnwarpModel from @mlx-node/vlm', async () => {
    const vlm = await import('@mlx-node/vlm');
    expect(vlm.DocOrientationModel).toBeDefined();
    expect(vlm.DocUnwarpModel).toBeDefined();
  });

  it('StructureV3Config should accept preprocessing model paths', () => {
    // Verify the config type accepts optional preprocessing paths
    // (This is a compile-time check - if it doesn't accept, TS would error)
    const config: StructureV3Config = {
      layoutModelPath: '/fake/layout',
      textDetModelPath: '/fake/det',
      textRecModelPath: '/fake/rec',
      dictPath: '/fake/dict',
      docOrientationModelPath: '/fake/ori',
      docUnwarpModelPath: '/fake/unwarp',
    };

    expect(config.docOrientationModelPath).toBe('/fake/ori');
    expect(config.docUnwarpModelPath).toBe('/fake/unwarp');
  });

  it('AnalyzeOptions should accept preprocessing flags', () => {
    const options: AnalyzeOptions = {
      useDocOrientationClassify: false,
      useDocUnwarping: false,
    };

    expect(options.useDocOrientationClassify).toBe(false);
    expect(options.useDocUnwarping).toBe(false);
  });
});
