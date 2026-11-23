/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { MODEL_CATALOG } from '../../agent/src/catalog.js';
import { ModelBadge } from '../ui/src/components/arc-x/model-badge/model-badge.js';
import { classifyModel, ModelLogo } from '../ui/src/components/arc-x/model-badge/model-logos.js';
import './render.js';

let mounted: { root: Root; container: HTMLElement } | undefined;

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  mounted = { root, container };
  return container;
}

afterEach(() => {
  if (mounted === undefined) return;
  act(() => mounted!.root.unmount());
  mounted.container.remove();
  mounted = undefined;
});

/** The colours of the official hy.tencent.com mark. */
const HY_COLOURS = new Set(['#B3DDF2', '#0053E0', '#2AB9FF', '#ECECEE']);

describe('classifyModel', () => {
  it('names the family of every model the catalog lists', () => {
    expect(Object.fromEntries(MODEL_CATALOG.map((entry) => [entry.label, classifyModel(entry.label)]))).toEqual({
      'Qwen3.8-27B': 'qwen',
      'Inkling-2-Mini': 'generic',
      'Qwen3.8-Flash-Next': 'qwen',
      'Hy4-flash': 'hy',
      'Gemma-4-12B': 'gemma',
      'Nemotron-3.5-Lightning-30B-A3B': 'generic',
      'Clairic Flash (9B)': 'cloudflare',
      'Clairic (27B)': 'cloudflare',
    });
  });

  it.each([
    ['clairic-flash-gguf', 'cloudflare'],
    ['clef-v1', 'cloudflare'],
    ['gemma-4-26b-a4b-it-gguf', 'gemma'],
    ['LFM2.5-1.2B', 'liquid'],
    ['ornith-7b', 'ornith'],
    ['Agents-A1-35B', 'internscience'],
    ['qwen-agentworld-35b-a3b-gguf', 'qwen'],
    ['Qwen3.8-Flash-Next-UD-Q4_K_XL', 'qwen'],
    ['Llama-3.2-3B-Instruct-4bit', 'generic'],
  ])('%s is %s', (name, family) => {
    expect(classifyModel(name)).toBe(family);
  });

  it.each([
    'Hy4-flash',
    'hy4-flash-gguf',
    'AngelSlim/Hy4-flash-GGUF',
    'tencent/Hy4-flash',
    'hy_v4_flash',
    'Hy4-flash-Q4_K_M',
    'tencent/Hy3',
    'tencent/Hunyuan-A13B-Instruct',
    'Hy-MT1.5-1.8B',
    'tencent/Hy-MT1.5-1.8B-2bit-GGUF',
  ])('gives the Tencent Hy mark to %s', (name) => {
    expect(classifyModel(name)).toBe('hy');
  });

  it.each([
    'Hybrid Mamba-2 + MoE, native MTP, 1M context',
    'Nemotron-3.5-Lightning-30B-A3B',
    'nemotron_h',
    'Inkling-2-Mini',
    'inkling-2-mini-gguf',
    'Qwen3.8-Flash-Next',
    'shy-7b',
    'org/shy-7b',
    'Physics-1B',
    'tencent/WeDLM-8B',
    'hy',
  ])('does not give the Tencent Hy mark to %s', (name) => {
    expect(classifyModel(name)).not.toBe('hy');
  });
});

describe('the Tencent Hy mark', () => {
  function expectHyMark(svg: SVGSVGElement | null): void {
    expect(svg).not.toBeNull();
    expect(svg!.getAttribute('viewBox')).toBe('0 0 16 16');
    expect(svg!.getAttribute('aria-hidden')).toBe('true');
    expect(svg!.hasAttribute('fill')).toBe(false);
    const paths = [...svg!.querySelectorAll('path')];
    expect(paths.length).toBe(7);
    for (const path of paths) expect(HY_COLOURS.has(path.getAttribute('fill') ?? '')).toBe(true);
    expect(svg!.querySelector('[id], mask, use')).toBeNull();
  }

  it('shows on the Hy4-flash catalog card tile, not the monogram', () => {
    const container = mount(createElement(ModelLogo, { model: 'Hy4-flash', className: 'size-[18px] shrink-0' }));
    expectHyMark(container.querySelector('svg'));
    expect(container.querySelector('svg')!.getAttribute('class')).toBe('size-[18px] shrink-0');
    expect(container.textContent).toBe('');
  });

  it('shows in the chat badge for a served Hy4 model id', () => {
    const container = mount(createElement(ModelBadge, { name: 'Hy4-flash-Q4_K_M', showLogo: true }));
    const badge = container.querySelector<HTMLElement>('[data-model]')!;
    expectHyMark(badge.querySelector('svg'));
    expect(badge.textContent).toBe('Hy4-flash-Q4_K_M');
  });

  it('leaves an unknown model on its monogram', () => {
    const container = mount(createElement(ModelLogo, { model: 'Inkling-2-Mini' }));
    expect(container.querySelector('svg')).toBeNull();
    expect(container.textContent).toBe('IN');
  });
});
