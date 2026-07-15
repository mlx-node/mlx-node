// model-blocked-copy.ts — the user-facing copy shown when a device can't run the
// in-browser model without crashing (see device-capability.ts for the WHY). Kept
// in one place so the chapter demo card (ModelConsentLayer) and the landing hero
// stay in sync. Glossary terms (WebGPU, iOS Safari, Chrome, Edge, Safari) stay
// English in both locales.

import type { ModelBlockReason } from './device-capability';
import type { Locale } from './i18n';

type Reason = Exclude<ModelBlockReason, null>;

const TITLE: Record<Locale, string> = {
  en: 'Best on a desktop',
  zh: '建议在桌面端体验',
};

const DETAIL: Record<Locale, Record<Reason, string>> = {
  en: {
    ios: 'This demo runs a real model live in your browser with WebGPU — it needs more memory than iOS Safari lets a tab use, so loading it here would crash the page. Open this page on a desktop or laptop (Chrome, Edge, or Safari) to run the live model. Every lesson on this page still works as you read.',
    'low-memory':
      'This demo runs a real model live in your browser and needs more memory than this device has. Open it on a machine with more RAM to run the live model. Every lesson on this page still works as you read.',
    'no-webgpu':
      'This demo needs WebGPU, which this browser doesn’t support. Try the latest Chrome, Edge, or Safari on a desktop to run the live model. Every lesson on this page still works as you read.',
  },
  zh: {
    ios: '这个演示会用 WebGPU 在你的浏览器里实时运行真实模型，所需内存超过了 iOS Safari 允许单个标签页使用的上限，在这里加载会导致页面崩溃。请在桌面或笔记本电脑（Chrome、Edge 或 Safari）上打开本页来运行实时模型。本页的每节课程照常可读。',
    'low-memory': '这个演示会在你的浏览器里实时运行真实模型，所需内存超过了本设备的容量。请在内存更大的机器上打开来运行实时模型。本页的每节课程照常可读。',
    'no-webgpu': '这个演示需要 WebGPU，而当前浏览器不支持。请在桌面端使用最新版的 Chrome、Edge 或 Safari 来运行实时模型。本页的每节课程照常可读。',
  },
};

// One-line form for compact affordances (the landing hero note under the CTA).
const SHORT: Record<Locale, string> = {
  en: 'This device can’t run the in-browser model — open on a desktop to try it live.',
  zh: '本设备无法运行浏览器内模型——请在桌面端打开以实时体验。',
};

export function modelBlockedTitle(locale: Locale): string {
  return TITLE[locale];
}

export function modelBlockedDetail(locale: Locale, reason: Reason): string {
  return DETAIL[locale][reason];
}

export function modelBlockedShort(locale: Locale): string {
  return SHORT[locale];
}
