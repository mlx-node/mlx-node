// localized.ts — locale overlays for the chapter registry. React-free (same
// rule as chapters.ts / seo-metadata.ts) so the prerender entry can import it.
//
// The English CHAPTERS array in learn/chapters.ts stays the single source of
// truth for ids, numbers, ordering, and availability. A locale overlay only
// re-words the display strings (title/blurb, plus section titles/blurbs);
// anything missing from the overlay falls back to English, so a half-finished
// translation never breaks the index or the sidebar.

import { CHAPTERS, findChapter, findSection, type ChapterMeta, type SectionMeta } from '../chapters';
import { DEFAULT_LOCALE, type Locale } from '../../lib/i18n';

/** Display-string overlay for one chapter. All fields optional → English fallback. */
export type ChapterMetaOverlay = {
  title?: string;
  blurb?: string;
  /** Keyed by section id, e.g. 'flash-attention'. */
  sections?: Record<string, { title?: string; blurb?: string }>;
};

/**
 * Chinese display strings, keyed by chapter id. Populated by the chrome
 * translation pass; chapter BODY translations live in bodies.zh.ts instead.
 */
export const ZH_CHAPTER_META: Record<string, ChapterMetaOverlay> = {
  overview: {
    title: '什么是大语言模型？',
    blurb: '全局视角：LLM 就是一个函数——输入 token，为每个候选的下一个 token 打分——循环运行。',
  },
  tokenization: {
    title: '分词（Tokenization）',
    blurb: '什么是 token？看 Qwen 的 BPE 分词器把字符串切成子词。',
  },
  embeddings: {
    title: '词嵌入（Embeddings）',
    blurb: 'token 变成向量。用 PCA 散点图展示模型真实的嵌入矩阵。',
  },
  attention: {
    title: '自注意力',
    blurb: 'softmax(QKᵀ / √d) V——让每个 token 都能看到其他所有 token 的机制。',
    sections: {
      // 显存带宽墙：该小节讲的是 GPU 的 HBM 带宽瓶颈（"显存墙"在中文语境里
      // 常被理解为显存容量不够，会带偏读者，所以点明"带宽"）。
      'memory-wall': {
        title: '显存带宽墙',
        blurb: 'softmax(QKᵀ) 到底有多大——以及为什么注意力的瓶颈是带宽，而不是算力。',
      },
      'flash-attention': {
        title: 'Flash Attention',
        blurb: '同样的 softmax(QKᵀ/√d)·V，却从不显式构造 N×N 分数矩阵。',
      },
    },
  },
  'multi-head-gqa': {
    title: '多头注意力与 GQA',
    blurb: '为什么需要多个头，以及 Qwen3.5 如何用分组查询注意力（GQA）让多个头共享 KV。',
  },
  rope: {
    title: '位置编码（RoPE）',
    blurb: '模型如何知道 token 的顺序——可视化为每对维度上的一次旋转。',
  },
  rmsnorm: {
    title: 'RMSNorm',
    blurb: '为什么要归一化？看真实层在归一化前后的激活分布。',
  },
  mlp: {
    title: 'MLP 模块',
    blurb: '门控 MLP 与残差连接——模型对每个 token 的前馈计算步骤。',
  },
  'full-block': {
    title: '完整的 Transformer 块',
    blurb: '注意力 + 归一化 + MLP + 残差。可旋转的 3D 层堆叠总览。',
  },
  'lm-head': {
    title: 'LM head 与权重共享',
    blurb: 'last_hidden 如何变成 logits——一次矩阵乘法，而且用的就是嵌入矩阵本身。',
  },
  sampling: {
    title: '采样',
    blurb: 'logits → softmax → token。实时 top-k 柱状图，可调 temperature 和 top-p。',
  },
  'kv-cache': {
    title: 'KV 缓存与混合注意力',
    blurb: '推理为什么快，以及 Qwen3.5 如何交错使用线性注意力和全量注意力。',
  },
  training: {
    title: '训练与 teacher forcing',
    blurb: '权重是怎么来的：下一个 token 的交叉熵、并行计算所有位置、用真实答案作输入。',
  },
  scaling: {
    title: '扩展与正则化',
    blurb: '学习率预热 + 余弦衰减、梯度裁剪、权重衰减——让训练收敛的工程细节。',
  },
  'post-training': {
    title: '从基础模型到助手',
    blurb: '预训练、指令微调与对话模板——自动补全如何变成有用的助手。',
  },
  architecture: {
    title: '整个模型，端到端',
    blurb: '在一张交互式全景图上看清每个组件——以及成品模型能力的真实边界。',
  },
};

const OVERLAYS: Record<Locale, Record<string, ChapterMetaOverlay>> = {
  en: {},
  zh: ZH_CHAPTER_META,
};

function applyOverlay(chapter: ChapterMeta, overlay: ChapterMetaOverlay | undefined): ChapterMeta {
  if (!overlay) return chapter;
  return {
    ...chapter,
    title: overlay.title ?? chapter.title,
    blurb: overlay.blurb ?? chapter.blurb,
    sections: chapter.sections?.map((section) => {
      const s = overlay.sections?.[section.id];
      return s ? { ...section, title: s.title ?? section.title, blurb: s.blurb ?? section.blurb } : section;
    }),
  };
}

/** CHAPTERS with the locale's display strings applied (English passes through untouched). */
export function localizedChapters(locale: Locale): ChapterMeta[] {
  if (locale === DEFAULT_LOCALE) return CHAPTERS;
  return CHAPTERS.map((chapter) => applyOverlay(chapter, OVERLAYS[locale][chapter.id]));
}

// NB: findChapter/findSection return `null` on a miss (and findSection wraps
// its hit in { chapter, section }) — normalized here so callers get a plain
// `ChapterMeta | null` / `SectionMeta | undefined`.
export function localizedChapter(locale: Locale, chapterId: string): ChapterMeta | null {
  const chapter = findChapter(chapterId);
  if (!chapter || locale === DEFAULT_LOCALE) return chapter;
  return applyOverlay(chapter, OVERLAYS[locale][chapterId]);
}

export function localizedSection(locale: Locale, chapterId: string, sectionId: string): SectionMeta | undefined {
  if (locale === DEFAULT_LOCALE) return findSection(chapterId, sectionId)?.section;
  return localizedChapter(locale, chapterId)?.sections?.find((s) => s.id === sectionId);
}
