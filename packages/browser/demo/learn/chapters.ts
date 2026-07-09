// Single source of truth for the 16 lesson chapters. Each entry maps a stable
// chapterId to display metadata. Only chapters with `available: true` have
// fully-authored content; the rest render a "coming soon" badge in the index
// and refuse to open.
export type ChapterMeta = {
  id: string;
  /** Position in the curriculum (1-indexed) — also displayed as a chip. */
  number: number;
  title: string;
  /** One-sentence teaser shown in the chapter index. */
  blurb: string;
  /** Whether the chapter body has been authored. */
  available: boolean;
  /**
   * Optional "go deeper" sub-chapters. Each is its own page at
   * /chapters/<id>/<section.id> — independently prerendered, SEO-indexed, and
   * shown nested under this chapter in the sidebar. Order here is the order
   * shown in the sidebar and the in-chapter "go deeper" nav.
   */
  sections?: SectionMeta[];
};

/**
 * A sub-chapter: a focused deep-dive that expands on one detail of its parent
 * chapter, served as its own sub-route. `id` is the URL slug (stable) and the
 * key into the SECTION_BODIES registry (`${chapterId}/${sectionId}`).
 */
export type SectionMeta = {
  id: string;
  title: string;
  /** One-sentence teaser shown in the sidebar tooltip + the in-chapter nav. */
  blurb: string;
};

export const CHAPTERS: ChapterMeta[] = [
  {
    id: 'overview',
    number: 1,
    title: 'What is an LLM?',
    blurb: 'The big picture: an LLM is one function — tokens in, a score for every next token — run in a loop.',
    available: true,
  },
  {
    id: 'tokenization',
    number: 2,
    title: 'Tokenization',
    blurb: "What is a token? Watch Qwen's BPE tokenizer slice a string into sub-words.",
    available: true,
  },
  {
    id: 'embeddings',
    number: 3,
    title: 'Embeddings',
    blurb: "Tokens become vectors. A PCA scatter of the model's actual embedding matrix.",
    available: true,
  },
  {
    id: 'attention',
    number: 4,
    title: 'Self-attention',
    blurb: 'softmax(QKᵀ / √d) V. The mechanism that lets every token look at every other one.',
    available: true,
    sections: [
      {
        id: 'memory-wall',
        title: 'The memory wall',
        blurb: 'How big softmax(QKᵀ) actually gets — and why attention is bandwidth-bound, not compute-bound.',
      },
      {
        id: 'flash-attention',
        title: 'Flash Attention',
        blurb: 'The same softmax(QKᵀ/√d)·V, computed without ever materializing the N×N score matrix.',
      },
      {
        id: 'roofline',
        title: 'The roofline',
        blurb: 'Put a number on "memory-bound": arithmetic intensity, the ops:byte line, and where batching moves you.',
      },
    ],
  },
  {
    id: 'multi-head-gqa',
    number: 5,
    title: 'Multi-head & GQA',
    blurb: 'Why heads exist, and how Qwen3.5 shares KV across them with grouped-query attention.',
    available: true,
  },
  {
    id: 'rope',
    number: 6,
    title: 'Positional encoding (RoPE)',
    blurb: 'How the model knows token order, visualized as a rotation per dimension pair.',
    available: true,
  },
  {
    id: 'rmsnorm',
    number: 7,
    title: 'RMSNorm',
    blurb: 'Why normalize? Pre- and post-norm activation distributions for a real layer.',
    available: true,
  },
  {
    id: 'mlp',
    number: 8,
    title: 'MLP block',
    blurb: "Gated MLP and residual connections — the model's per-token feed-forward step.",
    available: true,
  },
  {
    id: 'full-block',
    number: 9,
    title: 'Full transformer block',
    blurb: 'Attention + Norm + MLP + Residual. The 3D rotatable stack overview.',
    available: true,
  },
  {
    id: 'lm-head',
    number: 10,
    title: 'LM head & weight tying',
    blurb: 'How last_hidden becomes logits — one matmul, and the same matrix as the embedding.',
    available: true,
    sections: [
      {
        id: 'logit-lens',
        title: 'The logit lens',
        blurb: 'Point the LM head at a middle layer instead of the last one: read the vocabulary out of every depth and watch the answer emerge — the logit lens, live on your in-browser Qwen3.5.',
      },
    ],
  },
  {
    id: 'sampling',
    number: 11,
    title: 'Sampling',
    blurb: 'Logits → softmax → token. Live top-k bar chart, with temperature and top-p sliders.',
    available: true,
    sections: [
      {
        id: 'multi-token-prediction',
        title: 'Multi-token prediction',
        blurb: 'Get more than one token per forward pass — the MTP head Qwen3.5 ships, and the speculative decoding it powers.',
      },
      {
        id: 'reasoning',
        title: 'Reasoning & test-time compute',
        blurb: 'Chain-of-thought, self-consistency voting, and test-time compute scaling from o1 to DeepSeek-R1 — and why Qwen3.5-0.8B ships thinking off by default.',
      },
      {
        id: 'tool-calling',
        title: 'Tool calling & constrained decoding',
        blurb: "From ReAct to grammar-masked JSON decoding: how a validity mask makes tool calls unbreakable — and why this repo's own parser has no such guarantee.",
      },
    ],
  },
  {
    id: 'kv-cache',
    number: 12,
    title: 'KV cache & hybrid attention',
    blurb: 'Why inference is fast, and how Qwen3.5 interleaves linear and full attention.',
    available: true,
    sections: [
      {
        id: 'batching',
        title: 'Latency, throughput & batching',
        blurb: 'TTFT vs tokens/sec, why averages lie, and how batching turns a memory-bound decode into throughput — with the browser pinned at batch 1.',
      },
      {
        id: 'prefix-caching',
        title: 'Prefix caching',
        blurb: "Reuse one prompt's KV cache across requests — pay for the system prompt once, and put novel tokens last.",
      },
      {
        id: 'quantization',
        title: 'Quantization',
        blurb: 'Fewer bits per weight: how bf16 → fp8 / int8 / int4 shrinks the model and speeds the memory-bound decode — and what it costs in quality.',
      },
      {
        id: 'number-formats',
        title: 'Number formats & precision',
        blurb: 'Open the hood: how a weight is stored as bits, the precision and range of every format (fp32 → fp4, int8/int4, nf4, the MX/NVFP4 block formats), and how quantization snaps a real number onto a grid.',
      },
    ],
  },
  {
    id: 'training',
    number: 13,
    title: 'Training & teacher forcing',
    blurb: 'How the weights got there: next-token cross-entropy, parallel positions, ground-truth inputs.',
    available: true,
    sections: [
      {
        id: 'data-curation',
        title: 'Pretraining data curation',
        blurb: 'Deduplication, quality filtering, and Common Crawl/FineWeb sourcing — what "trillions of tokens" actually took, and what Qwen still keeps proprietary.',
      },
    ],
  },
  {
    id: 'scaling',
    number: 14,
    title: 'Scaling & regularization',
    blurb: 'LR warmup + cosine, gradient clipping, weight decay — the engineering that makes training converge.',
    available: true,
    sections: [
      {
        id: 'scaling-laws',
        title: 'Scaling laws: how many tokens for how many parameters?',
        blurb: 'Kaplan said grow the model; Chinchilla corrected it to grow tokens equally too — the 20:1 ratio, this model\'s own token math, and why Llama 3 ignores it.',
      },
    ],
  },
  {
    id: 'post-training',
    number: 15,
    title: 'From base model to assistant',
    blurb: 'Pretraining, instruction tuning, and chat templates — how autocomplete becomes a helpful assistant.',
    available: true,
    sections: [
      {
        id: 'distillation',
        title: 'Knowledge distillation',
        blurb: 'Train a smaller student on a bigger teacher\'s outputs — Hinton\'s 2015 soft targets, DeepSeek-R1\'s SFT-only recipe, and whether Qwen3.5 itself was distilled.',
      },
    ],
  },
  {
    id: 'architecture',
    number: 16,
    title: 'The whole model, end to end',
    blurb: 'Every piece on one interactive map — then the honest limits of what the finished model can do.',
    available: true,
    sections: [
      {
        id: 'local-edge',
        title: 'Local & edge inference',
        blurb: 'Why run a small model on your own device at all — the cloud-vs-edge trade in latency, privacy, cost, and what quantization makes possible.',
      },
      {
        id: 'evaluation',
        title: 'Evaluation',
        blurb: 'Perplexity is your own training loss, exponentiated. Then MMLU, HumanEval, GSM8K, LLM-as-judge, contamination — and an honest look at what Qwen3.5 does and doesn\'t publish.',
      },
    ],
  },
];

export function findChapter(id: string | null | undefined): ChapterMeta | null {
  if (!id) return null;
  return CHAPTERS.find((c) => c.id === id) ?? null;
}

/**
 * Resolve a chapter + one of its sub-chapters by ids. Returns null if the
 * chapter doesn't exist or has no section with that id — used by the section
 * route's beforeLoad (redirect on miss) and the prerenderer.
 */
export function findSection(
  chapterId: string | null | undefined,
  sectionId: string | null | undefined,
): { chapter: ChapterMeta; section: SectionMeta } | null {
  const chapter = findChapter(chapterId);
  if (!chapter || !sectionId) return null;
  const section = chapter.sections?.find((s) => s.id === sectionId);
  return section ? { chapter, section } : null;
}
