// Single source of truth for the 10 lesson chapters. Each entry maps a stable
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
};

export const CHAPTERS: ChapterMeta[] = [
  {
    id: "tokenization",
    number: 1,
    title: "Tokenization",
    blurb:
      "What is a token? Watch Qwen's BPE tokenizer slice a string into sub-words.",
    available: true,
  },
  {
    id: "embeddings",
    number: 2,
    title: "Embeddings",
    blurb:
      "Tokens become vectors. A 3D PCA scatter of the model's actual embedding matrix.",
    available: false,
  },
  {
    id: "attention",
    number: 3,
    title: "Self-attention",
    blurb:
      "softmax(QKᵀ / √d) V. The mechanism that lets every token look at every other one.",
    available: true,
  },
  {
    id: "multi-head-gqa",
    number: 4,
    title: "Multi-head & GQA",
    blurb:
      "Why heads exist, and how Qwen3.5 shares KV across them with grouped-query attention.",
    available: true,
  },
  {
    id: "rope",
    number: 5,
    title: "Positional encoding (RoPE)",
    blurb:
      "How the model knows token order, visualized as a rotation per dimension pair.",
    available: true,
  },
  {
    id: "rmsnorm",
    number: 6,
    title: "RMSNorm",
    blurb:
      "Why normalize? Pre- and post-norm activation distributions for a real layer.",
    available: true,
  },
  {
    id: "mlp",
    number: 7,
    title: "MLP block",
    blurb:
      "Gated MLP and residual connections — the model's per-token feed-forward step.",
    available: true,
  },
  {
    id: "full-block",
    number: 8,
    title: "Full transformer block",
    blurb:
      "Attention + Norm + MLP + Residual. The 3D rotatable stack overview.",
    available: true,
  },
  {
    id: "sampling",
    number: 9,
    title: "Sampling",
    blurb:
      "Logits → softmax → token. Live top-k bar chart, with temperature and top-p sliders.",
    available: true,
  },
  {
    id: "kv-cache",
    number: 10,
    title: "KV cache & hybrid attention",
    blurb:
      "Why inference is fast, and how Qwen3.5 interleaves linear and full attention.",
    available: false,
  },
];

export function findChapter(id: string | null | undefined): ChapterMeta | null {
  if (!id) return null;
  return CHAPTERS.find((c) => c.id === id) ?? null;
}
