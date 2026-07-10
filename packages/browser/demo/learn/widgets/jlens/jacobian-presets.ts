// jacobian-presets.ts — curated J-lens presets + band constants shared by BOTH
// the offline bake script (scripts/jlens/bake.mts) and the live widget (T4.4).
//
// PLAIN data/TS: NO node- or browser-only imports (no `node:*`, no React, no DOM)
// so both a Node/oxnode script and the browser bundle can import it unchanged.
//
// Sources of truth (controller-verified):
//   - Featured set: scripts/jlens/PHASE3-VERDICT.md:122-129 (French season = headline;
//     Spanish opposite = feature-with-framing; arithmetic = optional number-slot demo;
//     do NOT hard-feature multihop/association/poetry).
//   - Prompts: scripts/jlens/data/lens-eval-multilingual.json (french-season-summer,
//     spanish-opposite-big) and lens-eval-order-ops.json (mult-parens-add). Copied
//     byte-exact (ASCII apostrophe/quotes).
//   - Band numbers: band-report.json `lens_legibility_gap` — legibility onset at
//     boundary 6, curve peak at boundary 17.
//
// Prompts are RAW (fed to `encodeTokens` verbatim, NO chat template — design
// decision D9). Concepts are pinned the way the widget does (LogitLensLive.tsx:
// 269-274): each concept is tokenized as ` ${concept}` and its FIRST token id is
// pinned. Every preset has <=5 concepts, well under the 8-pin LENS_MAX_PINNED cap,
// so a single readout call per mode suffices.

/** One curated prompt plus the concepts whose full-vocab rank we track. */
export type JacobianPreset = {
  /** URL/file-safe id; also the baked filename `<slug>.json`. */
  slug: string;
  /** Raw prompt fed to `encodeTokens` verbatim — NO chat template (D9). */
  prompt: string;
  /** Concepts to pin. Each is tokenized as ` ${concept}`; its first token id
   *  becomes a pinned id (mirrors LogitLensLive.tsx:269-274). Pin order === this
   *  array's order (bake asserts a 1:1 concept↔pin mapping). */
  concepts: string[];
  /** One-line, honest reader-facing framing (only observed phenomena). */
  blurb: string;
};

/** The curated featured set. Order = display order; french-season is the headline. */
export const JACOBIAN_PRESETS: JacobianPreset[] = [
  {
    slug: 'french-season',
    prompt: "La saison après l'été est l'",
    concepts: ['season', 'summer', 'autumn', 'automne'],
    blurb:
      "Headline: mid-stack the J-lens surfaces the abstract concept ('season' near rank 1, " +
      "'summer' near rank 2, around boundaries 16-17) where the plain logit lens is still ranked " +
      "999+. A clean mid-band concept cluster. The target token is 'automne' (autumn).",
  },
  {
    slug: 'spanish-opposite',
    prompt: 'Lo opuesto de "grande" es "',
    concepts: ['opposite', 'big', 'small', 'pequeño'],
    blurb:
      "Framed honestly: the J-lens holds the English concept ('opposite' / 'small') while the logit " +
      "lens moves toward the Spanish target 'pequeño'. This is a LATE-boundary effect, not a clean " +
      'mid-band story.',
  },
  {
    slug: 'arithmetic-parens',
    prompt: '2 * (3 + 4) = ',
    concepts: ['7', 'multiplication'],
    blurb:
      'A number-slot demo (NOT a concept claim): the inner sum 3 + 4 = 7 shows up as a digit cluster ' +
      'mid-stack where the logit lens is garbage. The final answer is 14.',
  },
];

/** Legibility band constants from band-report.json (`lens_legibility_gap`).
 *  `boundaries` is the full residual-boundary axis (1..24; 0 is unfitted/errors). */
export const BAND = {
  /** Boundary where intermediate concepts start becoming legible under the J-lens. */
  onsetBoundary: 6,
  /** Boundary of peak legibility gap (J-lens vs logit lens). */
  legibilityPeak: 17,
  /** Full boundary axis, ascending (1..24). */
  boundaries: Array.from({ length: 24 }, (_, i) => i + 1),
} as const;

/** DEFAULT boundary subset baked/read for Jacobian mode: band onset (6) through
 *  the output boundary (24), evenly spaced. The early band (1..5) is intentionally
 *  excluded from the default view — T4.4 puts it behind a "show all layers" toggle. */
export const JACOBIAN_LAYERS: number[] = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24];
