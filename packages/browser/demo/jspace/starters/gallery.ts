// demo/jspace/starters/gallery.ts
// The /jspace gallery: the single source of truth for the vetted "unspoken word"
// examples. Drives the headless bake (scripts/jlens/bake-gallery.mts) AND the
// runtime launcher. Kept OUT of jlens-core/jacobian-presets.ts so the lesson's
// bake path stays untouched ("core instrument unchanged"). `concepts` are the
// pin words fed to derivePins (first-token id each); `band` is 1..24-scale.
export interface GalleryEntry {
  /** URL/file-safe id; the baked filename `<slug>.json`. */
  slug: string;
  /** Raw prompt fed to encodeTokens verbatim — NO chat template. */
  prompt: string;
  /** Concept words whose full-vocab rank we pin/thread (derivePins → first-token id). */
  concepts: string[];
  /** Legibility band on the layer axis, 1..24 scale (peak must be a displayed layer). */
  band: { onset: number; peak: number };
  /** Which lens the tile opens in. All 8 vetted tiles are jacobian. */
  defaultMode: 'jacobian' | 'logit';
  /** STRONG = clean top-3 mid-stack; WEAK = faint/floor but genuine. */
  grade: 'strong' | 'weak';
}

export const GALLERY: readonly GalleryEntry[] = [
  {
    slug: 'french-season',
    prompt: "La saison après l'été est l'",
    concepts: ['season', 'summer', 'autumn', 'automne'],
    band: { onset: 6, peak: 17 },
    defaultMode: 'jacobian',
    grade: 'strong',
  },
  {
    slug: 'arith-inner-sum',
    prompt: '2 * (1 + 2) = ',
    concepts: ['3'],
    band: { onset: 15, peak: 18 },
    defaultMode: 'jacobian',
    grade: 'strong',
  },
  {
    slug: 'arith-precedence',
    prompt: '3 + 4 * 2 = ',
    concepts: ['8'],
    band: { onset: 18, peak: 20 },
    defaultMode: 'jacobian',
    grade: 'strong',
  },
  {
    slug: 'arith-fewshot',
    prompt: '(1+2)*2=6. (2+3)*2= ',
    concepts: ['5'],
    band: { onset: 20, peak: 20 },
    defaultMode: 'jacobian',
    grade: 'strong',
  },
  {
    // Concept is the CAPITALIZED ' Incorrect' (the token that actually surfaces on
    // this base model — the lowercase ' incorrect' never reaches top-K; see the
    // bake's non-drift assertion + task-2 report). Peaks ℓ17 rank ~2.
    slug: 'grammar-error',
    prompt: "The plural of 'child' is childs.",
    concepts: ['Incorrect'],
    band: { onset: 17, peak: 18 },
    defaultMode: 'jacobian',
    grade: 'strong',
  },
  {
    // The answer 'Africa' surfaces as the STANDALONE vocab token (rank ~2 at
    // ℓ17–18), NOT derivePins' ' Africa' fragment — the bake's robust pin
    // resolution re-pins to it. Graded WEAK (not strong): only ~2 legible band
    // layers, the greedy output is a degenerate '1', and the Egypt bridge never
    // surfaces — the same faint-but-genuine class as eiffel-capital, so it is
    // graded WEAK for consistency (Task-2 fix #2). First of the WEAK group.
    slug: 'giza-continent',
    prompt: 'Fact: The continent where the pyramids of Giza are located is ',
    concepts: ['Africa'],
    band: { onset: 16, peak: 17 },
    defaultMode: 'jacobian',
    grade: 'weak',
  },
  {
    // Concept is the CAPITALIZED ' Error' (the surfacing token; lowercase ' error'
    // never reaches top-K on this base model). Peaks ℓ18 rank ~2.
    slug: 'int-cast-error',
    prompt: ">>> int('hello')\n",
    concepts: ['Error'],
    band: { onset: 18, peak: 18 },
    defaultMode: 'jacobian',
    grade: 'weak',
  },
  {
    // NO trailing space: with a trailing space the model surfaces the space-less
    // "Paris" continuation token (id 57590), which derivePins (always ` ${concept}`)
    // cannot pin. Without it, the space-form ' Paris' (id 11751) is the next token
    // and peaks ℓ18 rank 1 — the multi-hop answer as an unspoken word.
    slug: 'eiffel-capital',
    prompt: 'Fact: The capital of the country where the Eiffel Tower stands is',
    concepts: ['Paris'],
    band: { onset: 18, peak: 18 },
    defaultMode: 'jacobian',
    grade: 'weak',
  },
];

export const STARTER_SLUGS: string[] = GALLERY.map((g) => g.slug);
