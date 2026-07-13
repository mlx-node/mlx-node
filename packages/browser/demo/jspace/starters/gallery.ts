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
    slug: 'grammar-error',
    prompt: "The plural of 'child' is childs.",
    concepts: ['incorrect'],
    band: { onset: 17, peak: 18 },
    defaultMode: 'jacobian',
    grade: 'strong',
  },
  {
    slug: 'giza-continent',
    prompt: 'Fact: The continent where the pyramids of Giza are located is ',
    concepts: ['Africa'],
    band: { onset: 16, peak: 17 },
    defaultMode: 'jacobian',
    grade: 'strong',
  },
  {
    slug: 'int-cast-error',
    prompt: ">>> int('hello')\n",
    concepts: ['error'],
    band: { onset: 18, peak: 18 },
    defaultMode: 'jacobian',
    grade: 'weak',
  },
  {
    slug: 'eiffel-capital',
    prompt: 'Fact: The capital of the country where the Eiffel Tower stands is ',
    concepts: ['Paris'],
    band: { onset: 18, peak: 18 },
    defaultMode: 'jacobian',
    grade: 'weak',
  },
];

export const STARTER_SLUGS: string[] = GALLERY.map((g) => g.slug);
