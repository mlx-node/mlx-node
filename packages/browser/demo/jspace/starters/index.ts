// starters/index.ts — the model-free /jspace GALLERY frames, as one typed map.
//
// A COLD visitor to /jspace (no model, no permalink) sees a real argmax grid
// built entirely from these committed baked frames — no Worker, no weight
// download. Each file carries BOTH a logit and a jacobian `LensReadoutRun` for a
// curated preset, so the LOGIT | JACOBIAN toggle works offline too.
//
// WHY imported, NOT demo/public: vite.config.ts sets `publicDir: false` on a
// production build (vite.config.ts:59), so anything under demo/public/ is DROPPED
// from the shipped bundle. These JSON modules live in the SOURCE tree so Vite
// bundles them and the starter grid ships to mlx.void.app. Typed arrays are
// serialized as plain `number[]` (Constraint 15) — the same envelope the offline
// bake writes and `reviveRun` rehydrates.
//
// These 7 frames are the vetted gallery, baked from `demo/jspace/starters/gallery.ts`
// by `scripts/jlens/bake-gallery.mts` (the honesty artifact `vetting.json` lists
// which candidates shipped and which were dropped — incl. eiffel-capital, cut for
// being a SPOKEN answer). Slugs + display order come from `gallery.ts`
// (STARTER_SLUGS === GALLERY.map(g => g.slug)); keep this map's keys in that order.

import type { BakedFile } from '../../jlens-core/revive';
import arithFewshot from './arith-fewshot.json';
import arithInnerSum from './arith-inner-sum.json';
import arithPrecedence from './arith-precedence.json';
import frenchSeason from './french-season.json';
import { STARTER_SLUGS } from './gallery';
import gizaContinent from './giza-continent.json';
import grammarError from './grammar-error.json';
import intCastError from './int-cast-error.json';

/** Starter frame per gallery slug (keys === `STARTER_SLUGS`, display order). */
export const STARTERS: Record<string, BakedFile> = {
  'french-season': frenchSeason as unknown as BakedFile,
  'arith-inner-sum': arithInnerSum as unknown as BakedFile,
  'arith-precedence': arithPrecedence as unknown as BakedFile,
  'arith-fewshot': arithFewshot as unknown as BakedFile,
  'grammar-error': grammarError as unknown as BakedFile,
  'giza-continent': gizaContinent as unknown as BakedFile,
  'int-cast-error': intCastError as unknown as BakedFile,
};

/**
 * Resolve a possibly-untrusted slug (e.g. from a shared `#s=` permalink) to a REAL
 * gallery tile. `STARTERS` is a plain object, so a bare `STARTERS[slug]` lookup
 * resolves INHERITED keys: `s=toString` / `s=__proto__` / `s=constructor` return
 * `Object.prototype` members instead of `undefined`, so a `?? default` fallback
 * never fires and the caller gets a non-frame that crashes `reviveRun` at render.
 * `Object.hasOwn` accepts ONLY the map's own keys; anything else → the default tile.
 */
export function resolveStarterSlug(slug: string | null | undefined): string {
  return slug != null && Object.hasOwn(STARTERS, slug) ? slug : STARTER_SLUGS[0]!;
}

export { STARTER_SLUGS };
