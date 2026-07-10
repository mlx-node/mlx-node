// starters/index.ts — the model-free /jspace starter frames, as one typed map.
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
// bake writes (scripts/jlens/bake.mts) and `reviveRun` rehydrates.
//
// These three files are seeded from the committed per-preset lesson bakes
// (demo/learn/widgets/jlens/baked/<slug>.json — real GPU bakes with the exact
// envelope). Gate C1 re-bakes them for /jspace via `bake.mts` (which now also
// writes this directory); the shapes are identical either way.

import arithmeticParensJson from './arithmetic-parens.json';
import frenchSeasonJson from './french-season.json';
import spanishOppositeJson from './spanish-opposite.json';
import type { BakedFile } from '../../jlens-core/revive';

/** Starter frame per preset slug (keys === `JACOBIAN_PRESETS[].slug`). Display
 *  order = insertion order; french-season is the headline. */
export const STARTERS: Record<string, BakedFile> = {
  'french-season': frenchSeasonJson as unknown as BakedFile,
  'spanish-opposite': spanishOppositeJson as unknown as BakedFile,
  'arithmetic-parens': arithmeticParensJson as unknown as BakedFile,
};

/** Slugs in display order (drives the starter chip row). */
export const STARTER_SLUGS: string[] = ['french-season', 'spanish-opposite', 'arithmetic-parens'];
