// baked/index.ts — the 3 committed offline-baked J-lens frames, as one typed map.
//
// Each JSON file (baked by scripts/jlens/bake.mts) carries BOTH a logit and a
// jacobian `LensReadoutRun` for its curated prompt, so the widget's DEFAULT view
// renders with NO model and NO lens pack — it revives the baked frame and hands
// it to the existing jlens children. Vite bundles these JSON imports and the SSG
// prerender resolves them synchronously.
//
// `resolveJsonModule` is already enabled (tsconfig.json), so these imports carry
// the inferred literal JSON type. We narrow each to `BakedFile` through
// `unknown`: the on-disk shape is fixed by the bake script and asserted at bake
// time, and this keeps the widget on the single hand-authored `BakedFile` type
// instead of three giant inferred literal types.

import arithmeticParensJson from './arithmetic-parens.json';
import frenchSeasonJson from './french-season.json';
import spanishOppositeJson from './spanish-opposite.json';
import type { BakedFile } from '../revive';

/** Baked frame per preset slug (keys === `JACOBIAN_PRESETS[].slug`). */
export const BAKED: Record<string, BakedFile> = {
  'french-season': frenchSeasonJson as unknown as BakedFile,
  'spanish-opposite': spanishOppositeJson as unknown as BakedFile,
  'arithmetic-parens': arithmeticParensJson as unknown as BakedFile,
};
