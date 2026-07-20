import * as React from 'react';

/**
 * usePrefersReducedMotion — the course's single source of truth for the
 * reduced-motion preference.
 *
 * This is the `matchMedia`-in-useState idiom that ~30 widgets in this course
 * each declared privately (verbatim, and imported by nobody). New animated
 * widgets should import THIS one. The existing private copies are intentionally
 * left alone for now — swapping 30 files is a separate, mechanical change.
 *
 * Two properties matter and must not be "simplified" away:
 *
 *  - The SSR guard is TWO-part: `typeof window === 'undefined'` AND
 *    `typeof window.matchMedia !== 'function'`. The prerender step runs this in
 *    Node (no window) and some very old engines expose window without
 *    matchMedia; either alone is not enough.
 *
 *  - The default is `false`, i.e. the SERVER always renders the animated-start
 *    branch. Deterministic server output is the whole point: the prerendered
 *    HTML must not depend on a preference the server cannot know. Widgets then
 *    retarget after mount (see useStepPlayer).
 */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });

  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);

  return reduced;
}
