/**
 * Stands in for every optional renderer markstream loads on demand (KaTeX, Mermaid, D2, infographics, stream-diffs):
 * loading it fails, every markstream loader catches that, and nothing is fetched or bundled. `vite.config.ts` points
 * those specifiers here; the chat renders such blocks as code.
 */
throw new Error('markstream optional renderers are not bundled');
