import katex from 'katex';
import * as React from 'react';
import 'katex/dist/katex.min.css';

/**
 * Render a LaTeX math expression. Block-level (display) by default; pass
 * `inline` to render inside flowing prose without breaking the line.
 *
 * KaTeX is rendered into innerHTML — we feed it strings we control, so this
 * is safe and avoids the SSR-vs-client mismatch from dangerouslySetInnerHTML
 * that other React wrappers have. The horizontal-scroll container handles
 * wide formulas gracefully on narrow viewports.
 */
export function MathDisplay({ latex, inline = false }: { latex: string; inline?: boolean }) {
  const html = React.useMemo(
    () =>
      katex.renderToString(latex, {
        throwOnError: false,
        displayMode: !inline,
        output: 'html',
      }),
    [latex, inline],
  );

  if (inline) {
    return (
      <span
        className="katex-inline"
        // KaTeX-generated HTML is safe; the input comes from us, not user data.
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <div className="my-3 overflow-x-auto rounded-md border border-border bg-background/40 px-4 py-3">
      <div
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{ __html: html }}
      />
    </div>
  );
}
