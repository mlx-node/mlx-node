import { cn } from '@/lib/utils';
import * as React from 'react';

/**
 * Shared typography wrapper for chapter bodies. Each chapter renders JSX
 * (headings, paragraphs, code blocks) inside <Prose>, and inherits a
 * consistent reading layout — comfortable measure, generous line height,
 * accent on code and inline math, dark-mode-friendly contrast.
 *
 * Keep this restrained: chapters use plain HTML tags (h1/h2/p/ul/code/pre).
 * Anything fancier should be a named component the chapter imports.
 */
export function Prose({ className, children, ...props }: React.ComponentProps<'div'>) {
  return (
    <div
      data-slot="prose"
      className={cn(
        // base color + type scale. Width is intentionally NOT capped here: the
        // chapter body fills its layout column so the prose right-edge lines up
        // with the scaffolding cards (objective / pipeline / glossary), which
        // span the same column. The reading width is set once, upstream — the
        // panel-less reading wrapper in LessonLayout (`max-w-3xl`) and the grid
        // reading column for try-it chapters. A separate `max-w-prose` here was
        // ~150px narrower than that column, so the prose stranded left of the
        // cards' right edge. (Architecture's poster opts out via `lg:max-w-none`,
        // now a no-op, and stays full-width.)
        'text-[15px] leading-7 text-foreground/90',
        // headings
        '[&_h1]:text-3xl [&_h1]:font-semibold [&_h1]:tracking-tight [&_h1]:mt-0 [&_h1]:mb-4 [&_h1]:text-foreground',
        '[&_h2]:text-xl [&_h2]:font-semibold [&_h2]:tracking-tight [&_h2]:mt-8 [&_h2]:mb-3 [&_h2]:text-foreground',
        '[&_h3]:text-base [&_h3]:font-semibold [&_h3]:mt-6 [&_h3]:mb-2 [&_h3]:text-foreground',
        // paragraphs + lists
        '[&_p]:my-3 [&_p]:text-foreground/85',
        '[&_ul]:my-3 [&_ul]:pl-6 [&_ul]:list-disc [&_ul]:text-foreground/85',
        '[&_ol]:my-3 [&_ol]:pl-6 [&_ol]:list-decimal [&_ol]:text-foreground/85',
        '[&_li]:my-1',
        // inline code / math
        '[&_code]:rounded [&_code]:bg-muted [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:text-[0.92em] [&_code]:font-mono',
        // block code
        '[&_pre]:my-4 [&_pre]:rounded-lg [&_pre]:bg-muted [&_pre]:p-4 [&_pre]:overflow-x-auto',
        '[&_pre_code]:bg-transparent [&_pre_code]:p-0',
        // misc
        '[&_em]:italic [&_strong]:font-semibold',
        '[&_a]:underline [&_a]:underline-offset-2 [&_a]:text-primary hover:[&_a]:text-primary/80',
        '[&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:text-muted-foreground',
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
