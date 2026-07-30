import { cn } from '@/lib/utils';
import type * as React from 'react';

/**
 * Placeholder for content that has not arrived.
 *
 * Sized by the caller, and the caller's job is to make that size EXACTLY the box
 * the real content will occupy — otherwise the swap reflows everything below it.
 * For a line of text that means `h-[1lh]` inside the element carrying the font
 * classes: `1lh` resolves against that element's own line-height, so the bar is
 * one line box by construction and stays correct if the type scale is ever
 * changed. A hard-coded `h-4` beside `text-sm` is 4px short, and four of those
 * stacked in a tile grid is a visible jolt.
 *
 * The pulse — and the brief hold before the bar becomes visible at all — live in
 * `index.css` against `data-slot="skeleton"` rather than in a class here, so
 * they apply to every placeholder without a call site having to remember them.
 * See the comment there for why the delay exists.
 */
function Skeleton({ className, ...props }: React.ComponentProps<'div'>) {
  return <div data-slot="skeleton" className={cn('bg-accent rounded-md', className)} {...props} />;
}

export { Skeleton };
