# Provenance of `components/arc/`

Every file in this folder is copied from Arc, the free (MIT) component library.

| | |
|---|---|
| Source registry | `https://uiarc.dev/r/<item>.json` |
| Repository | https://github.com/kuratlielia/arc-library (MIT, branch `main`, pushed 2026-11-01T14:16:10Z) |
| Fetched | 2026-11-10 |
| License | `LICENSE` in this folder (MIT, Copyright (c) 2026 Elia Kuratli) |

Files are copied by hand (no CLI) to the paths the registry names under `components/arc/`. Imports between items stay
relative, as shipped. Formatting and linting skip this folder so every file stays diffable against upstream.

## Vendored items (25)

| Item | Files |
|---|---|
| arc-foundation | `foundation.css` |
| arc-motion-tokens | `motion-tokens.ts`, `lib/motion-tokens.ts` |
| button | `button/button.tsx`, `button/button.module.css` |
| badge | `badge/badge.tsx`, `badge/badge.module.css` |
| segmented-control | `segmented-control/segmented-control.tsx`, `segmented-control/segmented-control.module.css` |
| tabs | `tabs/tabs.tsx`, `tabs/tabs.module.css` |
| tooltip | `tooltip/tooltip.tsx`, `tooltip/tooltip.module.css` |
| dialog | `dialog/dialog.tsx`, `dialog/dialog.module.css` |
| drawer | `drawer/drawer.tsx`, `drawer/drawer.module.css` |
| popover | `popover/popover.tsx`, `popover/popover.module.css` |
| select | `select/select.tsx`, `select/select.module.css` |
| input | `input/input.tsx`, `input/input.module.css` |
| search-field | `search-field/search-field.tsx`, `search-field/search-field.module.css` |
| progress | `progress/progress.tsx`, `progress/progress.module.css`. Note: `components/download-progress.module.css` paints the running download's shine on `[role="progressbar"] > :last-child`, so it depends on the track being the last child of the progressbar element; if an update moves the track, the shine disappears and nothing else breaks |
| alert | `alert/alert.tsx`, `alert/alert.module.css` |
| empty-state | `empty-state/empty-state.tsx`, `empty-state/empty-state.module.css` |
| scroll-area | `scroll-area/scroll-area.tsx`, `scroll-area/scroll-area.module.css` |
| toast-stack | `toast-stack/toast-stack.tsx`, `toast-stack/toast-stack.module.css` |
| copy-button | `copy-button/copy-button.tsx`, `copy-button/copy-button.module.css`, `lib/use-copy-feedback.ts` |
| command-palette (block) | `command-palette/command-palette.tsx`, `command-palette/command-palette.module.css` |
| breadcrumb | `breadcrumb/breadcrumb.tsx`, `breadcrumb/breadcrumb.module.css`, `lib/arc-provider.tsx` |
| animated-counter | `animated-counter/animated-counter.tsx`, `animated-counter/animated-counter.module.css` |
| text-shimmer | `text-shimmer/text-shimmer.tsx`, `text-shimmer/text-shimmer.module.css` |
| expandable-text | `expandable-text/expandable-text.tsx`, `expandable-text/expandable-text.module.css` |
| switch | `switch/switch.tsx`, `switch/switch.module.css` (dependency `@radix-ui/react-switch` 1.4.1) |

## Local changes

Every `.ts` / `.tsx` file that shipped with a first line `"use client";` has that line removed (the bundler warns on
module-level directives): `lib/arc-provider.tsx`, `lib/use-copy-feedback.ts`, and the `.tsx` file of every item above
except `arc-foundation` and `arc-motion-tokens`. No other file changed except the regex replacements below
(this app commits no regex; the replacements are hand-written scanners in `ui/src/lib/text-scan.ts`).

| File | Line (here) | Upstream | Here |
|---|---|---|---|
| `button/button.tsx` | 7 | — | `import { hasNonWhitespace, stripTags } from "../../../lib/text-scan";` |
| `button/button.tsx` | 122 | `icon={!/\S/.test(key.replace(/<[^>]*>/g, ""))}` | `icon={!hasNonWhitespace(stripTags(key))}` |
| `input/input.tsx` | 6 | — | `import { isNumericWord, stripCommas } from "../../../lib/text-scan";` |
| `input/input.tsx` | 21 | `const isNumber = (word: string) => /^\d[\d.,/:]*%?$/.test(word);` | `const isNumber = (word: string) => isNumericWord(word);` |
| `input/input.tsx` | 27 | `parseFloat(word.replace(/,/g, "")) < parseFloat(shown.word.replace(/,/g, ""))` | `parseFloat(stripCommas(word)) < parseFloat(stripCommas(shown.word))` |
| `command-palette/command-palette.tsx` | 8 | — | `import { isWordChar, splitShortcut, splitWhitespace } from "../../../lib/text-scan";` |
| `command-palette/command-palette.tsx` | 25 | `!/[\p{L}\p{N}]/u.test(text.charAt(at - 1))` | `!isWordChar(text.charAt(at - 1))` |
| `command-palette/command-palette.tsx` | 32 | `/[\p{L}\p{N}]/u.test(text.charAt(start - 1))` | `isWordChar(text.charAt(start - 1))` |
| `command-palette/command-palette.tsx` | 59 | `query.trim().toLowerCase().split(/\s+/).filter(Boolean)` | `splitWhitespace(query.trim().toLowerCase())` |
| `command-palette/command-palette.tsx` | 91 | `shortcut.split(/\s+\|\+(?=.)/).filter(Boolean)` | `splitShortcut(shortcut)` |

All files compile under `ui/tsconfig.json` as shipped; no type fix was needed.
