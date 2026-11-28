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

`combobox` and `number-field` were added on 2026-11-27 with the shadcn CLI 4.23.1 in a scratch project
(`shadcn add https://uiarc.dev/r/combobox.json https://uiarc.dev/r/number-field.json`, the command the Arc MCP tool
`get_install_command` gives) and copied from there. That CLI writes `number-field.module.css` without the two comment
lines the registry file starts with. It also wrote newer `motion-tokens.ts` and `lib/motion-tokens.ts` (shorter durations
and springs); those were not taken, the copies here stay as fetched on 2026-11-10.

## Vendored items (27)

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
| combobox | `combobox/combobox.tsx`, `combobox/combobox.module.css` |
| number-field | `number-field/number-field.tsx`, `number-field/number-field.module.css` |

## Local changes

Every `.ts` / `.tsx` file that shipped with a first line `"use client";` has that line removed (the bundler warns on
module-level directives): `lib/arc-provider.tsx`, `lib/use-copy-feedback.ts`, and the `.tsx` file of every item above
except `arc-foundation` and `arc-motion-tokens`. No other file changed except the regex replacements below
(this app commits no regex; the replacements are hand-written scanners in `ui/src/lib/text-scan.ts`, or a plain
list where the pattern matched four fixed key names), the `button/button.module.css` overflow clip, the
`number-field/number-field.tsx` draft fix in the row for line 400, the `combobox/combobox.tsx` fixes and the one
`drawer/drawer.tsx` fix in the last rows of the table.

| File | Line (here) | Upstream | Here |
|---|---|---|---|
| `button/button.module.css` | 2–3 | `.button { position: relative; display: inline-flex; … }` | comment line above `.button` plus `overflow: clip;` on `.button` |
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
| `number-field/number-field.tsx` | 7 | — | `import { hasAsciiDigit, keepDigitsAnd } from "../../../lib/text-scan";` |
| `number-field/number-field.tsx` | — | line 63: `const escape = (text: string) => text.replace(/[.*+?^${}()\|[\]\\-]/g, "\\$&");` | removed: `keepDigitsAnd` takes its characters as they are, so nothing needs escaping |
| `number-field/number-field.tsx` | 138 | `const digits = word?.replace(/[^\d.-]/g, "");` | `const digits = word === undefined ? undefined : keepDigitsAnd(word, ".-");` |
| `number-field/number-field.tsx` | 138 | `return digits && /\d/.test(digits) && …` | `return digits && hasAsciiDigit(digits) && …` |
| `number-field/number-field.tsx` | 287 | `….replace(symbols.decimal, ".").replace(/[^\d.-]/g, "");` | `keepDigitsAnd(….replace(symbols.decimal, "."), ".-");` |
| `number-field/number-field.tsx` | 288 | `if (!/\d/.test(normalized)) return null;` | `if (!hasAsciiDigit(normalized)) return null;` |
| `number-field/number-field.tsx` | 400 | ``const allowed = new RegExp(`[^0-9${escape(symbols.group)}${decimals \|\| maxFraction ? escape(symbols.decimal) : ""}${min < 0 ? "\\-" : ""}]`, "g");`` | ``const allowed = `${symbols.group}${symbols.decimal}-`;`` (the draft keeps the decimal point and the minus sign, see below) |
| `number-field/number-field.tsx` | 401 | `const next = text.replace(allowed, "");` | `const next = keepDigitsAnd(text, allowed);` |
| `number-field/number-field.tsx` | 468 | `if (/^(Arrow(Up\|Down)\|Page(Up\|Down))$/.test(event.key)) stopHold();` | `if (["ArrowUp", "ArrowDown", "PageUp", "PageDown"].includes(event.key)) stopHold();` |
| `combobox/combobox.tsx` | 81 | `const [activeIndex, setActiveIndex] = useState(-1);` | `const [activeValue, setActiveValue] = useState<string \| null>(null);` |
| `combobox/combobox.tsx` | 108 | — | `const activeIndex = filteredOptions.findIndex((option) => option.value === activeValue);` |
| `combobox/combobox.tsx` | 109 | — | `const setActiveIndex = (index: number) => setActiveValue(filteredOptions[index]?.value ?? null);` |
| `combobox/combobox.tsx` | 199 | `<div ref={rootRef} className={styles.field}>` | the same `div` with `onBlur` that closes the menu and clears the query when focus moves outside it |
| `combobox/combobox.tsx` | 258 | — | `onMouseDown={(event) => event.preventDefault()}` on the popover |
| `drawer/drawer.tsx` | 97 | — | the comment line above the `offset` value |
| `drawer/drawer.tsx` | 98 | `const offset = useMotionValue<number \| string>(0);` (four lines higher, above `pan`) | ``const offset = useMotionValue<number \| string>(reduced ? 0 : `${sign * 100}%`);`` |

The button fix: `Button` crossfades labels inside `<AnimatePresence mode="popLayout">`. Motion's `PopChild` measures
past `position: sticky` ancestors, so when a button inside a sticky bar swaps children during a scroll-root reset, the
exiting `.labelPhase` can be positioned hundreds of pixels below `.button` and inflate the scroll root's `scrollHeight`.
`overflow: clip;` on `.button` contains exiting labels without clipping focus outlines or box shadows.

The drawer fix: the panel's motion value started at 0 while the entrance starts at `100%`. Motion writes that first
frame into the value when the panel mounts, the jump read as a velocity of 100 % in 30 ms, and the entrance spring
inherited it, so the first open of each drawer went about 30 % of the panel width past its resting edge (143 px on a
480 px panel) and came back. Later opens were right because the exit leaves the value at `100%`.

The number field fix: upstream lets the decimal point into the draft only when `step` or `formatOptions` has fraction
digits, and the minus sign only when `min` is below 0. With whole steps from 11, typing "13.5" left the draft "135",
which Enter clamped to the maximum, and "-16" left "16", which applied at once. The draft now keeps both characters,
so the commit snaps "13.5" to 14 and clamps "-16" to the minimum.

The combobox fixes: upstream keeps the highlighted option as an index, so when `options` changed under an open menu
(a list that arrives while the menu is open) Enter picked whatever option then sat at that index. The highlight is now
kept by value, and an option that left the list has none. Upstream also closed the menu only on a pointer press
outside it, so Tab left the open list over the next control. The menu now closes when focus moves outside the field,
and a press on the popover (its padding, or the list's scrollbar when scroll bars always show) no longer takes focus
from the input, so that press does not close it and the arrow keys still work after it.

All files compile under `ui/tsconfig.json` as shipped; no type fix was needed.
