# Playground Redesign — Capybara Library

**Status:** Draft
**Date:** 2026-05-05
**Owner:** @LongYinan (brooklyn)
**Scope:** `packages/browser/demo` only

## Goal

Replace the current cool-blue 3-panel workspace at `packages/browser/demo/app.tsx` with a HF-style 3-screen flow (landing → loading → chat) styled around Qwen's official capybara mascot. Add a WebGL hero on the landing screen that reads as "Qwen's library" — pastel-colored books drifting through space.

The redesign is purely a presentation rewrite. Streaming, tool calls, image attach, reasoning effort, decode counters, and local-model loading must all keep working. No changes to `@mlx-node/browser` runtime, `chat-stream-sab`, `generated-text` sanitisers, or the worker bridge.

## Non-goals

- No new chat features (no tools beyond `create_app_preview`, no model switching beyond what we already support, no history persistence).
- No backend changes — Rust, C++, NAPI, WASM untouched.
- No `@mlx-node/lm` API changes.
- No accessibility audit beyond keeping current keyboard / focus behaviour intact.
- No marketing site, no homepage, no router.

## Brand direction

The HF reference (`webml-community/Qwen3.5-WebGPU`) uses a warm-amber accent that is **not** Qwen-brand — it is the community author's choice. Qwen's actual identity has two faces:

1. **Corporate:** purple `#615CED` on near-black, the rotating chevron-ribbon Penrose knot. Cold, geometric.
2. **Mascot (吉祥物):** a sleepy capybara reading books with pastel covers (pink, blue, cream, lilac). Cozy, bookish, calm.

The redesign leans into the mascot — pastel pinks/blues/creams pulled from the mascot image — with Qwen purple `#615CED` as a single accent for telemetry and Qwen-knot moments. The result reads as "the model that has read everything and is happy to chat about it," not "AI cyberpunk."

## Visual system

### Palette

```css
:root {
  --bg:           #0F0D11;
  --surface:      #1B1820;
  --surface-2:    #2A2630;
  --border:       #2F2A36;
  --text:         #ECE4DA;
  --text-dim:     #9E928A;
  --text-muted:   #6A6168;

  /* Mascot — primary accent family */
  --book-pink:    #F4C8D7;  /* primary accent */
  --book-blue:    #BAD4E8;
  --book-cream:   #F0DDB8;
  --book-lilac:   #E0D0E8;
  --book-peach:   #F6D2BC;
  --book-sage:    #CFE0CC;

  /* Qwen corporate — telemetry + knot watermarks only */
  --qwen:         #615CED;
  --qwen-light:   #8884F6;
  --qwen-deep:    #4F2DDA;
}

body {
  background:
    radial-gradient(1200px 600px at 80% -10%, rgba(244,200,215,0.08) 0%, transparent 60%),
    radial-gradient(900px 500px at 10% 110%, rgba(97,92,237,0.10) 0%, transparent 65%),
    radial-gradient(207% 123% at 50% 0%, #2A2530 0%, #0F0D11 100%) fixed;
}
```

### Typography

Three Google Fonts (matches HF reference shell):

- **Instrument Serif** (`ital@0;1`) — display: H1 hero title and section H3 in cards. Italic em is used for the brand-warm word in each title.
- **Inter** (400, 500, 600) — body, buttons.
- **DM Mono** (400, 500) — kicker pills, telemetry, pill labels.

### Motion

All transitions ease-in-out 200–300ms unless noted. Books in the WebGL hero ease their per-book bob phase. No spring physics, no scrub-on-scroll. Reduced-motion respects `prefers-reduced-motion: reduce` by freezing the WebGL after one frame.

## Information architecture — three screens

State machine:

```
landing  ──[user clicks Load Model]──▶  loading
loading  ──[model ready]─────────────▶  chat
loading  ──[model error]─────────────▶  landing  (with error banner)
chat     ──[user clicks Reset Chat]──▶  chat     (in-place; clears messages + SAB ring)
```

A single React state value `screen: "landing" | "loading" | "chat"` drives which screen renders. Only one is mounted at a time. The WebGL renderer is created and disposed with the landing screen lifecycle.

There is no "switch model" flow once the model is loaded — to change models the user reloads the page (matches HF reference). This keeps screen transitions simple and avoids the cost of disposing/recreating the worker + WASM module mid-session.

## Screen 1 — Landing

Centered, full viewport.

- WebGL hero canvas fills the background (Drifting Library, see below).
- Foreground stack, vertically centered, max-width 640px:
  - `landing-tag` pill — DM Mono kicker `MULTIMODAL · 100% LOCAL · WEBGPU`. Pink dot prefix, glow.
  - `landing-title` H1 — `Qwen 3.5 ` in regular + `Vision` italic em in `--book-pink`. Instrument Serif clamp(3rem, 8vw, 5.5rem).
  - `landing-sub` — one paragraph, max-width 520px, `--text-dim`, weight 300.
  - `landing-specs` — three centered columns: `Vision + Language / Unified Multimodal`, `201 Languages / Global Coverage`, `Reasoning / Code · Agents · Visual`. Mono value, dim small-caps label.
  - `btn-load-group` — pill button `Load Model (0.8B ▾)` in `--book-pink` background with `--bg` text. Right edge has a separator and the model-size dropdown trigger. Box-shadow uses `--book-pink` glow on hover.
  - `landing-footer-row` — DM Mono `Built with @mlx-node/browser` + a secondary `Local model…` link that opens the file picker (replaces today's header `Local model` button).
- Capybara mascot image (`./capybara.png`, 90KB, already in `packages/browser/demo/public/` after this redesign) rendered as `<img>` in the bottom-right of the viewport at 18% width with `filter: drop-shadow()`. Acts as a small "host" presence, not the hero.

The Drifting Library hero canvas sits behind everything at `position: fixed; inset: 0; z-index: 0`. All foreground content sits at `z-index: 1+`.

## Screen 2 — Loading

- Centered ring spinner (1px border, 72×72, `border-top-color: --book-pink`, 1s rotation).
- Below: `loader-text` (DM Mono small, `--text-dim`) — receives the live status updates currently emitted to the status pill (`Initializing model...`, `Loading weights...`, `Compiling shaders...`, `Ready.`).
- Sub-line: `Model weights are cached for future visits.` (small, muted).
- No WebGL during loading — Drifting Library has been disposed; the page is plain dark.

## Screen 3 — Chat

Full viewport, three flex rows:

```
┌──────────────────────────────────────┐
│  Header                               │  ← 56px
├──────────────────────────────────────┤
│  Messages                             │  ← flex 1, scrollable
│  ┌─────────────────────────────────┐  │
│  │ User bubble                      │  │
│  │ Assistant bubble                 │  │
│  │ ▶ [Inline preview card] ←tool   │  │
│  │ Assistant bubble                 │  │
│  └─────────────────────────────────┘  │
├──────────────────────────────────────┤
│  Telemetry strip                      │  ← 26px, --qwen tinted
├──────────────────────────────────────┤
│  Power-bar composer                   │  ← ~96px (input + pill row)
└──────────────────────────────────────┘
```

Solid `--bg` surface — **no WebGL** (decision Q5=A, hero only).

### Chat header

- Left: 34×34 rounded-square avatar with capybara `Q` glyph in serif italic (Instrument Serif), background `linear-gradient(135deg, --book-pink, --book-cream)`, `--bg` text.
- Title `Qwen 3.5 Vision` (Instrument Serif 1.15rem) + status sub-line `● Ready on WebGPU` (DM Mono, `--book-pink` dot + label, glow on dot).
- Right: `Reset Chat` outline pill button (DM Mono small caps). Clicking clears the message list and resets the underlying chat session (existing SAB ring reset path) — stays on the chat screen.

### Messages area

- Same `chat-messages` container as today but restyled. User bubble uses `rgba(244,200,215,0.18)` with `--book-pink` border. Assistant bubble uses `--surface` with `--border`. Bottom-corner radius reduced (4px) on the matching side.
- Streamdown markdown rendering preserved as-is. Streamdown stylesheet still imported.
- `splitAssistantThinking` gives us the `<think>` block — it renders as a collapsible left-rule block above the answer (existing pattern, restyled with `--book-pink` accent on hover).
- Generation stats (existing `msg-stats` row) restyled to DM Mono, muted.

### Inline preview card (tool: `create_app_preview`)

When the model emits a `create_app_preview` tool call, a card renders **inside** the assistant bubble that triggered it (instead of the current dedicated Preview panel). The card:

- Has a thin `--book-pink` border, slightly tinted background.
- Header row: DM Mono `▶ APP PREVIEW · {title}` (left), `↗ open full-screen` button (right).
- Body: the existing `<iframe sandbox="allow-scripts allow-forms allow-popups allow-modals allow-downloads">` rendered at 320px height by default; "open full-screen" promotes it to a modal overlay covering the full chat area with a backdrop blur and a close (`Esc`) button.
- The current Preview panel and its visibility toggle are removed entirely. The `tools · on/off` pill in the composer (see below) gates whether the model is allowed to call the tool at all.

### Telemetry footer strip

A 26px strip between messages and composer:

```
[ 21 tok/s ]  •  2,070 gpu-rpc/tok  •  pool 62%        qwen3.5-0.8b · bf16
```

- Background `rgba(97,92,237,0.06)`, top border `rgba(97,92,237,0.20)` — single Qwen-purple accent on the page. DM Mono 10px, uppercase.
- Left: live `tok/s` in a Qwen-blue pill, then `gpu-rpc/tok`, `pool` hit-rate, in dot-separated mono.
- Right: model name + dtype, dimmed.
- Counter source: existing `ProfileStats` reducer in `app.tsx` plus `decode_profiler` events. No new instrumentation. Updated on each `done` chunk; values shown are from the last completed turn (or `—` before the first turn).

### Power-bar composer

Two-row composer, glass-morphism background:

```
┌───────────────────────────────────────────────────────┐
│  📎  [ textarea — Ask Qwen anything…                ] 🎤  ↑  │
│  [think · med] [temp · 0.6] [max · 1024] [tools · on]      │
└───────────────────────────────────────────────────────┘
```

- Top row: image-attach (📎) — same `imageInputRef` flow as today; textarea (auto-grow up to 140px); voice mic (placeholder, currently no-op — kept because the existing component has the slot wired); send button (`--book-pink` solid, `--bg` arrow icon, transitions to red square on stop).
- Pill row: four pills, all DM Mono 10.5px:
  - `think · {off|low|med|high}` — clicking cycles, mirrors `reasoningEffortRef.current`.
  - `temp · {value}` — clicking opens a small numeric stepper popover (0.0–2.0 step 0.1).
  - `max · {value}` — same stepper, 1–36864.
  - `tools · {on|off}` — toggles `appToolsEnabled` (controls inclusion of the `APP_PREVIEW_TOOL` definition in the request).
- The `model · qwen3.5-0.8b · bf16` info has moved to the telemetry strip; it is **not** a pill, since switching models requires a reset/load cycle.

## WebGL hero — Drifting Library

Loaded on the landing screen only. Disposed in `useEffect` cleanup when the screen transitions.

### Visual

- ~120 individual book Groups (`THREE.Group` containing two `BoxGeometry` meshes — pages + cover). Six pastel cover colours from the mascot palette, cycled.
- Each book is placed on a cylindrical orbit at radius 2.3–5.3 from the origin, randomised height ±2.5, randomised initial rotation.
- Per-book animation:
  - Angular velocity around Y axis (lazy orbit, ~0.05 rad/s).
  - Vertical bob: `y = y0 + sin(t * 0.3 + bob_phase) * 0.15`.
  - Slow tumble: rotation deltas on X and Y of `(rand - 0.5) * 0.4 * dt`.
- Lighting: `HemisphereLight(0xFFE8D8, 0x2A2030, 0.55)` + warm `DirectionalLight` from upper-right + cool Qwen-purple rim light from back-left.
- A purple radial sprite (additive blending) sits at z=−3 to give the hero its `--qwen` corona.

### Implementation

- Library: `three` (latest stable at install time, currently `^0.160.0` series — verify with `npm info three` before adding). ≈600 KB minified. Add as a runtime dep on `packages/browser/demo` only — not a workspace-level dep.
- Single `WebGLRenderer({ antialias: true })`, pixel ratio capped at 2.
- Renders at full screen-rate while landing is mounted.
- No shaders required for this concept — `MeshStandardMaterial` everywhere; cheap.
- A `prefers-reduced-motion: reduce` check renders one static frame and pauses the rAF loop.

### File location

- New file `packages/browser/demo/components/landing/drifting-library.tsx` — exports `<DriftingLibrary />` React component that mounts the renderer and cleans up.
- All Three.js imports stay inside this component so tree-shaking keeps it out of the chat-screen bundle.

## Mascot usage

- Copy `capybara.png` (the user-provided mascot, 86×86 PNG ~90 KB) into `packages/browser/demo/public/capybara.png`.
- Used in:
  - Landing screen — bottom-right at 18% viewport width, `pointer-events: none`.
  - Browser tab favicon — generated 32×32 / 180×180 from this PNG via a small `vp run` script (or hand-cropped once and committed).
- **Not** used inside the WebGL scene (decision Q2=C — Drifting Library is mascot-implied, not literal). The capybara is a 2D friend on the landing edge.

## File-level changes

| File | Change |
|---|---|
| `packages/browser/demo/index.html` | Add Google Fonts `<link>` for Instrument Serif / Inter / DM Mono. |
| `packages/browser/demo/styles.css` | Full rewrite. Drop the existing OKLCH cool-blue tokens. New tokens listed in §Visual system. Tailwind v4 retained (still using `@import "tailwindcss"`); custom CSS replaces the `@layer components` block. |
| `packages/browser/demo/app.tsx` | Top-level becomes a `<Screen>` switcher reading a single `screen` state. Tool-call display buffer, sanitisers, SAB ring wiring, profiling reducer, model-load worker plumbing all stay. The current single-page JSX (header / workspace-grid / composer-bar) is deleted and replaced. |
| `packages/browser/demo/components/landing/Landing.tsx` | New. Renders WebGL hero + foreground stack. |
| `packages/browser/demo/components/landing/drifting-library.tsx` | New. Three.js component. |
| `packages/browser/demo/components/loading/Loading.tsx` | New. Spinner + status text. |
| `packages/browser/demo/components/chat/ChatScreen.tsx` | New. Header + messages + telemetry strip + power-bar composer. |
| `packages/browser/demo/components/chat/InlinePreviewCard.tsx` | New. Replaces the dedicated Preview Card surface. |
| `packages/browser/demo/components/chat/TelemetryStrip.tsx` | New. |
| `packages/browser/demo/components/chat/PowerComposer.tsx` | New. Houses the textarea + four pills. Numeric stepper popovers reuse current shadcn-ish primitives. |
| `packages/browser/demo/components/ui/*` | Existing shadcn primitives (Button, Card, Select, Switch, Textarea, Badge) stay. Card and Badge will likely be unused after the rewrite — leave them in place for the next person who needs them; do not delete pre-emptively. |
| `packages/browser/demo/public/capybara.png` | New asset. |
| `packages/browser/demo/package.json` | Add `three` (^0.160.0) and `@types/three` (dev). |
| `.gitignore` | Already updated to ignore `.superpowers/` (this PR). |

## Risks & open questions

1. **three.js bundle weight.** Adds ~600 KB minified to the demo bundle. Acceptable for a playground, but log it in the PR description. If we end up needing to ship this elsewhere, the Three.js import lives in one file and can be lazy-loaded behind `React.lazy(() => import('./drifting-library'))` on the landing screen.
2. **Mascot rendering rights.** The capybara image was provided by the user as Qwen's official mascot; we should confirm it's safe to ship inside the playground (likely fine — it's a public Qwen mascot — but flag during PR review).
3. **Voice mic** stays as a placeholder (the current playground also has it as a no-op). The redesign doesn't ship Web Speech wiring; the slot exists for future work.
4. **Reduced-motion.** Current playground has zero motion preferences. The redesign respects `prefers-reduced-motion: reduce` for the WebGL hero only; chat-screen animations are minimal enough to not need it.
5. **Profile-stats display when empty.** The telemetry strip needs sensible placeholders before any decode has run (`— tok/s`, `0.8b · bf16`). Not visually different — the strip is always there.

## Out of scope (keep in mind for follow-ups)

- Persisting chat history across sessions.
- Switching models at runtime without a full reset.
- Mobile layout (the playground stays desktop-only; chat UI on mobile is acceptable but not tuned).
- Wiring real voice input.
- Drifting Library variations beyond the single picked concept.

## Decisions log

| ID | Question | Outcome |
|---|---|---|
| Q1 | Single-page workspace vs. HF-style 3-screen flow? | **HF-style 3-screen flow** |
| Q2 | WebGL aesthetic? | **Drifting Library** — pastel book-cloud, mascot-grounded |
| Q3 | Where do Telemetry / Preview live? | **Minimal** — telemetry footer strip, preview inline in chat thread |
| Q4 | Composer density? | **Power bar** — all knobs visible as DM Mono pills |
| Q5 | WebGL persistence on chat screen? | **Hero only** — disposed when chat opens |
