# Void 0.9 upgrade + TanStack Router migration

**Status:** approved
**Date:** 2026-05-28
**Package:** `packages/browser` (the in-browser MLX learning app deployed at https://mlx.void.app)

## Context

The browser app is a Vite SPA hosted on Void. Its 10-chapter LLM education flow has shipped, but "navigation" is encoded entirely in query parameters:

| Today's URL | Logical screen |
| --- | --- |
| `/?screen=landing` | Landing |
| `/?screen=chapter_index` | Chapter index |
| `/?screen=chapter&chapterId=rmsnorm` | Single chapter |
| `/?screen=chat` | Free chat |
| `/?screen=loading` | Transient model-load state |

The screen state is parsed in `packages/browser/demo/lib/screen-state.ts` and synchronized to `history.replaceState()` inside `packages/browser/demo/app.tsx`. A `popstate` listener restores state on back/forward. This works but has three concrete problems:

1. **Reload mid-load yanks the user back to landing.** `?screen=loading` doesn't survive a refresh, so a learner who reloads `/?screen=chapter&chapterId=rmsnorm` while the model is still warming up lands on `/?screen=landing` instead of the chapter they were reading.
2. **`replaceState` (not `pushState`) means the browser back button exits the app**, never walks back through chapter → chapter index → landing. Users can't undo "open chapter".
3. **The URL is ugly and brittle.** Bookmarks like `?screen=chapter&chapterId=rmsnorm` mix two unrelated concerns (which screen, which chapter) into a flat query string; the app can't easily share or deep-link a chapter URL.

Independently, Void's npm package has moved from `0.7.5` (currently installed) to `0.9.0` (latest as of 2026-05-28). Two minor bumps (0.7 → 0.8 → 0.9) with unread changelog. Both changes are confined to `packages/browser` and orthogonal to the underlying MLX runtime, so they're bundled into one project.

## Goals

- Replace query-param screen state with real pathname routes powered by **TanStack Router** (`@tanstack/react-router` 1.170.x).
- Upgrade `void` to `0.9.0` and confirm `voidPlugin()`, the `/api/model/*` Hono routes, the `storage` binding, and the COOP/COEP/CORP headers still work.
- Preserve every existing demo, widget, inspector, and chapter component byte-identical — only the routing layer changes.
- Keep old `?screen=…` bookmarks working via a one-shot redirect.

## Non-goals

- Section-level anchor deep-links inside chapters (`/chapters/rmsnorm#scale-invariance`) — no heading IDs exist yet; out of scope.
- Renaming `chapters` → `learn` or nesting chapters under `/learn/chapters/<id>` — flat URL shape chosen.
- Moving any backend functionality. `/api/model/*` and `/api/model-upload/*` stay where they are.
- Replacing the `screen-state.ts` reducer entirely on day one — model-load state (`'loading'` | `'ready'` | `'error'`) and reasoning-effort cycle stay reducer-managed; only the URL-derived `ScreenState` variants are removed.
- Adding tests for routing. The existing manual + Chrome MCP verification is the test surface for this work; no new automated route tests.

## URL shape

Approved by user: **flat**.

| New URL | Component |
| --- | --- |
| `/` | Landing |
| `/chapters` | Chapter index |
| `/chapters/:chapterId` | Single chapter (id matched against `findChapter()` from `learn/chapters.ts`) |
| `/chat` | Free chat |

`:chapterId` validation: in the route's `loader`, call `findChapter(params.chapterId)`. If it returns `null`, throw `notFound()` and redirect to `/chapters`. The `chapters.ts` registry stays the single source of truth for valid IDs.

**Search params are preserved** for model config. The root route declares a `validateSearch` (Zod-style) for `model_url`, `model_label`, `max_new_tokens`, `temperature`, `tools` / `app_preview`, and the legacy aliases (`modelUrl`, `modelLabel`, `temp`, `maxOutputTokens`, `model`). Every route reads them via `Route.useSearch()` so they're type-safe and survive navigation. They're orthogonal to which screen the user is on — no path migration for them.

## Loading is not a route

Today `?screen=loading` is a transient state with two purposes: showing a loading UI between `landing → chat` or `chapter → chat`, and gating the chat UI on a warm model.

After migration:
- The model-load state lives in a top-level provider mounted on the `__root` route (see "Providers" below).
- Routes that need a warm model (`/chat`, `/chapters/:chapterId`) render an inline loading state when the model is still warming up. The URL stays at the destination.
- Reload behavior changes for the better: reloading `/chat` while the model loads keeps the user on `/chat` with a "loading model…" overlay, instead of bouncing them to landing.
- The transition `chapter → chat` becomes: click "Open free chat" → kick off model load if needed → `navigate({ to: '/chat' })`. The user sees the chat page in loading state, then content fades in once weights are ready.

This is the only meaningful behavior change in Phase 2 — it improves the reload story and removes a special-case route.

## Phasing

The work splits into two phases that can land independently.

### Phase 1 — Void 0.7.5 → 0.9.0

Mechanical, isolated. Goal: green build on the new Void with no other changes.

1. Bump `void` to `0.9.0` in `packages/browser/package.json`. Run `vp install` (or the project's equivalent) at the repo root.
2. Run `void prepare` to regenerate `.void/*.d.ts`.
3. Open `node_modules/void/CHANGELOG.md` (or the repo equivalent if absent) and read the 0.7 → 0.9 entries. For each breaking change, file a sub-task. Likely candidates to verify:
   - `voidPlugin()` call signature in `packages/browser/vite.config.ts`.
   - `void.json` schema — `appType: "void"`, `storage: true` binding, the COOP/COEP/CORP headers.
   - Hono route handler signatures in `packages/browser/routes/api/model/[...path]/*.ts` and `packages/browser/routes/api/model-upload/*.ts`.
4. Local verification:
   - `vp dev` starts cleanly.
   - `curl -H "Range: bytes=0-1023" http://localhost:<port>/api/model/<weight>.safetensors` returns `206 Partial Content` with the right slice (this is the gate for chapter demos to work).
   - `yarn workspace @mlx-node/browser build` produces a clean bundle.
5. Do **not** deploy. User owns deploys.

### Phase 2 — TanStack Router migration

Add the router, build out file-based routes, migrate `app.tsx`, delete the URL half of `screen-state.ts`.

1. **Add dependencies.** `@tanstack/react-router` 1.170.x runtime, `@tanstack/router-plugin` 1.168.x Vite plugin (file-based code-gen). Both go in `packages/browser/package.json`.
2. **Wire the Vite plugin.** Edit `packages/browser/vite.config.ts` to add `tanstackRouter()` before `voidPlugin()`. Both transform Vite config; this order matches the TanStack docs.
3. **Add generated tree to `.gitignore`.** The plugin writes `packages/browser/demo/routeTree.gen.ts`.
4. **Create file-based routes.**
   ```
   packages/browser/demo/routes/
   ├── __root.tsx              App shell: providers + <Outlet />, validateSearch for model config
   ├── index.tsx               /             → <Landing />
   ├── chapters.tsx            /chapters     layout (header/footer wrapper)
   ├── chapters.index.tsx      /chapters     → <ChapterIndex />
   ├── chapters.$chapterId.tsx /chapters/:id → loader calls findChapter, renders chapter component
   └── chat.tsx                /chat         → <FreeChat />
   ```
5. **Extract providers from `app.tsx` into `__root.tsx`.**
   - Model loader (currently inline in `app.tsx`'s ~1700-line component) becomes `ModelLoaderProvider`. Hooks: `useModelLoader()` exposes `{ status, progress, model, load, reset }`.
   - Free-chat session state (sampler config, conversation history) becomes `FreeChatProvider`. Hook: `useFreeChat()`.
   - Telemetry + reasoning-effort cycle become `TelemetryProvider`.
   - Each provider is one file in `packages/browser/demo/providers/` and is mounted in `__root.tsx`'s component tree above `<Outlet />`.
6. **Shrink `screen-state.ts`.** Remove all `ScreenState` variants and the `parseScreenFromUrl` / `screenToUrlQuery` / `popstate` plumbing. Keep only what serves model-load state and reasoning-effort cycle. Consider renaming to `model-state.ts` once the URL parsing is gone.
7. **Migrate `app.tsx`.** It becomes a thin entry point that mounts `<RouterProvider />` with the generated tree. All current screen-switching logic disappears. Imperative navigation (e.g. `dispatchScreen({ type: 'open_chapter', chapterId })`) becomes `navigate({ to: '/chapters/$chapterId', params: { chapterId } })`.
8. **Update `<Link>` and `useNavigate()` call sites.** Audit `ChapterIndex.tsx`, the landing screen, the chat screen, and the chapter header/footer for in-app navigation; convert each to TanStack `<Link>` or `useNavigate()`.
9. **Back-compat redirect.** In `__root.tsx`'s `beforeLoad` (or equivalent one-shot effect), parse legacy `?screen=…&chapterId=…` and `throw redirect(...)` to the corresponding new path. Strip the legacy query keys; preserve model-config search params. Mapping:
   - `?screen=landing` → `/`
   - `?screen=chapter_index` → `/chapters`
   - `?screen=chapter&chapterId=X` → `/chapters/X` (after `findChapter(X)` validates; otherwise `/chapters`)
   - `?screen=chat` → `/chat`
   - `?screen=loading` → `/` (loading is no longer a screen)
10. **Documentation sweep.** Update inline `// /?screen=…` comment examples in `app.tsx` line 51 and any references in `docs/` to the new URLs.

## File-level scope summary

**New files** (Phase 2):
- `packages/browser/demo/routes/__root.tsx`
- `packages/browser/demo/routes/index.tsx`
- `packages/browser/demo/routes/chapters.tsx`
- `packages/browser/demo/routes/chapters.index.tsx`
- `packages/browser/demo/routes/chapters.$chapterId.tsx`
- `packages/browser/demo/routes/chat.tsx`
- `packages/browser/demo/providers/model-loader.tsx`
- `packages/browser/demo/providers/free-chat.tsx`
- `packages/browser/demo/providers/telemetry.tsx`
- `packages/browser/demo/router.ts` (createRouter wiring, types)

**Edited files:**
- `packages/browser/package.json` (deps in both phases; lockfile in both)
- `packages/browser/vite.config.ts` (Phase 2: add tanstackRouter plugin)
- `packages/browser/.gitignore` (Phase 2: ignore `demo/routeTree.gen.ts`)
- `packages/browser/demo/app.tsx` (Phase 2: collapses to a `<RouterProvider />` entry point)
- `packages/browser/demo/lib/screen-state.ts` (Phase 2: shrinks; remove URL-derived state)
- `packages/browser/demo/learn/ChapterIndex.tsx` (Phase 2: `<Link>` / `useNavigate()`)
- `packages/browser/void.json` (Phase 1: only if 0.9 schema changes require it)

**Untouched** (by intent):
- All chapter components under `packages/browser/demo/learn/chapters/*.tsx`.
- All widgets under `packages/browser/demo/learn/widgets/*.tsx`.
- All scaffolding under `packages/browser/demo/learn/scaffolding/*.tsx`.
- All inspectors under `packages/browser/demo/learn/inspector/*.tsx`.
- `packages/browser/demo/learn/chapters.ts` registry.
- `packages/browser/routes/api/model/[...path]/*.ts` and `packages/browser/routes/api/model-upload/*.ts`.
- `packages/browser/demo/public/_headers`.
- All MLX runtime crates and packages (`crates/*`, `packages/core`, `packages/lm`, etc.).

## Verification

After each phase:
- `vp lint`, `vp check`, `vp test` are clean.
- `cargo test` not required (no Rust touched).

Phase 1 specific:
- `vp dev` boots and serves the existing app at the same URLs as before.
- Range-GET against `/api/model/<weight>.safetensors` returns `206` with correct bytes.
- `yarn workspace @mlx-node/browser build` succeeds.

Phase 2 specific (Chrome MCP walk-through):
- `/` shows landing. Click "Start learning" → URL becomes `/chapters`. Browser back → URL returns to `/`.
- `/chapters` shows the index with the forward-pass diagram. Click any chapter card → `/chapters/<id>`.
- `/chapters/rmsnorm` deep-loaded in a fresh tab renders the chapter without bouncing to landing, even while the model is still loading (loading overlay visible until weights warm up).
- "Open free chat" from a chapter navigates to `/chat`. Browser back returns to `/chapters/<id>`.
- `/chapters/bogus` → redirects to `/chapters`.
- `/?screen=chapter&chapterId=rmsnorm` → URL replaces with `/chapters/rmsnorm`.
- `/chapters/rmsnorm?model_url=http://localhost:8080&temperature=0.3` parses both search params and they survive a navigation to `/chat` and back.
- COOP/COEP headers present in `vp dev` response headers and in the built bundle's `_headers` file.
- Existing demos still work: RMSNorm slider drags, attention heatmap renders, RoPE dot trail animates, sampling controls react, KV cache panel shows growth. (Smoke pass — no per-widget regression sweep beyond a visual check.)

**Not deployed until user explicitly says so.**

## Risks and open questions

- **Void 0.7 → 0.9 changelog is unread.** Phase 1's first task is to read it; any breaking change becomes a sub-task. We don't speculate before reading.
- **TanStack Router + Vite + Void plugin order.** Both are Vite plugins; ordering is empirical. Plan defaults to `tanstackRouter()` first, `voidPlugin()` second per TanStack docs. If conflicts surface, swap.
- **`app.tsx` is ~1700 lines.** The extraction into providers is the riskiest part of Phase 2. Plan keeps it mechanical: copy logic verbatim into providers, change the import in `app.tsx`, verify, commit. No refactoring during extraction.
- **Model-config search params with case variants.** The current parser accepts both snake_case (`model_url`) and camelCase (`modelUrl`) plus a `model` alias. The `validateSearch` schema must accept all variants and normalize to one canonical shape internally.
- **Reduced-motion accessibility.** Existing chapter components use `prefers-reduced-motion` CSS opt-outs. Router transitions add no animation, so this is preserved by default — flag here only because it's easy to regress.
