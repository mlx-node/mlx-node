# Provenance of `components/arc-x/wait-orb/`

`engine.ts` holds geometry derived from `thinking-orbs`, the free (MIT) dotted-orb package. `wait-orb.tsx` (the React
component, the painter, the states, the loop and its gates) is our own code.

|                |                                                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Source package | npm `thinking-orbs` 0.3.2                                                                                                               |
| Repository     | https://github.com/Jakubantalik/Libraries.dev (MIT), commit `5e7afad5fc8f3661e0bf0b4cd3f87de482ffcf16`, folder `packages/thinking-orbs` |
| Fetched        | 2026-11-13                                                                                                                              |
| License        | `LICENSE` in this folder (MIT, Copyright (c) 2026 Jakub Antalik), the same text as the package's `LICENSE`                              |

## Derived from

| Upstream file                              | What `engine.ts` keeps                                                                                                          |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `src/engine/core.ts`                       | `makeProj` (spin, tilt, orthographic projection), `radiusScale`, `fibDir`, `finalizeFrame` (radius floor 0.3, sort far to near) |
| `src/engine/ribbon.ts`                     | `frameRibbon`: the band (`composing`) and, with `faceOn`, the ring (`breathing`)                                                |
| `src/engine/lattice.ts`                    | `frameWave`: the pulsing sphere (`listening`)                                                                                   |
| `src/presets.ts`, `src/engine/profiles.ts` | the numbers `resolvePreset(state, 20)` returns for `breathing`, `composing` and `listening`                                     |

Our state names map to the package's: `hold` = `breathing` (ring), `think` = `composing` (band), `load` = `listening`
(wave).

## Local changes

| #   | Change                                                                                                                                                                                                                                                                        |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 3 of the 9 shapes are kept (`ring`, `ribbon`, `wave`). The other frame functions, the `web` lines, `morph`, `braid`, `orbits` and `rubik` are not copied                                                                                                                      |
| 2   | The presets are resolved to numbers; `profiles.ts`, `presets.ts`, `scaleCounts` and `scaleRadii` are not copied                                                                                                                                                               |
| 3   | Only the 20 px size                                                                                                                                                                                                                                                           |
| 4   | Every shipped `ribbon` and `ring` preset has `spin: 0`, so the band's plane is fixed and its basis is written out. Every dot equals the package's within 3.6e-15 px at 0, 0.6, 1.234, 7.5 and 123.456 s (dots of equal depth may swap their draw order)                       |
| 5   | `orbFrame(shape, seconds)` multiplies by the preset's speed itself, as the package's component does with `resolvePreset().speed`                                                                                                                                              |
| 6   | The painter is ours: one ink color for the whole orb, depth by alpha (`a · (1 − white)`). The package's `inkColor`, `paint`, `paintLines`, `paintFrame` and its tint parsing are not copied                                                                                   |
| 7   | `ThinkingOrb.tsx` (the React wrapper), `theme.ts` (its page-wide MutationObserver), `gravity.ts`, `registry.ts` and the native ports are not copied                                                                                                                           |
| 8   | `finalizeFrame`'s drop of dots with alpha under 0.02 and the painter's clamp of `white` to [0, 1] are not copied: for these 3 shapes no dot reaches them (alpha ≥ 0.125, `white` in [0.004, 0.78] over 200,000 instants; `__test__/wait-orb-engine.test.ts` holds the bounds) |

## Not used

- The Studio, Studio exports, Pro presets and recipes of libraries.dev (paid): not fetched, not copied.
- The states `weaving` and `shaping` and the 32 px size, which the site's playground marks as Pro.
- The libraries.dev site's own page code, including its text shimmer. The line beside the orb uses the app's
  `LiveLabel`.
