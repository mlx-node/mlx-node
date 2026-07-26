/**
 * Shared motion kit for the course's animated diagrams.
 *
 * Before this module, every animated widget declared its own
 * `usePrefersReducedMotion` (30 verbatim copies, imported by nobody) and its own
 * `frame` + `playing` + `setInterval` + play/step buttons (~32 copies). New
 * animated widgets should build on these instead.
 *
 * What is here:
 *   usePrefersReducedMotion  the preference, with the two-part SSR guard
 *   useStepPlayer            frame stepper + the SSR/reduced-motion contract
 *   StepControls             the standard play / step / reset row
 *   FlowDots                 dots travelling along a path ("data moving now")
 *   HatchDefs / hatchFill    diagonal hatch ("this box is busy this step")
 *   skin                     the LOOK: two hues, one radius, two stroke widths,
 *                            two dasharrays, a named opacity ladder, and the
 *                            PanelFrame / FrameLabel / DiagramFrame / PanBox
 *                            primitives
 *
 * Everything is SSR-safe: no window access during render, deterministic first
 * frame, and the animation itself is either CSS (compositor) or a post-mount
 * interval. The prerenderer gets a meaningful still frame in the HTML.
 */
export { FLOW_BLUE, FlowDots, type FlowDotsProps } from './FlowDots';
export { HatchDefs, hatchFill, type HatchTone } from './HatchPattern';
export { DASH, DIM, DiagramFrame, EMERALD, elbow, FrameLabel, PanBox, PanelFrame, RED, RX, SW } from './skin';
export { StepControls } from './StepControls';
export { usePrefersReducedMotion } from './use-prefers-reduced-motion';
export { useStepPlayer, type StepPlayer } from './use-step-player';
