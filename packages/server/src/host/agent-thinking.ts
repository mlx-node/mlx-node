/**
 * The thinking bounds of the two hosts that run agents: the desktop engine and the server `mlx agent` starts. Both pass
 * these three values as their server options; `mlx serve` by hand sets none of them.
 *
 * `tokenBudget`: thinking tokens of a Messages request that sets no budget (Splash's example for the Qwen line).
 * `cycleGuard`: an exact repeat inside a thought closes it. `endText`: the sentence written before the template's
 * close when a cut ends thinking, so the model turns to its answer (Splash `--thinking-end-text`).
 */
export const AGENT_HOST_THINKING = {
  tokenBudget: 12_000,
  cycleGuard: true,
  endText: '\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.',
} as const;

/** The longest thinking end text a server takes, in characters. */
export const THINKING_END_TEXT_MAX = 512;

/** A thinking end text of 1 to {@link THINKING_END_TEXT_MAX} characters; throws on any other. */
export function parseThinkingEndText(text: string): string {
  if (text.length === 0 || text.length > THINKING_END_TEXT_MAX) {
    throw new Error(`Invalid thinking end text: expected 1 to ${THINKING_END_TEXT_MAX} characters.`);
  }
  return text;
}
