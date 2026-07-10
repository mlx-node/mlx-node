/** A prompt is "cold" (show the model-free starter grid) ONLY when it is exactly
 *  empty. Whitespace is content — a custom prompt not yet run shows the skeleton,
 *  never a starter grid under someone else's text. Matches the permalink write gate. */
export const isColdPrompt = (prompt: string): boolean => prompt === '';
