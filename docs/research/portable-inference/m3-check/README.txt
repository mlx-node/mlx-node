M3 Pro performance check — use the existing signed test3 app

1. Quit mlx-node and its coding agent so the model is not loaded twice.
2. Double-click “Run M3 check.command”. Select the test3 app, then the original
   Qwen3.8-27B-UD-Q4_K_XL.gguf file. Allow several minutes for three runs.
3. Send M3-results.json from the folder opened on your Desktop.

The runs compare test3 and stock matrix kernels with MTP off, then measure
MTP separately. Each uses the same public code-review fixture and a cached
follow-up of about 1.5k new tokens. The model and app are not modified. No
packages are installed, no network requests are made, and no private agent
conversation is read. Results contain timings, token counts, memory settings,
software/hardware identifiers and output hashes; generated text is omitted.
The accompanying inference logs may include local paths. Send only the
summary JSON initially. This is a diagnostic, not a new speedup claim.
