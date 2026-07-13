/**
 * Ad-hoc headless J-lens run for ONE prompt, both logit-lens and fitted-Jacobian.
 * Run: env PATH="/opt/homebrew/bin:$PATH" JLENS_PACK=lens-pack-v1.safetensors \
 *   oxnode packages/browser/scripts/jlens/adhoc-planet.mts
 */
const MODEL_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const OUT_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';
import { join } from 'node:path';
const PACK_ENV = process.env.JLENS_PACK ?? 'lens-pack-v1.safetensors';
const PACK_PATH = PACK_ENV.includes('/') ? PACK_ENV : join(OUT_DIR, PACK_ENV);

const PROMPT = 'The color of the planet fourth from the sun is';
const LAYERS = Array.from({ length: 24 }, (_, i) => i + 1); // 1..24
const TOP_K = 6;

async function main() {
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL_PATH} ...`);
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;

  console.log(`Loading lens pack from ${PACK_PATH} ...`);
  const loaded = await model.loadLensPackFromFile(PACK_PATH);
  console.log(`pack Jacobians = ${loaded}`);

  const promptIds: number[] = await model.encodeTokens(PROMPT);
  const texts: string[] = await model.decodeTokens(promptIds);
  console.log(`prompt tokens (${promptIds.length}): ${texts.map((t) => JSON.stringify(t)).join(' ')}`);

  for (const useJacobian of [false, true]) {
    const ro = (await model.lensReadout(promptIds, {
      layers: LAYERS,
      topK: TOP_K,
      pinnedIds: [],
      useJacobian,
    })) as any;
    const P = ro.promptLen;
    const lastPos = P - 1;
    console.log(`\n===== ${useJacobian ? 'FITTED-JACOBIAN' : 'LOGIT'} LENS  (jacobianApplied=${ro.jacobianApplied}) =====`);
    console.log(`promptLen=${P} layers=${ro.layers.length} topK=${ro.topK}`);
    console.log(`per-layer top-${TOP_K} at LAST position ("... sun is" -> next):`);
    for (let li = 0; li < ro.layers.length; li++) {
      const cell = ro.cells[li * P + lastPos];
      const toks = (cell.topKTexts as string[]).map((t) => JSON.stringify(t)).join(' ');
      console.log(`  L${String(ro.layers[li]).padStart(2)}: ${toks}`);
    }
  }
  console.log('\nDONE');
  process.exit(0);
}
main().catch((e) => {
  console.error('FAILED:', e);
  process.exit(1);
});
