/** Usage: vp exec oxnode examples/tts/instruct.ts <1.7B-CustomVoice-path> <preset-id> */
import { loadTtsModel, createAudioPlayer } from '@mlx-node/tts';
import type { AudioPlayer, TtsInputEvent } from '@mlx-node/tts';
const [path, preset] = process.argv.slice(2);
if (!path || !preset) throw new Error('Supply a 1.7B CustomVoice path and preset ID');
const model = await loadTtsModel(path);
async function* text(): AsyncGenerator<TtsInputEvent> {
  yield '现在开始播报。';
  yield { type: 'instruct', value: '用兴奋、充满期待的语气说。' };
  yield { type: 'text', text: '列车即将到站，我们终于可以出发了！' };
  yield { type: 'instruct', value: null };
  yield '请携带好随身物品。';
}
let player: AudioPlayer | undefined;
try {
  player = await createAudioPlayer(model.capabilities.sampleRate, 1);
  const stream = model.synthesizeStream(text(), {
    voice: preset,
    instruct: '用平静、清晰的语气说。',
    language: 'chinese',
  });
  for await (const chunk of stream) await player.write(chunk.samples);
  console.log(await stream.completed, await player.finish());
} finally {
  player?.cancel();
  await model.dispose();
}
