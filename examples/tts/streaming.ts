/** Run: vp exec oxnode examples/tts/streaming.ts <model-directory> <voice-id>
 * Pipe UTF-8 text into stdin; sentences can play before stdin reaches EOF.
 */
import { StringDecoder } from 'node:string_decoder';

import { loadTtsModel, createAudioPlayer } from '@mlx-node/tts';

const [path, voice] = process.argv.slice(2);
if (!path || !voice) throw new Error('Supply the model directory and a preset voice ID');
const model = await loadTtsModel(path);
const abort = new AbortController();
const cancel = () => abort.abort();
process.once('SIGINT', cancel);
try {
  const player = await createAudioPlayer(model.capabilities.sampleRate, model.capabilities.channels);
  try {
    async function* text() {
      const decoder = new StringDecoder('utf8');
      for await (const bytes of process.stdin) yield decoder.write(bytes as Buffer);
      yield decoder.end();
    }
    const stream = model.synthesizeStream(text(), { voice, signal: abort.signal });
    for await (const chunk of stream) await player.write(chunk.samples);
    console.error(await stream.completed, await player.finish());
  } finally {
    player.cancel();
  }
} finally {
  process.removeListener('SIGINT', cancel);
  await model.dispose();
}
