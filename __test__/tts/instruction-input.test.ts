import { expect, it } from 'vite-plus/test';

import { paragraphInstructions } from '../../scripts/tts/instruction-input.js';
it('preserves text and paragraph instruction positions at every chunk boundary', async () => {
  const text = '第一段🙂。\r\n\r\n第二段。\n \nThird paragraph.';
  for (let size = 1; size < 20; size++) {
    const source = (async function* () {
      for (let i = 0; i < text.length; i += size) yield text.slice(i, i + size);
    })();
    let reconstructed = '';
    const changes = [];
    for await (const event of paragraphInstructions(source, text, ['calm', 'excited', null])) {
      if (typeof event === 'string') reconstructed += event;
      else if (event.type === 'instruct') changes.push([reconstructed.length, event.value]);
    }
    expect(reconstructed).toBe(text);
    expect(changes).toEqual([
      [0, 'calm'],
      [text.indexOf('第二'), 'excited'],
      [text.indexOf('Third'), null],
    ]);
  }
});
