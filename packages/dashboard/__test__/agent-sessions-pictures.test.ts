/**
 * Pictures in tool results in the shared reducer: a `read_file` of a picture or a PDF (mlx-agent v0.3.4
 * `acp_conversion.rs` `ReadFile`) gives `pictures` (sizes, never bytes) on the tool entry, the same from live lines and
 * from disk; the reducer keeps no base64; `toolPictureBlocks` finds a call's bytes in a whole file, rewinds applied.
 */

import { describe, expect, it } from 'vite-plus/test';

import {
  parseUpdates,
  slimRawOutput,
  toolPictureBlocks,
  TranscriptReducer,
  type UpdateLine,
} from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';

const SID = '01a20000-0000-7000-8000-0000000000f1';
const T0 = 1_791_643_279_000;

type Update = Record<string, unknown>;
type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;

/** Base64 of `bytes` bytes, each `fill`. */
function b64(bytes: number, fill = 0x41): string {
  return Buffer.alloc(bytes, fill).toString('base64');
}

const SHOT = b64(4_000, 0x42);
const PAGES = [b64(3_001, 0x43), b64(2_000, 0x44), b64(1_000, 0x45)];

function diskText(updates: readonly Update[], xai: readonly boolean[] = []): string {
  return updates
    .map(
      (update, n) =>
        `${JSON.stringify({
          timestamp: Math.floor(T0 / 1000),
          method: xai[n] === true ? '_x.ai/session/update' : 'session/update',
          params: { sessionId: SID, update, _meta: { eventId: `${SID}-${n}`, agentTimestampMs: T0 + n } },
        })}\n`,
    )
    .join('');
}

function liveLines(updates: readonly Update[]): UpdateLine[] {
  return updates.map((update, n) => ({ xai: false, update, meta: { agentTimestampMs: T0 + n } }));
}

function both(updates: readonly Update[]): { stored: Tool[]; live: Tool[]; reducer: TranscriptReducer } {
  const stored = parseUpdates(diskText(updates)).transcript.filter((e): e is Tool => e.kind === 'tool');
  const reducer = new TranscriptReducer();
  for (const line of liveLines(updates)) reducer.apply(line);
  const live = reducer.snapshot().entries.filter((e): e is Tool => e.kind === 'tool');
  return { stored, live, reducer };
}

const READ_TOOL = {
  version: 1,
  name: 'read_file',
  kind: 'read',
  namespace: 'grok_build',
  label: 'Read',
  read_only: true,
};

/** The three lines of a `read_file` call on the real agent: the call, its title update, its result. */
function readCall(id: string, path: string, result: Update): Update[] {
  return [
    {
      sessionUpdate: 'tool_call',
      toolCallId: id,
      title: 'read_file',
      rawInput: { target_file: path },
      _meta: { 'x.ai/tool': READ_TOOL },
    },
    {
      sessionUpdate: 'tool_call_update',
      toolCallId: id,
      kind: 'read',
      title: `Read \`${path}\``,
      locations: [{ path }],
      rawInput: { variant: 'ReadFile', target_file: path },
      _meta: { 'x.ai/tool': { ...READ_TOOL, input: { path } } },
    },
    { sessionUpdate: 'tool_call_update', toolCallId: id, ...result },
  ];
}

const image = (data: string, mimeType: string): Update => ({
  type: 'content',
  content: { type: 'image', data, mimeType },
});

const pictureResult: Update = {
  status: 'completed',
  content: [image(SHOT, 'image/png')],
  rawOutput: { type: 'ReadFile', ImageContent: { data: SHOT, mime_type: 'image/png' } },
};

const pdfResult: Update = {
  status: 'completed',
  content: PAGES.map((data) => image(data, 'image/jpeg')),
  rawOutput: {
    type: 'ReadFile',
    PdfPageImages: {
      pages: PAGES.map((data, i) => ({ data, mime_type: 'image/jpeg', page_number: 4 + i })),
      total_pages: 12,
      file_size: 1_468_006,
    },
  },
};

describe('readPictures', () => {
  it('one image block: pictures[0] with its decoded size, no output and no "[image]", live = stored', () => {
    const { stored, live } = both(readCall('call_png', 'docs/shot.png', pictureResult));
    expect(live).toEqual(stored);
    expect(live).toHaveLength(1);
    expect(live[0]!.pictures).toEqual([{ n: 0, mediaType: 'image/png', bytes: 4_000 }]);
    expect(live[0]!.pdfPages).toBeUndefined();
    expect(live[0]!.output).toBeUndefined();
    expect(JSON.stringify(live)).not.toContain('[image]');
    expect(JSON.stringify(live)).not.toContain('ImageContent');
  });

  it('a PDF: one picture a page, its page number from rawOutput, the file page count as pdfPages', () => {
    const { stored, live } = both(readCall('call_pdf', 'docs/spec.pdf', pdfResult));
    expect(live).toEqual(stored);
    expect(live[0]!.pictures).toEqual([
      { n: 0, mediaType: 'image/jpeg', bytes: 3_001, page: 4 },
      { n: 1, mediaType: 'image/jpeg', bytes: 2_000, page: 5 },
      { n: 2, mediaType: 'image/jpeg', bytes: 1_000, page: 6 },
    ]);
    expect(live[0]!.pdfPages).toBe(12);
    expect(live[0]!.output).toBeUndefined();
  });

  it('the entries, the reducer kept state and the parsed file hold no base64', () => {
    const updates = [...readCall('call_png', 'a.png', pictureResult), ...readCall('call_pdf', 'b.pdf', pdfResult)];
    const { reducer } = both(updates);
    const kept = JSON.stringify([...(Reflect.get(reducer, 'tools') as Map<string, unknown>).values()]);
    const entries = JSON.stringify(reducer.snapshot());
    const parsed = JSON.stringify(parseUpdates(diskText(updates)));
    for (const data of [SHOT, ...PAGES]) {
      const probe = data.slice(0, 40);
      expect(kept).not.toContain(probe);
      expect(entries).not.toContain(probe);
      expect(parsed).not.toContain(probe);
    }
    // What the result keeps of a PDF read: page numbers, page count and file size.
    expect(kept).toContain('"page_number":5');
    expect(kept).toContain('"total_pages":12');
    expect(kept).toContain('"file_size":1468006');
  });

  it('a text read and a picture that could not be sent stay text rows', () => {
    const text = both(
      readCall('call_txt', 'a.ts', {
        status: 'completed',
        content: [{ type: 'content', content: { type: 'text', text: '1→const a = 1;' } }],
        rawOutput: { type: 'ReadFile', FileContent: { content: '1→const a = 1;' } },
      }),
    ).live[0]!;
    expect(text.pictures).toBeUndefined();
    expect(text.output).toBe('1→const a = 1;');
    const failed = both(
      readCall('call_big', 'huge.png', {
        status: 'failed',
        content: [{ type: 'content', content: { type: 'text', text: 'Could not embed image in conversation: x' } }],
        rawOutput: { type: 'ReadFile', ImageSizeError: 'Could not embed image in conversation: x' },
      }),
    ).live[0]!;
    expect(failed.pictures).toBeUndefined();
    expect(failed.status).toBe('failed');
    expect(failed.output).toBe('Could not embed image in conversation: x');
  });

  it('a picture beside text keeps the text as output; a media type is cut at 100 characters', () => {
    const { live } = both(
      readCall('call_mix', 'a.png', {
        status: 'completed',
        content: [
          { type: 'content', content: { type: 'text', text: 'note' } },
          image('QUJD', `image/${'x'.repeat(200)}`),
        ],
      }),
    );
    expect(live[0]!.output).toBe('note');
    expect(live[0]!.pictures).toEqual([{ n: 0, mediaType: `image/${'x'.repeat(94)}`, bytes: 3 }]);
  });
});

describe('slimRawOutput', () => {
  it('drops every data of a ReadFile picture result and keeps the rest', () => {
    expect(slimRawOutput(pictureResult.rawOutput)).toEqual({
      type: 'ReadFile',
      ImageContent: { mime_type: 'image/png' },
    });
    expect(slimRawOutput(pdfResult.rawOutput)).toEqual({
      type: 'ReadFile',
      PdfPageImages: {
        pages: [4, 5, 6].map((page_number) => ({ mime_type: 'image/jpeg', page_number })),
        total_pages: 12,
        file_size: 1_468_006,
      },
    });
    const text = { type: 'ReadFile', FileContent: { content: 'x' } };
    expect(slimRawOutput(text)).toBe(text);
  });
});

describe('toolPictureBlocks', () => {
  const updates = [
    { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'look' }, _meta: { promptIndex: 0 } },
    ...readCall('call_png', 'docs/shot.png', pictureResult),
    ...readCall('call_pdf', 'docs/spec.pdf', pdfResult),
  ];

  it("gives a call's image blocks in order, compared by call id; none for another id", () => {
    const text = diskText(updates);
    expect(toolPictureBlocks(text, 'call_png')).toEqual([{ mediaType: 'image/png', data: SHOT }]);
    expect(toolPictureBlocks(text, 'call_pdf')?.map((b) => b.data)).toEqual(PAGES);
    expect(toolPictureBlocks(text, 'call_')).toBeNull();
    expect(toolPictureBlocks(text, 'call_txt')).toBeNull();
  });

  it("takes the call's newest line with image blocks, as the reducer does", () => {
    const again: Update = {
      sessionUpdate: 'tool_call_update',
      toolCallId: 'call_png',
      content: [image('TkVX', 'image/webp')],
    };
    const text = diskText([...updates, again]);
    expect(toolPictureBlocks(text, 'call_png')).toEqual([{ mediaType: 'image/webp', data: 'TkVX' }]);
    const reducer = new TranscriptReducer();
    reducer.applyText(text);
    const entry = reducer.snapshot().entries.find((e): e is Tool => e.kind === 'tool' && e.id === 'call_png');
    expect(entry?.pictures).toEqual([{ n: 0, mediaType: 'image/webp', bytes: 3 }]);
  });

  it('a call a rewind removed has no blocks', () => {
    const rewind = { sessionUpdate: 'rewind_marker', target_prompt_index: 0 };
    const text = diskText([...updates, rewind], [...updates.map(() => false), true]);
    expect(parseUpdates(text).transcript).toEqual([]);
    expect(toolPictureBlocks(text, 'call_png')).toBeNull();
  });
});
