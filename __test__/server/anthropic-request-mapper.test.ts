import { describe, expect, it } from 'vite-plus/test';

import {
  canonicalizeSystemForCacheKey,
  mapAnthropicRequest,
} from '../../packages/server/src/mappers/anthropic-request.js';
import type { AnthropicContentBlock } from '../../packages/server/src/types-anthropic.js';

describe('mapAnthropicRequest', () => {
  it('maps a simple string user message to a single user ChatMessage', () => {
    const { messages, config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(messages).toEqual([{ role: 'user', content: 'Hello' }]);
    expect(config.reportPerformance).toBe(true);
  });

  it('prepends system prompt string as first message', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      system: 'You are a helpful assistant.',
      messages: [{ role: 'user', content: 'Hi' }],
    });

    expect(messages).toEqual([
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'Hi' },
    ]);
  });

  it('prepends system prompt array of blocks as concatenated system message', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      system: [
        { type: 'text', text: 'You are helpful.' },
        { type: 'text', text: ' Be concise.' },
      ],
      messages: [{ role: 'user', content: 'Hi' }],
    });

    expect(messages).toEqual([
      { role: 'system', content: 'You are helpful. Be concise.' },
      { role: 'user', content: 'Hi' },
    ]);
  });

  it('drops Anthropic x-anthropic-billing-header system blocks before they reach the model', () => {
    // Claude Code injects a leading system block of the shape
    // `"x-anthropic-billing-header: cc_version=...; cch=<rotating-token>;"`
    // where the cch= token rotates per request. Leaving it in the prompt
    // defeats prefix caching at both the warm-slot gate and the native
    // token-prefix verifier. Mirrors vLLM's exact strip logic
    // (`vllm/entrypoints/anthropic/serving.py`, commit 262b76a0): drop any
    // text block whose content starts with the prefix.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      system: [
        { type: 'text', text: 'x-anthropic-billing-header: cc_version=2.1.119.806; cc_entrypoint=cli; cch=4a8a9;' },
        { type: 'text', text: 'You are Claude Code.' },
        { type: 'text', text: ' Be helpful.' },
      ],
      messages: [{ role: 'user', content: 'Hi' }],
    });

    expect(messages).toEqual([
      { role: 'system', content: 'You are Claude Code. Be helpful.' },
      { role: 'user', content: 'Hi' },
    ]);
  });

  it('does NOT strip a string-typed system field even when it starts with the billing header', () => {
    // String-system path is unfiltered (mirrors vLLM, which only strips at
    // the block level). Real Claude Code traffic always sends the header in
    // an array block, so a string-system caller copying the prefix is on
    // their own — but the contract is to leave string `system` alone.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      system: 'x-anthropic-billing-header: cch=AAAA; You are Claude.',
      messages: [{ role: 'user', content: 'Hi' }],
    });

    expect(messages).toEqual([
      { role: 'system', content: 'x-anthropic-billing-header: cch=AAAA; You are Claude.' },
      { role: 'user', content: 'Hi' },
    ]);
  });

  describe('canonicalizeSystemForCacheKey', () => {
    it('returns null for a missing system field', () => {
      expect(canonicalizeSystemForCacheKey(undefined)).toBeNull();
    });

    it('passes through a string-typed system field unchanged', () => {
      // String path mirrors the mapper's string branch — no filtering.
      expect(canonicalizeSystemForCacheKey('You are helpful.')).toBe('You are helpful.');
    });

    it('produces the same joined text the mapper bakes into the system message (billing header dropped)', () => {
      // Cross-check: the cache-key view MUST equal the mapper's mapped
      // system content for the same input, otherwise the warm-slot gate
      // misses on requests the model would otherwise treat as identical.
      const system = [
        { type: 'text' as const, text: 'x-anthropic-billing-header: cc_version=...; cch=ROT1;' },
        { type: 'text' as const, text: 'You are Claude.' },
        { type: 'text' as const, text: ' Be concise.' },
      ];
      const cacheKey = canonicalizeSystemForCacheKey(system);
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        system,
        messages: [{ role: 'user', content: 'Hi' }],
      });
      expect(cacheKey).toBe('You are Claude. Be concise.');
      expect(messages[0]).toEqual({ role: 'system', content: cacheKey });
    });

    it('matches across rotating cch= tokens (key-stability witness)', () => {
      // The whole point: the cache key MUST be byte-identical across two
      // turns whose only difference is the rotating cch= token in the
      // first block. A regression that leaves the header in would yield
      // distinct cache keys here.
      const t1 = canonicalizeSystemForCacheKey([
        { type: 'text', text: 'x-anthropic-billing-header: cch=AAAA;' },
        { type: 'text', text: 'You are Claude.' },
      ]);
      const t2 = canonicalizeSystemForCacheKey([
        { type: 'text', text: 'x-anthropic-billing-header: cch=BBBB;' },
        { type: 'text', text: 'You are Claude.' },
      ]);
      expect(t1).toBe('You are Claude.');
      expect(t1).toBe(t2);
    });

    it('returns null when array system contains only billing header blocks', () => {
      // A request whose only system block is the rotating Anthropic
      // billing-header line is semantically equivalent to "no system" —
      // collapse to `null` rather than the empty string so (a) the
      // chat template does not emit two extra wrapper tokens for an
      // empty `system` message, and (b) the warm-slot cache key compares
      // byte-equal to an absent-system request (`null === null`, where
      // `'' !== null` would otherwise miss the slot).
      expect(
        canonicalizeSystemForCacheKey([{ type: 'text', text: 'x-anthropic-billing-header: cc_version=2.1.119.806;' }]),
      ).toBeNull();
    });

    it('all-stripped array is equivalent to absent system for cache-key purposes', () => {
      // Cache-key invariant: a request whose system field is absent
      // and a request whose only system block is the rotating
      // billing-header line MUST produce the same cache key (both
      // `null`). Any divergence here re-introduces the warm-slot drift
      // this fix targets.
      expect(canonicalizeSystemForCacheKey(undefined)).toBe(
        canonicalizeSystemForCacheKey([
          { type: 'text', text: 'x-anthropic-billing-header: cc_version=2.1.119.806; cch=ROT1;' },
        ]),
      );
    });
  });

  it('mapAnthropicRequest emits no system message when all blocks are stripped', () => {
    // When every system block is stripped (here the only block is the
    // rotating `x-anthropic-billing-header` line), the mapper MUST NOT
    // push a `{ role: 'system', content: '' }` placeholder. The chat
    // template wraps every pushed message with `<|im_start|>{role}\n
    // {content}<|im_end|>\n`, so an empty system content emits two extra
    // wrapper tokens that an absent-system request would not — perturbing
    // the prefix and breaking prefix caching across two semantically
    // equivalent requests.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      system: [{ type: 'text', text: 'x-anthropic-billing-header: cc_version=2.1.119.806; cch=4a8a9;' }],
      messages: [{ role: 'user', content: 'Hi' }],
    });

    expect(messages.some((m) => m.role === 'system')).toBe(false);
    expect(messages).toEqual([{ role: 'user', content: 'Hi' }]);
  });

  it('maps multi-turn conversation (user → assistant → user)', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        { role: 'user', content: 'What is 2+2?' },
        { role: 'assistant', content: '4' },
        { role: 'user', content: 'Are you sure?' },
      ],
    });

    expect(messages).toEqual([
      { role: 'user', content: 'What is 2+2?' },
      { role: 'assistant', content: '4' },
      { role: 'user', content: 'Are you sure?' },
    ]);
  });

  it('maps user message with tool_result content blocks to tool role messages', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'call_abc', content: '72F and sunny' }],
        },
      ],
    });

    expect(messages).toEqual([{ role: 'tool', content: '72F and sunny', toolCallId: 'call_abc' }]);
  });

  it('rejects text/image blocks that precede a tool_result in the same user turn', () => {
    // A tool_result must appear as a CONTIGUOUS PREFIX of the user
    // turn. Text/image blocks that appear BEFORE any tool_result
    // are still rejected — emitting them in-place would put a
    // user `content` message between the preceding assistant
    // fan-out and the tool_result it is answering, orphaning the
    // fan-out. The mapper also cannot reorder blocks to satisfy
    // the prefix invariant without silently changing authorship.
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Here is the weather result:' },
              { type: 'tool_result', tool_use_id: 'call_123', content: 'Rainy' },
              { type: 'tool_result', tool_use_id: 'call_456', content: 'Sunny' },
            ],
          },
        ],
      }),
    ).toThrow(/tool_result blocks must appear as a contiguous prefix/i);
  });

  it('rejects an image block that precedes a tool_result in the same user turn', () => {
    // Same prefix invariant as the text-before-tool_result case
    // above — an image block before a tool_result is still
    // rejected because it would orphan the fan-out.
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
              { type: 'tool_result', tool_use_id: 'call_abc', content: 'ok' },
            ],
          },
        ],
      }),
    ).toThrow(/tool_result blocks must appear as a contiguous prefix/i);
  });

  it('accepts tool_result followed by trailing text and emits a tool block + user turn', () => {
    // Accept the common Anthropic shape `[tool_result, text("now do X")]`: emit the
    // tool message(s) first, then a trailing user message carrying text/image content.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_123', content: 'Sunny, 72F' },
            { type: 'text', text: 'Now tell me a joke about the weather.' },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      { role: 'tool', content: 'Sunny, 72F', toolCallId: 'call_123' },
      { role: 'user', content: 'Now tell me a joke about the weather.' },
    ]);
  });

  it('accepts multiple tool_result blocks followed by a trailing image', () => {
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_a', content: 'one' },
            { type: 'tool_result', tool_use_id: 'call_b', content: 'two' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(3);
    expect(messages[0]).toEqual({ role: 'tool', content: 'one', toolCallId: 'call_a' });
    expect(messages[1]).toEqual({ role: 'tool', content: 'two', toolCallId: 'call_b' });
    expect(messages[2].role).toBe('user');
    expect(messages[2].content).toBe('');
    expect(messages[2].images).toHaveLength(1);
  });

  it('rejects a tool_result whose content array carries a nested image block', () => {
    // Anthropic allows `tool_result.content` to mix text and image blocks, but the
    // internal `ChatMessage` shape is a NAPI-generated Rust struct with no `images`
    // field on `role: 'tool'`. Refuse this shape outright — hoisting the image onto
    // a trailing user turn would reorder images relative to the caller's declaration
    // and lose the per-tool association after canonicalization reorders tool rows.
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'call_screenshot',
                content: [
                  { type: 'text', text: 'Here is the screenshot you asked for:' },
                  { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
                ],
              },
            ],
          },
        ],
      }),
    ).toThrow(/nested image content in tool_result blocks is not representable/i);
  });

  it('rejects a tool_result nested image even when another tool_result in the same turn is plain text', () => {
    // When the caller mixes a plain-text tool_result with one that carries a nested
    // image, reject the whole turn rather than silently hoisting the image onto a
    // trailing user message that loses its owner after canonicalization.
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_ok', content: 'ok' },
              {
                type: 'tool_result',
                tool_use_id: 'call_shot',
                content: [
                  { type: 'text', text: 'screenshot:' },
                  { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
                ],
              },
            ],
          },
        ],
      }),
    ).toThrow(/nested image content in tool_result blocks is not representable/i);
  });

  // The flat NAPI `ChatMessage` shape cannot represent interleaved text/image blocks
  // after a tool_result prefix, so the mapper rejects ambiguous shapes. These tests
  // pin the narrow accepted set (pure trailing text, single trailing image) and pin
  // rejection for every interleaved / multi-image shape.
  const iter28Png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

  it('rejects trailing text followed by an image after a tool_result prefix', () => {
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
              { type: 'text', text: 'see the attached screenshot' },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
            ],
          },
        ],
      }),
    ).toThrow(/mixing trailing text and image blocks after a tool_result prefix/i);
  });

  it('rejects trailing image followed by text after a tool_result prefix', () => {
    // The text-after-image guard fires during the main loop (before the
    // trailing-mixed check) because the serializer cannot preserve text
    // that follows an image in any turn, not just after a tool_result
    // prefix. Either rejection is equivalent for the caller.
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
              { type: 'text', text: 'that was the screenshot' },
            ],
          },
        ],
      }),
    ).toThrow(/text block after an image block in the same user turn/i);
  });

  it('rejects multiple trailing image blocks after a tool_result prefix', () => {
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
            ],
          },
        ],
      }),
    ).toThrow(/multiple trailing image blocks after a tool_result prefix/i);
  });

  it('accepts a single trailing text block after a tool_result prefix', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
            { type: 'text', text: 'now summarise the above' },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(2);
    expect(messages[0]).toEqual({ role: 'tool', content: 'alpha', toolCallId: 'call_a' });
    expect(messages[1]).toEqual({ role: 'user', content: 'now summarise the above' });
  });

  it('accepts a single trailing image block after a tool_result prefix', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: iter28Png } },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(2);
    expect(messages[0]).toEqual({ role: 'tool', content: 'alpha', toolCallId: 'call_a' });
    expect(messages[1].role).toBe('user');
    expect(messages[1].content).toBe('');
    expect(messages[1].images).toHaveLength(1);
    expect(messages[1].images![0]).toEqual(Buffer.from(iter28Png, 'base64'));
  });

  it('accepts multiple trailing text blocks after a tool_result prefix (concatenated)', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_a', content: 'alpha' },
            { type: 'text', text: 'part one ' },
            { type: 'text', text: 'part two' },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(2);
    expect(messages[0]).toEqual({ role: 'tool', content: 'alpha', toolCallId: 'call_a' });
    expect(messages[1]).toEqual({ role: 'user', content: 'part one part two' });
  });

  it('maps a pure-tool_result user turn to a contiguous tool block (counter-test)', () => {
    // A user turn whose blocks are ALL tool_result maps cleanly to contiguous `tool`
    // ChatMessage blocks in caller-supplied order (downstream canonicalization then
    // reorders against the assistant's declared sibling order).
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_123', content: 'Rainy' },
            { type: 'tool_result', tool_use_id: 'call_456', content: 'Sunny' },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      { role: 'tool', content: 'Rainy', toolCallId: 'call_123' },
      { role: 'tool', content: 'Sunny', toolCallId: 'call_456' },
    ]);
  });

  it('maps a pure text/image user turn to a single user message (counter-test)', () => {
    // A user turn containing only text and image blocks maps to a single `user`
    // ChatMessage with both fields populated.
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What is in this image?' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    expect(messages[0].role).toBe('user');
    expect(messages[0].content).toBe('What is in this image?');
    expect(messages[0].images).toHaveLength(1);
  });

  it('rewrites incoming toolu_<uuid> ids back to internal call_<uuid> on tool_result blocks', () => {
    // Anthropic-spec clients echo back the same `toolu_<uuid>` we emitted
    // on the prior assistant turn. The internal session store and the
    // native session continue paths see `call_<uuid>` ids, so the
    // mapper must translate inbound `toolu_*` back to `call_*` to keep
    // historical tool_call_id lookups matching.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'toolu_abc123', content: 'sunny' }],
        },
      ],
    });
    expect(messages).toEqual([{ role: 'tool', content: 'sunny', toolCallId: 'call_abc123' }]);
  });

  it('rewrites incoming toolu_<uuid> ids on assistant.tool_use blocks too', () => {
    // Assistant turns the client echoes back carry the `toolu_*` ids
    // we previously emitted. The mapper must translate them back so
    // the internal `ChatMessage.toolCalls[].id` stays in the `call_*`
    // family the native paths expect.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'assistant',
          content: [{ type: 'tool_use', id: 'toolu_xyz789', name: 'get_weather', input: { city: 'SF' } }],
        },
      ],
    });
    expect(messages).toEqual([
      {
        role: 'assistant',
        content: '',
        toolCalls: [{ id: 'call_xyz789', name: 'get_weather', arguments: '{"city":"SF"}' }],
      },
    ]);
  });

  it('passes through tool_use_id values that do not have the toolu_ prefix', () => {
    // Defensive: legacy callers that already speak the internal `call_*`
    // shape (or any other unrecognised id family) must round-trip
    // unchanged.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'call_already', content: 'ok' }],
        },
      ],
    });
    expect(messages).toEqual([{ role: 'tool', content: 'ok', toolCallId: 'call_already' }]);
  });

  it('encodes tool_result.is_error=true via the structured isError field, leaving content verbatim', () => {
    // Earlier revisions used an in-band `[error] ` content prefix to
    // surface tool-call failure. Codex review flagged the in-band marker
    // as theoretically collision-prone — a successful tool_result whose
    // literal content begins with the same prefix would be
    // indistinguishable from an errored one on later read-back. The fix
    // is the structured `isError` field on `ChatMessage` (mirroring
    // `toolCallId`): the mapper passes content through unchanged and
    // surfaces the error condition on a dedicated field, eliminating
    // the read-back ambiguity. The wire-format renderers on the Rust
    // side still inject a `[tool error]` cue into the model prompt at
    // serialization time, but the structured field stays the source of
    // truth and the original payload is preserved byte-for-byte.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'call_fail',
              content: 'boom: connection refused',
              is_error: true,
            },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      {
        role: 'tool',
        content: 'boom: connection refused',
        toolCallId: 'call_fail',
        isError: true,
      },
    ]);
  });

  it('preserves a JSON tool_result payload verbatim under isError=true', () => {
    // The structured `isError` field is the failure signal; `content`
    // stays byte-for-byte equal to the original payload so downstream
    // consumers (replay logic, audit logs, the model's own template
    // rendering) get the unmodified bytes back.
    const jsonPayload = '{"error_code":500,"message":"upstream unavailable"}';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'call_json_fail',
              content: jsonPayload,
              is_error: true,
            },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const toolMsg = messages[0]!;
    expect(toolMsg.role).toBe('tool');
    expect(toolMsg.toolCallId).toBe('call_json_fail');
    expect(toolMsg.content).toBe(jsonPayload);
    expect(toolMsg.isError).toBe(true);
  });

  it('does not set isError on successful tool_result content even when it resembles a legacy error marker', () => {
    // A successful tool_result whose content happens to start with the
    // legacy `[error]` text MUST NOT be re-marked. With the structured
    // field, this case can no longer be confused with an errored result —
    // `isError` is undefined (or false) on success regardless of what the
    // content text looks like. Guard the round-trip explicitly.
    const suspicious = '[error] this was supplied by a tool that returned success';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'call_ok', content: suspicious, is_error: false }],
        },
      ],
    });

    expect(messages).toEqual([{ role: 'tool', content: suspicious, toolCallId: 'call_ok' }]);
    expect(messages[0]!.isError).toBeUndefined();
  });

  it('leaves tool_result content untouched and isError unset when is_error is absent or false', () => {
    // Successful tool_result entries — including ones whose content text
    // happens to begin with `[error] ` or the new model-facing
    // `[tool error] ` cue — must survive verbatim and MUST NOT carry an
    // `isError` flag.
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'call_ok_1', content: '72F and sunny', is_error: false },
            { type: 'tool_result', tool_use_id: 'call_ok_2', content: '68F and cloudy' },
            { type: 'tool_result', tool_use_id: 'call_ok_3', content: '[error] this is a legitimate payload' },
            { type: 'tool_result', tool_use_id: 'call_ok_4', content: '[tool error] also legitimate' },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      { role: 'tool', content: '72F and sunny', toolCallId: 'call_ok_1' },
      { role: 'tool', content: '68F and cloudy', toolCallId: 'call_ok_2' },
      { role: 'tool', content: '[error] this is a legitimate payload', toolCallId: 'call_ok_3' },
      { role: 'tool', content: '[tool error] also legitimate', toolCallId: 'call_ok_4' },
    ]);
    for (const m of messages) {
      expect(m.isError).toBeUndefined();
    }
  });

  it('maps assistant message with tool_use to assistant with toolCalls', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'call_xyz',
              name: 'get_weather',
              input: { city: 'San Francisco' },
            },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      {
        role: 'assistant',
        content: '',
        toolCalls: [{ id: 'call_xyz', name: 'get_weather', arguments: '{"city":"San Francisco"}' }],
      },
    ]);
  });

  it('maps assistant message with thinking to assistant with reasoningContent', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'Let me reason through this...' },
            { type: 'text', text: 'The answer is 42.' },
          ],
        },
      ],
    });

    expect(messages).toEqual([
      {
        role: 'assistant',
        content: 'The answer is 42.',
        reasoningContent: 'Let me reason through this...',
      },
    ]);
  });

  it('maps mixed assistant message (thinking + text + tool_use) into a single message', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'I should call the weather tool.' },
            { type: 'text', text: 'Let me check the weather.' },
            {
              type: 'tool_use',
              id: 'call_abc',
              name: 'get_weather',
              input: { city: 'NYC' },
            },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const msg = messages[0];
    expect(msg.role).toBe('assistant');
    expect(msg.content).toBe('Let me check the weather.');
    expect(msg.reasoningContent).toBe('I should call the weather tool.');
    expect(msg.toolCalls).toEqual([{ id: 'call_abc', name: 'get_weather', arguments: '{"city":"NYC"}' }]);
  });

  it('maps tool definition from Anthropic input_schema to internal format with JSON.stringify', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tools: [
        {
          name: 'get_weather',
          description: 'Get the weather for a city',
          input_schema: {
            type: 'object',
            properties: { city: { type: 'string', description: 'City name' } },
            required: ['city'],
          },
        },
      ],
    });

    expect(config.tools).toHaveLength(1);
    const tool = config.tools![0];
    expect(tool.type).toBe('function');
    expect(tool.function.name).toBe('get_weather');
    expect(tool.function.description).toBe('Get the weather for a city');
    expect(tool.function.parameters).toEqual({
      type: 'object',
      properties: JSON.stringify({ city: { type: 'string', description: 'City name' } }),
      required: ['city'],
    });
  });

  it('maps tool choice auto → passes all tools', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tool_choice: { type: 'auto' },
      tools: [
        { name: 'tool_a', input_schema: {} },
        { name: 'tool_b', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('maps tool choice any → passes all tools', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tool_choice: { type: 'any' },
      tools: [
        { name: 'tool_a', input_schema: {} },
        { name: 'tool_b', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('maps tool choice tool with name → filters to only the named tool', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tool_choice: { type: 'tool', name: 'tool_b' },
      tools: [
        { name: 'tool_a', input_schema: {} },
        { name: 'tool_b', input_schema: {} },
        { name: 'tool_c', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(1);
    expect(config.tools![0].function.name).toBe('tool_b');
  });

  it('passes all tools when tool_choice is absent', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tools: [
        { name: 'tool_a', input_schema: {} },
        { name: 'tool_b', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('rejects tool_choice that names a tool not present in tools[] (silent contract violation)', () => {
    // Anthropic semantics: `{type:'tool', name:'X'}` is a HARD constraint —
    // the model MUST call X and only X. Previously, an unmatched name fell
    // through to the all-tools branch and the model silently received every
    // tool. That violates the caller's contract; reject with a 400 instead.
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hello' }],
        tool_choice: { type: 'tool', name: 'C' },
        tools: [
          { name: 'A', input_schema: {} },
          { name: 'B', input_schema: {} },
        ],
      }),
    ).toThrow(/"C".*tools list/i);
  });

  it('rejects tool_choice with type=tool but no name', () => {
    // Malformed shape — without a name we cannot select a tool, and falling
    // through to the all-tools path would silently accept the request.
    expect(() =>
      mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hello' }],
        tool_choice: { type: 'tool' },
        tools: [
          { name: 'A', input_schema: {} },
          { name: 'B', input_schema: {} },
        ],
      }),
    ).toThrow(/no name was provided/i);
  });

  it('regression: tool_choice tool with matching name selects only that tool', () => {
    // Pin the happy path against the rejection logic added above.
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tool_choice: { type: 'tool', name: 'B' },
      tools: [
        { name: 'A', input_schema: {} },
        { name: 'B', input_schema: {} },
        { name: 'C', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(1);
    expect(config.tools![0].function.name).toBe('B');
  });

  it('regression: tool_choice type=auto sends all tools', () => {
    // Pin the auto branch — the rejection logic for `type=tool` MUST NOT
    // affect `type=auto`, which should continue to forward every tool.
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      tool_choice: { type: 'auto' },
      tools: [
        { name: 'A', input_schema: {} },
        { name: 'B', input_schema: {} },
      ],
    });

    expect(config.tools).toHaveLength(2);
  });

  it('accepts non-empty stop_sequences and threads them out as stopSequences', () => {
    // `stop_sequences` is now carried out of the mapper (on the widened
    // return) so a downstream consumer can honour the stop strings. It must
    // NOT be added to `ChatConfig` (that would need a native rebuild).
    const { config, stopSequences } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      stop_sequences: ['STOP'],
    });
    expect(stopSequences).toEqual(['STOP']);
    expect(config).not.toHaveProperty('stopSequences');
    expect(config).not.toHaveProperty('stop');
  });

  it('treats empty stop_sequences array as no stop strings', () => {
    const { stopSequences } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      stop_sequences: [],
    });
    expect(stopSequences).toEqual([]);
  });

  it('treats absent stop_sequences as no stop strings', () => {
    const { stopSequences } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
    });
    expect(stopSequences).toEqual([]);
  });

  it('filters empty strings out of stop_sequences', () => {
    const { stopSequences } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      stop_sequences: ['', 'X'],
    });
    expect(stopSequences).toEqual(['X']);
  });

  it('filters whitespace-only strings out of stop_sequences', () => {
    // The real Anthropic API rejects whitespace-only stops with a 400, and
    // keeping them live would silently truncate normal output at the first
    // space/newline. Dropping them makes such entries a no-op.
    const { stopSequences } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      stop_sequences: [' ', '\n', '\t', '  ', 'X'],
    });
    expect(stopSequences).toEqual(['X']);
  });

  it('treats an all-whitespace stop_sequences array as no stop strings', () => {
    const { stopSequences } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
      stop_sequences: [' ', '\n'],
    });
    expect(stopSequences).toEqual([]);
  });

  it('maps max_tokens to maxNewTokens in config', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 512,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.maxNewTokens).toBe(512);
  });

  it('maps temperature to config', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      temperature: 0.7,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.temperature).toBe(0.7);
  });

  it('maps top_p to topP in config', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      top_p: 0.9,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.topP).toBe(0.9);
  });

  it('maps top_k to topK in config', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      top_k: 50,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.topK).toBe(50);
  });

  it('always sets reportPerformance to true', () => {
    const { config } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [{ role: 'user', content: 'Hello' }],
    });

    expect(config.reportPerformance).toBe(true);
  });

  it('maps content array with only text blocks to concatenated text', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Part one. ' },
            { type: 'text', text: 'Part two.' },
          ],
        },
      ],
    });

    expect(messages).toEqual([{ role: 'user', content: 'Part one. Part two.' }]);
  });

  it('maps user message with a single image block to images array with decoded Uint8Array', () => {
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } }],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const msg = messages[0];
    expect(msg.role).toBe('user');
    expect(msg.content).toBe('');
    expect(msg.images).toHaveLength(1);
    expect(msg.images![0]).toEqual(Buffer.from(imageData, 'base64'));
  });

  it('maps user message with text + image to content and images populated', () => {
    const imageData =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'What is in this image?' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } },
          ],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const msg = messages[0];
    expect(msg.role).toBe('user');
    expect(msg.content).toBe('What is in this image?');
    expect(msg.images).toHaveLength(1);
    expect(msg.images![0]).toEqual(Buffer.from(imageData, 'base64'));
  });

  it('maps user message with only image (no text) to empty content with images', () => {
    const imageData = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoH';
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [{ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: imageData } }],
        },
      ],
    });

    expect(messages).toHaveLength(1);
    const msg = messages[0];
    expect(msg.role).toBe('user');
    expect(msg.content).toBe('');
    expect(msg.images).toBeDefined();
    expect(msg.images).toHaveLength(1);
    expect(msg.images![0]).toBeInstanceOf(Uint8Array);
    expect(msg.images![0]).toEqual(Buffer.from(imageData, 'base64'));
  });

  it('maps tool_result with text block array content', () => {
    const { messages } = mapAnthropicRequest({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'call_789',
              content: [
                { type: 'text', text: 'Result: ' },
                { type: 'text', text: 'success' },
              ],
            },
          ],
        },
      ],
    });

    expect(messages).toEqual([{ role: 'tool', content: 'Result: success', toolCallId: 'call_789' }]);
  });

  describe('text/image ordering in pure user turns', () => {
    // Existing tool_result-prefix rejection at line ~160 catches mixed
    // trailing content after tool_result, but a PURE user turn (no
    // tool_result blocks) was previously silently concatenating all text
    // and stacking all images, reordering the caller's content. These
    // tests pin the uniform rejection for both call patterns.
    const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

    it('rejects [image, text] in a pure user turn', () => {
      // Image-first-then-text gets reordered to text-first-then-image by
      // the flat ChatMessage + Jinja serializer pipeline. Reject rather
      // than silently rewrite the caller's intent.
      expect(() =>
        mapAnthropicRequest({
          model: 'claude-3-5-sonnet-20241022',
          max_tokens: 1024,
          messages: [
            {
              role: 'user',
              content: [
                { type: 'image', source: { type: 'base64', media_type: 'image/png', data: png } },
                { type: 'text', text: 'describe this' },
              ],
            },
          ],
        }),
      ).toThrow(/text block after an image block in the same user turn/i);
    });

    it('rejects [text, image, text] in a pure user turn', () => {
      expect(() =>
        mapAnthropicRequest({
          model: 'claude-3-5-sonnet-20241022',
          max_tokens: 1024,
          messages: [
            {
              role: 'user',
              content: [
                { type: 'text', text: 'before' },
                { type: 'image', source: { type: 'base64', media_type: 'image/png', data: png } },
                { type: 'text', text: 'after' },
              ],
            },
          ],
        }),
      ).toThrow(/text block after an image block/i);
    });

    it('accepts [text, image] in a pure user turn (representable by the flat ChatMessage)', () => {
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'what colour is this?' },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: png } },
            ],
          },
        ],
      });
      expect(messages).toHaveLength(1);
      expect(messages[0].role).toBe('user');
      expect(messages[0].content).toBe('what colour is this?');
      expect(messages[0].images).toHaveLength(1);
    });

    it('accepts [text, text, image] (all text parts before the image)', () => {
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'part one. ' },
              { type: 'text', text: 'part two.' },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: png } },
            ],
          },
        ],
      });
      expect(messages).toHaveLength(1);
      expect(messages[0].content).toBe('part one. part two.');
      expect(messages[0].images).toHaveLength(1);
    });

    it('accepts multiple images with no text (no ordering ambiguity)', () => {
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: png } },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: png } },
            ],
          },
        ],
      });
      expect(messages).toHaveLength(1);
      expect(messages[0].content).toBe('');
      expect(messages[0].images).toHaveLength(2);
    });
  });

  // -------------------------------------------------------------------
  // W7 (MTP): `extra_body.generation_mode` + `extra_body.mtp_depth`
  // -------------------------------------------------------------------
  describe('extra_body MTP overrides', () => {
    it('maps generation_mode "mtp" to enableMtp=true', () => {
      const { config } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hello' }],
        extra_body: { generation_mode: 'mtp' },
      });
      expect(config.enableMtp).toBe(true);
    });

    it('maps generation_mode "ar" to enableMtp=false', () => {
      const { config } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hello' }],
        extra_body: { generation_mode: 'ar' },
      });
      expect(config.enableMtp).toBe(false);
    });

    it('leaves enableMtp untouched when extra_body is absent', () => {
      const { config } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hello' }],
      });
      expect(config.enableMtp).toBeUndefined();
    });

    it('forwards a valid mtp_depth onto config.mtpDepth', () => {
      const { config } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hello' }],
        extra_body: { mtp_depth: 2 },
      });
      expect(config.mtpDepth).toBe(2);
    });

    it('forwards mtp_depth 8 (gemma4 assistant max; native owns per-family clamps)', () => {
      const { config } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hello' }],
        extra_body: { mtp_depth: 8 },
      });
      expect(config.mtpDepth).toBe(8);
    });

    it('rejects non-integer, non-positive, and > 64 mtp_depth values', () => {
      for (const depth of [0, -1, 65, 2.5, Number.NaN, null]) {
        const { config } = mapAnthropicRequest({
          model: 'claude-3-5-sonnet-20241022',
          max_tokens: 1024,
          messages: [{ role: 'user', content: 'Hello' }],
          extra_body: { mtp_depth: depth as never },
        });
        expect(config.mtpDepth).toBeUndefined();
      }
    });
  });

  // -------------------------------------------------------------------
  // system-role message folding
  //
  // `system` is not a role in the Anthropic Messages spec, but Claude
  // Code's SessionStart hooks (e.g. superpowers) inject a
  // `{ role: 'system' }` message carrying "additional context" into the
  // `messages` array. The mapper previously rejected this with HTTP 400
  // ("Unsupported message role"). It now folds such a message's text into
  // the single leading system prompt instead. The content is positionless
  // additional context, so its location in the array does not matter — we
  // accumulate in encounter order and prepend after the message loop.
  // -------------------------------------------------------------------
  describe('system-role message folding', () => {
    it('folds a trailing system-role message into a leading system prompt (the 400 repro)', () => {
      // Exact shape that produced `400 Unsupported message role: "system"`:
      // a SessionStart hook appends a `{ role: 'system' }` message after the
      // user turn. It must now be accepted and hoisted to the front.
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'system', content: 'Additional context: the repo is mlx-node.' },
        ],
      });

      expect(messages).toEqual([
        { role: 'system', content: 'Additional context: the repo is mlx-node.' },
        { role: 'user', content: 'Hi' },
      ]);
    });

    it('joins a top-level system and a folded system-role message with a blank line', () => {
      // The separator between distinct contributions is `'\n\n'`. Pin it.
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        system: 'You are a helpful assistant.',
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'system', content: 'Session note: be concise.' },
        ],
      });

      expect(messages).toEqual([
        { role: 'system', content: 'You are a helpful assistant.\n\nSession note: be concise.' },
        { role: 'user', content: 'Hi' },
      ]);
    });

    it('folds multiple system-role messages in encounter order after the top-level system', () => {
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        system: 'Base prompt.',
        messages: [
          { role: 'system', content: 'context A' },
          { role: 'user', content: 'Hi' },
          { role: 'system', content: 'context B' },
        ],
      });

      expect(messages).toEqual([
        { role: 'system', content: 'Base prompt.\n\ncontext A\n\ncontext B' },
        { role: 'user', content: 'Hi' },
      ]);
    });

    it('concatenates a system-role array content with no separator within the message, tolerating extra block fields', () => {
      // Within a single array-content message, text blocks join with `''`
      // (matching the top-level `system`-array behaviour); the `'\n\n'`
      // separator only appears BETWEEN distinct contributions. Extra block
      // fields a client may attach (e.g. `cache_control`) are ignored — the
      // cast models the wire shape Claude Code can send even though the
      // text-block type does not declare the field.
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        system: 'Base prompt.',
        messages: [
          {
            role: 'system',
            content: [
              { type: 'text', text: 'part one ', cache_control: { type: 'ephemeral' } },
              { type: 'text', text: 'part two' },
            ] as unknown as AnthropicContentBlock[],
          },
          { role: 'user', content: 'Hi' },
        ],
      });

      expect(messages).toEqual([
        { role: 'system', content: 'Base prompt.\n\npart one part two' },
        { role: 'user', content: 'Hi' },
      ]);
    });

    it('hoists a mid-conversation system-role message to the front while preserving the rest of the order', () => {
      // CONTRACT: a system-role message is positionless and hoisted to the
      // single leading system prompt regardless of where it appears. This is
      // deliberate — the Anthropic wire format defines no positional `system`
      // role, the only known producer (Claude Code SessionStart hooks) emits
      // positionless additional context, and the internal
      // `ChatMessage`/`primeHistory` pipeline can represent only a single
      // leading system message. See the matching block comment in the mapper.
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          { role: 'user', content: 'What is 2+2?' },
          { role: 'assistant', content: '4' },
          { role: 'system', content: 'midstream note' },
          { role: 'user', content: 'Are you sure?' },
        ],
      });

      expect(messages).toEqual([
        { role: 'system', content: 'midstream note' },
        { role: 'user', content: 'What is 2+2?' },
        { role: 'assistant', content: '4' },
        { role: 'user', content: 'Are you sure?' },
      ]);
    });

    it('drops an empty system-role message rather than corrupting the system prompt with a trailing blank line', () => {
      // An empty hook context message must neither append a dangling `'\n\n'`
      // to a real system prompt nor synthesise a bare empty system message.
      const withTopLevel = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        system: 'Real system prompt.',
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'system', content: '' },
        ],
      });
      expect(withTopLevel.messages).toEqual([
        { role: 'system', content: 'Real system prompt.' },
        { role: 'user', content: 'Hi' },
      ]);

      // With no top-level system and only an empty folded message, no system
      // message is emitted at all (mirrors the all-stripped-array contract).
      const withoutTopLevel = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        messages: [
          { role: 'user', content: 'Hi' },
          { role: 'system', content: '' },
        ],
      });
      expect(withoutTopLevel.messages).toEqual([{ role: 'user', content: 'Hi' }]);
    });

    it('rejects a non-text content block inside a system-role message', () => {
      const imageData =
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
      expect(() =>
        mapAnthropicRequest({
          model: 'claude-3-5-sonnet-20241022',
          max_tokens: 1024,
          messages: [
            {
              role: 'system',
              content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: imageData } }],
            },
          ],
        }),
      ).toThrow(/Unsupported content block type "image" in system-role message/i);
    });

    it('regression: a request with no system-role message is unaffected by the folding path', () => {
      // The folding path must be inert for normal traffic — a top-level
      // system plus an ordinary multi-turn conversation maps exactly as
      // before, with no `'\n\n'` artifacts introduced.
      const { messages } = mapAnthropicRequest({
        model: 'claude-3-5-sonnet-20241022',
        max_tokens: 1024,
        system: 'You are a helpful assistant.',
        messages: [
          { role: 'user', content: 'What is 2+2?' },
          { role: 'assistant', content: '4' },
          { role: 'user', content: 'Are you sure?' },
        ],
      });

      expect(messages).toEqual([
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is 2+2?' },
        { role: 'assistant', content: '4' },
        { role: 'user', content: 'Are you sure?' },
      ]);
    });
  });
});
