//! The generation prompt a chat render appends, and the replay point a turn
//! resumes from when the next request re-renders the same conversation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::tokenizer::RenderInputs;

/// Probed generation suffixes of one tokenizer, keyed by the full render
/// inputs. A `None` result is cached as well.
pub(crate) type GenerationSuffixCache = Mutex<HashMap<RenderInputs, Option<Arc<[u32]>>>>;

/// `with_prompt[without_prompt.len()..]` when `without_prompt` is a strict token
/// prefix of `with_prompt`; otherwise `None`.
pub(crate) fn generation_suffix_from_renders(
    with_prompt: &[u32],
    without_prompt: &[u32],
) -> Option<Arc<[u32]>> {
    (with_prompt.len() > without_prompt.len() && with_prompt.starts_with(without_prompt))
        .then(|| Arc::from(&with_prompt[without_prompt.len()..]))
}

/// The last whole block before the generation suffix: `floor(len - suffix, bs)`.
/// `None` unless `tokens` ends with the (non-empty) suffix.
pub(crate) fn replay_point(tokens: &[u32], suffix: &[u32], block_size: usize) -> Option<usize> {
    if suffix.is_empty() || block_size == 0 || !tokens.ends_with(suffix) {
        return None;
    }
    let body = tokens.len() - suffix.len();
    Some(body - body % block_size)
}

/// Splash `MAX_THINKING_END_TOKENS` (engine `kMaximumThinkingEndTokens`): the most
/// tokens a thinking budget may end thinking with.
pub(crate) const MAX_THINKING_END_TOKENS: usize = 64;

/// What a render writes from the end of `reasoning` through the first `close`
/// after it (Splash `chat_templates._thinking_end`). `None` when the render holds
/// no `reasoning`, or no `close` after it.
pub(crate) fn thinking_end_from_render<'r>(
    rendered: &'r str,
    reasoning: &str,
    close: &str,
) -> Option<&'r str> {
    let start = rendered.find(reasoning)?;
    let end = start + rendered[start..].find(close)?;
    rendered.get(start + reasoning.len()..end + close.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::ChatConfig;
    use crate::tokenizer::{ChatMessage, Qwen3Tokenizer, RenderContextOptions};

    const QWEN38_TEMPLATE: &str =
        include_str!("../tokenizer/fixtures/qwen3.8-27b/chat_template.jinja");
    const QWEN36_TEMPLATE: &str =
        include_str!("../tokenizer/fixtures/qwen3.6-27b/chat_template.jinja");

    struct Fixture {
        dir: std::path::PathBuf,
        tok: Qwen3Tokenizer,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// A word-level vocabulary that keeps `\n` and spaces as their own tokens and
    /// the ChatML markers as added tokens, next to `template`.
    fn fixture(template: &str) -> Fixture {
        fixture_with(
            template,
            &["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
            true,
        )
    }

    /// [`fixture`] over the added tokens `added` (ids from 100). With `reply`,
    /// `<think>`/`</think>` are not special (Qwen's) and decoding fuses the
    /// pieces, so a decoded reply is its exact text.
    fn fixture_with(template: &str, added: &[&str], reply: bool) -> Fixture {
        let dir = write_fixture(template, added, reply);
        let tok = Qwen3Tokenizer::load_from_file_sync(dir.join("tokenizer.json").to_str().unwrap())
            .unwrap();
        Fixture { dir, tok }
    }

    fn write_fixture(template: &str, added: &[&str], reply: bool) -> std::path::PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "mlx-gen-prompt-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let added: Vec<serde_json::Value> = added
            .iter()
            .enumerate()
            .map(|(i, content)| {
                let special = !(reply && content.contains("think>"));
                serde_json::json!({
                    "id": 100 + i, "content": content, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": special,
                })
            })
            .collect();
        let decoder = if reply {
            serde_json::json!({ "type": "Fuse" })
        } else {
            serde_json::Value::Null
        };
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {
                "type": "Split",
                "pattern": { "Regex": "\n|[^\n ]+| " },
                "behavior": "Isolated",
                "invert": false,
            },
            "post_processor": null,
            "decoder": decoder,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0, "\n": 1, " ": 2, "user": 3, "assistant": 4, "Hello": 5,
                    "system": 6, "Hi": 7, "again": 8,
                },
                "unk_token": "<unk>",
            },
        });
        std::fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer_json).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("chat_template.jinja"), template).unwrap();
        dir
    }

    fn encode(tok: &Qwen3Tokenizer, text: &str) -> Vec<u32> {
        tok.encode_sync(text, Some(false)).unwrap()
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: None,
            tool_call_id: None,
            is_error: None,
            reasoning_content: None,
            thinking_enabled: None,
            images: None,
            audio: None,
        }
    }

    fn effort(effort: Option<&str>) -> ChatConfig {
        ChatConfig {
            reasoning_effort: effort.map(str::to_owned),
            ..Default::default()
        }
    }

    /// The suffix the real render appends, and the replay point it gives on a
    /// multi-turn prompt rendered the way a turn renders it.
    fn assert_suffix_and_replay_point(tok: &Qwen3Tokenizer, config: &ChatConfig, expected: &str) {
        let suffix = tok.generation_suffix(config).unwrap().expect("suffix");
        assert_eq!(&*suffix, encode(tok, expected).as_slice());

        let conversation = [
            msg("user", "Hello"),
            msg("assistant", "Hi"),
            msg("user", "Hello again"),
        ];
        let prompt = tok
            .apply_chat_template_with_config(&conversation, true, config, false)
            .unwrap();
        let body = tok
            .apply_chat_template_with_config(&conversation, false, config, false)
            .unwrap();
        assert_eq!(prompt.len(), body.len() + suffix.len());
        assert!(prompt.starts_with(&body));
        let r = replay_point(&prompt, &suffix, 16).expect("replay point");
        assert_eq!(r, body.len() / 16 * 16);
        assert!(r > 0, "fixture prompt must span at least one block");
    }

    #[test]
    fn thinking_on_suffix_and_replay_point() {
        let f = fixture(QWEN38_TEMPLATE);
        assert_suffix_and_replay_point(&f.tok, &effort(None), "<|im_start|>assistant\n<think>\n");
        assert_suffix_and_replay_point(
            &f.tok,
            &effort(Some("low")),
            "<|im_start|>assistant\n<think>\n",
        );
    }

    #[test]
    fn thinking_off_suffix_and_replay_point() {
        let f = fixture(QWEN38_TEMPLATE);
        let suffix = f
            .tok
            .generation_suffix_for(&RenderInputs {
                render_ctx: RenderContextOptions::default(),
                enable_thinking: Some(false),
            })
            .unwrap()
            .expect("suffix");
        assert_eq!(
            &*suffix,
            encode(&f.tok, "<|im_start|>assistant\n<think>\n\n</think>\n\n").as_slice()
        );
    }

    #[test]
    fn reasoning_effort_none_suffix_and_replay_point() {
        let f = fixture(QWEN38_TEMPLATE);
        assert_suffix_and_replay_point(
            &f.tok,
            &effort(Some("none")),
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        );
    }

    #[test]
    fn prompt_without_the_suffix_has_no_replay_point() {
        let f = fixture(QWEN38_TEMPLATE);
        let config = effort(None);
        let suffix = f.tok.generation_suffix(&config).unwrap().expect("suffix");
        let no_gen = f
            .tok
            .apply_chat_template_with_config(&[msg("user", "Hello again")], false, &config, false)
            .unwrap();
        assert!(no_gen.len() > suffix.len() + 16);
        assert_eq!(replay_point(&no_gen, &suffix, 16), None);
        assert_eq!(replay_point(&suffix[1..], &suffix, 16), None);
        assert_eq!(replay_point(&no_gen, &[], 16), None);
    }

    #[test]
    fn template_whose_no_gen_render_is_not_a_prefix_has_no_suffix() {
        let f = fixture(
            "{%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor -%}\
             {%- if add_generation_prompt -%}<|im_start|>assistant\n{%- else -%}system{%- endif -%}",
        );
        assert_eq!(f.tok.generation_suffix(&effort(None)).unwrap(), None);
        assert_eq!(generation_suffix_from_renders(&[1, 2], &[1, 2]), None);
        assert_eq!(generation_suffix_from_renders(&[1], &[1, 2]), None);
    }

    #[test]
    fn suffix_cache_hits_on_repeat_and_misses_on_any_render_input() {
        let f = fixture(QWEN38_TEMPLATE);
        let base = RenderInputs {
            render_ctx: RenderContextOptions::default(),
            enable_thinking: None,
        };
        let first = f.tok.generation_suffix_for(&base).unwrap().unwrap();
        let again = f.tok.generation_suffix_for(&base).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &again), "repeat must hit the cache");
        let via_config = f.tok.generation_suffix(&effort(None)).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &via_config));
        let clone = f.tok.clone();
        let from_clone = clone.generation_suffix_for(&base).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &from_clone));

        // Without `..`, a new render-context field fails to compile here until
        // it gets its own miss case below.
        let RenderContextOptions {
            reasoning_effort: _,
            current_date: _,
            reasoning_strength: _,
            preserve_thinking: _,
        } = RenderContextOptions::default();
        let variants = [
            RenderInputs {
                render_ctx: RenderContextOptions {
                    reasoning_effort: Some("low".into()),
                    ..Default::default()
                },
                ..base.clone()
            },
            RenderInputs {
                render_ctx: RenderContextOptions {
                    current_date: Some("2026-10-29".into()),
                    ..Default::default()
                },
                ..base.clone()
            },
            RenderInputs {
                render_ctx: RenderContextOptions {
                    reasoning_strength: Some("low".into()),
                    ..Default::default()
                },
                ..base.clone()
            },
            RenderInputs {
                render_ctx: RenderContextOptions {
                    preserve_thinking: true,
                    ..Default::default()
                },
                ..base.clone()
            },
            RenderInputs {
                enable_thinking: Some(true),
                ..base.clone()
            },
        ];
        for inputs in &variants {
            let suffix = f.tok.generation_suffix_for(inputs).unwrap().unwrap();
            assert!(
                !Arc::ptr_eq(&first, &suffix),
                "{inputs:?} must miss the cache"
            );
        }
    }

    #[test]
    fn replay_point_floors_to_the_block_size() {
        let suffix = [100, 4, 1];
        let mut tokens: Vec<u32> = vec![5; 50];
        tokens.extend_from_slice(&suffix);
        assert_eq!(replay_point(&tokens, &suffix, 16), Some(48));
        assert_eq!(replay_point(&tokens, &suffix, 32), Some(32));
        let mut tokens: Vec<u32> = vec![5; 64];
        tokens.extend_from_slice(&suffix);
        assert_eq!(replay_point(&tokens, &suffix, 32), Some(64));
        assert_eq!(replay_point(&tokens, &suffix, 0), None);
    }

    #[test]
    fn probe_finds_the_template_close() {
        for template in [QWEN38_TEMPLATE, QWEN36_TEMPLATE] {
            let f = fixture(template);
            let end = f.tok.think_end().thinking_end;
            assert_eq!(&*end, encode(&f.tok, "\n</think>").as_slice());
            assert_eq!(&*end, &[1, f.tok.think_end_id().unwrap()]);
        }
    }

    #[test]
    fn probe_falls_back_and_refuses_a_bad_close() {
        let close = "'\\n</think>\\n\\n' + content";
        assert!(QWEN38_TEMPLATE.contains(close));
        let rewritten = |to: &str| QWEN38_TEMPLATE.replace(close, to);

        let none = fixture(&rewritten("'\\n\\n' + content"));
        assert_eq!(
            &*none.tok.think_end().thinking_end,
            &[none.tok.think_end_id().unwrap()]
        );

        let load = |template: &str| -> String {
            let dir = write_fixture(
                template,
                &["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
                false,
            );
            let loaded =
                Qwen3Tokenizer::load_from_file_sync(dir.join("tokenizer.json").to_str().unwrap());
            let _ = std::fs::remove_dir_all(&dir);
            match loaded {
                Ok(_) => String::from("loaded"),
                Err(error) => error.reason,
            }
        };
        let special = load(&rewritten("'<|im_end|></think>\\n\\n' + content"));
        assert!(
            special.contains("must end with </think> and hold no other special token"),
            "{special}"
        );
        let long = load(&rewritten(&format!(
            "'{}</think>\\n\\n' + content",
            " x".repeat(40)
        )));
        assert!(long.contains("takes 81 tokens; the most is 64"), "{long}");

        let no_close = fixture_with(QWEN38_TEMPLATE, &["<|im_start|>", "<|im_end|>"], false);
        assert!(no_close.tok.think_end().thinking_end.is_empty());
    }

    /// A budget cut, then the next request's re-render of the whole history:
    /// the reply's tokens come back through the forced close, so the paged
    /// prefix hit covers it.
    #[test]
    fn budget_cut_rerender_hits_through_the_close() {
        use crate::engine::backend::ThinkingSetup;
        use crate::engine::penalties::ReasoningTracker;
        use mlx_paged_attn::BlockAllocator;

        const BLOCK: u32 = 4;
        for template in [QWEN38_TEMPLATE, QWEN36_TEMPLATE] {
            let f = fixture_with(
                template,
                &["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
                true,
            );
            let tok = &f.tok;
            let thinking_on = ChatConfig::default();
            let prompt1 = tok
                .apply_chat_template_with_config(&[msg("user", "Hello")], true, &thinking_on, true)
                .unwrap();

            // A loop that observes, then takes; the budget cuts after "again".
            let close = tok.think_end();
            let mut tracker = ReasoningTracker::from_setup(
                &ThinkingSetup {
                    enabled: true,
                    budget: Some(3),
                },
                &close,
            );
            let im_end = encode(tok, "<|im_end|>")[0];
            let close_id = close.think_end_id.unwrap();
            let mut script = [7, 2, 8, 1, 1, 5, im_end].into_iter();
            let mut generated = Vec::new();
            loop {
                let token = match tracker.take_forced_token() {
                    Some(token) => token,
                    None => script.next().unwrap(),
                };
                tracker.observe_token(token);
                generated.push(token);
                if token == im_end {
                    break;
                }
            }

            let raw = tok.decode_sync(&generated, true).unwrap();
            let (content, _, reasoning) = crate::engine::finalize::parse_thinking_and_tools(
                &raw,
                &generated,
                true,
                close.think_end_id,
                close.think_end_str.as_deref(),
                true,
            );
            let reply = ChatMessage {
                reasoning_content: reasoning,
                ..msg("assistant", &content)
            };
            let history = [msg("user", "Hello"), reply, msg("user", "Hi again")];
            let turn2 = tok
                .apply_chat_template_with_config(&history, true, &thinking_on, true)
                .unwrap();

            let mut turn1 = prompt1.clone();
            turn1.extend_from_slice(&generated);
            assert!(turn2.starts_with(&turn1), "{turn1:?}\n{turn2:?}");

            // The last sampled token is never forwarded, so its KV is never cached.
            let cached = &turn1[..turn1.len() - 1];
            let mut allocator = BlockAllocator::new(64, 64, BLOCK);
            let blocks: Vec<_> = (0..cached.len() / BLOCK as usize)
                .map(|_| allocator.allocate().unwrap())
                .collect();
            allocator
                .cache_full_blocks(cached, &blocks, BLOCK, &[], 0)
                .unwrap();
            for block in blocks {
                allocator.free(block);
            }
            let (_, hit) =
                allocator.find_longest_cache_hit(&turn2[..turn2.len() - 1], BLOCK, &[], 0);
            let floor = |n: usize| n / BLOCK as usize * BLOCK as usize;
            assert_eq!(hit, floor(cached.len()));
            let cut = prompt1.len() + 3;
            assert!(hit > floor(cut), "the hit must run past the cut");
            assert_eq!(generated, [7, 2, 8, 1, close_id, 1, 1, 5, im_end]);
        }
    }
}
